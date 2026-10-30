#pragma once
// EqualDepthDir (2026-10-11): a region-scoped SEPARATOR DIRECTORY.
//
// WHAT IT IS. Inside one region: sort the keys, and place G-1 thresholds at equal RANKS
// (threshold j at rank floor(j*n/G), i.e. the first key of its depth band). A key's group is
// the number of thresholds its suffix compares greater-than-or-equal against. Nothing is
// fitted, nothing is learned: the map is the empirical quantiles of the region's own keys.
//
// WHY THIS SHAPE. The BlockLinearRank round measured a concentration failure (its mapping left
// 94 % of groups empty on mixed and collapsed low-entropy keys into one group), and the entry
// diagnosis that followed (RESULTS_THRESHOLD_ENTRY_20261011.md) measured the same regions under
// this candidate: maximum group load 1366/824/546 -> 12/12/12, empty groups -> 0, and the
// thresholds cost 0.6-5.0 KB per region once separators are truncated. The 63-bit COORDINATE
// form of the same idea was measured too and DOES NOT WORK (skew: 64/64 regions with duplicate
// thresholds, max load 1431 against Context16's 824) -- so this file stores separators, not
// coordinates. The TRUNCATED form is this version's single variable (design:
// TRUNCDIR_ENTRY_DESIGN.md): fit(..., truncate_separators = true) stores for each threshold the
// SHORTEST prefix of its suffix that still compares greater than the key immediately to its
// left, which leaves the routing answer of every stored key identical (lemma A of the design)
// while dropping the bytes the binary search never had to read. The default (false) is the
// untruncated directory, byte for byte.
//
// CORRECTNESS ARGUMENT (what the index relies on; the gate checks each):
//   1. Compare is the index's own unsigned-byte order on the SUFFIX (region prefix stripped),
//      which agrees with full-key order because every key the region accepts shares the prefix.
//      The SEARCH must obey the same order, byte by byte: it compares through
//      `static_cast<unsigned char>` on purpose, because a plain char comparison would be
//      signed on x86 and would order every byte >= 0x80 below every byte < 0x80.
//   2. group(q) = #{ j : sep_j <= q } is a count of a fixed, sorted list -- hence non-decreasing
//      in q, and equal q give equal answers. Placement and lookup call the same function of the
//      same coordinate, so a stored key cannot be routed to a different group by the query path.
//   3. Ranks: threshold j is the key at rank floor(j*n/G), so group i holds exactly the keys
//      with ranks in [floor(i*n/G), floor((i+1)*n/G)) -- depth differs by at most one key.
//      With duplicate keys the thresholds can repeat, which leaves groups empty; that is
//      reported rather than hidden, and it cannot be fixed by any map of the suffix alone.
//   4. Total: keys outside [min, max] land in group 0 / group G-1 by the same count, and the
//      answer stays inside [0, G-1].
//
// OWNERSHIP. The directory owns its bytes (one arena plus offsets). It keeps NOTHING that
// points into the caller's keys -- the lifetime rule the LITS round had to fix, applied here
// from the start.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace equivdir {

enum class Status {
    Ok = 0,
    NoKeys,       // n == 0: every prediction is group 0
    TooFewKeys,   // n == 1: one rank, every prediction is group 0
    SingleGroup,  // G == 1: every answer is 0 by definition
};

inline const char* to_string(Status s) {
    switch (s) {
        case Status::Ok: return "ok";
        case Status::NoKeys: return "no_keys";
        case Status::TooFewKeys: return "too_few_keys";
        case Status::SingleGroup: return "single_group";
    }
    return "?";
}

// The index's own order: unsigned bytes, shorter-is-smaller on a tie. Returns <0, 0, >0.
inline int compare(std::string_view a, std::string_view b) {
    const std::size_t n = std::min(a.size(), b.size());
    const int c = n ? std::memcmp(a.data(), b.data(), n) : 0;
    if (c) return c;
    return (a.size() > b.size()) - (a.size() < b.size());
}

inline std::size_t common_prefix(std::string_view a, std::string_view b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

#ifdef E25_DIAG
// P1 (2026-10-11) read-path diagnosis: the traced shadow of predict_group(). Diagnostic
// builds only, so the shipped search keeps exactly one implementation; the diagnosis
// asserts per query that the trace reproduces the production group and the walked-byte
// count, so a drifting shadow fails the run instead of being reported.
struct SearchTrace {
    std::uint32_t probes = 0;           // binary-search iterations
    std::uint64_t compared = 0;         // bytes examined (same definition as `walked`)
    std::uint32_t end_first_diff = 0;   // iterations decided at the first differing byte
    std::uint32_t end_sep_le_q = 0;     // overlap equal, separator <= query
    std::uint32_t end_sep_gt_q = 0;     // overlap equal, separator > query
    std::uint32_t sep_prefix_of_q = 0;  // separator is a strict prefix of the query
};
#endif

class RegionModel {
public:
    static RegionModel fit(const std::string_view* keys, std::size_t n,
                           std::size_t prefix_len, std::uint32_t groups,
                           bool truncate_separators = false) {
        RegionModel m;
        m.groups_ = groups;
        m.keys_ = n;
        m.truncate_ = truncate_separators;
        if (n == 0) { m.status_ = Status::NoKeys; return m; }
        if (n == 1) { m.status_ = Status::TooFewKeys; return m; }
        if (groups <= 1) { m.status_ = Status::SingleGroup; return m; }

        // Defensive: a caller-supplied prefix longer than the keys really share would make the
        // suffix comparison blind to bytes that decide their order.
        m.prefix_len_ = std::min(prefix_len, common_prefix(keys[0], keys[n - 1]));

        // Section 2 of the design, verbatim and with no free parameter: the stored threshold
        // t_j is the shortest prefix of s_j that is still > the key at rank-1; when s_j is
        // itself a prefix of that left key (a run of duplicate suffixes at the band boundary)
        // no shorter prefix can win, so s_j is kept as-is; such cases are reported by the
        // gate's own re-derivation rather than hidden here.
        const std::size_t sep_count = static_cast<std::size_t>(groups) - 1;
        std::vector<std::string_view> stored;
        stored.reserve(sep_count);
        for (std::size_t j = 1; j < groups; ++j) {
            const std::size_t rank = std::min(n - 1, (j * n) / groups);
            const std::string_view suffix = keys[rank].substr(m.prefix_len_);
            std::string_view t = suffix;
            if (m.truncate_ && rank > 0) {
                const std::string_view left = keys[rank - 1].substr(m.prefix_len_);
                const std::size_t c = common_prefix(left, suffix);
                if (c < suffix.size()) t = suffix.substr(0, c + 1);
            }
            stored.push_back(t);
        }

        m.offset_.assign(sep_count + 1, 0);
        std::size_t total = 0;
        for (std::size_t j = 0; j < sep_count; ++j) {
            total += stored[j].size();
            if (total > static_cast<std::size_t>(0xffffffffu))
                throw std::length_error("equal-depth directory: separator arena exceeds 4 GiB");
            m.offset_[j + 1] = static_cast<std::uint32_t>(total);
        }

        m.arena_.resize(total);
        for (std::size_t j = 0; j < sep_count; ++j) {
            const std::string_view t = stored[j];
            if (!t.empty())
                std::memcpy(m.arena_.data() + m.offset_[j], t.data(), t.size());
        }
        m.status_ = Status::Ok;
        return m;
    }

    // One prediction for a full key, plus the WORK the search did: the number of key bytes the
    // comparisons examined. Same field the other backends fill, so `result.steps` keeps its
    // published meaning ("what the model processed") without inventing a new one.
    std::uint32_t predict_group(std::string_view full_key, std::uint32_t& consumed) const {
        if (status_ != Status::Ok) { consumed = 0; return 0; }

        const std::string_view q = full_key.size() > prefix_len_
            ? full_key.substr(prefix_len_) : std::string_view();
        const std::size_t sep_count = offset_.size() - 1;

        std::uint64_t walked = 0;
        std::size_t lo = 0, hi = sep_count;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            const std::string_view sep(
                arena_.data() + offset_[mid], offset_[mid + 1] - offset_[mid]);
            const std::size_t n = std::min(q.size(), sep.size());
            std::size_t i = 0;
            while (i < n && q[i] == sep[i]) ++i;
            walked += i + 1;
            // UNSIGNED bytes, and explicitly so. A plain `sep[i] < q[i]` compares char,
            // which is signed on x86: every byte >= 0x80 would sort below every byte <
            // 0x80, so the count is wrong for stored keys and for probes alike (128/256
            // mismatches and one group inversion on the four-key fixture in
            // check_depth_dir_order). compare() below uses memcmp, which is unsigned by
            // the standard, which is why the fit and the duplicate-threshold check never
            // showed this.
            const bool sep_le_q =
                (i == n) ? (sep.size() <= q.size())
                         : (static_cast<unsigned char>(sep[i]) <
                            static_cast<unsigned char>(q[i]));
            if (sep_le_q) lo = mid + 1;
            else hi = mid;
        }

        consumed = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(walked, 0xffffffffull));
        const std::uint64_t g = lo;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(groups_ - 1, g));
    }

    std::uint32_t predict_group(std::string_view full_key) const {
        std::uint32_t consumed = 0;
        return predict_group(full_key, consumed);
    }

#ifdef E25_DIAG
    std::uint32_t predict_group_traced(std::string_view full_key, std::uint32_t& consumed,
                                       SearchTrace& tr) const {
        if (status_ != Status::Ok) { consumed = 0; return 0; }
        const std::string_view q = full_key.size() > prefix_len_
            ? full_key.substr(prefix_len_) : std::string_view();
        const std::size_t sep_count = offset_.size() - 1;
        std::uint64_t walked = 0;
        std::size_t lo = 0, hi = sep_count;
        while (lo < hi) {
            ++tr.probes;
            const std::size_t mid = lo + (hi - lo) / 2;
            const std::string_view sep(
                arena_.data() + offset_[mid], offset_[mid + 1] - offset_[mid]);
            const std::size_t n = std::min(q.size(), sep.size());
            std::size_t i = 0;
            while (i < n && q[i] == sep[i]) ++i;
            walked += i + 1;
            bool sep_le_q = false;
            if (i == n) {
                sep_le_q = sep.size() <= q.size();
                if (sep_le_q) ++tr.end_sep_le_q; else ++tr.end_sep_gt_q;
                if (sep.size() < q.size()) ++tr.sep_prefix_of_q;
            } else {
                sep_le_q = static_cast<unsigned char>(sep[i]) <
                           static_cast<unsigned char>(q[i]);
                ++tr.end_first_diff;
            }
            if (sep_le_q) lo = mid + 1;
            else hi = mid;
        }
        tr.compared = walked;
        consumed = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(walked, 0xffffffffull));
        const std::uint64_t g = lo;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(groups_ - 1, g));
    }
#endif

    Status status() const { return status_; }
    bool fitted() const { return status_ == Status::Ok; }
    // Whether this directory stores the truncated separators of section 2.
    bool truncated() const { return truncate_; }
    // Stored threshold idx (0 .. threshold_count()-1), i.e. j = idx + 1 of the design.
    std::string_view separator(std::size_t idx) const {
        if (idx + 1 >= offset_.size()) return std::string_view();
        return std::string_view(arena_.data() + offset_[idx], offset_[idx + 1] - offset_[idx]);
    }
    // Arena bytes actually stored (the truncated figure the round reports).
    std::size_t separator_bytes() const { return arena_.size(); }
    std::uint32_t groups() const { return groups_; }
    std::size_t keys() const { return keys_; }
    std::size_t prefix_len() const { return prefix_len_; }
    std::size_t threshold_count() const {
        return offset_.empty() ? 0 : offset_.size() - 1;
    }
    std::size_t owned_bytes() const {
        return arena_.capacity() + offset_.capacity() * sizeof(std::uint32_t);
    }

    // Identity of the mapping this directory implements; two directories with equal identities
    // answer identically for every key.
    std::uint64_t identity() const {
        std::uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ULL;
            }
        };
        const std::uint64_t st = static_cast<std::uint64_t>(status_);
        const std::uint64_t tr = truncate_ ? 1u : 0u;
        const std::uint64_t n = keys_;
        const std::uint64_t g = groups_;
        const std::uint64_t p = prefix_len_;
        mix(&st, sizeof(st));
        mix(&tr, sizeof(tr));
        mix(&n, sizeof(n));
        mix(&g, sizeof(g));
        mix(&p, sizeof(p));
        if (!arena_.empty()) mix(arena_.data(), arena_.size());
        for (std::uint32_t off : offset_) mix(&off, sizeof(off));
        return h;
    }

private:
    std::vector<char> arena_;
    std::vector<std::uint32_t> offset_;   // sep_count + 1 entries, offsets into arena_
    std::size_t prefix_len_ = 0;
    std::size_t keys_ = 0;
    std::uint32_t groups_ = 0;
    Status status_ = Status::NoKeys;
    bool truncate_ = false;
};

}  // namespace equivdir
