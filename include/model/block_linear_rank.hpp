#pragma once
// BlockLinearRank (2026-10-11): a region-scoped RANK model for this index.
//
// ORIGIN AND WHAT IS PORTED. The algorithm is the teammate's BlockLinearRank-style
// model, ported from `Zephyr761/Learned-Index-Model` @ 60837a88:
//   * `src/key_bytes.h::encode` -- the 9-bit, EOS-terminated, order-preserving
//     coordinate (byte values become 1..256, end-of-string becomes 0, at most
//     7 symbols so the coordinate fits in 63 bits);
//   * `src/block_model.cpp::BlockModel` -- the segmented linear fit and the
//     lower-bound segment lookup.
//
// WHAT IS DELIBERATELY NOT PORTED (first integration round, see
// docs/NEXT_ACTION_PLAN_20261011_BLOCK_MODEL.md section 4.1):
//   * the parameter search over block size / encoding width / segment count --
//     the fit here is FIXED: block_size = 1, encoding_bytes = 7, segments = 8;
//   * the residual tags, the delta maintenance and the candidate window search;
//   * `BlockModel::prefix_`, which is a `string_view` INTO the training keys. This
//     model keeps a prefix LENGTH (a number) and copies nothing that could dangle.
//
// WHY block_size = 1 AND WHAT IT MEANS. With block_size = 1 the model's y axis is
// the RANK of a key inside the region's sorted key array, so the model predicts a
// position, not a block. The group id the index needs is then a fixed integer map
// of that rank:  g = min(G-1, floor(p * G / N)).
//
// WHY THIS MAP IS ORDER-PRESERVING (the property the index relies on; the group id
// of a key must be non-decreasing in key order, or a group's records do not form a
// contiguous range and the sub-block boundary directory is meaningless):
//   1. `encode` is non-decreasing: it is a lexicographic encoding of the first
//      `min(7, suffix length)` bytes of the SUFFIX, with EOS = 0 below every real
//      byte (which is `byte + 1 >= 1`), and the suffix comparison agrees with the
//      full-key comparison because every key in a region shares the region prefix
//      (Region::prefix_matches refuses the rest).
//   2. Inside a segment, y = clamp(intercept + slope * dx, y_min, y_max) is
//      non-decreasing in x, because slope >= 0 by construction (x_max >= x_min and
//      y_max >= y_min are both guaranteed by the sorted input).
//   3. Across segments, y_max(i) = end_i - 1 and y_min(i+1) = end_i, so the
//      clamped ranges are disjoint and ordered: a key in segment i+1 can never
//      predict below a key in segment i. (Segments never split equal coordinates,
//      so the lower-bound lookup cannot return the wrong segment for a shared x.)
//   4. floor(p * G / N) and min(G-1, .) are non-decreasing in p.
//
// DOMAIN. Every byte string is supported, including the empty key, embedded NUL
// bytes and bytes >= 0x80 (everything is read as `unsigned char`). There is no
// refusal path: this is what lets the backend cover a whole region with no
// Context16 fallback, unlike the pinned LITS adapter.
//
// DETERMINISM. `fit` is a pure function of (keys, n, prefix_len, groups): no
// search, no floating-point mode switches, no dependence on addresses.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace blockrank {

// Why a region's model is in one state or another. Only Ok means "fitted from this
// region's own keys"; the other three are DEFINED BOUNDARY CASES whose answer is
// group 0 for every key, and they are counted rather than hidden.
enum class Status {
    Ok = 0,
    NoKeys,        // n == 0: nothing to fit, every prediction is 0
    TooFewKeys,    // n == 1: one rank, every prediction is 0
    SingleGroup,   // G == 1: every answer is 0 by definition
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

struct Segment {
    std::uint64_t x_min = 0, x_max = 0;
    // Rank bounds, inclusive. y_max(i) + 1 == y_min(i+1) across segments.
    std::uint32_t y_min = 0, y_max = 0;
    double slope = 0.0, intercept = 0.0;
};

inline std::size_t common_prefix(std::string_view a, std::string_view b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// 9-bit symbols, EOS = 0, real byte b becomes b + 1. Truncation after `bytes`
// symbols makes the coordinate NON-INJECTIVE (two keys sharing those symbols
// collide) but never non-monotone. `bytes` above 7 would overflow 63 bits, so the
// loop is clamped; `fit` only ever asks for 7.
inline std::uint64_t encode(std::string_view key, std::size_t prefix, unsigned bytes) {
    if (bytes > 7) bytes = 7;
    std::uint64_t x = 0;
    for (unsigned i = 0; i < bytes; ++i) {
        const std::size_t at = prefix + i;
        x = (x << 9) | (at < key.size()
            ? std::uint64_t(static_cast<unsigned char>(key[at])) + 1
            : 0);
    }
    return x;
}

class RegionModel {
public:
    static constexpr unsigned kEncodeBytes = 7;
    static constexpr std::size_t kSegmentBudget = 8;

    // `keys` must be in ascending unsigned-byte order and `prefix_len` must be a
    // common prefix of every key (the region's own prefix length is exactly that).
    // A longer prefix_len than the keys really share would make the encoding blind
    // to bytes that decide their order, so it is clamped down to the common prefix
    // of the first and last key -- a defensive reduction that can only make the map
    // see MORE bytes, never fewer.
    static RegionModel fit(const std::string_view* keys, std::size_t n,
                           std::size_t prefix_len, std::uint32_t groups) {
        RegionModel m;
        m.groups_ = groups;
        m.keys_ = n;
        if (n == 0) { m.status_ = Status::NoKeys; return m; }
        if (n == 1) { m.status_ = Status::TooFewKeys; return m; }
        if (groups <= 1) { m.status_ = Status::SingleGroup; return m; }

        m.prefix_len_ = std::min(prefix_len, common_prefix(keys[0], keys[n - 1]));

        std::vector<std::uint64_t> xs(n);
        for (std::size_t i = 0; i < n; ++i)
            xs[i] = encode(keys[i], m.prefix_len_, kEncodeBytes);

        // Segment split exactly as the source's fit: target size ceil(n / budget),
        // and a run of EQUAL coordinates is never split -- `segment()` finds a
        // coordinate's predictor by its x_max, so two segments sharing an x would
        // make the answer depend on which one the search happened to return.
        std::size_t first = 0;
        while (first < n) {
            const std::size_t target =
                std::max<std::size_t>(1, (n + kSegmentBudget - 1) / kSegmentBudget);
            std::size_t end = std::min(n, first + target);
            while (end < n && xs[end] == xs[end - 1]) ++end;

            Segment s;
            s.x_min = xs[first];
            s.x_max = xs[end - 1];
            s.y_min = static_cast<std::uint32_t>(first);
            s.y_max = static_cast<std::uint32_t>(end - 1);
            s.intercept = s.x_min == s.x_max
                ? (double(s.y_min) + double(s.y_max)) / 2.0
                : double(s.y_min);
            s.slope = s.x_min == s.x_max
                ? 0.0
                : double(s.y_max - s.y_min) / double(s.x_max - s.x_min);
            m.segments_.push_back(s);
            first = end;
        }

        m.status_ = Status::Ok;
        return m;
    }

    // One prediction for a full key, plus the number of SUFFIX bytes the model
    // examined (the `steps` quantity the index reports; not a hash contribution --
    // this model does not feed the fingerprint, so callers must hash the whole
    // suffix themselves, which is what reproduces the stored tag).
    std::uint32_t predict_group(std::string_view full_key, std::uint32_t& consumed) const {
        if (status_ != Status::Ok) { consumed = 0; return 0; }

        const std::size_t suffix_bytes =
            full_key.size() > prefix_len_ ? full_key.size() - prefix_len_ : 0;
        consumed = static_cast<std::uint32_t>(
            std::min<std::size_t>(kEncodeBytes, suffix_bytes));

        const std::uint64_t x = encode(full_key, prefix_len_, kEncodeBytes);
        const Segment& s = segment(x);

        // Integer subtraction BEFORE the conversion to double: a full 63-bit
        // coordinate loses its low bits in a double, and the difference does not.
        const double dx = x >= s.x_min ? double(x - s.x_min) : -double(s.x_min - x);
        double y = s.intercept + s.slope * dx;
        if (!(y > double(s.y_min))) y = double(s.y_min);   // also catches NaN
        if (y > double(s.y_max)) y = double(s.y_max);

        const std::uint64_t p = static_cast<std::uint64_t>(std::llround(y));
        const std::uint64_t g =
            std::min<std::uint64_t>(groups_ - 1, p * groups_ / keys_);
        return static_cast<std::uint32_t>(g);
    }

    std::uint32_t predict_group(std::string_view full_key) const {
        std::uint32_t consumed = 0;
        return predict_group(full_key, consumed);
    }

    Status status() const { return status_; }
    bool fitted() const { return status_ == Status::Ok; }
    std::uint32_t groups() const { return groups_; }
    std::size_t keys() const { return keys_; }
    std::size_t prefix_len() const { return prefix_len_; }
    std::size_t segment_count() const { return segments_.size(); }
    const std::vector<Segment>& segments() const { return segments_; }

    // Resident bytes this model owns (its own vector only; nothing is borrowed).
    std::size_t owned_bytes() const {
        return segments_.capacity() * sizeof(Segment);
    }

    // Identity of the fitted mapping, for the index's preset guard: two models with
    // equal identities answer the same for every key. FNV-1a over the scalar state
    // and the segment table, in order; deliberately independent of addresses and of
    // anything the fit did not read.
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
        const std::uint64_t n = keys_;
        const std::uint64_t g = groups_;
        const std::uint64_t p = prefix_len_;
        const std::uint64_t b = kEncodeBytes;
        const std::uint64_t sb = kSegmentBudget;
        mix(&st, sizeof(st));
        mix(&n, sizeof(n));
        mix(&g, sizeof(g));
        mix(&p, sizeof(p));
        mix(&b, sizeof(b));
        mix(&sb, sizeof(sb));
        for (const auto& s : segments_) {
            mix(&s.x_min, sizeof(s.x_min));
            mix(&s.x_max, sizeof(s.x_max));
            mix(&s.y_min, sizeof(s.y_min));
            mix(&s.y_max, sizeof(s.y_max));
            mix(&s.slope, sizeof(s.slope));
            mix(&s.intercept, sizeof(s.intercept));
        }
        return h;
    }

private:
    // First segment whose x_max is >= x, else the last. Same lower-bound shape as
    // the source; a key below every segment's range lands on segment 0 and clamps
    // to rank 0, a key above it lands on the last segment and clamps to rank n-1,
    // so the answer stays inside [0, n-1] for EVERY input.
    const Segment& segment(std::uint64_t x) const {
        auto it = std::lower_bound(
            segments_.begin(), segments_.end(), x,
            [](const Segment& s, std::uint64_t v) { return s.x_max < v; });
        return it == segments_.end() ? segments_.back() : *it;
    }

    std::vector<Segment> segments_;
    std::size_t prefix_len_ = 0;
    std::uint32_t groups_ = 0;
    std::size_t keys_ = 0;
    Status status_ = Status::NoKeys;
};

}  // namespace blockrank
