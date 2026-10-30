#pragma once
// E38 / P1.2: the LITS HPT adapter.
//
// WHAT THIS IS. A thin, checked layer over the PINNED LITS HPT that turns it into
// the one thing this index needs from a model: a logical group id for a full key, given a
// region. It is not a reimplementation -- every position and CDF value comes out of
// vendor/lits_model_safe/lits_model.hpp, the minimal-safety-patched copy of the pinned
// commit. E37's lesson applies directly here: a dry-run model and the mechanism that
// implements it must share ONE definition of the constraint, and the same algorithm written
// twice WILL drift. So this file computes nothing the source already computes.
//
// THE COORDINATE. The source's prediction and CDF loops start at `i = gcpl` and read
// `key[i - 1]`, where `gcpl` is the ABSOLUTE position at which the node's own partial key
// begins -- lits_node.hpp:292 passes `ccpl + icpl`, and lits_node.hpp:365 calibrates with the
// same value. This index's regions have exactly that quantity: the region's own common
// prefix length. So the adapter passes the FULL key and the REGION PREFIX LENGTH, and never
// a suffix -- LITS_AUDIT section 11.2 works through the four ways to get this wrong.
//
// WHY THE WALK IS RE-EXPRESSED, AND WHAT KEEPS IT HONEST.
//
// The first version of this adapter called `lits::HPT::getPos` directly, which forced the entry
// point to take `const std::string&` -- `getPos` takes `char *` and stops at the first NUL, so
// only a terminated buffer is safe. That worked for P1's gate, whose keys are owning strings,
// but it cannot serve the index: `Index::find`, `Region::find_impl`, the write paths and the
// scan start all carry `std::string_view`, and the task book's section 7 is explicit that the
// adapter is to use "a length-carrying view and unsigned bytes, not assuming a terminator
// behind it".
//
// So the adapter now takes a view and walks it with an explicit end. The calibration still
// calls the source's `getCdf` (its keys ARE owning strings), and only the position walk is
// re-expressed -- and the statement "it computes what `getPos` computes" is a MEASUREMENT
// rather than a claim: `check_lits_model` runs this walk and the source's own `getPos` over
// the same keys, every key of every real region plus the absent probes and the boundary cases,
// and fails on a single disagreement. That is also the answer to E37's lesson about a dry run
// drifting from the mechanism that implements it.
//
// WHAT THIS FILE DELIBERATELY DOES NOT DO.
//   * No fingerprint fusion. The model reads bytes; it does not advance anyone's hash state.
//     P1.3 gives the fingerprint its own field for exactly this reason.
//   * No table-layout change. The HPT's shape, masks, weight array and early stop are the
//     pinned source's, and P1 step 1 forbids touching them in the same step as the fix.
//   * No online learning. One model per index version; see task book section 8.

// lits_model.hpp includes ONLY lits_base.hpp, but its train() calls ucpl() and ustrlen(),
// which live in lits_gpkl.hpp. The upstream convention is that lits.hpp pulls in the whole
// set in the right order; this adapter needs only these two, and the order matters, so the
// dependency that lits_model.hpp leaves implicit is made explicit here.
#include "../vendor/lits_model_safe/lits_gpkl.hpp"
#include "../vendor/lits_model_safe/lits_model.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lits_hpt {

// ---------------------------------------------------------------------------
// The supported input domain.
//
// Two independent restrictions, both REFUSED rather than worked around, and both measured
// over every population of all three datasets in LITS_AUDIT section 8:
//
//   bytes >= 128   lits_model.hpp:170 indexes the table with a raw `char` while the training
//                  pass at :120 uses `unsigned char`. On a signed-char target a high byte is
//                  a NEGATIVE array index. A real defect in the pinned source, not a
//                  hypothetical (LITS_AUDIT section 11.4a). Measured maximum byte over the
//                  three corpora is 126, so the round never reaches it -- and if a future
//                  corpus does, the adapter refuses instead of reading out of bounds.
//
//   embedded NUL   The source's loops stop at the first NUL, so a NUL inside the key would
//                  silently shorten what the model sees. The task book's first round is
//                  "no embedded NUL" and this is that condition, enforced. Note that
//                  data.hpp::make_queries appends '\0' and is therefore OUT; the E34-E37
//                  family appends "~absent" and is in.
// ---------------------------------------------------------------------------

inline const char* unsupported_reason(std::string_view k) {
    for (std::size_t i = 0; i < k.size(); ++i) {
        const auto c = static_cast<unsigned char>(k[i]);

        if (c == 0) return "embedded NUL";
        if (c >= 128) return "byte >= 128";
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Why a region has no usable model. Every one of these is a NORMAL outcome that the caller
// must handle, not an exception -- the task book's section 3.4 lists them, and the pinned
// source treats the two calibration ones as `goto FAIL_TO_BULK` rather than as errors
// (lits_node.hpp:367,397). Refusing to fake a success is the requirement.
// ---------------------------------------------------------------------------

enum class Status {
    Ok = 0,
    NoKeys,               // no keys at all: nothing to calibrate on
    TooFewKeys,           // fewer than two keys: the pinned train()/calibrate() need two
    SingleGroup,          // G == 1: the answer is 0 by definition, no local division
    KeyShorterThanPrefix, // a key does not even contain this region's prefix
    KeyLongerThanLimit,   // longer than the hard ceiling kMaxKeyBytes
    UnsupportedByte,      // see unsupported_reason
    TooFewToTrain,        // the shared table needs at least two keys: train() reads keys[1]
    DegenerateCdf,        // hi <= lo -- the source's own FAIL_TO_BULK
    NonFinite,            // a coefficient that is inf or nan
    CoefficientTooLarge,  // refused by the CONSERVATIVE ADMISSION BOUND, not by the safety guard
    TooManyGroups,        // G does not fit the source's `int` size parameter
};

inline const char* to_string(Status s) {
    switch (s) {
        case Status::Ok: return "ok";
        case Status::NoKeys: return "no_keys";
        case Status::TooFewKeys: return "too_few_keys";
        case Status::SingleGroup: return "single_group";
        case Status::KeyShorterThanPrefix: return "key_shorter_than_prefix";
        case Status::KeyLongerThanLimit: return "key_longer_than_limit";
        case Status::UnsupportedByte: return "unsupported_byte";
        case Status::TooFewToTrain: return "too_few_to_train";
        case Status::DegenerateCdf: return "degenerate_cdf";
        case Status::NonFinite: return "non_finite";
        case Status::CoefficientTooLarge: return "coefficient_too_large";
        case Status::TooManyGroups: return "too_many_groups";
    }

    return "unknown";
}

// ---------------------------------------------------------------------------
// THE ADMISSION BOUND -- an admission policy, and NOT the safety mechanism.
//
// WHAT IT IS. From the walk's own recurrence,
//
//     c = size * beta                                 once
//     c += ps * uni.CDF                               per byte, with ps starting at size * alpha,
//                                                     only ever decreasing, and CDF <= 1
//
// so with L the bytes the walk covers,
//
//     |c|  <=  |size * beta|  +  L * |size * alpha|
//
// and the rule below refuses a region whose coefficients would let that exceed the limit for the
// longest key the region will be looked up with. `kMaxKeyBytes` is the hard ceiling on any key
// the adapter will look at, so the loop length is bounded by a number this file can name.
//
// WHAT IT IS NOT. It is NOT what makes the integer conversion defined, and it must not be
// described as such. That is done in `predict_group` on the ACTUAL value: a non-finite
// accumulator is refused, and a finite one is saturated in the DOUBLE domain against the group
// range before any conversion happens. A static, per-region decision cannot be the only
// protection for a runtime quantity -- a query key, an absent probe or a later insert may be
// longer than anything the region stored, and the reviewer's point is exactly that.
//
// WHY IT IS KEPT ANYWAY. A very large `|G * alpha|` means a very small CDF span, and the mapping
// over such a region is numerically ill-conditioned whatever the saturation then does with it.
// Refusing the region gives it a well-defined reference-model layout instead of an arbitrary
// one. That is a QUALITY decision, and it is registered as one: a refused region is counted,
// reported, and takes the Context16 fallback rather than disappearing.
//
// IT DOES CHANGE WHAT THE LITS ARM COVERS -- three regions of the 256 real ones, on mixed. That
// is a property of this admission rule and not of the model, and it has to be reported as such.
inline constexpr std::size_t kMaxKeyBytes = 4096;

// Nine orders of magnitude below int's range, which is now irrelevant to safety: the number is
// an admissions threshold, chosen so that an ill-conditioned region is caught long before the
// saturation would have to do anything.
inline constexpr double kMaxAccumulator = 1.0e9;

// ---------------------------------------------------------------------------
// THE POSITION WALK, re-expressed with a LENGTH instead of a NUL terminator.
//
// The pinned `HPT::getPos` takes `char *` and stops at the first NUL, so it can only be called
// on a terminated buffer. This index's query and write paths carry `std::string_view`, and the
// task book's section 7 says the adapter is to use "a length-carrying view and unsigned bytes,
// not assuming a terminator behind it" -- so the walk is written here with an explicit end.
//
// WHY THIS IS NOT A SECOND COPY OF THE ALGORITHM. The calibration still calls the source's
// `getCdf` directly (its keys are owning strings). Only the walk below is re-expressed, and the
// statement "it computes what getPos computes" is not left to inspection: check_lits_model
// compares this walk against the source's own `getPos` key by key, over every key of every real
// region plus the absent probes and the boundary cases. E37's lesson about a dry run and its
// mechanism drifting apart is answered by that comparison, not by keeping two solvers.
//
// The operation order and the early stop are the source's, verbatim in meaning:
//
//     double ps = size * k;                       lits_model.hpp:166-167
//     double c  = size * b;
//     for (i = gcpl; <not past the end> && ps >= 1; ++i) {
//         const auto &uni = m[i & PS_MASK][<-1 if i==0 else key[i-1]> & FC_MASK][key[i]];
//         c += ps * uni.CDF;   ps *= uni.PRO;
//     }
//
// TWO DELIBERATE DIFFERENCES, both confined to bytes this adapter REJECTS anyway:
//
//   * the character indexes with `unsigned char`, where the source uses a bare `char`. For a
//     byte < 128 the two are the same number; for a byte >= 128 the source's is negative
//     (LITS_AUDIT section 11.4a). Since the adapter refuses those keys, the walk never sees one.
//   * the walk is bounded by the view's length rather than by a NUL, which is the whole point.
// ---------------------------------------------------------------------------

// Returns the position as a DOUBLE, and that is the whole point: the caller saturates in the
// float domain and only then converts. A saturation applied to an `int` cannot repair a
// conversion that has already happened, and the source's own `getPos` ends with exactly such a
// conversion (`static_cast<int>(c)`, lits_model.hpp:175).
inline double walk_position(const lits::HPT& hpt, std::string_view key, std::size_t gcpl,
                            int size, double k, double b, std::uint32_t& walked) {
    double ps = double(size) * k;
    double c = double(size) * b;

    std::size_t i = gcpl;
    for (; i < key.size() && ps >= 1.0; ++i) {
        const auto ch = static_cast<unsigned>(static_cast<unsigned char>(key[i]));
        const unsigned fc = (i == 0)
            ? 0u
            : static_cast<unsigned>(static_cast<unsigned char>(key[i - 1])) & lits::HPT::FC_MASK;

        const auto& uni = hpt.m[i & lits::HPT::PS_MASK][fc][ch];

        c += ps * uni.CDF;
        ps *= uni.PRO;
    }

    // The bytes the walk actually COUNTED, which is not the same as the key's length: the loop
    // stops early once `ps < 1`, and that is the model's real processing count.
    walked = static_cast<std::uint32_t>(i - gcpl);

    return c;
}

// The bound the guard enforces, exposed so the gate can report it instead of re-deriving it.
// `walk_bytes` is the number of bytes the position loop may walk for the longest key that will
// be looked up in this region.
inline double accumulator_bound(std::uint32_t groups, double alpha, double beta,
                                std::size_t walk_bytes) {
    const double g = double(groups);

    return std::abs(g * alpha) * double(walk_bytes + 1) + std::abs(g * beta);
}

// ---------------------------------------------------------------------------
// The one shared table.
//
// The task book's section 2.2: one read-only HPT for the whole index, trained from the
// initial full key set, shared by every region -- NOT one 2 MiB table per region. Ownership
// is explicit and exclusive here; Index holds a unique_ptr in the wiring stage.
//
// The pinned `HPT` owns 1,024 separate `new UNI[128]` blocks plus a 32x32 pointer array
// (lits_model.hpp:44,53), and its destructor calls destroy(). So it must never be copied,
// and it must never be destroy()ed by hand -- a second release would free the same blocks
// twice. Holding it by unique_ptr and deleting both copy operations is what makes those two
// mistakes unrepresentable rather than merely discouraged.
// ---------------------------------------------------------------------------

class SharedModel {
public:
    // Train once from the initial, sorted, de-duplicated FULL key set.
    //
    // Throws std::invalid_argument when the key set cannot be trained on at all (fewer than
    // two keys, or an unsupported byte). A refusal that can be ignored is not a refusal, and
    // silently dropping the offending keys would turn a domain restriction into a silent
    // truncation -- which the task book forbids.
    static std::unique_ptr<SharedModel> train(const std::vector<std::string>& sorted_keys);

    // The form the index holds: a shared, immutable handle. `Index` uses this so that every
    // region, and the statistics replay, reference ONE object rather than each training an
    // identical one -- which is the difference between a guarantee and a coincidence.
    static std::shared_ptr<const SharedModel> train_shared(const std::vector<std::string>& sorted_keys) {
        return std::shared_ptr<const SharedModel>(train(sorted_keys));
    }

    ~SharedModel() = default;

    SharedModel(const SharedModel&) = delete;
    SharedModel& operator=(const SharedModel&) = delete;

    const lits::HPT& hpt() const { return *hpt_; }

    std::size_t trained_keys() const { return trained_; }

    // The prefix the TRAINING pass skipped, `ucpl(keys[0], keys[len-1])` over the whole key
    // set (lits_model.hpp:92). It is not the region prefix and must not be confused with it:
    // every region prefix is at least this long, because a region's keys are a subset of the
    // training keys. RegionModel asserts that relation.
    int training_common_prefix() const { return train_gcpl_; }

    // The table's payload only: 32 * 32 * 128 * sizeof(UNI). The task book's section 11.3
    // requires this to be kept apart from the full allocation -- the 1,024 separate
    // allocations and the pointer array are real bytes this figure does not contain.
    std::size_t table_payload_bytes() const { return table_bytes_; }

    std::size_t table_allocations() const { return 32 * 32; }

private:
    SharedModel() = default;

    std::unique_ptr<lits::HPT> hpt_;
    std::size_t trained_ = 0;
    int train_gcpl_ = 0;
    std::size_t table_bytes_ = 0;
};

// ---------------------------------------------------------------------------
// One region's view of the shared table.
//
// This is the whole "region model": a pointer to the shared table plus the three numbers the
// source's node carries (its partial-key offset, its local slope and intercept) and the
// group count. No table is copied and no per-region HPT exists.
//
// The calibration is the source's, verbatim in meaning (lits_node.hpp:365-370):
//
//     lo = getCdf(first, p)          alpha = 1 / (hi - lo)
//     hi = getCdf(last,  p)          beta  = -lo / (hi - lo)
//
// and `hi <= lo` is the source's FAIL_TO_BULK. It is NOT re-derived by a least-squares fit;
// the task book is explicit that this is endpoint calibration.
// ---------------------------------------------------------------------------

struct RegionModel {
    const SharedModel* shared = nullptr;
    std::size_t prefix_length = 0;
    std::uint32_t groups = 0;
    double alpha = 1.0;
    double beta = 0.0;
    Status status = Status::NoKeys;

    // The DEFINED BOUNDARY PATH of task book section 3.4: a region with fewer than two keys,
    // or with G == 1, is never handed to the calibration at all -- the pinned `train` reads
    // `keys[1]` and `keys[len-2]`, and there is nothing to divide by. Such a region's answer
    // is group 0, and that is stated by a factory rather than reached by passing degenerate
    // arguments into `calibrate`, so a caller cannot arrive here by accident.
    static RegionModel boundary(std::uint32_t groups, Status why);

    // `region_keys` is the number of LIVE keys in the region, passed explicitly because the
    // two-key requirement is about the region's population and not about the two strings:
    // `calibrate(shared, k, k, ...)` with a single-key region would otherwise be
    // indistinguishable from a degenerate two-key region.
    //
    // `region_max_key_bytes` is the longest FULL key in the region, and it is what makes the
    // coefficient guard tight instead of conservative: the accumulation bound grows with the
    // bytes the position loop may walk, and the loop walks a key. A generic upper limit here
    // would refuse regions that are in fact safe.
    static RegionModel calibrate(const SharedModel& shared,
                                 const std::string& first,
                                 const std::string& last,
                                 std::size_t region_prefix,
                                 std::uint32_t groups,
                                 std::size_t region_keys,
                                 std::size_t region_max_key_bytes);

    bool usable() const {
        return status == Status::Ok || status == Status::SingleGroup ||
               status == Status::TooFewKeys || status == Status::NoKeys;
    }

    // True when the region has no calibrated coefficients at all -- the caller must not
    // expect `alpha`/`beta` to mean anything.
    bool is_boundary() const { return shared == nullptr || status != Status::Ok; }

    // The logical group of a full key.
    //
    // The offset and the clamp are the source's own position arithmetic, with the two
    // sentinel slots removed -- the task book's section 3.3:
    //
    //   source  pos = getPos(key, len - 2, gcpl, k, b) + 1
    //           index = max(min(pos, len - 2), 1)              valid indices [0, len-1]
    //   ours    group = max(min(getPos(key, G, p, a, b), G - 1), 0)   valid groups [0, G-1]
    //
    // Substituting len = G + 2 turns the source's line into `max(min(getPos(key,G,p,a,b)+1,
    // G), 1) - 1`, which is exactly ours. check_lits_model asserts that equality key by key
    // rather than trusting this paragraph.
    //
    // NOTE ON ORDER: the clamp here is NOT the safety mechanism. `getPos` converts to int
    // internally, so the coefficient bound enforced at calibration is what keeps that
    // conversion defined; this clamp only removes the sentinel offsets.
    // Takes a VIEW, not an owning string: this index's query, write and scan paths all carry
    // `std::string_view`, and the walk above is bounded by the view's length rather than by a
    // terminator, so nothing here needs the bytes to be followed by a NUL.
    std::uint32_t predict_group(std::string_view full_key) const;

    // Same answer, and reports how many bytes of the key the walk actually CONSUMED. That count
    // is what `steps` means -- the model's own processing -- and it is deliberately NOT the
    // fingerprint offset, which is a separate field on the index's Prediction.
    std::uint32_t predict_group(std::string_view full_key, std::uint32_t& walked) const;
};

// ---------------------------------------------------------------------------
// implementation
// ---------------------------------------------------------------------------

inline std::unique_ptr<SharedModel> SharedModel::train(const std::vector<std::string>& sorted_keys) {
    const std::size_t n = sorted_keys.size();

    if (n < 2) {
        throw std::invalid_argument("lits_hpt: the shared table needs at least two keys "
                                    "(the pinned train() reads keys[1])");
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("lits_hpt: key count does not fit the source's `int len`");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (const char* why = unsupported_reason(sorted_keys[i])) {
            throw std::invalid_argument("lits_hpt: key " + std::to_string(i) + " is outside the "
                                        "supported domain (" + why + ")");
        }
    }

    auto model = std::unique_ptr<SharedModel>(new SharedModel());

    model->hpt_.reset(new lits::HPT());

    // `train` takes `const str *` = `char *const *` and only READS through it: ucpl() and
    // ustrlen() (lits_gpkl.hpp:14,35) are plain byte loops with no stores. The const_cast is
    // the usual accommodation for a C-era interface that is not const-correct; it is
    // confined to this block and nothing here writes through it.
    std::vector<char*> ptrs;
    ptrs.reserve(n);

    for (const auto& k : sorted_keys) {
        ptrs.push_back(const_cast<char*>(k.c_str()));
    }

    model->hpt_->train(ptrs.data(), static_cast<int>(n));

    model->trained_ = n;
    model->train_gcpl_ = lits::ucpl(const_cast<char*>(sorted_keys.front().c_str()),
                                    const_cast<char*>(sorted_keys.back().c_str()));
    model->table_bytes_ = model->hpt_->model_size();

    return model;
}

inline RegionModel RegionModel::boundary(std::uint32_t groups, Status why) {
    RegionModel m;
    m.groups = groups;
    m.status = why;

    return m;
}

inline RegionModel RegionModel::calibrate(const SharedModel& shared,
                                          const std::string& first,
                                          const std::string& last,
                                          std::size_t region_prefix,
                                          std::uint32_t groups,
                                          std::size_t region_keys,
                                          std::size_t region_max_key_bytes) {
    // The boundary cases are checked FIRST and against the region's POPULATION, because that
    // is what decides whether the pinned calibration functions may be called at all.
    if (groups == 0 || region_keys == 0) return boundary(groups, Status::NoKeys);
    if (region_keys < 2) return boundary(groups, Status::TooFewKeys);
    // Section 3.4: G == 1 needs no local division, and its answer is 0.
    if (groups == 1) return boundary(groups, Status::SingleGroup);
    if (groups > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return boundary(groups, Status::TooManyGroups);
    }

    RegionModel m;
    m.shared = &shared;
    m.prefix_length = region_prefix;
    m.groups = groups;

    for (const std::string* k : {&first, &last}) {
        if (unsupported_reason(*k) != nullptr) {
            m.status = Status::UnsupportedByte;
            return m;
        }
        if (k->size() < region_prefix) {
            m.status = Status::KeyShorterThanPrefix;
            return m;
        }
    }

    const auto& hpt = shared.hpt();

    const double lo = hpt.getCdf(const_cast<char*>(first.c_str()), static_cast<int>(region_prefix));
    const double hi = hpt.getCdf(const_cast<char*>(last.c_str()), static_cast<int>(region_prefix));

    // The source's own degenerate test, and NOT `hi == lo`: a `last` that sorts before
    // `first` in the model's coordinate gives hi < lo and is equally unusable.
    if (!(hi > lo)) {
        m.status = Status::DegenerateCdf;
        return m;
    }

    const double span = hi - lo;

    m.alpha = 1.0 / span;
    // Written as the SOURCE writes it (lits_node.hpp:370: `min_cdf / (min_cdf - max_cdf)`)
    // rather than as the algebraically equal `-lo / (hi - lo)`. The two differ in the last
    // bit for some inputs, and this gate compares coefficients exactly -- there is no reason
    // to introduce a difference that has to be explained later.
    m.beta = lo / (lo - hi);

    if (!std::isfinite(m.alpha) || !std::isfinite(m.beta)) {
        m.status = Status::NonFinite;
        return m;
    }

    // The bound that makes the source's internal `static_cast<int>(c)` defined. See the
    // constant's own comment: a clamp around the call cannot repair a conversion that has
    // already happened inside it, so the guarantee has to be on the inputs.
    //
    // Computed from the LONGEST KEY IN THIS REGION rather than from kMaxKeyBytes. The walk is
    // one accumulation per byte of the key being looked up, so the region's own longest key is
    // the tight bound for every lookup the region will serve; a generic limit is safe but
    // refuses regions that are in fact fine, which is what the first version did to three of
    // the 256 real regions.
    if (accumulator_bound(groups, m.alpha, m.beta, region_max_key_bytes) > kMaxAccumulator) {
        m.status = Status::CoefficientTooLarge;
        return m;
    }

    m.status = Status::Ok;
    return m;
}

inline std::uint32_t RegionModel::predict_group(std::string_view full_key) const {
    std::uint32_t ignored = 0;
    return predict_group(full_key, ignored);
}

inline std::uint32_t RegionModel::predict_group(std::string_view full_key,
                                                std::uint32_t& walked) const {
    walked = 0;

    // The boundary path: no coefficients were derived and none are needed. Checked before
    // anything else so a caller that took `boundary()` never reaches the table.
    if (status == Status::SingleGroup || status == Status::TooFewKeys || status == Status::NoKeys) {
        return 0;
    }
    if (groups <= 1) return 0;

    if (status != Status::Ok) {
        throw std::logic_error(std::string("lits_hpt: unusable region model (") +
                               to_string(status) + ")");
    }

    // Both length checks matter and for different reasons:
    //   shorter than the prefix -> the walk would read past the end of the view at i = p
    //   longer than kMaxKeyBytes -> the loop length is outside anything this file bounds
    if (full_key.size() < prefix_length) {
        throw std::invalid_argument("lits_hpt: key is shorter than the region prefix");
    }
    if (full_key.size() > kMaxKeyBytes) {
        throw std::invalid_argument("lits_hpt: key is longer than the accumulation bound "
                                    "was written for");
    }

    // The LOOKUP side of the same domain train() and calibrate() enforce. The walk indexes the
    // source's tables with the BYTE VALUE (`m[i & PS_MASK][fc][key[i]]`), and those per-cell
    // tables cover 0..127; a byte >= 0x80 -- or a NUL, which the shared table never trained on --
    // must be refused HERE, before the walk, not walked out of bounds. The comment above
    // walk_position assumed the build-side refusal made this unreachable; it does not, because a
    // QUERY key never passes through train() or calibrate(). ASan found the read one table cell
    // past the end of a trained block on 2026-10-11, from a scan start of "a\x80\xff"
    // (diagnostics/check_owner_lifecycle.cpp, domain/query probes).
    if (const char* why = unsupported_reason(full_key)) {
        throw std::invalid_argument(std::string("lits_hpt: key is outside the supported "
                                                "domain (") + why + ")");
    }

    // The calibration bound again, now with THIS key's length. A lookup key is not always one
    // of the region's stored keys -- the absent probes append to a live one -- so the check
    // has to be repeated here rather than inferred from calibration.
    if (accumulator_bound(groups, alpha, beta, full_key.size()) > kMaxAccumulator) {
        throw std::length_error("lits_hpt: this key is long enough to put the position "
                                "accumulator outside int range");
    }

    const double c = walk_position(shared->hpt(), full_key, prefix_length,
                                   static_cast<int>(groups), alpha, beta, walked);

    // THE ORDER OF THESE THREE STEPS IS THE SAFETY ARGUMENT, and it is why this function does
    // its own saturation rather than calling the source's `getPos`:
    //
    //   1. a non-finite accumulator is a NUMERIC ERROR and is refused. It is NOT mapped to some
    //      group, and the query is NOT quietly re-run through the reference model -- switching
    //      models mid-query would send the key to a group the build never placed it in, which is
    //      the one thing the task book's section 3.4 forbids outright.
    //   2. otherwise saturate IN THE DOUBLE DOMAIN against the group range.
    //   3. convert.
    //
    // The integer cast is the LAST step and the only one that can be undefined if the first two
    // are skipped. That is what "saturation before the conversion" means, and it cannot be had
    // by clamping the return of a function that has already converted.
    if (!std::isfinite(c)) {
        throw std::domain_error("lits_hpt: the position accumulator is not finite");
    }
    if (c <= 0.0) return 0;
    if (c >= double(groups - 1)) return groups - 1;

    return static_cast<std::uint32_t>(c);
}

}  // namespace lits_hpt
