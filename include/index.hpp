#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// E38: the LITS HPT backend. Included here rather than from the programs, because the backend
// is selected by Config and the Region constructor has to be able to build one. The header
// pulls in the pinned source's model only -- lits_model.hpp and its two dependencies -- and
// deliberately NOT lits_node.hpp, which would drag in HOT, cnode, iter and PMSS.
#include "model/lits_hpt_adapter.hpp"

// BlockLinearRank (2026-10-11): the third backend. Region-scoped -- it fits from the region's own
// sorted keys and keeps no shared state -- so unlike the LITS adapter it is included for the same
// structural reason (Region::calibrate_block_rank must exist) but brings no vendor dependency.
#include "model/block_linear_rank.hpp"

// EqualDepthDir (2026-10-11): the fourth backend. A region-scoped SEPARATOR DIRECTORY -- the
// empirical quantiles of the region's own keys -- chosen by the entry diagnosis
// (RESULTS_THRESHOLD_ENTRY_20261011.md): max group load 1366/824/546 -> 12/12/12, empty groups
// -> 0, and 0.6-5.0 KB per region once separators are truncated. The 63-bit COORDINATE form of
// the same idea was measured and rejected (skew: 64/64 regions with duplicate thresholds).
#include "model/equal_depth_dir.hpp"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

// ---------------------------------------------------------------------------
// E14: build-phase instrumentation.
//
// Everything below is compiled out unless -DE14_PHASE is defined. index.hpp is
// therefore unchanged in behaviour with and without the flag -- the only
// difference is whether these accumulators are incremented. The gates are built
// WITHOUT the flag, so their "unchanged" evidence still holds, and the layout a
// -DE14_PHASE build produces is asserted byte-equal to a build without it by
// bench_e14 before any number is reported.
//
// Plain integers, not atomics: every build measured here is single-threaded and
// the process is bound to one core.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// E18: activity counters for the construction path.
//
// NO clock calls here, on purpose. The operations being counted are very short
// (a model prediction, a byte copy, a descriptor shift), so a timer around each
// one would cost more than the thing it measures and would change the cost
// structure under study. Counts are cheap and are the input to deciding which
// stage to time coarsely next -- see the round's RESULTS.
//
// Compiled out unless -DE18_COUNT.
// ---------------------------------------------------------------------------
#ifdef E18_COUNT
namespace validation {
namespace e18 {
struct Counts {
    std::uint64_t group_calls = 0;          // model predictions for a group id
    std::uint64_t fingerprint_calls = 0;    // Fingerprint::of
    std::uint64_t region_ctors = 0;
    std::uint64_t subblock_probes = 0;      // put_primary loop iterations
    std::uint64_t insert_attempts = 0;      // insert_into calls
    std::uint64_t insert_ok = 0;            // ... of which found room
    std::uint64_t entry_shifts = 0;         // descriptors moved to make room
    std::uint64_t suffix_bytes_copied = 0;  // bytes memcpy'd into a block
    std::uint64_t overflow_inserts = 0;     // records sent to an overflow list
    std::uint64_t overflow_containers = 0;  // overflow lists created
    void reset() { *this = Counts{}; }
};
inline Counts& counts() { static Counts c; return c; }
}  // namespace e18
}  // namespace validation
#define E18_INC(field) (++::validation::e18::counts().field)
#define E18_ADD(field, n) (::validation::e18::counts().field += (n))
#else
#define E18_INC(field) ((void)0)
#define E18_ADD(field, n) ((void)0)
#endif

// ---------------------------------------------------------------------------
// E24: update-path and space diagnostics.
//
// The E23 audit found Region::update and Region::erase linear-scan every
// sub-block with a FULL-KEY compare, while find/insert use the fingerprint tag
// filter, the length pre-check, the boundary directory and the overflow Hybrid.
// These counters record what the two paths actually walk, so the asymmetry is
// MEASURED rather than read off the source.
//
// e24_group_report() exposes the live per-group geometry that the space
// decomposition and the reclaim analysis need. It walks the real structure --
// it is not the static GroupStats model, which describes an INPUT, not the
// current layout.
//
// Compiled out unless -DE24_DIAG. Nothing here is on the timing path.
// ---------------------------------------------------------------------------
#ifdef E30_TIME
// E30: how much of a replacement workload is spent INSIDE compact()?
// Direct measurement, so it does not have to be inferred from the difference between
// two configurations -- that inference is what produced two answers 5.5x apart.
#include <chrono>
namespace validation {
namespace e30 {
inline std::uint64_t now_ns();
#ifdef E30_WRITE
// Declared before Timers so the fields can be listed first; the constructor calls the
// forward-declared clock, whose definition follows in this same translation unit.
struct WriteScope {
    std::uint64_t t0;
    std::uint64_t* add;
    std::uint64_t* inc;
    WriteScope(std::uint64_t* a, std::uint64_t* i) : t0(now_ns()), add(a), inc(i) {}
    ~WriteScope() { *add += now_ns() - t0; ++*inc; }
};
#endif
struct Timers {
    std::uint64_t compact_ns = 0, compact_calls = 0;
    std::uint64_t collect_ns = 0;   // gathering live records out of the sub-blocks
    std::uint64_t replay_ns = 0;    // put_primary replay + directory rebuild
    std::uint64_t erase_ns = 0, erase_calls = 0;
    std::uint64_t insert_ns = 0, insert_calls = 0;
    // E31 item 3 -- the checklist's section 5 PRECONDITION. The ORDERED OVERFLOW
    // VECTOR's own time: put_overflow's lower_bound + element shift + std::string
    // construction, and erase_at's vector erase + destruction. Without it the share the
    // O1/O2/O3 choice depends on is an assumption, and the audit found it has never been
    // isolated (E25-E31 carry no timing for the container itself).
    //
    // These NEST inside erase_ns (and inside insert_slow's insert_ns for the placements
    // that come from it), so a ratio against either is a SHARE, never an additive term.
    // Note that insert_ns is 0 in the frozen arm -- V2 routes a new key through
    // place_with_ctx, which starts no clock -- which is exactly why the diagnostic times
    // the segment itself instead of dividing by insert_ns.
    std::uint64_t ov_insert_ns = 0, ov_insert_calls = 0;
    std::uint64_t ov_erase_ns = 0, ov_erase_calls = 0;
    // A3 entry diagnosis: the insert's own two halves, so the breakdown can say how
    // much of it is LOOKING and how much is CONSTRUCTING/SHIFTING. Both are sub-pieces
    // of ov_insert_ns, which keeps its old meaning so E32's numbers stay comparable.
    std::uint64_t ov_lookup_ns = 0, ov_lookup_calls = 0;
    std::uint64_t ov_alloc_ns = 0, ov_alloc_calls = 0;
#ifdef E30_WRITE
    // E3 entry diagnosis (2026-10-10): the WHOLE write path, so the five work kinds the
    // plan names -- real suffix writes, descriptor moves, container growth, short-lived
    // temporaries, output materialisation -- have a denominator each. Every one of these
    // NESTS (locate inside insert/erase, place inside insert, rebuild inside insert,
    // all() inside rebuild), so a ratio against the outer clock is a SHARE and the
    // components are never additive.
    std::uint64_t w_insert_ns = 0, w_insert_calls = 0;       // Index::insert, whole
    std::uint64_t w_erase_ns = 0, w_erase_calls = 0;         // Index::erase, whole
    std::uint64_t w_locate_ns = 0, w_locate_calls = 0;       // Index::locate, called by both
    // locate is SHARED by the two passes, so a per-pass budget cannot use the line above:
    // the same counter would be charged twice. These two attribute it at the call site.
    std::uint64_t w_locate_ins_ns = 0, w_locate_ins_calls = 0;
    std::uint64_t w_locate_era_ns = 0, w_locate_era_calls = 0;
    // The write path ALSO searches inside the region (find_located -> the model + the
    // sub-block walk). Untimed, it would sit in the insert budget as an unattributed
    // residual and the residual would be read as "something else".
    std::uint64_t w_place_ns = 0, w_place_calls = 0;         // place_with_ctx
    std::uint64_t w_rebuild_ns = 0, w_rebuild_calls = 0;     // rebuild / rebuild_with_caps
    std::uint64_t w_rebuild_all_ns = 0, w_rebuild_all_calls = 0;  // the r.all() materialisation
    std::uint64_t w_rebuild_all_records = 0;                 // records copied by all()
    std::uint64_t w_reclaim_ns = 0, w_reclaim_calls = 0;     // repack_subblock, ALL calls
    // Repack split (2026-10-11): the two segments of one FULL pass (n >= 2). sort =
    // order[] init + insertion sort; move = the packing memmoves + offset rewrites +
    // bottom. Each full pass reads one SHARED boundary between them and one at the end;
    // the n<2 early return stays in the parent alone. See repack_subblock for the shape.
    std::uint64_t w_rp_sort_ns = 0, w_rp_sort_calls = 0;     // sort end - entry
    std::uint64_t w_rp_move_ns = 0, w_rp_move_calls = 0;     // move end - sort end
    std::uint64_t w_findloc_ns = 0, w_findloc_calls = 0;     // Region::find_located
    // Stage D supplementary check: the two walks the placement path re-runs. Named
    // so the analyzer can read the re-derivation's price directly instead of
    // bounding it by place-without-repack.
    std::uint64_t w_walk_ns = 0, w_walk_calls = 0;           // put_primary's sub-block walk
    std::uint64_t w_fc_ns = 0, w_fc_calls = 0;               // first_candidate
#endif
    void reset() { *this = Timers{}; }
};
inline Timers& timers() { static Timers t; return t; }
inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace e30
}  // namespace validation
#define E30_T0() ::validation::e30::now_ns()
#define E30_ADD(f, t0) (::validation::e30::timers().f += ::validation::e30::now_ns() - (t0))
#define E30_INC(f) (++::validation::e30::timers().f)
#ifdef E30_WRITE
// E3 entry: one RAII clock per NAMED phase. E30_W(insert) declares a scope called
// e30w_insert, so a function can time exactly one thing without inventing a name -- and
// the name it uses is the field it writes, so a mistyped phase cannot compile.
#define E30_W(field) \
    ::validation::e30::WriteScope e30w_##field(&::validation::e30::timers().w_##field##_ns, \
                                               &::validation::e30::timers().w_##field##_calls)
#else
#define E30_W(field) ((void)0)
#endif
#else
#define E30_T0() 0
#define E30_ADD(f, t0) ((void)0)
#define E30_INC(f) ((void)0)
// The write-path clocks exist only with E30_TIME + E30_WRITE, but their CALL SITES are
// unconditional -- so the no-clock build needs the no-op too. Without this line the
// gates (built with neither macro) fail to compile: the first version of this change did
// exactly that, and only a build of an UNRELATED target caught it.
#define E30_W(field) ((void)0)
#endif

#ifdef E24_DIAG
namespace validation {
namespace e24 {
struct Counts {
    // Region::update -- the same key set find() is measured on.
    std::uint64_t upd_cmps = 0, upd_bytes = 0, upd_entries = 0, upd_subblocks = 0;
    // Region::erase -- on a separate, discarded index.
    std::uint64_t era_cmps = 0, era_bytes = 0, era_entries = 0, era_subblocks = 0;
    void reset() { *this = Counts{}; }
};
inline Counts& counts() { static Counts c; return c; }

// One live sub-block of a group, as it exists NOW.
struct SubBlock {
    std::size_t slots = 0;        // capacity in slots
    std::size_t used = 0;         // live entries
    std::size_t suffix_cap = 0;   // bb - start_of, the whole suffix area
    std::size_t suffix_live = 0;  // sum of live entry lengths
    std::size_t suffix_free = 0;  // bottom - start_of: bytes that can still take a record
    std::size_t occupied = 0;     // occupied region = suffix_cap - suffix_free
    std::size_t garbage = 0;      // occupied - suffix_live: deleted bytes not yet reclaimed
};
struct GroupReport {
    std::uint32_t group = 0;
    std::vector<SubBlock> subs;   // empty when the group has no primary sub-block
    std::size_t ov_records = 0;
    std::size_t ov_live_bytes = 0;    // sum of live overflow suffix lengths
    std::size_t ov_suffix_cap = 0;    // sum of suffix.capacity()+1 over live records
    std::size_t ov_container_cap = 0; // container capacity * sizeof(OverflowRecord)
    std::size_t ov_rec_size = 0;      // sizeof(OverflowRecord), so slack is exact
    // Reclaim: an overflow record may move back only into the ONE sub-block its key
    // order selects (M1), and only if THAT sub-block has a free slot AND room for its
    // length. "The group has free space" is not the same test.
    std::size_t ov_fits_ordered = 0;      // passes slot + bytes + key-order
    std::size_t ov_fits_ordered_bytes = 0;
    std::size_t ov_blocked_slots = 0;     // ordered sub-block exists, no free slot
    std::size_t ov_blocked_bytes = 0;     // slot free, not enough suffix bytes
    std::size_t ov_blocked_order = 0;     // sorts below every primary sub-block
    std::size_t ov_fits_if_any_subblock = 0;  // what the existing group_has_room() would say
};
}  // namespace e24
}  // namespace validation
// The compare macros expand at their use site (deep inside Region), where both
// compare_bytes and the counter struct are visible -- so they carry the work
// themselves rather than calling a helper that would have to be declared here.
#define E24_CMP_UPD(a, b) \
    (++::validation::e24::counts().upd_cmps, \
     compare_bytes((a), (b), ::validation::e24::counts().upd_bytes))
#define E24_CMP_ERA(a, b) \
    (++::validation::e24::counts().era_cmps, \
     compare_bytes((a), (b), ::validation::e24::counts().era_bytes))
#define E24_UP_ENTRY() (++::validation::e24::counts().upd_entries)
#define E24_ERA_ENTRY() (++::validation::e24::counts().era_entries)
#define E24_UP_BLOCK() (++::validation::e24::counts().upd_subblocks)
#define E24_ERA_BLOCK() (++::validation::e24::counts().era_subblocks)
#else
#define E24_CMP_UPD(a, b) compare((a), (b))
#define E24_CMP_ERA(a, b) compare((a), (b))
#define E24_UP_ENTRY() ((void)0)
#define E24_ERA_ENTRY() ((void)0)
#define E24_UP_BLOCK() ((void)0)
#define E24_ERA_BLOCK() ((void)0)
#endif

// ---------------------------------------------------------------------------
// E25: the call matrix. What each write path actually invokes, per operation
// category, so "repeated location" is a counted fact rather than an inference
// from a large comparison ratio.
//
// FINGERPRINT ACCOUNTING. Counting Fingerprint::of() calls alone would be
// misleading: the model prediction hashes the bytes it consumes (fp.byte) and
// find_impl finishes the key with fp.tail, so the FUSED path produces a full
// fingerprint without ever calling Fingerprint::of. A change in code shape could
// therefore make "fingerprint computation" look like it disappeared when it only
// moved. So bytes actually fed to the hash are counted at the byte()/tail() level,
// which is independent of which call site produced them.
//
// Branch counters exist so the gate can assert that a branch RAN, instead of
// inferring coverage from "the final content matched" -- which is exactly the
// inference that missed the Index-level control flow E25 broke once.
//
// Compiled out unless -DE25_DIAG. Never on a timing path.
// ---------------------------------------------------------------------------
#ifdef E25_DIAG
namespace validation {
namespace e25 {
struct Counts {
    std::uint64_t locate_calls = 0;        // Index::locate (outer routing)
    std::uint64_t predict_calls = 0;       // Model::predict
    std::uint64_t fp_of_calls = 0;         // Fingerprint::of, whole-key form
    std::uint64_t fp_hash_bytes = 0;       // bytes fed to Fingerprint::byte/tail
    std::uint64_t existence_searches = 0;  // Region::find_impl entries
    std::uint64_t cmp_calls = 0;           // full-key compares in the write paths
    std::uint64_t cmp_bytes = 0;
    // P1 read-path entry diagnosis (2026-10-11): site-split compare work, outer-navigation
    // work and the fingerprint's fused-vs-tail split. The totals above stay the totals, so
    // `primary + overflow == cmp_calls/cmp_bytes` on the read path is checked, not assumed.
    std::uint64_t locate_cmp_calls = 0;    // Index::locate boundary probes (outer routing)
    std::uint64_t locate_cmp_bytes = 0;    // ... logical bytes walked by those probes
    std::uint64_t fp_fused_bytes = 0;      // bytes the MODEL fed to the hash during prediction
    std::uint64_t cmp_calls_primary = 0;   // exact compares against primary entries
    std::uint64_t cmp_bytes_primary = 0;
    std::uint64_t cmp_calls_overflow = 0;  // exact compares against overflow records
    std::uint64_t cmp_bytes_overflow = 0;
    std::uint64_t place_attempts = 0;      // insert_into calls
    std::uint64_t place_fail_slots = 0;    // ... refused: no free slot
    std::uint64_t place_fail_bytes = 0;    // ... refused: not enough suffix bytes
    std::uint64_t compactions = 0;         // Region::compact runs
    std::uint64_t region_rebuilds = 0;     // Index::rebuild calls
    // Erase work. `erase_shifts` counts descriptors moved by the shift-down, and
    // `erase_released` counts suffix bytes whose record went away -- for a primary
    // erase those bytes stay in the block as garbage, for an overflow erase the
    // string is actually destroyed. The two are reported apart because only the
    // second is a real free().
    std::uint64_t erase_shifts = 0;
    std::uint64_t erase_released_primary = 0;
    std::uint64_t erase_released_overflow = 0;
    std::uint64_t string_allocs = 0;       // std::string constructed into a container
    // E3 entry: the KEY BYTES those constructions really copy. The count alone cannot say
    // whether the temporary is a 5-byte string or a 200-byte one, and the plan's first
    // category ("real suffix writes") is a byte quantity.
    std::uint64_t ov_suffix_bytes_written = 0;
    std::uint64_t string_frees = 0;        // record destroyed, its string released

    // ---- E26: what compaction actually does ---------------------------------
    // compact() is entered from ONE place: a failed primary placement in a group
    // that still reports room somewhere. Whether the retry then SUCCEEDS is the
    // direct measure of whether that compaction did anything for the operation that
    // triggered it. A failure to help is recorded as "did not enable this placement",
    // which is NOT the same as "the compaction had no value" -- it also redistributes
    // records across sub-blocks, and that can matter for later operations.
    std::uint64_t compact_entered = 0;      // times the branch was taken
    std::uint64_t compact_helped = 0;       // the retry placed the record in the primary
    std::uint64_t compact_no_help = 0;      // the retry still failed -> overflow
    std::uint64_t compact_records_replayed = 0;
    std::uint64_t compact_bytes_copied = 0;
    std::uint64_t compact_garbage_before = 0;   // un-referenced bytes in the group, pre-compact
    std::uint64_t compact_tag_mismatch = 0;     // E26 diag: stored tag != recomputed tag
    std::uint64_t compact_plan_violation = 0;   // I10 diag: plan_replay's guarantee broke
    std::uint64_t compact_plan_rejected = 0;    // I10: plan_replay said the group cannot hold
    // The three branches E25's rewrite touched. Asserted by the gate.
    std::uint64_t notlocated_fallback = 0; // V2 upsert: no context, took the old path
    std::uint64_t subblock_reclaims = 0;        // E31: in-place sub-block passes run
    std::uint64_t subblock_reclaim_helped = 0;  // ... of which made the retry fit
    // G3/A1': the WIDENED passes -- sub-blocks tightened after the rank-selected one had
    // already been tried and the placement still failed. Non-emptiness of the first is what
    // makes the gate's assertion about this branch meaningful; the ratio of the second to
    // the first is the mechanism's own success rate, which is the number E37 could not
    // separate from the migration's.
    std::uint64_t reclaim_widened = 0;
    std::uint64_t reclaim_widened_helped = 0;
    std::uint64_t reclaim_widened_blocks_examined = 0;
    // E37: what the drain actually moved. `helped` is the narrower question -- whether the
    // retry that triggered the drain then succeeded.
    std::uint64_t migrate_calls = 0;
    std::uint64_t migrated_records = 0;
    std::uint64_t migrated_bytes = 0;
    std::uint64_t migrate_helped = 0;
    // E37: records the drain took out of the container and could not place, so they went
    // straight back. Equal to migrated_records means the drain did nothing but repack.
    std::uint64_t migrate_rejected = 0;
    std::uint64_t reclaim_skipped_unchanged = 0;  // E31H: skipped, garbage not grown
    std::uint64_t reclaim_skipped_gap = 0;        // E31H: skipped, min operation gap unmet

    // ---- Hole-reuse entry diagnosis (2026-09-27) ----------------------------
    // A byte failure means the CONTIGUOUS free area is too small. It says nothing about
    // the dead ranges inside the occupied region, because insert_into never consults them.
    // `hole_fits` is the candidate's own match test; `repack_fits` is the existing
    // mechanism's, evaluated ON THE SAME EVENT, so the two are comparable one-for-one
    // rather than as two rates measured over different populations.
    std::uint64_t fail_bytes_hole_fits = 0;      // some gap in the failing block >= len
    std::uint64_t fail_bytes_hole_absent = 0;    // no gap in that block >= len
    std::uint64_t fail_bytes_repack_fits = 0;    // contiguous + garbage >= len
    std::uint64_t fail_bytes_len_sum = 0;        // the requested suffix lengths
    std::uint64_t fail_bytes_best_hole_sum = 0;  // the best gap available, per event
    std::uint64_t fail_bytes_garbage_sum = 0;    // this block's dead bytes, per event
    std::uint64_t fail_bytes_live_sum = 0;       // bytes a repack here would have to copy
    std::uint64_t hole_sweep_checks = 0;         // gap-sum vs garbage comparisons made
    std::uint64_t hole_sweep_mismatch = 0;       // ... that disagreed (0 is expected)
    std::uint64_t hole_events_recorded = 0;
    std::uint64_t hole_events_dropped = 0;       // beyond kMaxHoleEvents, reported not hidden
    // Low-activity reclaim: repack passes that the candidate LET THROUGH, i.e. blocks with
    // 0 or 1 live records. The gate asserts this is non-zero, because a branch that never
    // ran would make every downstream comparison vacuous -- the same hole the G0 audit
    // found in gates that reported success for a comparison that never happened.
    std::uint64_t low_activity_reclaimed = 0;
    // The same passes split by the block's live count, plus the garbage they recovered.
    // An empty block and a one-record block are different cases -- the first has nothing
    // to move, the second moves one suffix -- so a single "low activity" number would
    // hide which one the workload actually reaches. `bytes` is the block's garbage
    // BEFORE the pass runs (occupied - live, the same definition subblock_garbage uses),
    // because the pass is what removes it.
    std::uint64_t low_activity_n0 = 0;
    std::uint64_t low_activity_n1 = 0;
    std::uint64_t low_activity_bytes = 0;
    std::uint64_t low_activity_helped = 0;
    // Of the passes above, the ones whose retry then PLACED the record -- the number the
    // plan calls "promoted placements". Without it the only available reading was the
    // difference in overflow inserts between arms, which mixes in every other effect the
    // switch has. Counted at BOTH call sites next to subblock_reclaim_helped.
    // cannot answer "how many dead ranges does a failing block have". These counters are
    // exact over every refusal at no size cost.
    std::uint64_t hole_nholes0 = 0;            // no dead range at all
    std::uint64_t hole_nholes1 = 0;            // exactly one -- the run against bb_
    std::uint64_t hole_nholes2plus = 0;        // a SPLIT dead region, the case a table needs
    std::uint64_t hole_best_eq_garbage = 0;    // one run == all the dead bytes
    std::uint64_t hole_count_lt2 = 0;          // 0-1 live records: repack_subblock would no-op
    std::uint64_t hole_count_eq0 = 0;          //   ... of which the block held NOTHING
    std::uint64_t hole_count_eq1 = 0;          //   ... and the block held exactly one record
    std::uint64_t hole_slots_not_free = 0;     // must stay 0; recorded, not assumed
    // repack_subblock refuses to act below two live records (index.hpp subblock_garbage
    // applies the same guard), so this is the fitness test WITH that guard -- the version
    // that matches the mechanism rather than the arithmetic.
    std::uint64_t fail_bytes_repack_guarded = 0;
    // ---- Repack split (2026-10-11): the work side of one repack_subblock pass -----
    // The entry question for on-demand hole placement (candidate H) is what a reclaim
    // pass spends its time on. The E30 clocks carry the time side of the same split;
    // these count the WORK, exactly and machine-independently: a full pass processes n
    // records, issues one memmove per record and rewrites every offset. `moved` vs
    // `selfcopy` labels whether the record's offset actually changed -- a self-copy is
    // a same-address memmove, a call whose data movement does not exist. Function work,
    // not memory-bus traffic.
    std::uint64_t rp_calls = 0;            // full passes (sort + move reached)
    std::uint64_t rp_records = 0;          // records the packing loop handled
    std::uint64_t rp_bytes = 0;            // their suffix bytes (the memmove lengths)
    std::uint64_t rp_moved_records = 0;    // offset CHANGED: a real relocation
    std::uint64_t rp_moved_bytes = 0;
    std::uint64_t rp_selfcopy_records = 0;// offset unchanged: memmove(base+x, base+x, n)
    std::uint64_t rp_selfcopy_bytes = 0;
    // ---- Candidate H (primary_hole_reuse, 2026-10-11): the hole-placement branches -----
    // The bounded entry round justified ONE minimal candidate; these count its behaviour so
    // the gate can assert BOTH branches fired and the round can read how many repacks it
    // replaces instead of assuming it. counted under E25_DIAG only, like every counter here.
    std::uint64_t hole_reuse_attempts = 0;      // slot OK, bytes short, H applies
    std::uint64_t hole_reuse_placed = 0;        // ... and a dead range took the suffix
    std::uint64_t hole_reuse_absent = 0;        // ... and no gap was large enough
    std::uint64_t hole_reuse_skipped_lt2 = 0;   // < 2 live records: low-activity territory
    std::uint64_t hole_reuse_skipped_wide = 0;  // > 48 records: over the fixed stack bound
    std::uint64_t hole_reuse_bytes = 0;         // suffix bytes written into dead ranges
    std::uint64_t prefix_rebuilds = 0;     // Index::insert prefix-mismatch rebuild
    std::uint64_t threshold_splits = 0;    // Index::insert region split at 2*region_keys

    // ---- E27: what a range scan actually does -------------------------------
    // The four quantities the review named, kept apart so the round can pick the
    // ONE that matters instead of optimizing all four at once.
    std::uint64_t scan_regions = 0;          // regions entered by Index::scan
    std::uint64_t scan_groups = 0;           // groups iterated inside a region
    std::uint64_t scan_groups_empty = 0;     // ... of which held nothing at all
    std::uint64_t scan_blocks_skipped = 0;   // empty/exhausted sub-blocks stepped over
    std::uint64_t scan_merge_cmps = 0;       // primary-vs-overflow ordering compares
    std::uint64_t scan_result_allocs = 0;    // std::string constructions for results
    // ---- E31 item 3: the overflow vector's MOVEMENT, as bytes ------------------
    // "The ordered overflow vector's movement is a large share of the time" is the claim
    // the checklist wants before O1/O2/O3 is chosen, so the movement is recorded as a
    // number rather than an adjective. The element count is read at the one point where
    // the distance is still valid (before the container is touched), and the byte figure
    // is that count times the record's OWN size: a shift copies each OverflowRecord's 48
    // bytes and NOT the suffix buffer it points at, so a figure that counted the payload
    // would overstate what the container actually moves. The element size is reported as
    // ov_rec_bytes, so the byte counts can be read back as record counts.
    std::uint64_t ov_shifted_records_insert = 0;
    std::uint64_t ov_shifted_bytes_insert = 0;
    std::uint64_t ov_shifted_records_erase = 0;
    std::uint64_t ov_shifted_bytes_erase = 0;
    // ---- O1 (overflow_stable_store): RECORD moves vs HANDLE moves, and the space --
    // The E32 counters above measure the LOGICAL shift distance and are incremented at
    // the call site in put_overflow and erase_at, so both arms report the same distance
    // on the same workload and the byte figures are what changed. These two measure
    // what the container PHYSICALLY moved, and they are incremented inside the
    // container, at the line that does the moving: on the stable arm the order array
    // carries the shift in 4-byte handles and no stored record is relocated. The pair
    // is what the acceptance table needs to show that the 48 bytes per erase went
    // away rather than somewhere else -- and it is counted, not reasoned about.
    std::uint64_t overflow_record_moves_stable = 0;      // ON arm: stored records relocated
    std::uint64_t overflow_record_move_bytes_stable = 0;
    std::uint64_t overflow_handle_moves = 0;             // ON arm: handles shifted in order
    std::uint64_t overflow_handle_move_bytes = 0;
    // Space: the store's high-water mark and the largest live count seen, so what the
    // holes the free list keeps for reuse cost is visible. A store that reuses holes
    // holds live <= high_water; the gap at the peak is the worst case it produced.
    std::uint64_t ov_store_high_water = 0;
    std::uint64_t ov_store_live_water = 0;
    // ---- Branch C, shrink-to-peak (2026-10-10): the cost side ----------------
    // A policy that shrinks a container to its historical peak leaves capacity == peak,
    // so the NEXT record past the peak has to reallocate. These three count exactly that
    // moment: how often it happens, by how many elements the new peak exceeds the old one,
    // and how many bytes the resulting copy moves. The 0 -> 1 transition is excluded --
    // the container owned nothing before it, so both policies allocate there.
    std::uint64_t ov_new_peak_events = 0;
    std::uint64_t ov_new_peak_overshoot = 0;   // elements
    std::uint64_t ov_new_peak_copy_bytes = 0;  // bytes the realloc would memcpy
    // The same two quantities for the policy that is actually shipped: a change of
    // element capacity is a growth reallocation. Reported together so the extra cost of
    // the shrink is the DIFFERENCE, not the whole copy volume -- without this the new-peak
    // bytes read as if the current policy copied nothing.
    std::uint64_t ov_cap_change_events = 0;
    std::uint64_t ov_cap_change_copy_bytes = 0;  // bytes, approximate: (size-1) records
    // ---- E35 (prefix decidability): would an inline comparison prefix pay? ----
    // Every comparison in the binary overflow search reads the record: on the stable arm
    // that is store_[order_[mid]], two indirections. An entry that carried the suffix's
    // first W bytes could answer most of them from the entry alone -- but only if the
    // windows actually differ, which is a property of the DATA and is therefore counted
    // here rather than assumed. E6 already found that shared prefixes can defeat a filter.
    //
    // The window is measured on the STORED SUFFIX, the string being compared, not on the
    // original key: the container holds suffixes, so a full-key prefix distribution would
    // be describing a different string.
    std::uint64_t ov_queries_bin = 0;        // queries that took the binary path
    std::uint64_t ov_queries_linear = 0;     // queries that took the linear path
    std::uint64_t ov_bin_cmps = 0;           // comparisons inside the binary loop
    std::uint64_t ov_bin_decided4 = 0;       // both sides >= 4 B and the first 4 differ
    std::uint64_t ov_bin_decided8 = 0;       // both sides >= 8 B and the first 8 differ
    std::uint64_t ov_bin_short4 = 0;         // one side < 4 B: no window fixes these
    std::uint64_t ov_bin_lcp_bytes = 0;      // summed longest-common-prefix length
    // The LINEAR path, which is the one that actually runs in the measured workload: the
    // containers average about ten records, under the frozen Hybrid cutoff, so the binary
    // search is the minority path. The linear scan reads every record for its tag before it
    // compares anything, so the question there is different: what would a prefix add ON TOP
    // OF the tag? ov_linear_reject8 counts exactly that -- candidates that survived the tag
    // filter but that an 8-byte prefix would still have settled as "not equal".
    std::uint64_t ov_linear_recs = 0;         // records examined by the linear scan
    std::uint64_t ov_linear_tag_reject = 0;   // rejected by the tag, never compared
    std::uint64_t ov_linear_tag_pass = 0;     // survived the tag, reached the compare
    std::uint64_t ov_linear_reject8 = 0;      // of those, an 8-byte prefix settles them
    std::uint64_t ov_linear_lcp_bytes = 0;    // summed LCP over the tag survivors
    // ---- E36 (inline tag): the three questions the reviewer asked of this prototype ----
    //   1. how many order entries does one linear overflow query look at   -> ov_tag_entries
    //   2. how many are rejected BEFORE any store access                  -> ov_tag_reject_pre
    //   3. how many store accesses does that save                          -> ov_linear_recs,
    //      which counts store reads in every arm, so T against O is the answer directly.
    // ov_tag_entries equals ov_linear_recs in B and O, where the tag can only be read by
    // reading the record; the two separate exactly in T.
    std::uint64_t ov_tag_entries = 0;         // order entries examined by the linear scan
    std::uint64_t ov_tag_reject_pre = 0;      // rejected by the ENTRY's tag, store untouched
    std::uint64_t scan_start_locates = 0;    // E27 S2: locate(start) evaluations in scan()
    // P2/A4 entry diagnosis: the three pieces of per-call and per-group setup that the
    // plan suspects of repeating. Split so that "how many" can be read against "how
    // many records came back" -- a count that does not grow with the result is setup,
    // and a count that does is work the result forced.
    std::uint64_t scan_canonical_group = 0;  // group-number re-derivations in scan()
    std::uint64_t scan_cursor_starts = 0;    // per-group primary cursor initialisations
    std::uint64_t scan_ov_cursor_starts = 0; // ... of which had an overflow container
    // ---- Scan-entry round (2026-10-10). The one scan loop the earlier counters could
    // not see. Every other piece of scan work was already counted, and B and W measured
    // IDENTICAL on all of them (analyze_scandiag.py) while the wall clock was not -- so
    // the starting-cursor advance is the remaining traversal loop. Split by side and by
    // "comparison" vs "record consumed", because a lower_bound candidate replaces
    // comparisons, not records, and the two have to be readable against each other.
    // Every increment is E25_INC, i.e. compiled out of every timing build.
    std::uint64_t scan_adv_p_cmps = 0;       // comparisons in the primary start advance
    std::uint64_t scan_adv_p_skipped = 0;    // primary records passed (one compare each)
    std::uint64_t scan_adv_p_subblocks = 0;  // sub-blocks fully stepped over by the advance
    std::uint64_t scan_adv_ov_cmps = 0;      // comparisons in the overflow start advance
    std::uint64_t scan_adv_ov_skipped = 0;   // overflow records passed
    std::uint64_t scan_out_primary = 0;      // rows emitted from primary sub-blocks
    std::uint64_t scan_out_overflow = 0;     // rows emitted from overflow containers
    std::uint64_t scan_out_bytes = 0;        // full-key bytes materialised (prefix+suffix)
    // ---- Stage D supplementary check (2026-10-10) --------------------------
    // How much of the placement path is RE-DERIVATION of the sub-block index that
    // find_impl's locate_subblock already computed for this very key and dropped.
    // `put_primary_calls` and the two first_candidate counters price the walks;
    // the hint_* family measures whether the directory answer really equals the
    // block first_candidate picks -- the equality the candidate would rely on.
    std::uint64_t put_primary_calls = 0;   // put_primary entries (one or more per insert)
    std::uint64_t fc_calls = 0;            // first_candidate entries
    std::uint64_t fc_probes = 0;           // ... its boundary-compare iterations
    std::uint64_t fc_gap = 0;              // ... of which took the two-block (prev, i) path
    std::uint64_t hint_eq = 0;             // directory answer == first_candidate's block
    std::uint64_t hint_ne = 0;             // ... they disagreed (must stay 0)
    std::uint64_t hint_nodir = 0;          // no directory: no answer to compare
    std::uint64_t hint_below = 0;          // directory says below every primary key
    void reset() { *this = Counts{}; }
};
inline Counts& counts() { static Counts c; return c; }

#ifdef E25_DIAG
// Longest common prefix length, for the E35 prefix-decidability counters. Only defined in
// the diagnostic build: in the timing build the statistic must vanish entirely, and an
// extra byte loop inside the comparison is exactly the cost that must not be charged to
// one arm and not the other.
inline std::size_t ov_lcp(std::string_view a, std::string_view b) {
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}
#endif
}  // namespace e25
}  // namespace validation
#define E25_INC(f) (++::validation::e25::counts().f)
#define E25_ADD(f, n) (::validation::e25::counts().f += (n))
// One scale for BOTH search paths: the fast locate and the slow linear walk must
// report their string comparisons into the same counters, or the three variants
// would be measured with two different rulers. Expands at the use site, where
// compare_bytes is visible.
#define E25_CMP(a, b) \
    (++::validation::e25::counts().cmp_calls, \
     compare_bytes((a), (b), ::validation::e25::counts().cmp_bytes))
// A running maximum, for the O1 store's high-water marks. Separate from E25_ADD
// because the quantity is a peak and not a sum. Compiled out with the rest, so the
// timing build does not evaluate its argument either.
#define E25_MAX(f, n) \
    (::validation::e25::counts().f = \
     ::validation::e25::counts().f > (n) ? ::validation::e25::counts().f : (n))
#else
#define E25_INC(f) ((void)0)
#define E25_ADD(f, n) ((void)0)
#define E25_CMP(a, b) compare((a), (b))
#define E25_MAX(f, n) ((void)0)
#endif

#ifdef E14_PHASE
#include <chrono>

namespace validation {
namespace e14 {

struct Phases {
    // Region constructor
    std::uint64_t prefix = 0;        // common-prefix computation
    std::uint64_t model = 0;         // model_.train
    std::uint64_t suffix_scan = 0;   // suffix-byte total over the region
    std::uint64_t demand_pass = 0;   // E9 pass 1: per-group entries/bytes demand
    std::uint64_t layout = 0;        // group count, capacity table, off_ derivation
    std::uint64_t storage = 0;       // operator new + header/entry initialisation
    std::uint64_t place = 0;         // initial placement (put_primary/put_overflow)
    std::uint64_t dir = 0;           // boundary-directory generation
    std::uint64_t grp_prep = 0;      // E18: hoisted group-id + fingerprint pre-pass
    std::uint64_t ctor_total = 0;    // whole constructor, for reconciliation
    std::uint64_t ctor_calls = 0;

    // Index level
    std::uint64_t bulk_load = 0;     // Index::bulk_load, incl. the Region ctors
    std::uint64_t bulk_planned = 0;  // E15: Index::bulk_load_with_plan, ditto
    std::uint64_t rebuild = 0;       // rebuild_with_caps / rebuild, incl. Region ctor
    std::uint64_t rebuild_calls = 0;
    std::uint64_t region_all = 0;    // Region::all(), the sorted copy handed to a rebuild

    void reset() { *this = Phases{}; }
};

inline Phases& phases() { static Phases p; return p; }

inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace e14
}  // namespace validation

#define E14_T0() ::validation::e14::now_ns()
#define E14_ADD(field, t0) (::validation::e14::phases().field += ::validation::e14::now_ns() - (t0))
#define E14_INC(field) (++::validation::e14::phases().field)
#else
#define E14_T0() 0
#define E14_ADD(field, t0) ((void)0)
#define E14_INC(field) ((void)0)
#endif

namespace validation {

// ---------------------------------------------------------------------------
// Ordering contract: unsigned-byte lexicographic comparison, shared by every
// structure (model grouping, in-block ordering, outer routing, reference map).
// ---------------------------------------------------------------------------

inline int compare(std::string_view a, std::string_view b) {
    const auto n = std::min(a.size(), b.size());
    const int c = n ? std::memcmp(a.data(), b.data(), n) : 0;
    return c ? c : (a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0);
}

struct Less {
    bool operator()(std::string_view a, std::string_view b) const {
        return compare(a, b) < 0;
    }
};

using Pair = std::pair<std::string, std::uint64_t>;

inline bool pair_less(const Pair& a, const Pair& b) {
    return compare(a.first, b.first) < 0;
}

// ---------------------------------------------------------------------------
// Fingerprint: CRC-32C tag used only as a candidate filter; a matching tag never
// proves equality, every hit goes through the exact compare.
// ---------------------------------------------------------------------------

struct Fingerprint {
    std::uint32_t state = 0x9e3779b9U;

    void byte(std::uint8_t x) {
        E25_ADD(fp_hash_bytes, 1);
#if defined(__SSE4_2__)
        state = _mm_crc32_u8(state, x);
#else
        state ^= x;
        for (int i = 0; i < 8; ++i)
            state = (state >> 1) ^ (0x82f63b78U & (0U - (state & 1U)));
#endif
    }

    void tail(std::string_view x) {
#if defined(__SSE4_2__)
        // Count ONLY the bytes the 8-byte chunk loop consumes: the remainder goes
        // through byte(), which counts itself. Counting x.size() here as well would
        // double-count the tail.
        const std::size_t before = x.size();
        while (x.size() >= 8) {
            std::uint64_t w;
            std::memcpy(&w, x.data(), 8);
            state = static_cast<std::uint32_t>(_mm_crc32_u64(state, w));
            x.remove_prefix(8);
        }
        E25_ADD(fp_hash_bytes, before - x.size());
#endif
        for (unsigned char c : x) byte(c);
    }

    std::uint8_t tag() const { return static_cast<std::uint8_t>(state); }

    static std::uint8_t of(std::string_view s) {
        E25_INC(fp_of_calls);
        Fingerprint f;
        f.tail(s);
        return f.tag();
    }
};

// ---------------------------------------------------------------------------
// Lookup configuration and per-query result.
// ---------------------------------------------------------------------------

enum class Route { Full, Interval, Residual };
enum class Prefetch { Off, Early, Late };

// E5-2: mode.simd selects the batched SSE2 tag filter; identical results by
// construction.
// E7: overflow point-lookup policy rides on the per-query Mode (not Config):
// the overflow records and their sorted order are identical for every policy,
// so all variants measure the same records with a different lookup algorithm.
// E12: diagnostic counters are compiled OUT of the timing build. They never
// influence found/value, but a timed lookup should not pay for them either, so the
// benchmark is built without E12_COUNT while the gates are built with it.
#ifdef E12_COUNT
#define E12_TICK(x) (++(x))
#define E12_ADD(x, n) ((x) += (n))
#else
#define E12_TICK(x) ((void)0)
#define E12_ADD(x, n) ((void)0)
#endif

enum class OvSearch { Linear, Binary, Hybrid };

struct Mode {
    Route route;
    Prefetch prefetch;
    bool fingerprint;
    bool simd = false;
    OvSearch ov = OvSearch::Linear;
    std::uint32_t ov_thr = 32;
    // E11: for multi-block groups, use the boundary directory to pick at most one
    // candidate primary block instead of scanning them in order. Off by default,
    // so the path is bit-identical to E10 unless a caller asks for it.
    bool locate = false;
};

struct Result {
    bool found = false;
    std::uint64_t value = 0;
    std::uint32_t steps = 0;
    std::uint32_t overflow_checks = 0;
    std::uint32_t str_cmps = 0;   // E7: full string comparisons for this lookup
    std::uint8_t source = 0;      // E7: 0 miss, 1 primary hit, 2 overflow hit
    std::uint64_t cmp_bytes = 0;  // E7 diag: bytes walked by compares (E7_COUNT_BYTES builds)

    // E10 access counters: what the lookup path actually executes. Only meaningful
    // for a query that got past the prefix check -- one rejected there returns all
    // zeros. They never influence found/value, and E10 takes no timings, so leaving
    // them always on costs nothing that is measured.
    std::uint32_t subblocks_visited = 0;   // sub-blocks entered (empty ones count: header read)
    std::uint32_t simd_batches = 0;        // vector tag compares actually executed
    std::uint32_t tag_candidates = 0;      // tag matches surviving the live-slot mask
    std::uint32_t str_cmps_primary = 0;    // exact compares against in-block entries
    std::uint32_t str_cmps_overflow = 0;   // exact compares against overflow records
    std::uint32_t overflow_entered = 0;    // 1 if the overflow container was searched

    // E11: location cost kept separate from the block-check cost, because the goal
    // is "at most one candidate block checked", not "total access down to one".
    std::uint32_t locate_probes = 0;       // boundary comparisons during location
    std::uint64_t locate_cmp_bytes = 0;    // logical bytes walked by those comparisons
    std::uint32_t candidate_blocks = 0;    // primary blocks entered for tag/record checks
};

#ifdef E25_DIAG
// M4 entry diagnosis (2026-10-11): the shadow counts of the ordered-segment merge candidate.
// All counters are per call; nothing here is resident, and nothing is compiled into a build
// that does not define E25_DIAG.
struct MergeShadowCounts {
    std::uint64_t outputs = 0;            // rows the candidate emitted
    std::uint64_t merge_cmps = 0;         // per-row cross-side compares it still pays
    std::uint64_t probes = 0;             // boundary proofs executed (success + failure)
    std::uint64_t probe_ok = 0, probe_fail = 0;
    std::uint64_t probe_fail_equal = 0;   // failures that were a boundary TIE (c == 0)
    std::uint64_t seg_rows = 0;           // rows emitted through a proven segment
    std::uint64_t seg_rows_max = 0;
    std::uint64_t probe_bytes = 0;        // logical bytes of every proof (compare_bytes)
    std::uint64_t probe_fail_bytes = 0;   // ... of the failed ones
    std::uint64_t side_switches = 0;      // primary/overflow alternations in the rows
    std::uint64_t skip_short = 0;         // arrivals with r < 4
    std::uint64_t skip_no_ov = 0;         // arrivals with r >= 4 and no opposite side
    std::vector<std::uint32_t> seg_lens;  // successful segment lengths, for the distribution
    void add(const MergeShadowCounts& o) {
        outputs += o.outputs; merge_cmps += o.merge_cmps; probes += o.probes;
        probe_ok += o.probe_ok; probe_fail += o.probe_fail;
        probe_fail_equal += o.probe_fail_equal; seg_rows += o.seg_rows;
        if (o.seg_rows_max > seg_rows_max) seg_rows_max = o.seg_rows_max;
        probe_bytes += o.probe_bytes; probe_fail_bytes += o.probe_fail_bytes;
        side_switches += o.side_switches; skip_short += o.skip_short;
        skip_no_ov += o.skip_no_ov;
        seg_lens.insert(seg_lens.end(), o.seg_lens.begin(), o.seg_lens.end());
    }
};
#endif

// E5-1: slots is an explicit layout parameter (default 16 reproduces the E3 behavior).
enum class ModelKind { Context16, Position16, Unigram };

// E38: which MODEL FAMILY a region is mapped by. This is a separate axis from ModelKind,
// which selects a table shape INSIDE the Context16 family -- `model_kind` describes the
// reference model's own rows, and folding LITS into it would make "kind == Context16" mean two
// different things depending on another field.
//
// The task book's section 4 requires the two to be kept apart, and requires an unsupported
// combination to be REFUSED rather than silently mapped to some other group function.
enum class ModelBackend { Context16, LitsHpt, BlockLinearRank, EqualDepthDir,
                          EqualDepthDirTrunc };

// EqualDepthDirTrunc is the SAME directory as EqualDepthDir with section 2's truncated
// separators (model/equal_depth_dir.hpp). The mapping function has the same shape, so every
// wiring point asks this question instead of comparing against one enumerator; the two backends
// differ only in the bytes the directory stores, and the model identity mixes that flag in.
inline bool is_depth_dir_backend(ModelBackend b) {
    return b == ModelBackend::EqualDepthDir || b == ModelBackend::EqualDepthDirTrunc;
}

// The three backends answer the same question -- "which logical group does this key belong
// to" -- with different machinery, and the index refuses combinations it does not implement
// rather than substituting a near-enough mapping:
//   Context16       the reference table below; trained per region on suffixes
//   LitsHpt         one index-wide pinned HPT table, calibrated per region
//   BlockLinearRank a per-region segmented rank model (model/block_linear_rank.hpp),
//                   ported from the teammate model and fixed at 7 encoded bytes / 8 segments
//   EqualDepthDir   a per-region separator directory at equal key RANKS
//                   (model/equal_depth_dir.hpp): balances the group load by construction
//   EqualDepthDirTrunc  the same directory with the truncated separators of
//                   TRUNCDIR_ENTRY_DESIGN.md section 2: shortest prefix that still beats the
//                   left neighbour, so stored keys route exactly as in EqualDepthDir

struct Config {
    std::uint32_t block_bytes = 1024;
    double load = .75;
    std::size_t region_keys = 2048;
    Route canonical = Route::Full;
    std::uint32_t fixed_groups = 0;
    ModelKind model_kind = ModelKind::Context16;
    // E38: which model family maps a key to a group. Defaults to the reference model, so the
    // shipped configuration is unchanged and a program that never mentions this field builds
    // exactly what it built before.
    //
    // NOT free to combine with the rest: ModelBackend::LitsHpt supports only ModelKind::Context16
    // and Route::Full, and the Region constructor REFUSES any other pairing rather than quietly
    // falling back to a different group function (task book section 4).
    ModelBackend model_backend = ModelBackend::Context16;
    std::uint32_t slots = 16;
    // E8-11: false = legacy fixed-stride addressing (base + g*block_bytes, uniform
    // capacity only); true = offset-directory addressing (required for demand caps).
    bool offset_directory = true;
    // E9: when true, a group whose demand exceeds one sub-block grows to several
    // equal-sized sub-blocks within the same table shape. Off by default, so the
    // single-block path stays bit-identical to E5-E8 and their gates keep passing.
    bool multiblock = false;
    // E19 (R1): reuse the group ids the demand pass computed, instead of calling
    // canonical_group a second time for the same key in the placement loop.
    // Only meaningful with multiblock = true, which is where the demand pass runs.
    // The fingerprint is NOT affected -- it is computed once, in placement.
    bool reuse_group_ids = false;

    // E25 V1: update/erase search through the SAME machinery find() uses -- the
    // fingerprint tag filter, the SIMD batch filter, the boundary directory and the
    // overflow policy -- instead of walking every entry with a full-key compare.
    // Off reproduces the E24 behaviour exactly, so V0/V1 are one binary apart.
    // ---- FROZEN 2026-09-20: these five defaults are ON ----
    // Chosen on the same-round evidence in E25 (write path), E26 (compaction) and
    // E27 (scan), not on call counts. The build path (R1), the model, the logical
    // mapping and the main-block budget are unchanged.
    // Every gate pins its arms through `switches.hpp`, so these defaults are the
    // SHIPPED configuration and not silently the experimental baseline.
    bool write_fast_locate = true;

    // E25 V2: insert/upsert reuse the group id, the tag and the "not present"
    // conclusion the SAME operation's locate already produced, instead of asking
    // the model and the fingerprint again. Duplicate detection is NOT skipped.
    bool reuse_locate_ctx = true;

    // E25: the mode the WRITE paths search with (V1). Defaults mirror the query
    // mode the benchmarks use: fingerprint + SIMD + Hybrid overflow, and NOT the
    // boundary locate -- E13 established locate is a selectable component, not a
    // default. ov_threshold is the frozen Hybrid cutoff, set by the caller.
    std::uint32_t ov_threshold = 32;
    bool write_simd = true;
    bool write_locate = false;

    // E26 C1: compact() currently re-derives Fingerprint::of() for every record it
    // replays. The tag of a record does not depend on WHERE the record sits, and the
    // suffix bytes are copied verbatim, so the stored tag is still correct and can
    // travel with the record. Changing this must not alter membership, placement
    // order, capacity, the trigger, or the boundary maintenance -- it only removes a
    // recomputation.
    bool compact_reuse_tag = true;

    // E26 C2: compact() reclaims BYTES by re-packing suffixes; it creates no slots.
    // So when a group holds less reclaimable garbage than the record that just failed
    // to place, that compaction cannot free enough for it. This gate skips those. It
    // is a NECESSARY-condition test, not a model of when compaction pays: compaction
    // also redistributes records across sub-blocks, which this does not predict.
    bool compact_needs_garbage = true;

    // E27: the scan result is `prefix_ + std::string(suffix)`, which builds a
    // temporary string and then concatenates it -- two allocations per returned
    // record. Building the result string in place with one reservation does the same
    // job with one. Semantics (complete key/value pairs, key order, ownership) are
    // unchanged; only the intermediate copy goes away.
    bool scan_reserve_result = true;

    // -----------------------------------------------------------------------
    // Low-activity sub-block reclaim (candidate, default OFF).
    //
    // repack_subblock returns immediately for a block holding fewer than two live records,
    // and subblock_garbage excludes the same blocks from being reclaim candidates. Both are
    // optimisations, not preconditions: the packing loop already handles n == 0 (top stays at
    // bb_, the whole area free) and n == 1. On a replacement workload to 200%, 53%-98% of
    // block-level byte refusals land on exactly these blocks.
    //
    // Orthogonal to subblock_reclaim and reclaim_widest, so each can be measured with this as
    // the single varying field.
    bool reclaim_low_activity = false;

    // -----------------------------------------------------------------------
    // Primary-hole reuse (candidate H, default OFF).
    //
    // When an insert's contiguous space is short, the shipped path repacks the target
    // sub-block (subblock_reclaim) and retries. The bounded entry round measured that pass
    // at 40.1%/40.8% move+rewrite (mixed@200%, ctx16/lits), 96.7% of its memmove bytes
    // real relocations, ~4.6 KB per pass (RESULTS_REPACK_SPLIT_20261011.md). This candidate
    // instead finds a DEAD RANGE in the SAME sub-block and writes the new suffix there,
    // moving nothing else.
    //
    // Requires subblock_reclaim (the mechanism it substitutes for), never touches blocks
    // under two live records, and is inert when off. See insert_into_hole for the scope
    // locks.
    bool primary_hole_reuse = false;

    // -----------------------------------------------------------------------
    // Compact region routing (candidate, default OFF).
    //
    // locate() binary-searches the region fences. Each step dereferences a Node -- a
    // std::string plus a unique_ptr -- and then the fence's own heap buffer, so a step costs
    // two dependent loads of unrelated cache lines. This switch keeps an 8-byte
    // order-preserving digest of every fence in ONE contiguous array, so the same search
    // walks 8 bytes per step.
    //
    // The result must be IDENTICAL to the original search, not merely close: the digest is
    // order-preserving, so digest(a) < digest(b) implies a < b, but equal digests do not
    // imply equal keys -- ties fall back to the full unsigned-byte comparison. The
    // integration gate asserts equality of the two searches over every key, including short
    // keys, the empty key, embedded NULs, high bytes, common prefixes beyond 8 bytes and
    // deliberate digest collisions.
    //
    // Nothing about the region partition, the model, the group geometry, the blocks, the
    // records or the values changes. Only how a key is routed to a region.
    bool compact_region_routing = false;

    // -----------------------------------------------------------------------
    // LITS reference-table residency (2026-10-11, default OFF; candidate M3).
    //
    // A Region always trains and keeps a Context16 reference table (16 x 258 u32 =
    // 16,512 B). For a region whose mapping is the LITS backend's, that table is never
    // read: the query path (predict_for) and the placement path (canonical_group) both
    // dispatch on use_lits_, and the entry round measured the table entry count as exactly
    // the number of lookups that landed in a FALLBACK region -- zero on tranco/LITS across
    // 10,000 queries, 262,144 corpus keys, the whole construction, 2,000 erases and 50
    // scans (RESULTS_OWNER_ENTRY_20261011.md).
    //
    // With this switch ON, a region whose LITS calibration succeeded releases that table
    // right after the calibration decides the mapping. The mapping, the calibration result,
    // the rejection domain, the fallback coverage, the blocks and every answer are
    // unchanged -- only 16,512 B per such region stops being resident. The training itself
    // still runs, so the build time is unchanged: this is a SPACE candidate, and its
    // prototype measured the saving as exactly the theoretical table bytes of LITS-mapped
    // regions (2,113,536 B = 6.05% of the tranco/LITS accounting), with no resolvable
    // regression in build, query, update or scan (RESULTS_OWNER_PROTOTYPE_20261011.md).
    //
    // Combinations that cannot keep their meaning are REFUSED rather than degraded: a
    // preset offered for such a region cannot be validated against a digest of a table that
    // no longer exists, and group_of() would otherwise read an empty table. Callers that
    // need the reference model must use an independent reference object.
    bool lits_skip_reference_table = false;

    // E29 C3: relax the compaction TRIGGER. Compaction is currently attempted only
    // when some single sub-block has both a free slot and enough suffix bytes -- and
    // the E28 measurement showed that is exactly the condition that is usually false
    // while the group still holds plenty of garbage. C3 additionally compacts when a
    // full repack of the group would fit the record.
    //
    // The budget test is not optional: compact() replays every live record and throws
    // if the group cannot hold them, so it must never be invoked on a short budget.
    bool compact_repack_budget = false;

    // I10 (2026-09-23 review): make the COMMIT PHASE of compact() unable to be interrupted.
    //
    // The commit phase had two real throw sites, and one of them fired AFTER the old state
    // was destroyed. rebuild_dir() does dir.clear() and then push_back's freshly constructed
    // std::string boundaries, so a bad_alloc there leaves the group's records fully
    // re-placed and its boundary directory PARTIAL -- and locate_subblock binary searches
    // that directory, so the damage is a silently mis-routed lookup, not a visible lost
    // record. The other site is the explicit throw inside the replay loop.
    //
    // The fix is task book section 6.3: do everything that can fail before the point of no
    // return. The replay is PLANNED first (plan_replay): that both proves the group can hold
    // the records and builds the directory the replay will produce, entirely in local
    // scratch, while the group is still untouched. If the plan does not fit, compact()
    // returns false having changed nothing. The commit then only appends into pre-existing
    // block memory (append_planned: no capacity test, no allocation) and swaps the prepared
    // directory in, and vector::swap is noexcept.
    //
    // So the commit phase contains no allocation and no fallible call: it cannot be
    // interrupted.
    //
    // DEFAULT ON (2026-09-23). This is the one switch added since E24 that is a FIX rather
    // than a candidate, so it does not follow the "new switches default off" rule -- an
    // unsafe default is not something to leave opt-in. The paired measurement in
    // RESULTS_I10_COMMIT.md section 7 shows the plan is FASTER, not slower: it replaces
    // the per-record sub-block walk inside put_primary with one counter pass plus a direct
    // write, maintenance 0.649-0.924 and p99 0.418-0.942 against the old path.
    bool compact_plan_commit = true;

    // E31: reclaim the rank-selected sub-block IN PLACE before giving up on the
    // primary. erase() frees a slot but leaves the suffix bytes as garbage and does not
    // move the sub-block's bottom watermark, so free bytes do not increase -- which is
    // why a placement that needs bytes fails while the sub-block still holds plenty of
    // reclaimable space. E30 measured this pass at 6.1-11.4 ns/record against 163 ns for
    // the frozen arm's whole-group compaction, and the whole-run reclaim against the
    // compaction it displaces by 8.1x-21.3x.
    //
    // The sub-block BOUNDARY does not change, so order invariant M1 is untouched and
    // the E25 Located context stays valid.
    bool subblock_reclaim = false;

    // G3/A1' (2026-09-23): WIDEN WHICH SUB-BLOCKS THE RECLAIM MAY TOUCH.
    //
    // The placement path ever tightens exactly ONE sub-block: the one first_candidate
    // returns, which shares put_primary's enumeration on purpose (E31). E37 then measured
    // what actually moved the overflow numbers, and it was NOT the migration: its drain
    // tightens the sub-blocks that individual OVERFLOW RECORDS would land in, i.e. blocks
    // the insert path never reaches -- "reclaim enabled" rose by +40,000 to +163,000 while
    // the migrated record count was only 2,165-32,012, a factor of 15x-30x.
    //
    // So this is that observation as a single-variable change: after the rank-selected
    // block has been tightened and the placement STILL fails, tighten the group's remaining
    // sub-blocks, dirtiest first, up to a FIXED budget, retrying the placement after each.
    //
    // In-place tightening only. Members, slot order, the sub-block key range and the
    // boundary directory are all unchanged, so nothing a caller holds is invalidated --
    // and, unlike E37, nothing is migrated and no overflow record is moved.
    //
    // Requires subblock_reclaim: it is a wider trigger surface for that mechanism, not a
    // second mechanism. Default OFF, like every other candidate.
    bool reclaim_widest = false;

    // E31H (checklist item 4.3): hysteresis and a minimum operation interval on the
    // reclaim trigger. E31's own measurement (2026-09-20) fired it 127,780 times (mixed
    // same) to 255,882 (skew grow) at the 200% checkpoint and enabled the triggering
    // placement in only 20.4%-67.2% of them, so most of the rest re-ran the pass on a
    // sub-block the previous attempt had already left unchanged. With this ON a group's
    // reclaim is skipped unless at least kMinReclaimGap placements have happened since its
    // last attempt AND its garbage has grown since then. Default OFF: the frozen
    // combination and every existing gate keep the E31 behaviour byte for byte.
    bool reclaim_hysteresis = false;

    // E31P (checklist 4.3, the other half of the same rule): skip a repeated reclaim
    // only when the group's garbage has not grown since its last attempt -- the
    // checklist's "do not retry while the state has not materially changed" -- with no
    // minimum interval. The interval half is reclaim_hysteresis; measuring them apart
    // is the point, because the interval is what turned out to bind.
    bool reclaim_skip_repeat = false;

    // E27 S2 (checklist 7.1 / 9 row 4): Index::scan re-evaluated locate(start)
    // on every loop iteration AND once more in the body, so a scan spanning R
    // regions paid R+1 outer searches for the one answer it already had. Caching
    // the first evaluation makes it exactly one. The evaluations are counted
    // (scan_start_locates), so the saving is measured rather than argued: the S0
    // arm counts R+1 per scan, the S2 arm exactly 1.
    bool scan_cache_start = false;

    // Scan-entry round (2026-10-10), candidate S: the scan's START ADVANCE.
    //
    // The linear cursor walk from a group's head to the start position reads every record
    // it passes -- up to the whole group when the start sits near its end, and a
    // zero-result scan after the last key walks the group completely. Measured on the
    // microbenchmark: 136-492 comparisons in ONE call at the extreme starts, costing
    // 10-30 us; the walk exists in BOTH the baseline and the WIDE arms (moving records
    // between the primary sub-blocks and the overflow container only changes which side
    // pays it), so this is a general scan cost, not a WIDE artefact.
    //
    // ON, the position is found by ordered search instead: a two-level binary search
    // over the rank-ordered primary sub-blocks (block level first -- the last record of
    // the last non-empty block at or left of the midpoint -- then within the block), and
    // the overflow container's existing ordered lookup (lower_bound_pos). The number of
    // records passed is IDENTICAL to the linear walk (the counters prove it); the number
    // of comparisons becomes logarithmic.
    //
    // Default OFF: every existing arm keeps the linear walk byte for byte.
    bool scan_ordered_start = false;

    // O1 (2026-09-20, the overflow container's erase cost): stable record storage
    // plus an ordered handle array. E32 measured the ordered overflow vector at
    // 12.2 / 37.3 / 37.6 per cent of maintenance time on the three datasets,
    // concentrated in ERASE: at the 200 per cent checkpoint on the domain set the
    // container moved 3,099,948 records / 148,797,504 bytes through erases while the
    // largest single live container held 1,365 records. Each of those erases shifted
    // every later record by sizeof(OverflowRecord) == 48 bytes. Insert was cheap by
    // comparison: insertions almost always append.
    //
    // ON, a stored record never moves. The erase shifts 4-byte handles in order and
    // destroys the record where it sits; a free list reuses the hole. OFF is the
    // vector, byte for byte.
    //
    // NOT assumed to be faster. It still moves handles, and it adds one indirection
    // on every read, so the point query, the scan and the insert paths may all get
    // SLOWER. That is what the measurement is for.
    bool overflow_stable_store = false;
    // E36: keep a COPY of each record's tag in its order entry, so the linear scan can
    // reject a candidate without reading the record. Requires the stable store (there is
    // no order array otherwise); the arm setup asserts the pair is used together.
    bool overflow_inline_tag = false;
    // E37: on a placement that would otherwise go to overflow, first try to bring records the
    // other way -- out of this group's overflow container and back into the primary area.
    bool overflow_migrate = false;
};

#if defined(E7_COUNT_BYTES) || defined(E24_DIAG) || defined(E25_DIAG)
// Diagnostic builds only: mirrors compare() while counting bytes actually walked.
// memcmp stops at the first difference, so one call does not imply a full scan.
inline int compare_bytes(std::string_view a, std::string_view b, std::uint64_t& bytes) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && static_cast<unsigned char>(a[i]) == static_cast<unsigned char>(b[i])) ++i;
    bytes += (i < n) ? i + 1 : n;
    int c = 0;
    if (i < n)
        c = static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[i]) ? -1 : 1;
    else
        c = a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
    return c;
}
#endif

#ifdef E25_DIAG
// P1 (2026-10-11): site-split compare counting for the read path. Diagnostic builds
// only; in every other build both macros are plain compare().
inline int compare_bytes_site(std::string_view a, std::string_view b,
                              std::uint64_t& total, std::uint64_t& site) {
    const std::uint64_t before = total;
    const int c = compare_bytes(a, b, total);
    site += total - before;
    return c;
}
#define E25_CMP_P(a, b)                                                        \
    (++::validation::e25::counts().cmp_calls,                                  \
     ++::validation::e25::counts().cmp_calls_primary,                          \
     compare_bytes_site((a), (b), ::validation::e25::counts().cmp_bytes,       \
                        ::validation::e25::counts().cmp_bytes_primary))
#define E25_CMP_O(a, b)                                                        \
    (++::validation::e25::counts().cmp_calls,                                  \
     ++::validation::e25::counts().cmp_calls_overflow,                         \
     compare_bytes_site((a), (b), ::validation::e25::counts().cmp_bytes,       \
                        ::validation::e25::counts().cmp_bytes_overflow))
#else
#define E25_CMP_P(a, b) compare((a), (b))
#define E25_CMP_O(a, b) compare((a), (b))
#endif

// ---------------------------------------------------------------------------
// Conditional model: integer interval arithmetic over quantized probabilities.
// Q = 2^47 position scale, MASS = 2^16 probability mass, 16 context rows.
// ---------------------------------------------------------------------------

class Model {
public:
    static constexpr std::uint64_t Q = 1ULL << 47;
    static constexpr std::uint32_t MASS = 1U << 16;
    static constexpr std::size_t ROWS = 16;

    using Table = std::vector<std::array<std::uint32_t, 258>>;

    Table cumulative;
    ModelKind kind = ModelKind::Context16;

    std::size_t row(std::size_t pos, unsigned previous) const {
        if (kind == ModelKind::Unigram) return 0;
        if (kind == ModelKind::Position16) return std::min<std::size_t>(15, pos);
        return ((pos & 3U) << 2U) | (previous & 3U);
    }

    std::size_t allocated_bytes() const {
        return cumulative.capacity() * sizeof(Table::value_type);
    }

    // E16: a range overload. The body is the original loop with the container
    // iteration replaced by pointer iteration; the vector overload forwards, so
    // every existing caller trains exactly as before.
    void train(const std::vector<Pair>& keys, std::size_t prefix,
               ModelKind choice = ModelKind::Context16) {
        train(keys.data(), keys.data() + keys.size(), prefix, choice);
    }

    void train(const Pair* first, const Pair* last, std::size_t prefix,
               ModelKind choice = ModelKind::Context16) {
        kind = choice;
        const auto rows = kind == ModelKind::Unigram ? 1 : ROWS;
        cumulative.assign(rows, {});

        std::vector<std::array<std::uint64_t, 257>> counts(rows);
        for (auto& r : counts) r.fill(1);  // finite smoothing, including EOS and byte 0

        for (const Pair* it = first; it != last; ++it) {
            const auto s = std::string_view(it->first).substr(prefix);
            unsigned previous = 0;
            const auto lim = std::min<std::size_t>(16, s.size());
            for (std::size_t p = 0; p < lim; ++p) {
                unsigned d = static_cast<unsigned char>(s[p]) + 1U;
                ++counts[row(p, previous)][d];
                previous = d;
            }
            if (s.size() <= 16) ++counts[row(s.size(), previous)][0];
        }

        for (std::size_t r = 0; r < rows; ++r) {
            std::uint64_t total = 0;
            for (auto x : counts[r]) total += x;

            std::array<std::uint32_t, 257> frequency{};
            std::uint32_t sum = 0;
            for (std::size_t d = 0; d < 257; ++d) {
                frequency[d] = static_cast<std::uint32_t>(counts[r][d] * MASS / total);
                sum += frequency[d];
            }
            // Deterministic remainder assignment; at most 256 residual units.
            for (std::uint32_t d = 0; d < MASS - sum; ++d) ++frequency[d];

            cumulative[r][0] = 0;
            for (std::size_t d = 0; d < 257; ++d)
                cumulative[r][d + 1] = cumulative[r][d] + frequency[d];
            assert(cumulative[r][257] == MASS);
        }
    }

    // Interval step; both endpoints derive from the pre-update lo to avoid
    // ordering mistakes. W <= 2^47 and C <= 2^16, so the product fits uint64.
    static void step(const std::array<std::uint32_t, 258>& c, unsigned d,
                     std::uint64_t& lo, std::uint64_t& hi) {
        const std::uint64_t origin = lo, w = hi - lo;
        lo = origin + ((w * c[d]) >> 16);
        hi = origin + ((w * c[d + 1]) >> 16);
    }

    // Physical mapping h: monotone in x, hence group order follows key order.
    static std::uint32_t bucket(std::uint64_t x, std::uint32_t groups) {
        const auto g = static_cast<std::uint64_t>((static_cast<unsigned __int128>(x) * groups) >> 47);
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(groups - 1, g));
    }

    struct Prediction {
        std::uint32_t group;
        // How many bytes the MODEL processed. This is `result.steps`, a quantity that has been
        // reported since E5, and its meaning does not change with the backend.
        std::uint32_t consumed;
        // E38: how many bytes the FINGERPRINT state has absorbed -- the offset the caller
        // continues hashing from. On this fused path the model feeds every byte it reads into
        // the hash, so the two are equal; a backend that reads a full key from an absolute
        // position contributes NOTHING to the hash and reports 0 here instead. Keeping them
        // apart is what lets `steps` keep its published meaning while the hash keeps producing
        // the tags that were stored (see check_lits_fingerprint).
        std::uint32_t fp_consumed;
    };

    template <bool Hash>
    Prediction predict(std::string_view s, std::uint32_t groups, Route route,
                       Fingerprint& fp) const {
        E25_INC(predict_calls);
        std::uint64_t lo = 0, hi = Q;
        unsigned previous = 0;
        std::size_t p = 0;
        for (; p < s.size(); ++p) {
            const auto ch = static_cast<unsigned char>(s[p]);
            step(cumulative[row(p, previous)], ch + 1U, lo, hi);
            previous = ch + 1U;
            if constexpr (Hash) {
                fp.byte(ch);
                E25_ADD(fp_fused_bytes, 1);
            }

            const bool stop =
                (lo == hi) ||
                (route == Route::Interval && bucket(lo, groups) == bucket(hi, groups)) ||
                (route == Route::Residual && static_cast<unsigned __int128>(hi - lo) * groups < Q);
            if (stop) { ++p; break; }
        }
        // EOS has cumulative 0, so the final coordinate is the current lower bound.
        //
        // Field order: group, consumed, fp_consumed. The fused path hashes every byte it reads,
        // so the fingerprint offset IS the model's count -- stated here rather than left to a
        // default, which is the failure mode the review warned about (a stale two-field
        // aggregate would have zeroed `fp_consumed` and silently changed every query tag).
        const auto step = static_cast<std::uint32_t>(p);

        return {bucket(lo, groups), step, step};
    }
};

// ---------------------------------------------------------------------------
// On-block layout: one block per group, 64B-aligned. The suffix arena grows
// downward from the block end; tags sit at offset 16 (16B-aligned for SIMD).
// ---------------------------------------------------------------------------

struct Entry {
    std::uint32_t offset;
    std::uint32_t length;
    std::uint64_t value;
};

// E5-1: tags[48] keeps the header at 64B for every supported slot count (S<=48);
// tags start at offset 16, so every 16-tag chunk is 16-byte aligned for SIMD loads.
struct alignas(64) Header {
    std::uint64_t overflow_pointer;
    std::uint16_t count, bottom, flags, reserved0;
    std::uint8_t tags[48];
};

static_assert(sizeof(Entry) == 16 && sizeof(Header) == 64);
static_assert(sizeof(std::uintptr_t) == 8, "This validation layout assumes 64-bit pointers");

struct GroupStats {
    std::uint32_t group = 0;
    std::size_t primary_entries = 0, primary_bytes = 0, overflow_entries = 0, overflow_bytes = 0;
    std::size_t free_entries = 0, free_bytes = 0, garbage_bytes = 0;
    std::size_t blocked_slots_only = 0, blocked_bytes_only = 0, blocked_both = 0, oversized = 0, fits_now = 0;
};

// ---------------------------------------------------------------------------
// E20 (R2): what the statistics stage hands to the final construction.
//
// Carrying `group_ids` across the two stages is only sound if both map a string to
// the SAME logical group. That needs the prefix and the logical group count to
// agree, so they travel with the ids and the constructor checks them. Capacity
// tables and physical sub-block counts are deliberately NOT part of this: changing
// them keeps the group id, and the round states that distinction explicitly.
//
// The statistics stage's overflow membership is also deliberately not carried: it
// is a property of the PROBE's capacity, not of the final layout's.
// ---------------------------------------------------------------------------
struct RegionDemand {
    std::vector<GroupStats> stats;
    std::vector<std::uint32_t> group_ids;   // one per key, in input order
    std::string prefix;                     // the prefix the ids were computed under
    std::uint32_t groups = 0;               // the logical group count they were computed with
    // E21: identity of the string -> group mapping. Prefix and group count alone
    // are not enough -- a different input, or the same input under a different
    // model kind, can share both while training to a different table. See
    // Region::model_identity.
    std::uint64_t model_digest = 0;
    // E21/E22: a HELPER check on the key sequence, NOT an identity for it.
    //
    // Equal model tables do not imply equal inputs, so a check on the sequence is
    // needed. This one is deliberately cheap: one FNV step per key over its length
    // and its two boundary bytes. That is enough to catch an accidental reorder or
    // a resized key, and NOT enough to establish that the sequence is unchanged --
    // "aaba" and "abba" have the same length, the same first byte and the same last
    // byte, and are indistinguishable here before any mixing happens. The source
    // address and length do not close that gap either: they cannot see an in-place
    // edit.
    //
    // The guarantee therefore does NOT rest on this value. See
    // Index::bulk_load_reusing_ids: the context is created and consumed inside one
    // call and never reaches a caller, so it cannot be stored and replayed against
    // different data. This digest only makes some misuse visible.
    std::uint64_t key_digest = 0;
};

// E21/E22: the whole context, including the identity of the key array it came from.
//
// INTERNAL. A context is produced and consumed inside Index::bulk_load_reusing_ids
// and must not be held, copied out, or replayed against other data; the lifetime is
// what makes the reuse sound. The source address and size are a checked
// precondition, not the guarantee.
struct BuildContext {
    const Pair* source = nullptr;
    std::size_t source_size = 0;
    std::vector<RegionDemand> regions;
};

// ---------------------------------------------------------------------------
// Region: one contiguous lexicographic interval = model + capacity table + blocks.
// ---------------------------------------------------------------------------

// E38: how the LITS backend covered the regions it was asked to map. Counted, not asserted:
// the task book's section 3.4 requires the fallback to be reported with its region, record and
// query shares, so a round can say "LITS-HPT + Context16 fallback, N regions" instead of
// presenting a mixed result as if it were a pure one.
struct LitsStats {
    std::uint64_t regions_calibrated = 0;
    std::uint64_t regions_fallback = 0;
    // Regions with fewer than two keys, which take the DEFINED BOUNDARY PATH (group 0) rather
    // than a calibration. Kept apart from the two above because neither of those describes it:
    // it did not calibrate, and it did not fall back either -- it never needed a table.
    std::uint64_t regions_boundary = 0;
    std::uint64_t keys_calibrated = 0;
    std::uint64_t keys_fallback = 0;
    std::uint64_t keys_boundary = 0;
};

// BlockLinearRank: how the rank backend covered the regions it was asked to map. There is no
// fallback arm to report -- the 9-bit encoding accepts every byte string, so the only non-fitted
// outcomes are the three DEFINED boundary cases (no keys / one key / one group), and all three
// answer group 0, which is the answer the fit would give anyway. Counted for the same reason
// LitsStats is: a round must be able to say what mapping it actually timed.
struct BlockRankStats {
    std::uint64_t regions_fitted = 0;
    std::uint64_t keys_fitted = 0;
    std::uint64_t regions_boundary = 0;
    std::uint64_t keys_boundary = 0;
    std::uint64_t segments_total = 0;   // sum of segment counts over the fitted regions
    std::uint64_t fits = 0;             // fit() calls, boundary early-outs included
};

// EqualDepthDir: the same accounting for the directory backend. `duplicate_thresholds` counts
// the regions where two thresholds are equal, i.e. where the depth could NOT be spread -- the
// one place the balance is limited by the keys themselves rather than by the map.
struct DepthDirStats {
    std::uint64_t regions_fitted = 0;
    std::uint64_t keys_fitted = 0;
    std::uint64_t regions_boundary = 0;
    std::uint64_t keys_boundary = 0;
    std::uint64_t thresholds_total = 0;
    std::uint64_t regions_with_duplicate_thresholds = 0;
    std::uint64_t threshold_bytes_total = 0;   // arena bytes stored, truncated or not
    std::uint64_t regions_truncated = 0;       // fitted regions whose separators were truncated
    std::uint64_t fits = 0;
};

class Region {
    // E5-1: the entry-slot count S and the entries-area end are per-region values now.
    static constexpr std::size_t MAX_SLOTS = 48;
    static constexpr std::size_t NO_SUBBLOCK = ~std::size_t{0};   // E11 locate result

    struct OverflowRecord {
        std::string suffix;
        std::uint64_t value;
        std::uint8_t tag;
    };

    // E36: the ordered entry. In the plain stable store this is a bare 4-byte handle; in
    // the inline-tag arm it also carries a copy of the record's tag, so the linear scan can
    // reject a candidate from the ENTRY alone and never touch the store. The size is
    // reported by sizeof rather than assumed -- with uint32_t + uint8_t it is 8, not 5,
    // and that padding is part of what this round measures.
    struct OrderedEntry {
        std::uint32_t handle;
        std::uint8_t tag;
    };

    // O1: ONE container interface, TWO representations. The switch decides which is
    // live at construction; no call site tests it.
    //
    // OFF is the ordered vector this project has shipped since E7: sorted by suffix,
    // where a middle insert or erase shifts every later record by
    // sizeof(OverflowRecord) == 48 bytes. The OFF arm forwards to exactly that vector,
    // so the frozen build behaves and reports space as it did before O1.
    //
    // ON is the stable store. Records live in a deque, which never moves an existing
    // element when it grows, and they are never moved again. order holds sorted
    // 4-byte handles into the store, so an erase shifts handles instead of records,
    // and free_slots reuses the holes the erases leave so the store does not grow
    // with the number of records ever inserted.
    //
    // NOT assumed to be faster: it still moves handles, and every read now pays one
    // indirection through the handle array, so the point query, the scan and the
    // insert paths can all get slower. That is what the measurement is for.
    class OverflowList {
    public:
        // The two slow paths that walk the container -- Region::update and
        // Region::erase, reachable only when write_fast_locate is off -- iterate it,
        // and the erase does erase(it) and end() - it. Both arms expose this iterator
        // so those loops are written once, and its arithmetic means one thing in
        // either arm: a position in the container's own order.
        struct iterator {
            OverflowList* owner = nullptr;
            std::size_t pos = 0;

            OverflowRecord& operator*() const { return (*owner)[pos]; }
            OverflowRecord* operator->() const { return &(*owner)[pos]; }
            iterator& operator++() { ++pos; return *this; }

            std::ptrdiff_t operator-(const iterator& o) const {
                return static_cast<std::ptrdiff_t>(pos) - static_cast<std::ptrdiff_t>(o.pos);
            }

            bool operator==(const iterator& o) const { return pos == o.pos; }
            bool operator!=(const iterator& o) const { return pos != o.pos; }
        };

        struct const_iterator {
            const OverflowList* owner = nullptr;
            std::size_t pos = 0;

            const OverflowRecord& operator*() const { return (*owner)[pos]; }
            const OverflowRecord* operator->() const { return &(*owner)[pos]; }
            const_iterator& operator++() { ++pos; return *this; }

            bool operator==(const const_iterator& o) const { return pos == o.pos; }
            bool operator!=(const const_iterator& o) const { return pos != o.pos; }
        };

        // Read once, here, at construction; no operation tests them afterwards.
        explicit OverflowList(bool stable, bool inline_tag) : stable_(stable), tag_inline_(inline_tag) {}

        std::size_t size() const { return stable_ ? order_.size() : vec_.size(); }
        bool empty() const { return stable_ ? order_.empty() : vec_.empty(); }

        const OverflowRecord& operator[](std::size_t i) const {
            return stable_ ? store_[order_[i].handle] : vec_[i];
        }

        // E36: the tag copy that lives WITH the order entry. Reading it touches the order
        // array and nothing else -- which is the entire point, because the linear scan can
        // then reject a candidate without paying the store access. In the arms without the
        // inline copy this reads the record, which is exactly what they already did.
        bool tags_are_inline() const { return tag_inline_; }

        std::uint8_t entry_tag(std::size_t i) const {
            return tag_inline_ ? order_[i].tag : (*this)[i].tag;
        }

        // Named so the call site says what it costs: this one DOES reach the store.
        const OverflowRecord& record_at(std::size_t i) const { return (*this)[i]; }

        // Non-const: the ON arm returns a reference into the deque. A deque keeps
        // references to existing elements valid when it grows, and nothing is ever
        // removed from the store, so this reference stays valid -- which is what lets
        // store_at write a value in place without going through the handle array.
        OverflowRecord& operator[](std::size_t i) {
            return stable_ ? store_[order_[i].handle] : vec_[i];
        }

        iterator begin() { return iterator{this, 0}; }
        iterator end() { return iterator{this, size()}; }
        const_iterator begin() const { return const_iterator{this, 0}; }
        const_iterator end() const { return const_iterator{this, size()}; }

        // The position a record with suffix s belongs at: the first position whose
        // stored suffix is not less than s. The OFF arm is the exact std::lower_bound
        // call put_overflow always made, with the same comparator, so its ordering and
        // its comparison counts are unchanged.
        std::size_t lower_bound_pos(std::string_view s) const {
            if (!stable_) {
                const auto it = std::lower_bound(vec_.begin(), vec_.end(), s,
                    [](const OverflowRecord& a, std::string_view b) { return E25_CMP(a.suffix, b) < 0; });
                return static_cast<std::size_t>(it - vec_.begin());
            }
            // Ordered by the FULL suffix. The inline tag is never consulted here: a tag
            // does not order anything, and using it would break the ordering the binary
            // search depends on.
            std::size_t lo = 0, hi = order_.size();
            while (lo < hi) {
                const std::size_t mid = lo + (hi - lo) / 2;
                if (E25_CMP(store_[order_[mid].handle].suffix, s) < 0) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }

        // Insert rec before position pos.
        void insert_at(std::size_t pos, OverflowRecord rec) {
            if (!stable_) {
                vec_.insert(vec_.begin() + static_cast<std::ptrdiff_t>(pos), std::move(rec));
                track_insert();
                return;
            }
            // The tag is read BEFORE the move, because the entry and the record must carry
            // the same one. This is the only place that writes the store, so it is the only
            // place that can put a stale tag in the order array -- and it does not.
            const std::uint8_t rec_tag = rec.tag;
            std::uint32_t slot = 0;
            if (!free_slots_.empty()) {
                slot = free_slots_.back();
                free_slots_.pop_back();
                store_[slot] = std::move(rec);
            } else {
                store_.push_back(std::move(rec));
                slot = static_cast<std::uint32_t>(store_.size() - 1);
            }
            const std::size_t shifted = order_.size() - pos;
            // Handle and tag move as ONE entry, in one array, so they cannot come apart.
            order_.insert(order_.begin() + static_cast<std::ptrdiff_t>(pos),
                          OrderedEntry{slot, rec_tag});
            // The mechanism check. overflow_handle_moves counts the handles the order
            // array just shifted and overflow_handle_move_bytes what that cost. The
            // record pair is written at this line, the only place that touches the
            // store, and it is 0 because an insertion either appends to the deque or
            // writes into a hole: neither displaces a stored record. The E32 counters
            // (ov_shifted_records_*) keep reporting the LOGICAL shift distance in both
            // arms, so the two representations are compared on the same workload.
            E25_ADD(overflow_handle_moves, shifted);
            E25_ADD(overflow_handle_move_bytes, shifted * sizeof(OrderedEntry));
            E25_ADD(overflow_record_moves_stable, 0);
            E25_ADD(overflow_record_move_bytes_stable, 0);
            // Space. Both quantities are monotone in an insert, so a running max is
            // the peak: the store's slots -- holes included -- and the live count.
            E25_MAX(ov_store_high_water, store_.size());
            E25_MAX(ov_store_live_water, order_.size());
            track_insert();
        }

        // Erase the record at position pos.
        void erase_at(std::size_t pos) {
            if (!stable_) {
                vec_.erase(vec_.begin() + static_cast<std::ptrdiff_t>(pos));
                track_erase();
                return;
            }
            const std::uint32_t slot = order_[pos].handle;
            const std::size_t shifted = order_.size() - 1 - pos;
            // The entry carries its tag out with it, so an erased position leaves no tag
            // behind; the slot it names is handed to the free list and its next occupant
            // writes a fresh tag at insert_at above.
            order_.erase(order_.begin() + static_cast<std::ptrdiff_t>(pos));
            // Destroy the payload where it sits -- the caller has already counted the
            // free -- and hand the slot to the free list. The record did not move;
            // the handles after pos did, and that is the whole point of the store.
            store_[slot] = OverflowRecord{};
            free_slots_.push_back(slot);
            E25_ADD(overflow_handle_moves, shifted);
            E25_ADD(overflow_handle_move_bytes, shifted * sizeof(OrderedEntry));
            E25_ADD(overflow_record_moves_stable, 0);
            E25_ADD(overflow_record_move_bytes_stable, 0);
            track_erase();
        }

        void erase(iterator it) { erase_at(it.pos); }

        // E37: remove a set of positions in ONE compaction pass. `pos` must be sorted and
        // distinct. Erasing them one at a time makes every erase walk the tail that is left,
        // and the feasibility scan measured that at 11x to 165x the entry moves of this
        // single pass on the same set -- so the difference is not a constant factor to be
        // optimised away later, it is the order of the cost.
        void erase_positions(const std::vector<std::size_t>& pos) {
            if (pos.empty()) return;

            std::size_t k = 0, w = 0;

            if (!stable_) {
                for (std::size_t i = 0; i < vec_.size(); ++i) {
                    if (k < pos.size() && pos[k] == i) {
                        vec_[i] = OverflowRecord{};   // destroy the payload here, not later
                        ++k;
                        continue;
                    }

                    if (w != i) vec_[w] = std::move(vec_[i]);

                    ++w;
                }

                vec_.resize(w);
                track_erase();
                return;
            }

            for (std::size_t i = 0; i < order_.size(); ++i) {
                if (k < pos.size() && pos[k] == i) {
                    store_[order_[i].handle] = OverflowRecord{};
                    free_slots_.push_back(order_[i].handle);
                    ++k;
                    continue;
                }

                if (w != i) order_[w] = order_[i];

                ++w;
            }

            order_.resize(w);
            track_erase();
        }

#ifdef E25_DIAG
        // E36: the tag copy in every order entry against the tag in the record it names.
        // Returns the number of mismatches and reports how many pairings it compared.
        // Called with the clock stopped; it is a structural check, not a measurement.
        std::size_t diag_tag_pairing(std::size_t& checked) const {
            if (!stable_) return 0;
            std::size_t bad = 0;
            for (std::size_t i = 0; i < order_.size(); ++i) {
                ++checked;
                if (order_[i].tag != store_[order_[i].handle].tag) ++bad;
            }
            return bad;
        }
#endif

        // The container's own slot arrays, without the object or the payloads. The OFF
        // arm is the vector capacity in record units, which is what e24_report's
        // ov_container_cap has always meant. The ON arm reports the store's slots --
        // holes included, because that is the space the holes occupy -- plus both
        // handle vectors.
        std::size_t container_slot_bytes() const {
            if (!stable_) return vec_.capacity() * sizeof(OverflowRecord);
            return store_.size() * sizeof(OverflowRecord) +
                   order_.capacity() * sizeof(OrderedEntry) +
                   free_slots_.capacity() * sizeof(std::uint32_t);
        }

        // What this container owns, in the terms the E9 decomposition already uses: the
        // container object, its slot arrays, and each LIVE record's suffix buffer. The
        // OFF branch is the expression memory_breakdown used inline before O1, so the
        // frozen build's space numbers do not move. The ON branch counts the deque by
        // its slot count and does not model the deque's own chunk map, so the ON figure
        // is a lower bound on the real footprint -- never an overstatement.
        std::size_t allocated_bytes() const {
            if (!stable_) {
                std::size_t total = sizeof(std::vector<OverflowRecord>) +
                                    vec_.capacity() * sizeof(OverflowRecord);
                for (const auto& e : vec_) total += e.suffix.capacity() + 1;
                return total;
            }
            std::size_t total = sizeof(OverflowList) + container_slot_bytes();
            for (const OrderedEntry& e : order_) total += store_[e.handle].suffix.capacity() + 1;
            return total;
        }


        // Branch C: the element capacity, WITH ITS UNIT STATED. The OFF arm is the
        // vector's element capacity; the ON arm's order array is the handle vector whose
        // capacity is the analogous quantity. There is deliberately no attempt to give the
        // ON arm a single "element capacity": container_slot_bytes() sums three arrays
        // whose elements are 48, 8 and 4 bytes, and collapsing them would invent a number.
        std::size_t capacity_elems() const { return stable_ ? order_.capacity() : vec_.capacity(); }

        // Branch C: which representation this container latched at construction. Recorded
        // per checkpoint because stable_ comes from the Config at construction time, so it
        // is a per-RUN fact and not a property anyone may assume from the shipped default.
        bool stable_representation() const { return stable_; }

        // Branch C: Term A needs the capacity, Term B needs the high-water. hw_ is this
        // container's running maximum live count -- the peak it has to be compared against.
        std::size_t size_high_water() const {
#ifdef E25_DIAG
            return hw_;
#else
            return size();
#endif
        }

        // drops    -- how many times the live count FELL
        // regrowth -- how many times it came back to the peak it had fallen from
        std::uint64_t drop_events() const {
#ifdef E25_DIAG
            return drops_;
#else
            return 0;
#endif
        }

        std::uint64_t regrowth_events() const {
#ifdef E25_DIAG
            return regrowth_;
#else
            return 0;
#endif
        }

        // Branch C (2026-10-10): how many times this container exceeded its OWN previous
        // peak. That is the count of reallocations a shrink-to-peak policy would have paid
        // and the current geometric-capacity policy does not.
        std::uint64_t new_peak_events() const {
#ifdef E25_DIAG
            return peaks_;
#else
            return 0;
#endif
        }

        // How many times the SHIPPED policy had to grow this container. The difference
        // against new_peak_events() is the reallocation count the shrink would add.
        std::uint64_t cap_change_events() const {
#ifdef E25_DIAG
            return cap_changes_;
#else
            return 0;
#endif
        }

    private:
#ifdef E25_DIAG
        // Called at the END of every insert_at and erase_at, both arms, so the trackers see
        // every size change and nothing else.
        void track_insert() {
            const std::size_t now = size();
            const std::size_t cap = capacity_elems();
            if (cap != last_cap_) {
                if (last_cap_ != 0) {
                    ++cap_changes_;
                    E25_INC(ov_cap_change_events);
                    E25_ADD(ov_cap_change_copy_bytes, (now - 1) * sizeof(OverflowRecord));
                }
                last_cap_ = cap;
            }
            if (now > hw_) {
                if (hw_ != 0) {
                    ++peaks_;
                    E25_INC(ov_new_peak_events);
                    E25_ADD(ov_new_peak_overshoot, now - hw_);
                    E25_ADD(ov_new_peak_copy_bytes, now * sizeof(OverflowRecord));
                }
                hw_ = now;
            }
            if (fell_since_peak_ && now >= hw_) { ++regrowth_; fell_since_peak_ = false; }
        }

        void track_erase() {
            ++drops_;
            if (size() < hw_) fell_since_peak_ = true;
        }

        std::size_t hw_ = 0;
        std::size_t last_cap_ = 0;
        std::uint64_t peaks_ = 0;
        std::uint64_t cap_changes_ = 0;
        std::uint64_t drops_ = 0;
        std::uint64_t regrowth_ = 0;
        bool fell_since_peak_ = false;
#else
        void track_insert() {}
        void track_erase() {}
#endif

    public:
        bool stable_ = false;
        bool tag_inline_ = false;                // E36: the order entry carries a tag copy
        std::vector<OverflowRecord> vec_;        // OFF arm: the ordered vector
        std::deque<OverflowRecord> store_;       // ON arm: records never move once stored
        std::vector<OrderedEntry> order_;        // ON arm: sorted entries into store_
        std::vector<std::uint32_t> free_slots_;  // ON arm: the holes an erase left behind
    };

    using Overflow = OverflowList;

    Config config_;

    // E8: per-group capacity table. Uniform builds fill every group with the same
    // (slots, block bytes); demand builds give crowded groups more of both.
    //
    // E9: a group owns subblocks(g) consecutive, EQUAL-SIZED sub-blocks, and the
    // tables below describe ONE sub-block rather than the whole group. The count
    // is derived from off_ and bb_, so multi-block layouts add no locator metadata
    // at all. nb == 1 reproduces the E8 behaviour exactly.

    std::vector<std::uint16_t> sg_;   // slots per sub-block (<= MAX_SLOTS)
    std::vector<std::uint32_t> bb_;   // bytes per sub-block (64-aligned)
    std::vector<std::size_t> off_;    // group start offsets, size groups_+1
    // E31H: per-group reclaim hysteresis state, sized like the other per-group tables and
    // allocated only when the switch is on, so a build with it off pays nothing. Remembering
    // per group the operation counter and the garbage at its last attempt is the checklist's
    // "record the layout version and the reason the last attempt failed" -- a repeat is
    // allowed only once the group's state has materially moved.
    std::vector<std::uint64_t> reclaim_last_op_;     // placements_since_ at the last attempt
    std::vector<std::size_t> reclaim_last_garbage_;  // garbage bytes at the last attempt
    std::uint64_t placements_since_ = 0;             // successful placements, the interval scale
    static constexpr std::uint64_t kNoReclaimAttempt = ~std::uint64_t(0);
    // 64 successful placements. E30 measured the reclaim pass at 6.1-11.4 ns/record against
    // 163 ns for the whole-group compaction it displaces, and E31 measured the trigger firing
    // 127,780-255,882 times at the 200% checkpoint while enabling only 20.4%-67.2% of the
    // placements that raised it. Bounding a group's repeat attempts to one per 64 writes is
    // small next to a group's own live count (up to MAX_SLOTS records), so a group whose
    // layout did change is retried long before it could matter, while one whose layout did
    // not is not retried at all.
    static constexpr std::uint64_t kMinReclaimGap = 64;

    std::string prefix_;
    Model model_;

    // ---- E38: the LITS HPT backend -------------------------------------------------------
    //
    // The shared table is trained ONCE for the whole index and owned by Index; every region
    // holds a reference to the same object and adds its own three numbers. There is no
    // per-region 2 MiB table (task book section 2.2).
    //
    // `shared_ptr<const ...>` rather than a raw pointer, because the lifetime question is the
    // one the task book calls out: a region must not outlive the table it reads. The const
    // makes "a region never trains or mutates the shared table" a compile-time property.
    std::shared_ptr<const lits_hpt::SharedModel> lits_shared_;
    lits_hpt::RegionModel lits_;
    // Resolved ONCE, in the constructor. Every later decision -- the group mapping, the
    // fingerprint offset, the demand pass -- reads this flag rather than re-deriving it.
    bool use_lits_ = false;

    // ---- BlockLinearRank (2026-10-11) ---------------------------------------------------
    // A region-SCOPED rank model: no shared table, fitted from this region's own sorted keys,
    // and it owns everything it keeps -- a prefix LENGTH plus its segments, never a view into
    // the training keys. That lifetime rule is the one the LITS round had to fix, applied
    // before the fact instead of after.
    blockrank::RegionModel block_rank_;
    // Resolved ONCE, in the constructor, exactly like use_lits_: every later decision (group
    // mapping, demand pass, query path) reads this flag rather than re-deriving it. All four
    // model statuses define an answer, so this is set on every construction path.
    bool use_block_rank_ = false;
    inline static BlockRankStats block_rank_stats_{};

    // ---- EqualDepthDir (2026-10-11) -----------------------------------------------------
    // A region-scoped separator directory. It owns its arena and offsets; nothing it keeps
    // points into the training keys. Same lifetime rule as the rank model above.
    equivdir::RegionModel depth_dir_;
    bool use_depth_dir_ = false;
    inline static DepthDirStats depth_dir_stats_{};

    // Fallback coverage, counted rather than argued about. The task book's section 3.4 requires
    // a region whose calibration fails to fall back to Context16 and the experiment to REPORT
    // how many regions, records and queries that affects -- so the numbers are kept here and
    // read back through Region::lits_stats(). The struct itself lives at namespace scope: a
    // nested one with default member initializers cannot be used by an inline static member of
    // its own enclosing class (the class is not complete yet at that point).
    inline static LitsStats lits_stats_{};
    std::byte* storage_ = nullptr;
    std::uint32_t groups_ = 1;
    std::size_t size_ = 0, overflow_count_ = 0;
    std::size_t compactions = 0;   // E9 gate instrumentation: cross-sub-block repacks
    std::vector<std::unique_ptr<Overflow>> overflow_owners_;

    // E11 boundary directory. One entry per NON-EMPTY sub-block of a multi-block
    // group: the sub-block's minimum suffix and its index, in sub-block order.
    // Boundaries own their bytes (a copy, not a view into a block) so compaction
    // can never leave a dangling reference. Single-block groups keep this empty
    // and pay nothing but the outer vector slot.
    struct Boundary {
        std::string key;
        std::uint32_t subblock = 0;
    };
    std::vector<std::vector<Boundary>> dirs_;
    std::size_t dir_rebuilds = 0, dir_entries_built = 0;
    bool suppress_dir_rebuild = false;   // set while compact() replays a whole group

    // --- per-group geometry ---

    std::size_t slots_of(std::uint32_t g) const { return sg_[g]; }

    std::size_t start_of(std::uint32_t g) const {
        return sizeof(Header) + std::size_t(sg_[g]) * sizeof(Entry);
    }

    // E9: sub-blocks of a group are consecutive and equal-sized, so their number
    // and base addresses follow from the two existing tables. No new metadata.
    std::size_t subblocks(std::uint32_t g) const {
        return (off_[g + 1] - off_[g]) / bb_[g];
    }

    std::size_t subblock_offset(std::uint32_t g, std::size_t i) const {
        return off_[g] + i * bb_[g];
    }

    std::byte* base_of(std::uint32_t g, std::size_t i) {
        return config_.offset_directory
            ? storage_ + subblock_offset(g, i)
            : storage_ + std::size_t(g) * config_.block_bytes;
    }

    const std::byte* base_of(std::uint32_t g, std::size_t i) const {
        return config_.offset_directory
            ? storage_ + subblock_offset(g, i)
            : storage_ + std::size_t(g) * config_.block_bytes;
    }

    Header& header(std::uint32_t g, std::size_t i) {
        return *reinterpret_cast<Header*>(base_of(g, i));
    }
    const Header& header(std::uint32_t g, std::size_t i) const {
        return *reinterpret_cast<const Header*>(base_of(g, i));
    }

    Entry* entries(std::uint32_t g, std::size_t i) {
        return reinterpret_cast<Entry*>(base_of(g, i) + sizeof(Header));
    }
    const Entry* entries(std::uint32_t g, std::size_t i) const {
        return reinterpret_cast<const Entry*>(base_of(g, i) + sizeof(Header));
    }

    // The overflow container belongs to the logical group and is anchored in
    // sub-block 0; the other sub-blocks leave overflow_pointer at zero.
    Overflow* overflow(std::uint32_t g) const {
        return reinterpret_cast<Overflow*>(static_cast<std::uintptr_t>(header(g, 0).overflow_pointer));
    }

    std::string_view suffix(std::uint32_t g, std::size_t i, const Entry& e) const {
        assert(e.offset >= start_of(g) && std::uint64_t(e.offset) + e.length <= bb_[g]);
        return {reinterpret_cast<const char*>(base_of(g, i) + e.offset), e.length};
    }

    bool prefix_matches(std::string_view s) const {
        return s.size() >= prefix_.size() &&
               (!prefix_.size() || std::memcmp(s.data(), prefix_.data(), prefix_.size()) == 0);
    }

    // E38: THE ONE PLACE THE BACKEND IS CHOSEN.
    //
    // TAKES ONLY THE FULL KEY, and derives the suffix itself.
    //
    // The earlier version took both, on the theory that a second parameter would catch a caller
    // who passed a suffix where a full key was wanted. It does force every OLD call site to be
    // edited -- but the two quantities have the SAME TYPE, so `canonical_group(suffix, full_key)`
    // and `canonical_group(suffix, suffix)` compile just as happily. Two parameters of one type
    // are not evidence that the coordinates are right; only the implementation and its tests
    // are. Deriving the suffix HERE removes the second argument, and with it the whole class of
    // mistake: there is no longer a coordinate for a caller to supply wrongly.
    //
    //   reference model  maps the SUFFIX (its table is trained on suffixes and its rows are
    //                    indexed by a suffix-relative position)
    //   LITS HPT         maps the FULL KEY (the source's walk starts at an absolute position:
    //                    lits_node.hpp:292 passes `ccpl + icpl`, and this region's prefix
    //                    length is that same quantity -- see the adapter's header)
    //
    // M0 IS UNCHANGED BY THE DERIVATION: every call site used to compute
    // `full_key.substr(prefix_.size())` itself and pass it, so this is the identical value.
    std::uint32_t canonical_group(std::string_view full_key) const {
        E18_INC(group_calls);

        Fingerprint dummy;
        const std::uint32_t group = use_depth_dir_
            ? depth_dir_.predict_group(full_key)
            : use_block_rank_
                ? block_rank_.predict_group(full_key)
                : use_lits_
                    ? lits_.predict_group(full_key)
                    : model_.predict<false>(full_key.substr(prefix_.size()), groups_,
                                            config_.canonical, dummy).group;

#ifdef E25_DIAG
        // check_scan_contract's negative control. See the flag's own comment for what
        // this is for; with the flag false -- every other gate, every timed build --
        // this is the same function it was.
        if (sar_rotate_groups && groups_ > 1) return (group + 1) % groups_;
#endif

        return group;
    }

    // E38: the FUSED query path's model call, backend-dispatched.
    //
    // This is the one place a prediction also touches the fingerprint, and the two quantities
    // the caller needs are separate fields on purpose:
    //
    //   consumed     what the MODEL processed  -> `result.steps`, published since E5
    //   fp_consumed  what the HASH has absorbed -> where `fp.tail` continues from
    //
    // The reference model reads a prefix of the suffix and feeds every byte into the hash, so
    // the two are equal. The LITS walk reads the FULL key from an absolute position, so the
    // bytes it touches are not a prefix of the suffix at all -- hashing them would produce a tag
    // matching no stored record, which is the failure check_lits_fingerprint exists to catch.
    // It therefore reports fp_consumed = 0 and lets the caller hash the whole suffix.
    template <bool Hash>
    Model::Prediction predict_for(std::string_view full_key, std::string_view suffix, Route route,
                                  Fingerprint& fp) const {
        if (use_depth_dir_) {
            // The directory reads the full key from arbitrary positions of the suffix, so it
            // contributes nothing to the fingerprint -- same split as the two branches below.
            Model::Prediction p;
            p.group = depth_dir_.predict_group(full_key, p.consumed);
            p.fp_consumed = 0;

            return p;
        }

        if (use_block_rank_) {
            // Same two-field split as the LITS branch, for the same reason: the rank model reads
            // the full key from an absolute position, so the bytes it touches are NOT a prefix of
            // the suffix, and hashing them would produce a tag no stored record has. It reports
            // fp_consumed = 0 and the caller hashes the whole suffix.
            Model::Prediction p;
            p.group = block_rank_.predict_group(full_key, p.consumed);
            p.fp_consumed = 0;

            return p;
        }

        if (use_lits_) {
            Model::Prediction p;
            p.group = lits_.predict_group(full_key, p.consumed);
            p.fp_consumed = 0;

            return p;
        }

        return model_.predict<Hash>(suffix, groups_, route, fp);
    }

    // --- E11 boundary directory ---

    // Rebuild one group's directory from its current sub-blocks. Called whenever a
    // sub-block's minimum key can have changed: a front insert, a front erase, or a
    // repack. O(sub-blocks), and only for multi-block groups.
    void rebuild_dir(std::uint32_t g) {
        if (dirs_.empty()) return;

        auto& dir = dirs_[g];
        dir.clear();

        const std::size_t nb = subblocks(g);
        if (nb < 2) return;   // single-block groups keep no directory

        ++dir_rebuilds;
        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            if (!h.count) continue;
#ifdef E25_DIAG
            // I10 injection point 2: the pre-I10 design reaches here AFTER the group has
            // been rewritten, and `dir.clear()` has already destroyed the old directory --
            // so an interruption here is exactly the half-update the review found.
            if (maybe_inject(2)) throw std::bad_alloc();
#endif
            dir.push_back({std::string(suffix(g, i, entries(g, i)[0])),
                           static_cast<std::uint32_t>(i)});
            ++dir_entries_built;
        }
    }

    void rebuild_all_dirs() {
        for (std::uint32_t g = 0; g < groups_; ++g) rebuild_dir(g);
    }

    // The sub-block that can hold s, or NO_SUBBLOCK when s sorts below every
    // primary key. M1 makes this the ONLY candidate: everything in an earlier
    // sub-block is smaller than that sub-block's minimum, and everything in a
    // later one is larger. NOT finding s here says nothing about the overflow
    // container -- the caller must still search it.
    std::size_t locate_subblock(std::uint32_t g, std::string_view s, Result& r) const {
        const auto& dir = dirs_[g];
        std::size_t lo = 0, hi = dir.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            E12_TICK(r.locate_probes);
#ifdef E7_COUNT_BYTES
            const int c = compare_bytes(dir[mid].key, s, r.locate_cmp_bytes);
#else
            const int c = compare(dir[mid].key, s);
#endif
            if (c <= 0) lo = mid + 1;
            else hi = mid;
        }
        return lo == 0 ? NO_SUBBLOCK : dir[lo - 1].subblock;
    }

    // --- E16: derivations shared by the constructor and the statistics-only path
    //
    // Extracted verbatim from the constructor so the statistics-only path below
    // uses the SAME formulas instead of a copy of them. Each is exactly the
    // expression the constructor used inline, in the same order. check_e16_probe
    // compares the two paths field by field over many configurations, so a
    // divergence introduced here is caught rather than silently absorbed.
    //
    // They take a [first, last) range rather than a vector so a region's slice can
    // be addressed IN PLACE. Materialising that slice as a vector<Pair> is itself
    // measurable work (E14/E15 §3b), and the whole point of this path is to avoid
    // work the probe was doing.
    // -----------------------------------------------------------------------

    static std::string compute_prefix(const Pair* first, const Pair* last) {
        if (first == last) return {};
        const auto& a = first->first;
        const auto& b = (last - 1)->first;
        std::size_t p = 0;
        while (p < std::min(a.size(), b.size()) && a[p] == b[p]) ++p;
        return a.substr(0, p);
    }

    static std::uint64_t suffix_bytes_of(const Pair* first, const Pair* last, std::size_t prefix_len) {
        std::uint64_t n = 0;
        for (const Pair* it = first; it != last; ++it) n += it->first.size() - prefix_len;
        return n;
    }

    // E21: an identity for the string -> group mapping, without predicting anything.
    //
    // The mapping is (prefix, trained table, logical group count, routing mode).
    // Folding all four into one value means a single comparison decides whether a
    // context may be reused -- and it catches the two cases prefix+count cannot:
    // a different key set, and the same keys under a different model kind. If the
    // digest matches, the predictions are equal by construction.
    //
    // Cost is a pass over the trained table (16 x 258 u32 = 16.5 KiB per region),
    // and it is only paid when a preset is actually offered.
    static std::uint64_t model_identity(const Model& m, std::uint32_t groups, Route canonical) {
        std::uint64_t h = 1469598103934665603ULL;
        // One FNV step per BYTE. An earlier version stepped once per 8-byte value,
        // which cost eight multiplies per table entry -- about 3 ms per build across
        // two passes, i.e. the whole of what R2 saved. Byte-wise is the same hash
        // strength at roughly an eighth of the cost.
        auto mix_bytes = [&h](const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ULL;
            }
        };
        const std::uint64_t kind = static_cast<std::uint64_t>(m.kind);
        const std::uint64_t g = groups;
        const std::uint64_t c = static_cast<std::uint64_t>(canonical);
        mix_bytes(&kind, sizeof(kind));
        mix_bytes(&g, sizeof(g));
        mix_bytes(&c, sizeof(c));
        for (const auto& row : m.cumulative)
            mix_bytes(row.data(), row.size() * sizeof(std::uint32_t));
        return h;
    }

    // The identity the R1/R2 preset guard compares.
    //
    // For the reference and LITS backends this is the Context16 table digest it has always been
    // (the LITS table is index-wide, so both sides hold the identical object). For
    // BlockLinearRank the region's OWN fitted model is mixed in, because under that backend the
    // rank model IS the mapping: two different rank models that happened to leave the reference
    // table equal must not look like the same mapping to the guard.
    static std::uint64_t mapping_identity(const Model& m, std::uint32_t groups, Route canonical,
                                          ModelBackend backend,
                                          const blockrank::RegionModel& block_rank,
                                          const equivdir::RegionModel& depth_dir) {
        std::uint64_t h = model_identity(m, groups, canonical);
        if (is_depth_dir_backend(backend)) {
            const std::uint64_t dtag = depth_dir.identity();
            auto mix = [&h](const void* p, std::size_t n) {
                const auto* b = static_cast<const unsigned char*>(p);
                for (std::size_t i = 0; i < n; ++i) {
                    h ^= b[i];
                    h *= 1099511628211ULL;
                }
            };
            mix(&dtag, sizeof(dtag));
            return h;
        }
        if (backend != ModelBackend::BlockLinearRank) return h;
        const std::uint64_t tag = block_rank.identity();
        auto mix_bytes = [&h](const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ULL;
            }
        };
        mix_bytes(&tag, sizeof(tag));
        return h;
    }

    static std::uint32_t derive_groups(std::size_t n, std::uint64_t suffix_bytes, const Config& cfg) {
        const double count_groups = n / (double(cfg.slots) * cfg.load);
        const double byte_groups = suffix_bytes /
            ((cfg.block_bytes - sizeof(Header) - std::size_t(cfg.slots) * sizeof(Entry)) * cfg.load);
        const auto estimate = static_cast<std::uint64_t>(std::max(count_groups, byte_groups)) + 1;
        if (estimate > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("too many groups");
        return cfg.fixed_groups
            ? cfg.fixed_groups
            : static_cast<std::uint32_t>(std::max<std::uint64_t>(1, estimate));
    }

    // --- write paths ---

    // Insert into one sub-block at its sorted position; false if it has no room.

#ifdef E25_DIAG
    // Walk the occupied region [h.bottom, bb_) and measure the gaps between live suffixes.
    //
    // Intervals are disjoint by construction -- an insert takes fresh space below the
    // bottom and a repack re-packs without overlap -- so one sorted sweep is exact. The
    // `cur` guard keeps a hypothetical overlap from producing a negative gap rather than
    // an obviously wrong one.
    std::size_t diag_holes(std::uint32_t g, std::size_t blk, std::size_t& best,
                           std::size_t& gap_sum, std::size_t& live_out) const {
        const auto& h = header(g, blk);
        const auto* es = entries(g, blk);
        const std::size_t n = h.count;

        std::size_t iv[MAX_SLOTS][2];
        std::size_t live = 0;

        for (std::size_t j = 0; j < n; ++j) {
            iv[j][0] = es[j].offset;
            iv[j][1] = std::size_t(es[j].offset) + es[j].length;
            live += es[j].length;
        }

        for (std::size_t a = 1; a < n; ++a) {
            const std::size_t lo = iv[a][0], hi = iv[a][1];
            std::size_t b = a;
            while (b > 0 && iv[b - 1][0] > lo) {
                iv[b][0] = iv[b - 1][0];
                iv[b][1] = iv[b - 1][1];
                --b;
            }
            iv[b][0] = lo;
            iv[b][1] = hi;
        }

        best = 0;
        gap_sum = 0;
        std::size_t nholes = 0;
        std::size_t cur = h.bottom;

        for (std::size_t j = 0; j < n; ++j) {
            if (iv[j][0] > cur) {
                const std::size_t gap = iv[j][0] - cur;
                ++nholes;
                gap_sum += gap;
                if (gap > best) best = gap;
            }
            if (iv[j][1] > cur) cur = iv[j][1];
        }

        if (std::size_t(bb_[g]) > cur) {
            const std::size_t gap = std::size_t(bb_[g]) - cur;
            ++nholes;
            gap_sum += gap;
            if (gap > best) best = gap;
        }

        live_out = live;
        return nholes;
    }

    // Called from the ONE branch that refuses a record for want of bytes.
    void diag_note_byte_failure(std::uint32_t g, std::size_t blk, std::size_t len) {
        const auto& h = header(g, blk);
        std::size_t best = 0, gap_sum = 0, live = 0;
        const std::size_t nholes = diag_holes(g, blk, best, gap_sum, live);

        const std::size_t contig = std::size_t(h.bottom) - start_of(g);
        const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
        const std::size_t garbage = occupied - live;

        // The sweep and the index's own accounting describe the same region, so they must
        // agree. A mismatch means the sweep is wrong, and a wrong sweep would make every
        // downstream ratio meaningless while still looking plausible.
        E25_INC(hole_sweep_checks);
        if (gap_sum != garbage) E25_INC(hole_sweep_mismatch);

        const bool fits_hole = best >= len;
        const bool fits_repack = (contig + garbage) >= len;
        const bool slots_free = h.count < slots_of(g);

        if (fits_hole) E25_INC(fail_bytes_hole_fits);
        else E25_INC(fail_bytes_hole_absent);

        if (fits_repack) E25_INC(fail_bytes_repack_fits);

        // Structure, over the FULL population. A histogram built from the capped event log
        // cannot answer this, and the difference is not academic: the retained rows are the
        // earliest, least-dead-byte refusals.
        if (nholes == 0) E25_INC(hole_nholes0);
        else if (nholes == 1) E25_INC(hole_nholes1);
        else E25_INC(hole_nholes2plus);

        if (best == garbage) E25_INC(hole_best_eq_garbage);
        if (h.count < 2) E25_INC(hole_count_lt2);
        // The split the plan asks for: an EMPTY block and a ONE-record block are different
        // opportunities (nothing to move vs. one suffix to move), and hole_count_lt2 alone
        // cannot say which one the refusals land on. Both are subsets of hole_count_lt2.
        if (h.count == 0) E25_INC(hole_count_eq0);
        else if (h.count == 1) E25_INC(hole_count_eq1);
        if (!slots_free) E25_INC(hole_slots_not_free);
        if (h.count >= 2 && fits_repack) E25_INC(fail_bytes_repack_guarded);

        E25_ADD(fail_bytes_len_sum, len);
        E25_ADD(fail_bytes_best_hole_sum, best);
        E25_ADD(fail_bytes_garbage_sum, garbage);
        E25_ADD(fail_bytes_live_sum, live);

        if (diag_hole_total() < kMaxHoleEvents) {
            HoleEvent e;
            e.group = g;
            e.blk = static_cast<std::uint16_t>(blk);
            e.len = static_cast<std::uint16_t>(len);
            e.contig = static_cast<std::uint16_t>(contig);
            e.garbage = static_cast<std::uint16_t>(garbage);
            e.live = static_cast<std::uint16_t>(live);
            e.nholes = static_cast<std::uint16_t>(nholes);
            e.best_hole = static_cast<std::uint16_t>(best);
            e.fits_hole = fits_hole;
            e.fits_repack = fits_repack;
            e.slots_free = slots_free;
            hole_events_.push_back(e);
            ++diag_hole_total();
            E25_INC(hole_events_recorded);
        } else {
            E25_INC(hole_events_dropped);
        }
    }

#endif
    bool insert_into(std::uint32_t g, std::size_t i, std::string_view s,
                     std::uint64_t value, std::uint8_t tag) {
        E18_INC(insert_attempts);
        E25_INC(place_attempts);
        auto& h = header(g, i);
        // The two capacity failures are counted apart: "no free slot" and "not enough
        // suffix bytes" call for different remedies, and a combined counter cannot
        // tell them apart.
        if (h.count >= slots_of(g)) { E25_INC(place_fail_slots); return false; }
        if (s.size() > h.bottom - start_of(g)) {
            E25_INC(place_fail_bytes);
#ifdef E25_DIAG
            diag_note_byte_failure(g, i, s.size());
#endif
            // Candidate H (primary_hole_reuse): before the caller's repack machinery runs,
            // try the block's own dead ranges. On failure this returns false exactly as the
            // original code did, with the block untouched.
            if (config_.primary_hole_reuse && config_.subblock_reclaim &&
                insert_into_hole(g, i, s, value, tag))
                return true;
            return false;
        }
        E18_INC(insert_ok);
        E18_ADD(suffix_bytes_copied, s.size());

        auto* es = entries(g, i);
        std::size_t p = 0;
        while (p < h.count && E25_CMP(suffix(g, i, es[p]), s) < 0) ++p;
        E18_ADD(entry_shifts, h.count - p);
        for (std::size_t j = h.count; j > p; --j) {
            es[j] = es[j - 1];
            h.tags[j] = h.tags[j - 1];
        }
        h.bottom = static_cast<std::uint16_t>(h.bottom - s.size());
        if (!s.empty()) std::memcpy(base_of(g, i) + h.bottom, s.data(), s.size());
        es[p] = {h.bottom, static_cast<std::uint32_t>(s.size()), value};
        h.tags[p] = tag;
        ++h.count;

        // A front insert becomes this sub-block's new minimum.
        if (p == 0 && !suppress_dir_rebuild) rebuild_dir(g);
        return true;
    }

    // Candidate H (primary_hole_reuse, default OFF): place the suffix into a DEAD RANGE of
    // the SAME sub-block instead of repacking the block for contiguous room.
    //
    // H does not touch existing suffixes -- it finds a gap between them, writes the new
    // suffix there, inserts the descriptor in KEY order and leaves h.bottom where it was.
    // The bytes H consumes come from a dead range, not from the contiguous region below
    // h.bottom, so the block's occupancy does not grow.
    //
    // Scope is FIXED and enforced here: same block only; at least two live records; at most
    // 48 (a fixed stack bound, not a tuned parameter); the first fitting gap scanning
    // PHYSICALLY from high to low; no persistent hole table, no new header field, nothing
    // cached across calls. On failure nothing is modified and the caller's original control
    // flow continues.
    bool insert_into_hole(std::uint32_t g, std::size_t i, std::string_view s,
                          std::uint64_t value, std::uint8_t tag) {
        auto& h = header(g, i);
        if (h.count >= slots_of(g)) return false;   // slots come first: H creates none
        if (h.count < 2) { E25_INC(hole_reuse_skipped_lt2); return false; }
        if (h.count > 48) { E25_INC(hole_reuse_skipped_wide); return false; }
        E25_INC(hole_reuse_attempts);

        const auto* es = entries(g, i);
        const std::size_t n = h.count;
        std::size_t order[48];
        for (std::size_t j = 0; j < n; ++j) order[j] = j;
        for (std::size_t a = 1; a < n; ++a) {
            // The same descending-by-offset insertion sort repack_subblock uses; the
            // physical order is NOT assumed to match the descriptor order.
            const std::size_t key = order[a];
            const std::uint32_t koff = es[key].offset;
            std::size_t b = a;
            while (b > 0 && es[order[b - 1]].offset < koff) { order[b] = order[b - 1]; --b; }
            order[b] = key;
        }

        // Gaps, from the top of the block downward. A gap [low, cur) sits above the record
        // whose top is `low`; the new suffix goes at the gap's LOW end, so it never crosses
        // bb_ and whatever remains of the gap stays directly above it.
        std::size_t pos = SIZE_MAX;
        std::size_t cur = bb_[g];
        for (std::size_t a = 0; a < n; ++a) {
            const auto& e = es[order[a]];
            const std::size_t low = std::size_t(e.offset) + e.length;
            if (cur >= low && cur - low >= s.size()) { pos = low; break; }
            cur = e.offset;
        }
        if (pos == SIZE_MAX && cur >= h.bottom && cur - h.bottom >= s.size())
            pos = h.bottom;                          // the gap below the lowest record
        if (pos == SIZE_MAX) { E25_INC(hole_reuse_absent); return false; }

        auto* esw = entries(g, i);
        std::size_t p = 0;
        while (p < h.count && E25_CMP(suffix(g, i, esw[p]), s) < 0) ++p;
        E18_ADD(entry_shifts, h.count - p);
        for (std::size_t j = h.count; j > p; --j) {
            esw[j] = esw[j - 1];
            h.tags[j] = h.tags[j - 1];
        }
        if (!s.empty()) std::memcpy(base_of(g, i) + pos, s.data(), s.size());
        esw[p] = {static_cast<std::uint32_t>(pos), static_cast<std::uint32_t>(s.size()), value};
        h.tags[p] = tag;
        ++h.count;
        if (p == 0 && !suppress_dir_rebuild) rebuild_dir(g);
        E25_INC(hole_reuse_placed);
        E25_ADD(hole_reuse_bytes, s.size());
        return true;
    }

    // E9: a group's sub-blocks are rank-ordered, so every key in sub-block i is
    // smaller than every key in sub-block i+1 (invariant M1, which range scan
    // relies on). Placement below keeps M1 without ever shifting across blocks:
    // s goes into the first sub-block whose upper bound is >= s, or into the gap
    // just before it when that sub-block is full, or after every existing key.
    // E31: which sub-block would put_primary try FIRST for this key? Factored out of
    // put_primary's own enumeration so the reclaim and the placement cannot drift apart.
    std::size_t first_candidate(std::uint32_t g, std::string_view s) const {
        E25_INC(fc_calls);
        E30_W(fc);
        const std::size_t nb = subblocks(g);
        std::size_t prev = SIZE_MAX;
        for (std::size_t i = 0; i < nb; ++i) {
            E25_INC(fc_probes);
            const auto& h = header(g, i);
            if (h.count == 0) continue;
            const auto* es = entries(g, i);
            if (compare(suffix(g, i, es[h.count - 1]), s) < 0) { prev = i; continue; }
            if (prev != SIZE_MAX && compare(s, suffix(g, i, es[0])) < 0) {
                E25_INC(fc_gap);   // the two-block path: put_primary tries prev, then i
                return prev;
            }
            return i;
        }
        if (prev == SIZE_MAX) return 0;
        E25_INC(fc_gap);           // s above every block: the append path
        return prev;
    }

    // E31: pack this sub-block's live suffixes toward the block end, in place.
    //
    // Entries are processed in DECREASING offset order. That ordering is what makes a
    // single buffer safe: each record moves to a destination above the region still
    // holding un-read records, and memmove covers the self-overlap. Descriptors do not
    // move (only their offsets change), nothing is allocated, no hash is recomputed, and
    // the boundary directory is untouched because the sub-block's key range is
    // unchanged -- its first and last entries keep their slots.
    void repack_subblock(std::uint32_t g, std::size_t blk) {
        // Repack split (2026-10-11): one reading at entry, ONE SHARED boundary after the
        // sort (sort's end == move's start) and one at the very end. The accumulation
        // happens after the last reading, so the write-back is outside the window, and no
        // second RAII pair is nested inside the parent clock. The parent keeps its
        // stage-D meaning: every call in w_reclaim_calls, and the n<2 early return is
        // charged to the parent alone, closed at the return itself.
#if defined(E30_TIME) && defined(E30_WRITE)
        const std::uint64_t rp_t0 = E30_T0();
#endif
        auto& h = header(g, blk);
        const std::size_t n = h.count;

        // An optimisation, not a precondition. The loop below already handles n == 0 (it
        // does not run, so top stays at bb_ and bottom becomes bb_, the whole area free) and
        // n == 1 (the single record is packed to the top). The candidate lifts it because a
        // block holding 0 or 1 records is where most late-checkpoint byte refusals land.
        if (n < 2) {
            if (!config_.reclaim_low_activity) {
#if defined(E30_TIME) && defined(E30_WRITE)
                E30_ADD(w_reclaim_ns, rp_t0);
                E30_INC(w_reclaim_calls);
#endif
                return;
            }
            E25_INC(low_activity_reclaimed);
            // The recovered bytes, measured BEFORE the packing loop moves anything, and
            // with subblock_garbage's own definition of garbage. The entries are read here
            // rather than in subblock_garbage() because the loop below rewrites offsets.
            const auto* ea = entries(g, blk);
            std::size_t la_live = 0;
            for (std::size_t j = 0; j < n; ++j) la_live += ea[j].length;
            const std::size_t la_occupied = std::size_t(bb_[g]) - h.bottom;
            E25_ADD(low_activity_bytes, la_occupied > la_live ? la_occupied - la_live : 0);
            if (n == 0) E25_INC(low_activity_n0); else E25_INC(low_activity_n1);
        }
        E25_INC(rp_calls);
        auto* es = entries(g, blk);
        std::byte* base = base_of(g, blk);

        std::size_t order[48];
        for (std::size_t j = 0; j < n; ++j) order[j] = j;
        for (std::size_t a = 1; a < n; ++a) {
            const std::size_t key = order[a];
            const std::uint32_t koff = es[key].offset;
            std::size_t b = a;
            while (b > 0 && es[order[b - 1]].offset < koff) { order[b] = order[b - 1]; --b; }
            order[b] = key;
        }

#if defined(E30_TIME) && defined(E30_WRITE)
        const std::uint64_t rp_t1 = E30_T0();   // sort end == move start (shared)
#endif
        std::size_t top = bb_[g];
        for (std::size_t a = 0; a < n; ++a) {
            Entry& e = es[order[a]];
            top -= e.length;
            std::memmove(base + top, base + e.offset, e.length);
            // Work counters (count build only; the macros evaluate their argument only
            // there, so no comparison lands in the timed loop). The offset is read BEFORE
            // the rewrite: an unchanged offset is a same-address memmove -- counted, and
            // labelled self-copy rather than movement.
            E25_ADD(rp_records, 1);
            E25_ADD(rp_bytes, e.length);
            E25_ADD(rp_moved_records, (e.offset != top) ? 1 : 0);
            E25_ADD(rp_moved_bytes, (e.offset != top) ? e.length : 0);
            E25_ADD(rp_selfcopy_records, (e.offset == top) ? 1 : 0);
            E25_ADD(rp_selfcopy_bytes, (e.offset == top) ? e.length : 0);
            e.offset = static_cast<std::uint32_t>(top);
        }
        h.bottom = static_cast<std::uint16_t>(top);
#if defined(E30_TIME) && defined(E30_WRITE)
        const std::uint64_t rp_t2 = E30_T0();   // move end == whole pass end
        auto& rpT = ::validation::e30::timers();   // write-back, after the window
        rpT.w_reclaim_ns += rp_t2 - rp_t0;
        ++rpT.w_reclaim_calls;
        rpT.w_rp_sort_ns += rp_t1 - rp_t0;
        ++rpT.w_rp_sort_calls;
        rpT.w_rp_move_ns += rp_t2 - rp_t1;
        ++rpT.w_rp_move_calls;
#endif
    }

    // G3/A1': what one sub-block could give back if it were tightened -- the bytes it
    // occupies beyond its live suffixes. Zero means repack_subblock would do nothing (it
    // returns immediately for fewer than two live records, and a block with no slack has
    // nothing to reclaim either).
    std::size_t subblock_garbage(std::uint32_t g, std::size_t blk) const {
        const auto& h = header(g, blk);

        // The same guard as repack_subblock's, and it has to move with it: this function is
        // what reclaim_widen uses to CHOOSE candidates, so leaving it in place would keep the
        // mechanism from ever looking at the blocks the candidate exists for.
        if (h.count < 2 && !config_.reclaim_low_activity) return 0;
        const auto* es = entries(g, blk);
        std::size_t live = 0;
        for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
        const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
        return occupied > live ? occupied - live : 0;
    }

    // G3/A1': how many MORE sub-blocks one failed placement may tighten. FIXED, not tuned
    // per corpus -- the task book's rule for this package is that the budget is a fixed
    // configuration, so it is a constant and not a Config field.
    static constexpr std::size_t kReclaimWidenBudget = 4;

    // G3/A1': tighten the group's OTHER sub-blocks after the rank-selected one was already
    // tried and the placement still failed. Dirtiest first, fixed budget, retry after each.
    //
    // `already_tried` is the block first_candidate chose -- excluded here, because the call
    // site already ran it. Returns true when the placement finally fits.
    //
    // In-place only: repack_subblock changes Entry::offset and Header::bottom and nothing
    // else, so members, slot order, the sub-block key range and the boundary directory are
    // all untouched. Nothing is migrated and no overflow record moves -- that is the whole
    // difference from E37's drain, and the reason this is a clean single-variable change.
    bool reclaim_widen(std::uint32_t g, std::string_view s, std::uint64_t value,
                       std::uint8_t tag, std::size_t already_tried) {
        const std::size_t nb = subblocks(g);
        if (nb < 2) return false;               // nothing the rank-selected pass did not cover

        // Which blocks are worth touching, dirtiest first. A group has ~1.6 sub-blocks on
        // average (S0 snapshot) and at most a handful, so a small fixed array is enough and
        // the budget bounds the work regardless.
        std::size_t order[64];
        std::size_t count = 0;

        for (std::size_t i = 0; i < nb && count < 64; ++i) {
            if (i == already_tried) continue;
            if (subblock_garbage(g, i) == 0) continue;
            order[count++] = i;
        }

        for (std::size_t a = 1; a < count; ++a) {
            const std::size_t key = order[a];
            const std::size_t kg = subblock_garbage(g, key);
            std::size_t b = a;
            while (b > 0 && subblock_garbage(g, order[b - 1]) < kg) {
                order[b] = order[b - 1];
                --b;
            }
            order[b] = key;
        }

        E25_ADD(reclaim_widened_blocks_examined, count);

        const std::size_t budget = count < kReclaimWidenBudget ? count : kReclaimWidenBudget;

        for (std::size_t a = 0; a < budget; ++a) {
            E25_INC(reclaim_widened);
            repack_subblock(g, order[a]);
            if (put_primary(g, s, value, tag)) {
                E25_INC(reclaim_widened_helped);
                return true;
            }
        }

        return false;
    }

    // E37: how many records one drain call may move. A CAP ON RECORDS, not a bound on the
    // call's cost -- see the report's caliber note.
    static constexpr std::size_t kMigrateBudget = 64;

    // E37: move up to `budget` records out of this group's overflow container and back into
    // the primary area it came from.
    //
    // Returns the number moved. A record is taken only when the block that would hold it
    // has a free slot and enough bytes AFTER that block's own garbage is recovered, and the
    // block is repacked before the first record lands in it, because bytes are what binds.
    //
    // Removal is one compaction pass over the container. Candidates are copied out first:
    // the container is about to forget them.
    std::size_t drain_overflow_bounded(std::uint32_t g, std::size_t budget) {
        auto* o = overflow(g);

        if (!o || o->empty() || budget == 0) return 0;

        const std::size_t n = o->size();
        const std::size_t nb = subblocks(g);
        std::vector<std::size_t> pos;
        std::vector<std::size_t> blk_of;
        std::vector<OverflowRecord> moved;
        std::vector<char> reclaimed(nb, 0);

        // The reservation. Without it, 64 candidates each see the room the previous 63 are
        // about to consume, the later inserts fail their own room test, and the records go
        // straight back to overflow -- which is what the repair program caught: 1,239 moved
        // and a net change of 21.
        std::vector<std::uint32_t> vcount(nb);
        std::vector<std::size_t> vfree(nb);

        for (std::size_t i = 0; i < nb; ++i) {
            vcount[i] = header(g, i).count;
            vfree[i] = diag_free_after_reclaim(g, i);
        }

        for (std::size_t j = 0; j < n && pos.size() < budget; ++j) {
            const std::string_view sfx = (*o)[j].suffix;

            // Judged against the VIRTUAL state, so a record only fits if the ones already
            // accepted in this call have left room for it.
            const auto room_virtual = [&](std::size_t i) {
                return vcount[i] < slots_of(g) && sfx.size() <= vfree[i];
            };

            const std::size_t blk = diag_target_with(g, sfx, room_virtual);

            if (blk == SIZE_MAX) continue;

            // Repack before the first record lands here, so the block's real bytes match the
            // virtual view that accepted it. A sub-block with fewer than two entries is left
            // alone by repack_subblock, and for those the virtual view is already the real
            // one.
            if (!reclaimed[blk]) {
                repack_subblock(g, blk);
                reclaimed[blk] = 1;
            }

            ++vcount[blk];
            vfree[blk] -= sfx.size();
            pos.push_back(j);
            blk_of.push_back(blk);
        }

        if (pos.empty()) return 0;

        moved.reserve(pos.size());

        for (const std::size_t p : pos) moved.push_back((*o)[p]);

        for (std::size_t k = 0; k < moved.size(); ++k) {
            E25_ADD(migrated_bytes, moved[k].suffix.size());
            E25_ADD(erase_released_overflow, moved[k].suffix.size());
            E25_INC(string_frees);
            E25_INC(migrated_records);
        }

        o->erase_positions(pos);
        overflow_count_ -= pos.size();

        for (std::size_t k = 0; k < moved.size(); ++k) {
            if (!insert_into(g, blk_of[k], moved[k].suffix, moved[k].value, moved[k].tag)) {
                // The reservation should have made this unreachable; it is counted rather
                // than assumed away, because a drain whose inserts all fail moves nothing and
                // still pays for the repack. The record goes back rather than being dropped
                // -- a lost record would be a silent correctness bug.
                E25_INC(migrate_rejected);
                put_overflow(g, moved[k].suffix, moved[k].value, moved[k].tag);
            }
        }

        return pos.size();
    }

    bool put_primary(std::uint32_t g, std::string_view s, std::uint64_t value, std::uint8_t tag) {
        E25_INC(put_primary_calls);
        E30_W(walk);
        const std::size_t nb = subblocks(g);
        std::size_t prev = SIZE_MAX;   // last non-empty sub-block whose max < s

        for (std::size_t i = 0; i < nb; ++i) {
            E18_INC(subblock_probes);
            const auto& h = header(g, i);
            if (h.count == 0) continue;

            auto* es = entries(g, i);
            if (compare(suffix(g, i, es[h.count - 1]), s) < 0) { prev = i; continue; }

            // s is at or below this sub-block's upper bound: it belongs here, or
            // in the gap immediately before it when it sorts below the first key.
            if (prev != SIZE_MAX && compare(s, suffix(g, i, es[0])) < 0 &&
                insert_into(g, prev, s, value, tag))
                return true;
            return insert_into(g, i, s, value, tag);
        }

        // Greater than every existing key: append to the first sub-block with room
        // at or after `prev`. Starting at `prev` itself is correct (appending there
        // keeps the rank order) and covers the single-sub-block case.
        for (std::size_t j = prev == SIZE_MAX ? 0 : prev; j < nb; ++j)
            if (insert_into(g, j, s, value, tag)) return true;
        return false;
    }

    void put_overflow(std::uint32_t g, std::string_view s, std::uint64_t value, std::uint8_t tag) {
        E18_INC(overflow_inserts);
        auto* o = overflow(g);
        if (!o) {
            E18_INC(overflow_containers);
            // The representation is fixed for the container's whole life, from here.
            overflow_owners_.push_back(std::make_unique<Overflow>(config_.overflow_stable_store, config_.overflow_inline_tag));
            o = overflow_owners_.back().get();
            header(g, 0).overflow_pointer = reinterpret_cast<std::uintptr_t>(o);
        }
#ifdef E30_TIME
        const auto ov_t0 = E30_T0();
#endif
        // The search belongs to the container, so put_overflow does not care which
        // representation is live: lower_bound_pos returns the position in the
        // container's own order, and the OFF arm is the same std::lower_bound call
        // this line used to make.
#ifdef E30_TIME
        const auto ov_l0 = E30_T0();
#endif
        const std::size_t pos = o->lower_bound_pos(s);
#ifdef E30_TIME
        E30_ADD(ov_lookup_ns, ov_l0);
        E30_INC(ov_lookup_calls);
#endif
        // E31 item 3: the distance the insertion has to push up, in records AND in
        // bytes, read while the distance is still valid. In the OFF arm those records
        // are the ones vector::insert relocates. In the ON arm the same distance is
        // the number of 4-byte handles the order array shifts and the records do not
        // move at all -- which is why both arms report the same logical distance and
        // different byte figures. Two integer ops, and only in the counter build.
        E25_ADD(ov_shifted_records_insert, o->size() - pos);
        E25_ADD(ov_shifted_bytes_insert, (o->size() - pos) * sizeof(OverflowRecord));
        E25_INC(string_allocs);
        E25_ADD(ov_suffix_bytes_written, s.size());
#ifdef E30_TIME
        // The construction is timed apart from the shift: the string is a heap allocation
        // when the key exceeds SSO, while the shift is a memmove. E32 measured the shift;
        // A3 needs to know which of the two the container's cost actually is.
        const auto ov_a0 = E30_T0();
        OverflowRecord rec{std::string(s), value, tag};
        E30_ADD(ov_alloc_ns, ov_a0);
        E30_INC(ov_alloc_calls);
        o->insert_at(pos, std::move(rec));
#else
        o->insert_at(pos, OverflowRecord{std::string(s), value, tag});
#endif
        ++overflow_count_;
#ifdef E30_TIME
        E30_ADD(ov_insert_ns, ov_t0);
        E30_INC(ov_insert_calls);
#endif
    }


    // I10: reproduce the replay's placement decisions WITHOUT touching the group, or return
    // false when the group cannot hold every record.
    //
    // EXACT, NOT AN ESTIMATE. `live` is gathered sub-block by sub-block and M1 makes
    // sub-blocks rank-ordered, so `live` is globally ascending. For an ascending input every
    // insertion goes to the END of whichever sub-block it lands in -- a new key is >= every
    // key already placed -- so the in-block ordering inside insert_into is never exercised
    // and the decision reduces to per-sub-block counters. This pass therefore reproduces
    // put_primary's walk exactly, and `dir_out` is the directory the replay will produce.
    //
    // ALL scratch is local: every allocation this needs happens while the group is intact,
    // which is the whole point. Nothing here writes to the group.
    bool plan_replay(std::uint32_t g, const std::vector<Pair>& live,
                     std::vector<std::uint32_t>& blk_out,
                     std::vector<Boundary>& dir_out) const {
        const std::size_t nb = subblocks(g);
        const std::size_t cap_slots = slots_of(g);
        const std::size_t cap_bytes = std::size_t(bb_[g]);

        // `used[j]` counts from the block's own start, exactly as insert_into's
        // `h.bottom - start_of(g)` does, so the byte test below is the same test.
        std::vector<std::size_t> used(nb, start_of(g));
        std::vector<std::size_t> cnt(nb, 0);
        std::vector<std::uint32_t> first_rec(nb, 0);
        std::vector<std::uint32_t> last_rec(nb, 0);
        std::vector<char> has(nb, 0);

        blk_out.assign(live.size(), 0);

        auto fits = [&](std::size_t j, std::size_t len) {
            return cnt[j] < cap_slots && used[j] + len <= cap_bytes;
        };

        auto place = [&](std::size_t j, std::size_t k) {
            cnt[j] += 1;
            used[j] += live[k].first.size();
            if (!has[j]) { has[j] = 1; first_rec[j] = static_cast<std::uint32_t>(k); }
            last_rec[j] = static_cast<std::uint32_t>(k);
            blk_out[k] = static_cast<std::uint32_t>(j);
        };

        for (std::size_t k = 0; k < live.size(); ++k) {
            const std::string_view s = live[k].first;
            const std::size_t len = s.size();

            std::size_t prev = SIZE_MAX;
            bool done = false;

            for (std::size_t i = 0; i < nb; ++i) {
                if (!has[i]) continue;

                if (compare(std::string_view(live[last_rec[i]].first), s) < 0) {
                    prev = i;
                    continue;
                }

                if (prev != SIZE_MAX &&
                    compare(s, std::string_view(live[first_rec[i]].first)) < 0) {
                    if (fits(prev, len)) { place(prev, k); done = true; }
                    break;
                }

                if (fits(i, len)) { place(i, k); done = true; }
                break;
            }

            if (!done) {
                for (std::size_t j = (prev == SIZE_MAX ? 0 : prev); j < nb; ++j) {
                    if (fits(j, len)) { place(j, k); done = true; break; }
                }
            }

            // Unreachable for a well-formed group: it held these very records before the
            // pack, and a pack only ever frees bytes. Reaching it means the caller must NOT
            // commit -- which is exactly why the plan is computed before anything moves.
            if (!done) return false;
        }

        dir_out.clear();

        if (!dirs_.empty() && nb >= 2) {
            for (std::size_t i = 0; i < nb; ++i) {
                if (!has[i]) continue;
#ifdef E25_DIAG
                // I10 injection point 3: the same directory construction as point 2, but
                // under the plan it runs BEFORE the group is touched -- so an interruption
                // here must leave the index exactly as it was.
                if (maybe_inject(3)) throw std::bad_alloc();
#endif
                dir_out.push_back(Boundary{std::string(live[first_rec[i]].first),
                                           static_cast<std::uint32_t>(i)});
            }
        }

        return true;
    }

    // I10: the commit phase's only write primitive. It performs the append the plan already
    // proved feasible, so it has NO capacity test to fail and allocates nothing. Its
    // preconditions are exactly plan_replay's guarantees, and the diagnostic build checks
    // them (below) rather than trusting them.
    void append_planned(std::uint32_t g, std::size_t i, std::string_view s,
                        std::uint64_t value, std::uint8_t tag) {
        auto& h = header(g, i);
#ifdef E25_DIAG
        // The plan is a transcription of insert_into's own arithmetic; if it ever drifts,
        // this fires in the gates instead of writing past the block.
        if (h.count >= slots_of(g) || s.size() > h.bottom - start_of(g))
            E25_INC(compact_plan_violation);
#endif
        auto* es = entries(g, i);
        h.bottom = static_cast<std::uint16_t>(h.bottom - s.size());
        if (!s.empty()) std::memcpy(base_of(g, i) + h.bottom, s.data(), s.size());
        es[h.count] = {h.bottom, static_cast<std::uint32_t>(s.size()), value};
        h.tags[h.count] = tag;
        ++h.count;
        E18_INC(insert_ok);
        E18_ADD(suffix_bytes_copied, s.size());
    }

    //
    // I10: returns FALSE when the plan says the group cannot hold its own records, which
    // is decided BEFORE anything moves -- in that case the caller sees an unchanged group.
    // Both call sites discard the result today; the value exists so the failure has a
    // defined, inspectable outcome rather than a throw from inside the commit.
    bool compact(std::uint32_t g) {
        const auto e30_t0 = E30_T0();
        E30_INC(compact_calls);
        // `compactions` counts compactions that COMPLETED, so it is incremented at the end
        // and not here. It used to be incremented on entry, which meant a compaction that
        // threw before touching the group was still counted -- a counter half-update of
        // exactly the kind task book invariant I10 forbids. Found by the G1 failure
        // injection: with the PREPARE point armed, the index came back structurally
        // identical but with `compactions` one higher.
        //
        // The E25/E30 counters below are NOT moved: they measure work that was actually
        // performed before the failure (the gather really did read every live record), so
        // counting that work is correct for them even when the compaction then fails.
        const std::size_t nb = subblocks(g);
        std::vector<Pair> live;
        // E26 C1. The tags are collected ONLY when they are going to be reused: making
        // the deferred version pay for a vector it does not read would tilt the A/B.
        const bool want_tags = config_.compact_reuse_tag;
        std::vector<std::uint8_t> live_tags;
        for (std::size_t i = 0; i < nb; ++i)
            live.reserve(live.size() + header(g, i).count);
        if (want_tags) live_tags.reserve(live.capacity());

        std::size_t garbage_before = 0, bytes_copied = 0;

        // Sub-blocks are rank-ordered, so collecting them in order yields the
        // group's keys already sorted; replaying them refills from sub-block 0.
        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            const auto* es = entries(g, i);
            std::size_t live_in_block = 0;
            for (std::size_t j = 0; j < h.count; ++j) {
                live_in_block += es[j].length;
                bytes_copied += es[j].length;
                live.emplace_back(suffix(g, i, es[j]), es[j].value);
                if (want_tags) live_tags.push_back(h.tags[j]);
            }
            // Occupied region is bb - bottom; whatever it holds beyond the live
            // suffixes is deleted bytes the block has not reclaimed.
            const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
            if (occupied > live_in_block) garbage_before += occupied - live_in_block;
        }
        E25_ADD(compact_garbage_before, garbage_before);
        E25_ADD(compact_records_replayed, live.size());
        E25_ADD(compact_bytes_copied, bytes_copied);
        E30_ADD(collect_ns, e30_t0);   // entry through the gather, before the re-fill

#ifdef E25_DIAG
        // PREPARE point: everything allocated and read, nothing in the group touched yet.
        if (maybe_inject(0)) throw std::bad_alloc();
#endif

        // ---- PREPARE (I10) ----------------------------------------------------------
        // Everything that can fail happens HERE, while the group is still intact: the
        // replay is planned, and the directory the replay will produce is built. Both
        // allocate; both are allowed to fail; neither has touched the group. If the plan
        // does not fit, NOTHING has changed and compact() reports failure.
        //
        // Without this switch the directory is rebuilt after the replay instead, which is
        // where the old code could be interrupted with the group already rewritten.
        std::vector<std::uint32_t> plan_blk;
        std::vector<Boundary> plan_dir;

        if (config_.compact_plan_commit && !plan_replay(g, live, plan_blk, plan_dir)) {
            // I10: the group cannot hold its own records. NOTHING has been touched, so this
            // is a clean refusal and the caller simply keeps its old state.
            E25_INC(compact_plan_rejected);
            return false;
        }

        // ---- COMMIT ----------------------------------------------------------------
        // From here to the end of the replay there is no allocation and no fallible call:
        // append_planned cannot fail, and the directory swap below is noexcept. The commit
        // is therefore not interruptible, which is the property task book invariant I10
        // asks for.
        for (std::size_t i = 0; i < nb; ++i) {
            header(g, i).count = 0;
            header(g, i).bottom = static_cast<std::uint16_t>(bb_[g]);
        }

        // The replay front-inserts into every sub-block; rebuilding the directory
        // on each of those would be pure waste, so it is rebuilt once at the end.
        suppress_dir_rebuild = true;
        for (std::size_t k = 0; k < live.size(); ++k) {
#ifdef E25_DIAG
            // COMMIT point: half the records are already back in the group.
            if (k == live.size() / 2 && maybe_inject(1)) throw std::bad_alloc();
#endif
            std::uint8_t tg;
            if (want_tags) {
                tg = live_tags[k];
#ifdef E25_DIAG
                // The reuse is sound only if the stored tag still equals the
                // recomputed one. Checked in the DIAGNOSTIC build alone -- the timed
                // build must not pay for the very hashing this change removes.
                if (Fingerprint::of(live[k].first) != live_tags[k])
                    E25_INC(compact_tag_mismatch);
#endif
            } else {
                tg = Fingerprint::of(live[k].first);
            }

            if (config_.compact_plan_commit) {
                append_planned(g, plan_blk[k], live[k].first, live[k].second, tg);
            } else if (!put_primary(g, live[k].first, live[k].second, tg)) {
                throw std::logic_error("compaction lost capacity");
            }
        }
        suppress_dir_rebuild = false;

        if (config_.compact_plan_commit) {
            // Prepared above; vector::swap is noexcept, so this cannot be interrupted and
            // the OLD directory is destroyed only after the new one is in place.
            if (!dirs_.empty()) {
                ++dir_rebuilds;
                dir_entries_built += plan_dir.size();
                dirs_[g].swap(plan_dir);
            }
        } else {
            rebuild_dir(g);
        }

        ++compactions;   // see the note at the top of compact(): completed, not entered
        E30_ADD(compact_ns, e30_t0);          // whole call, collect + replay + directory
        return true;
    }

    // E38: calibrate this region's view of the shared table, or take the Context16 fallback.
    //
    // THE FALLBACK IS PER REGION AND IS RECORDED, NOT SILENT. Task book section 3.4: a region
    // whose calibration fails uses the reference model for the WHOLE region, and the experiment
    // reports how many regions, records and queries that covers. A region that quietly mapped
    // everything to group 0 instead would be a different layout wearing the LITS arm's name --
    // and a content comparison would not notice, because group 0 is still a valid group.
    void calibrate_lits(const std::vector<Pair>& sorted,
                        const std::shared_ptr<const lits_hpt::SharedModel>& shared) {
        // Combinations the adapter does not implement are REFUSED rather than mapped to
        // something near enough (task book section 4).
        if (config_.model_kind != ModelKind::Context16) {
            throw std::invalid_argument("LitsHpt supports ModelKind::Context16 only");
        }
        if (config_.canonical != Route::Full) {
            throw std::invalid_argument("LitsHpt supports Route::Full only");
        }

        // FEWER THAN TWO KEYS IS A DEFINED BOUNDARY CASE, NOT A FAILURE AND NOT A TRAINING.
        //
        // The pinned calibration reads `keys[1]` and divides by a span, so with one key there is
        // nothing to do; with none there is not even a key. Task book section 3.4 gives both a
        // defined answer -- group 0 -- and this branch takes it WITHOUT requiring a shared table
        // at all, which is what lets an index built from an empty key set exist.
        if (sorted.size() < 2) {
            lits_ = lits_hpt::RegionModel::boundary(
                groups_, sorted.empty() ? lits_hpt::Status::NoKeys : lits_hpt::Status::TooFewKeys);
            use_lits_ = true;
            ++lits_stats_.regions_boundary;
            lits_stats_.keys_boundary += sorted.size();

            return;
        }

        if (!shared) {
            throw std::invalid_argument("model backend is LitsHpt but no shared table was given");
        }

        // The coefficient bound is computed from the region's own longest key, so it is tight
        // rather than conservative -- check_lits_model measured a generic limit rejecting three
        // regions that were in fact safe.
        std::size_t max_key_bytes = 0;
        for (const auto& kv : sorted) {
            max_key_bytes = std::max(max_key_bytes, kv.first.size());
        }

        lits_ = sorted.empty()
            ? lits_hpt::RegionModel::boundary(groups_, lits_hpt::Status::NoKeys)
            : lits_hpt::RegionModel::calibrate(*shared, sorted.front().first,
                                               sorted.back().first, prefix_.size(), groups_,
                                               sorted.size(), max_key_bytes);

        if (lits_.usable()) {
            use_lits_ = true;
            ++lits_stats_.regions_calibrated;
            lits_stats_.keys_calibrated += sorted.size();
        } else {
            // The fallback, for this region's whole lifetime: the group mapping, the query
            // path and the demand pass all read `use_lits_`, which is resolved here once.
            use_lits_ = false;
            ++lits_stats_.regions_fallback;
            lits_stats_.keys_fallback += sorted.size();
        }
    }

    // BlockLinearRank (2026-10-11): fit this region's rank model.
    //
    // WHERE THIS RUNS. After `prefix_` and `groups_` exist and BEFORE the demand pass, exactly
    // where the LITS calibration runs and for the same reason: the pass below asks
    // canonical_group for every key, and the plan it computes has to describe the layout this
    // region will actually build.
    //
    // NO FALLBACK ARM. The rank model's input domain is every byte string -- empty key, embedded
    // NUL, high bytes, arbitrary length -- so there is no admission decision to get wrong and no
    // region is silently mapped by the reference model under this backend's name. The three
    // boundary statuses still answer group 0, which is a defined answer rather than a failure.
    //
    // The combination refusals mirror calibrate_lits: a pairing this backend does not implement
    // is REFUSED, not mapped to something near enough.
    void calibrate_block_rank(const std::vector<Pair>& sorted) {
        if (config_.model_kind != ModelKind::Context16) {
            throw std::invalid_argument("BlockLinearRank supports ModelKind::Context16 only");
        }
        if (config_.canonical != Route::Full) {
            throw std::invalid_argument("BlockLinearRank supports Route::Full only");
        }

        std::vector<std::string_view> keys;
        keys.reserve(sorted.size());
        for (const auto& kv : sorted) keys.push_back(kv.first);

        block_rank_ = blockrank::RegionModel::fit(keys.data(), keys.size(), prefix_.size(), groups_);
        use_block_rank_ = true;
        ++block_rank_stats_.fits;

        if (block_rank_.fitted()) {
            ++block_rank_stats_.regions_fitted;
            block_rank_stats_.keys_fitted += sorted.size();
            block_rank_stats_.segments_total += block_rank_.segment_count();
        } else {
            ++block_rank_stats_.regions_boundary;
            block_rank_stats_.keys_boundary += sorted.size();
        }
    }

    // EqualDepthDir (2026-10-11): build this region's separator directory.
    //
    // Runs where the other two calibrations run -- after `prefix_` and `groups_` exist and
    // BEFORE the demand pass, which asks canonical_group for every key. The plan it computes has
    // to describe the layout this region will build.
    //
    // No fallback arm: the directory accepts every byte string (it compares arbitrary suffixes),
    // so the only non-fitted outcomes are the three DEFINED boundary cases, all of which answer
    // group 0. Combination refusals mirror the other backends.
    void calibrate_depth_dir(const std::vector<Pair>& sorted) {
        if (config_.model_kind != ModelKind::Context16) {
            throw std::invalid_argument("the depth directory supports ModelKind::Context16 only");
        }
        if (config_.canonical != Route::Full) {
            throw std::invalid_argument("the depth directory supports Route::Full only");
        }

        std::vector<std::string_view> keys;
        keys.reserve(sorted.size());
        for (const auto& kv : sorted) keys.push_back(kv.first);

        // The truncated form is the same directory built from the same keys: one flag, fixed by
        // the backend the configuration names, so there is no per-region or per-run degree of
        // freedom here (the design's section 3 argues why the rule is not a parameter).
        const bool truncate = config_.model_backend == ModelBackend::EqualDepthDirTrunc;
        depth_dir_ = equivdir::RegionModel::fit(keys.data(), keys.size(), prefix_.size(), groups_,
                                                truncate);
        use_depth_dir_ = true;
        ++depth_dir_stats_.fits;

        if (depth_dir_.fitted()) {
            ++depth_dir_stats_.regions_fitted;
            depth_dir_stats_.keys_fitted += sorted.size();
            depth_dir_stats_.thresholds_total += depth_dir_.threshold_count();
            depth_dir_stats_.threshold_bytes_total += depth_dir_.separator_bytes();
            if (depth_dir_.truncated()) ++depth_dir_stats_.regions_truncated;

            // A duplicate threshold means the depth band boundary fell inside a run of equal
            // suffixes: that group can hold no key, and no map of the suffix alone can fix it.
            // Counted here so a round can say whether the balance was limited by the keys.
            std::vector<std::string_view> seps;
            for (std::size_t j = 1; j <= depth_dir_.threshold_count(); ++j) {
                const auto rank = std::min(sorted.size() - 1, (j * sorted.size()) / groups_);
                seps.push_back(sorted[rank].first);
            }
            std::sort(seps.begin(), seps.end(), [](std::string_view a, std::string_view b) {
                return equivdir::compare(a, b) < 0;
            });
            for (std::size_t j = 1; j < seps.size(); ++j)
                if (equivdir::compare(seps[j - 1], seps[j]) == 0) {
                    ++depth_dir_stats_.regions_with_duplicate_thresholds;
                    break;
                }
        } else {
            ++depth_dir_stats_.regions_boundary;
            depth_dir_stats_.keys_boundary += sorted.size();
        }
    }

public:
    Region(const std::vector<Pair>& sorted, Config cfg,
           std::shared_ptr<const lits_hpt::SharedModel> lits_shared = nullptr)
        : Region(sorted, cfg, {}, {}, {}, nullptr, std::move(lits_shared)) {}

    // nb_cap_in is an optional per-group ceiling on the sub-block count, used by
    // the strict-budget allocator to keep an unfunded group on its single-block
    // layout. It is build-time configuration only: nothing is stored, and
    // subblocks(g) still derives from off_ and bb_.
    //
    // E38: `lits_shared` is the ONE table the whole index shares. It is passed by value and
    // moved into the member, so a region either holds a reference to the table it was built
    // against or holds none -- there is no window in which it holds a dangling one.
    Region(const std::vector<Pair>& sorted, Config cfg,
           std::vector<std::uint16_t> sg_in, std::vector<std::uint32_t> bb_in,
           std::vector<std::uint16_t> nb_cap_in = {},
           const RegionDemand* preset = nullptr,
           std::shared_ptr<const lits_hpt::SharedModel> lits_shared = nullptr)
        : config_(cfg), lits_shared_(std::move(lits_shared)) {
        const auto e14_ctor_t0 = E14_T0();
        E14_INC(ctor_calls);
        E18_INC(region_ctors);
        if (cfg.block_bytes > 65535 || cfg.block_bytes % 64 || cfg.load <= 0 || cfg.load >= 1)
            throw std::invalid_argument("layout config");

        {
            const auto t0 = E14_T0();
            prefix_ = compute_prefix(sorted.data(), sorted.data() + sorted.size());
            E14_ADD(prefix, t0);
        }

        {
            const auto t0 = E14_T0();
            model_.train(sorted, prefix_.size(), config_.model_kind);
            E14_ADD(model, t0);
        }

        std::uint64_t suffix_bytes = 0;
        {
            const auto t0 = E14_T0();
            suffix_bytes = suffix_bytes_of(sorted.data(), sorted.data() + sorted.size(), prefix_.size());
            E14_ADD(suffix_scan, t0);
        }

        // All suffix bytes count in this average-capacity estimate, including long ones.
        //
        // E14: `layout` is accumulated in TWO segments -- before and after the E9
        // demand pass -- so that it EXCLUDES the demand pass. Measuring it as one
        // span would count the demand pass twice and the phases would then sum to
        // more than ctor_total. With the split, the eight phases partition the
        // constructor and their sum reconciles with ctor_total.
        const auto e14_layout_t0 = E14_T0();
        groups_ = derive_groups(sorted.size(), suffix_bytes, cfg);

        // E8: per-group capacity table (uniform default or explicit demand table)
        if (sg_in.empty()) {
            sg_in.assign(groups_, static_cast<std::uint16_t>(cfg.slots));
            bb_in.assign(groups_, cfg.block_bytes);
        }
        if (sg_in.size() != groups_ || bb_in.size() != groups_)
            throw std::invalid_argument("capacity table size");
        sg_ = std::move(sg_in);
        bb_ = std::move(bb_in);

        // E38: the LITS calibration, placed HERE because it needs `prefix_` and `groups_`, both
        // of which exist by now -- and because it must run BEFORE the demand pass, which asks
        // canonical_group for every key. Its cost is charged to the `layout` phase deliberately:
        // E14's eight phases must keep partitioning the constructor for their sum to reconcile
        // with ctor_total, and a ninth phase would break that reconciliation.
        if (config_.model_backend == ModelBackend::LitsHpt) {
            calibrate_lits(sorted, lits_shared_);
        } else if (config_.model_backend == ModelBackend::BlockLinearRank) {
            calibrate_block_rank(sorted);
        } else if (is_depth_dir_backend(config_.model_backend)) {
            calibrate_depth_dir(sorted);
        }
        // Candidate M3 (default OFF): the mapping is known here, so a LITS-mapped region can
        // release the reference table it will never read. The preset path is refused first:
        // its digest is computed from that table, and a silent mismatch would quietly turn
        // the R1 reuse into a full recomputation.
        if (config_.lits_skip_reference_table && use_lits_) {
            if (preset != nullptr)
                throw std::logic_error(
                    "lits_skip_reference_table: a preset was offered for a LITS region whose "
                    "reference table is dropped; refused rather than silently degraded");
            Model::Table().swap(model_.cumulative);
        }
        // E31H: hysteresis state, allocated only when the switch asks for it.
        if (reclaim_gated()) {
            reclaim_last_op_.assign(groups_, kNoReclaimAttempt);
            reclaim_last_garbage_.assign(groups_, 0);
        }
        E14_ADD(layout, e14_layout_t0);   // segment A: group estimate + capacity table

        // E9 pass 1: each group's real demand (entries and suffix bytes) decides
        // how many equal-sized sub-blocks it needs. Both constraints must hold,
        // and the resulting count stays derivable from off_ and bb_ alone. This
        // pass runs only in multi-block mode; otherwise every group has nb == 1.
        std::vector<std::uint32_t> entries_need;
        std::vector<std::uint64_t> bytes_need;
        // E19 (R1): the group id of every key in this region, recorded by the
        // demand pass and read back by the placement loop. Sized to the region
        // (region_keys entries at most), so the peak is a few KiB, not the whole
        // key set. Empty when the demand pass does not run.
        std::vector<std::uint32_t> group_of;
        // E20 (R2): accept the preset only if this region's own prefix and logical
        // group count match the ones the ids were computed under. A length check
        // alone would not catch a changed prefix or group count, both of which
        // change the mapping while leaving the count of ids equal.
        // E21: the model digest is checked LAST and only when a preset is offered,
        // so the default and R1 paths never pay for it.
        const bool preset_ok = preset != nullptr && preset->group_ids.size() == sorted.size() &&
                               preset->prefix == prefix_ && preset->groups == groups_ &&
                               preset->model_digest ==
                                   mapping_identity(model_, groups_, config_.canonical,
                                                    config_.model_backend, block_rank_,
                                                    depth_dir_);
        if (cfg.multiblock) {
            const auto t0 = E14_T0();
            entries_need.assign(groups_, 0);
            bytes_need.assign(groups_, 0);
            if (config_.reuse_group_ids && !preset_ok) group_of.resize(sorted.size());
            for (std::size_t i = 0; i < sorted.size(); ++i) {
                auto s = std::string_view(sorted[i].first).substr(prefix_.size());
                const auto g = preset_ok ? preset->group_ids[i] : canonical_group(sorted[i].first);
                if (!group_of.empty()) group_of[i] = g;
                ++entries_need[g];
                bytes_need[g] += s.size();
            }
            E14_ADD(demand_pass, t0);
        }
        // The reuse is conditional on the recorded ids covering exactly this key
        // sequence. Anything else falls back to computing them, so a future change
        // that breaks the correspondence degrades to the old behaviour instead of
        // silently misplacing records.
        const bool reuse_ids = cfg.multiblock && config_.reuse_group_ids &&
                               group_of.size() == sorted.size();

        const auto e14_layout2_t0 = E14_T0();
        off_.assign(std::size_t(groups_) + 1, 0);
        for (std::uint32_t g = 0; g < groups_; ++g) {
            const auto so = sizeof(Header) + std::size_t(sg_[g]) * sizeof(Entry);
            if (sg_[g] == 0 || sg_[g] > MAX_SLOTS || bb_[g] % 64 || bb_[g] <= so || bb_[g] > 65535)
                throw std::invalid_argument("layout config");

            std::uint64_t nb = 1;
            if (cfg.multiblock) {
                if (!nb_cap_in.empty() && nb_cap_in.size() != groups_)
                    throw std::invalid_argument("sub-block cap size");

                const std::uint64_t per_block = bb_[g] - so;
                nb = (entries_need[g] + sg_[g] - 1) / sg_[g];
                nb = std::max<std::uint64_t>(nb, (bytes_need[g] + per_block - 1) / per_block);
                nb = std::max<std::uint64_t>(nb, 1);

                // 0 means uncapped; any positive value is the ceiling. The strict
                // allocator uses 1 to pin an unfunded group to a single block.
                if (!nb_cap_in.empty() && nb_cap_in[g] != 0)
                    nb = std::min<std::uint64_t>(nb, nb_cap_in[g]);

                if (nb > 65535) throw std::length_error("too many sub-blocks");
            }
            off_[g + 1] = off_[g] + std::size_t(nb) * bb_[g];
        }

        // Legacy fixed-stride addressing cannot express more than one block per
        // group, so a multi-block layout requires the offset directory.
        if (!config_.offset_directory)
            for (std::uint32_t g = 0; g < groups_; ++g)
                if (subblocks(g) != 1)
                    throw std::invalid_argument("fixed stride requires one sub-block per group");
        E14_ADD(layout, e14_layout2_t0);   // segment B: offset derivation

        // E14 note: `storage` covers the storage allocation, the header/entry
        // initialisation AND the per-group outer directory slots -- all three are
        // one pass over `groups_`/`subblocks`.
        const auto e14_storage_t0 = E14_T0();
        storage_ = static_cast<std::byte*>(::operator new(off_.back(), std::align_val_t{64}));
        for (std::uint32_t g = 0; g < groups_; ++g) {
            for (std::size_t i = 0, nb = subblocks(g); i < nb; ++i) {
                new (base_of(g, i)) Header{};
                header(g, i).bottom = static_cast<std::uint16_t>(bb_[g]);
                for (std::size_t j = 0; j < sg_[g]; ++j) new (entries(g, i) + j) Entry{};
            }
        }

        // Only multi-block layouts carry a boundary directory; a single-block
        // layout must not pay even the outer vector slots.
        if (cfg.multiblock) dirs_.assign(groups_, {});
        E14_ADD(storage, e14_storage_t0);

        suppress_dir_rebuild = true;
#ifdef E18_PREPASS
        // Diagnostic variant: compute every group id and fingerprint FIRST, into
        // one pass, then run the placement loop over the prepared arrays. The two
        // spans partition the placement work with TWO clock calls per region
        // instead of per key. canonical_group and Fingerprint::of are pure, so the
        // placement decisions -- and the resulting structure -- are unchanged; the
        // round verifies that by digest.
        std::vector<std::uint32_t> pre_g(sorted.size());
        std::vector<std::uint8_t> pre_tag(sorted.size());
        {
            const auto t0 = E14_T0();
            for (std::size_t i = 0; i < sorted.size(); ++i) {
                auto s = std::string_view(sorted[i].first).substr(prefix_.size());
                pre_g[i] = reuse_ids ? group_of[i] : canonical_group(sorted[i].first);
                E18_INC(fingerprint_calls);
                pre_tag[i] = static_cast<std::uint8_t>(Fingerprint::of(s));
            }
            E14_ADD(grp_prep, t0);
        }
        const auto e14_place_t0 = E14_T0();
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            auto s = std::string_view(sorted[i].first).substr(prefix_.size());
            if (!put_primary(pre_g[i], s, sorted[i].second, pre_tag[i]))
                put_overflow(pre_g[i], s, sorted[i].second, pre_tag[i]);
            ++size_;
        }
        suppress_dir_rebuild = false;
        E14_ADD(place, e14_place_t0);
#else
        const auto e14_place_t0 = E14_T0();
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            const auto& kv = sorted[i];
            auto s = std::string_view(kv.first).substr(prefix_.size());
            // E19 (R1): read the id the demand pass computed rather than asking
            // the model the same question again.
            const auto g = preset_ok ? preset->group_ids[i]
                                     : (reuse_ids ? group_of[i] : canonical_group(kv.first));
            E18_INC(fingerprint_calls);
            const auto tag = Fingerprint::of(s);
            if (!put_primary(g, s, kv.second, static_cast<std::uint8_t>(tag)))
                put_overflow(g, s, kv.second, static_cast<std::uint8_t>(tag));
            ++size_;
        }
        suppress_dir_rebuild = false;
        E14_ADD(place, e14_place_t0);
#endif

        const auto e14_dir_t0 = E14_T0();
        rebuild_all_dirs();
        E14_ADD(dir, e14_dir_t0);

        E14_ADD(ctor_total, e14_ctor_t0);
    }

    ~Region() { ::operator delete(storage_, std::align_val_t{64}); }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    // --- read-only accessors ---

    std::size_t size() const { return size_; }
    std::size_t overflow_count() const { return overflow_count_; }
    // E9 gate: lets a test assert that cross-sub-block compaction really ran.
    std::size_t compaction_count() const { return compactions; }
    std::size_t prefix_length() const { return prefix_.size(); }
    const std::string& prefix() const { return prefix_; }

    // E38: how the LITS backend covered the regions built so far. A round reports these rather
    // than asserting the fallback count is zero -- the point of section 3.4 is that a fallback
    // is allowed and must be VISIBLE, not that it never happens.
    static const LitsStats& lits_stats() { return lits_stats_; }
    static void reset_lits_stats() { lits_stats_ = LitsStats{}; }
    // True when this region was mapped by the LITS table rather than by the reference model.
    bool uses_lits() const { return use_lits_; }
    // True when this region was mapped by the rank backend.
    bool uses_block_rank() const { return use_block_rank_; }
    // The group the PLACEMENT path would use for this full key, exposed for gates and
    // diagnostics that must check order-preservation directly instead of inferring it from
    // code. It is deliberately the same call the demand pass and the placement loop make.
    std::uint32_t placement_group(std::string_view full_key) const {
        return canonical_group(full_key);
    }
#ifdef E25_DIAG
    // P1 read-path diagnosis: run THIS region's directory search under a trace for the same
    // full key the find path receives. False when the region is not on the directory backend
    // or its model did not fit, in which case the production path would not search a
    // directory either.
    bool diag_depth_dir_trace(std::string_view full_key, equivdir::SearchTrace& tr,
                              std::uint32_t& group, std::uint32_t& consumed) const {
        if (!use_depth_dir_ || !depth_dir_.fitted()) return false;
        group = depth_dir_.predict_group_traced(full_key, consumed, tr);
        return true;
    }
#endif
    blockrank::Status block_rank_status() const { return block_rank_.status(); }
    static const BlockRankStats& block_rank_stats() { return block_rank_stats_; }
    static void reset_block_rank_stats() { block_rank_stats_ = BlockRankStats{}; }
    bool uses_depth_dir() const { return use_depth_dir_; }
    equivdir::Status depth_dir_status() const { return depth_dir_.status(); }
    std::size_t depth_dir_thresholds() const { return depth_dir_.threshold_count(); }
    // TRUNCDIR: which stored form this region's directory uses, and how many arena bytes
    // that form cost. Read-only, like the neighbours above: a gate must be able to assert
    // that the LIVE regions are the truncated form instead of inferring it from the config.
    bool depth_dir_truncated() const { return depth_dir_.truncated(); }
    std::size_t depth_dir_separator_bytes() const { return depth_dir_.separator_bytes(); }
    static const DepthDirStats& depth_dir_stats() { return depth_dir_stats_; }
    static void reset_depth_dir_stats() { depth_dir_stats_ = DepthDirStats{}; }
    // Candidate M3: whether the Context16 reference table is still resident.
    bool reference_table_resident() const { return !model_.cumulative.empty(); }

    // The status of this region's calibration, for a per-reason breakdown. Read from the region
    // itself rather than from `lits_stats_`, which accumulates over every region ever
    // CONSTRUCTED -- splits, prefix rebuilds and erases included -- and is therefore the wrong
    // denominator for "what does the index I just timed look like".
    lits_hpt::Status lits_status() const { return lits_.status; }
    std::uint32_t groups() const { return groups_; }
    // E33/SA0: read-only exposure of the group mapping, next to the other read-only
    // accessors (slot_table / block_table / boundary_dir / subblock_keys) that exist so a
    // diagnostic can CHECK the mapping instead of reading the code and hoping. The
    // structure review asks for one trace from a real key to its group and for I1
    // (the query group equals the placement group) to be verified, and both need this
    // callable from outside without making the placement path itself public.
    // The REFERENCE model's answer for a SUFFIX, for gates and diagnostics. Deliberately not
    // backend-dispatched: its callers use it as the frozen model's own mapping to compare
    // against, and routing it through the backend would make those comparisons compare a
    // model with itself.
    std::uint32_t group_of(std::string_view suffix) const {
        E18_INC(group_calls);

        // Candidate M3: a region that released its reference table cannot answer this. Fail
        // loudly instead of reading an empty table; callers that need the reference model
        // must hold an independent object. Never reached with the switch off.
        if (model_.cumulative.empty())
            throw std::logic_error(
                "group_of: this region has no resident reference table "
                "(lits_skip_reference_table); use an independent reference object");

        Fingerprint dummy;
        return model_.predict<false>(suffix, groups_, config_.canonical, dummy).group;
    }

    std::size_t slots() const { return sg_.empty() ? 0 : sg_[0]; }

    std::size_t block_bytes_total() const { return off_.back(); }

    // E8-6: read-only exposure of the capacity tables for structural validation.
    const std::vector<std::uint16_t>& slot_table() const { return sg_; }
    const std::vector<std::uint32_t>& block_table() const { return bb_; }
    const std::vector<std::size_t>& offset_table() const { return off_; }

    // E9: the sub-block count is derived, not stored, so there is no table to
    // expose — the gate recomputes it from off_/bb_ and compares.
    std::size_t subblock_count(std::uint32_t g) const { return subblocks(g); }

    // E11: whether this group has a boundary directory (only multi-block groups do),
    // and the directory's contents for the targeted differential.
    bool has_dir(std::uint32_t g) const { return !dirs_.empty() && !dirs_[g].empty(); }

    std::vector<std::pair<std::string, std::uint32_t>> boundary_dir(std::uint32_t g) const {
        std::vector<std::pair<std::string, std::uint32_t>> out;
        if (dirs_.empty()) return out;
        for (const auto& b : dirs_[g]) out.emplace_back(b.key, b.subblock);
        return out;
    }

    std::size_t directory_rebuilds() const { return dir_rebuilds; }
    std::size_t directory_entries_built() const { return dir_entries_built; }

    std::size_t subblock_entries(std::uint32_t g, std::size_t i) const {
        return header(g, i).count;
    }

    // Keys of one sub-block in stored order, as suffixes (all sub-blocks of a
    // group share the region prefix, so suffixes compare like full keys).
    std::vector<std::string> subblock_keys(std::uint32_t g, std::size_t i) const {
        std::vector<std::string> out;
        const auto& h = header(g, i);
        const auto* es = entries(g, i);
        for (std::size_t j = 0; j < h.count; ++j) out.emplace_back(suffix(g, i, es[j]));
        return out;
    }

    struct SubBlockStats {
        std::size_t entries, bytes, free_bytes, garbage_bytes;
    };

    // E37: what the current layout would accept back, if the overflow records were offered
    // to it. Read-only; no mechanism lives here.
    struct MigrationScan {
        std::uint64_t containers = 0;          // containers holding at least one record
        std::uint64_t overflow_records = 0;    // records sitting in those containers
        std::uint64_t no_target = 0;           // the group has no non-empty sub-block at all
        std::uint64_t fail_slots = 0;          // the target sub-block has no free slot
        std::uint64_t fail_bytes = 0;          // slot free, but not enough free suffix bytes
        std::uint64_t migratable = 0;          // passes the range, the slot AND the byte test
        std::uint64_t migratable_bytes = 0;    // total suffix bytes of the migratable records
        // The same three tests against the byte watermark the block WOULD have once its own
        // garbage is recovered. Reclaim frees bytes and never slots, so only the byte test
        // differs -- and the two failure counters below say which one binds afterwards.
        std::uint64_t fail_slots_after = 0;
        std::uint64_t fail_bytes_after = 0;
        std::uint64_t migratable_after = 0;
        std::uint64_t migratable_after_bytes = 0;
        // The dry run that CONSUMES what it places. This is the opportunity under a stated
        // policy; migratable_after above is the same records judged independently, so the
        // gap between the two is what mutual exclusion costs.
        std::uint64_t greedy_migratable = 0;
        std::uint64_t greedy_bytes = 0;
        // Removal cost, counted both ways, because the answer is a design decision:
        //   one_by_one  erasing them from the front, each erase walking the tail that is
        //               left -- the shape the reviewer warned about
        //   compaction  a single pass over the container, moving each survivor at most once
        std::uint64_t shift_one_by_one = 0;
        std::uint64_t shift_compaction = 0;
    };

    bool diag_room(std::uint32_t g, std::size_t i, std::string_view s) const {
        const auto& h = header(g, i);
        if (h.count >= slots_of(g)) return false;
        return s.size() <= h.bottom - start_of(g);
    }

    // The bytes a sub-block would have free once its own garbage is recovered: its whole
    // byte region minus the suffixes that are live in it. That is the watermark
    // repack_subblock produces for a block it processes. Reclaim frees BYTES and never
    // SLOTS -- packing suffixes does not change count -- so the two tests are separate here
    // for the same reason insert_into separates them.
    // WHAT A REPACK WOULD FREE IF IT ALWAYS RAN. (bb_ - start_of) - live is the whole suffix
    // area minus the live bytes, and it does NOT model repack_subblock's n<2 early return.
    // Under the shipped configuration that makes it an OVER-estimate for a block holding
    // fewer than two live records -- it becomes the real figure only with
    // reclaim_low_activity on. The gate measures the real capability by running the
    // mechanism; nothing should read this number as "what the current code can reach".
    std::size_t diag_free_after_reclaim(std::uint32_t g, std::size_t i) const {
        const auto& h = header(g, i);
        std::size_t live = 0;
        const auto* es = entries(g, i);
        for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
        return std::size_t(bb_[g]) - start_of(g) - live;
    }

    // put_primary's decision order, written ONCE. The three passes below differ only in the
    // room predicate they hand in, so none of them can drift away from put_primary or from
    // each other -- which is exactly the failure mode the two earlier copies invited.
    template <class RoomFn>
    std::size_t diag_target_with(std::uint32_t g, std::string_view s, RoomFn room) const {
        const std::size_t nb = subblocks(g);
        std::size_t prev = SIZE_MAX;

        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            if (h.count == 0) continue;

            const auto* es = entries(g, i);

            if (compare(suffix(g, i, es[h.count - 1]), s) < 0) { prev = i; continue; }

            if (prev != SIZE_MAX && compare(s, suffix(g, i, es[0])) < 0) {
                if (room(prev)) return prev;
                return room(i) ? i : SIZE_MAX;
            }

            return room(i) ? i : SIZE_MAX;
        }

        for (std::size_t j = prev == SIZE_MAX ? 0 : prev; j < nb; ++j)
            if (room(j)) return j;

        return SIZE_MAX;
    }

    void diag_migration_scan(MigrationScan& out) const {
        for (std::uint32_t g = 0; g < groups_; ++g) {
            const auto* o = overflow(g);

            if (!o || o->empty()) continue;

            const std::size_t n = o->size();
            const std::size_t nb = subblocks(g);
            ++out.containers;
            out.overflow_records += n;

            // The greedy pass's virtual view of the target blocks: what they would hold once
            // reclaimed, then decremented as the dry run accepts.
            std::vector<std::uint32_t> vcount(nb);
            std::vector<std::size_t> vfree(nb);
            std::vector<char> greedy_take(n, 0);
            std::size_t greedy_first = SIZE_MAX, greedy_removed = 0;

            for (std::size_t i = 0; i < nb; ++i) {
                vcount[i] = header(g, i).count;
                vfree[i] = diag_free_after_reclaim(g, i);
            }

            for (std::size_t j = 0; j < n; ++j) {
                const std::string_view s = (*o)[j].suffix;

                bool any_block = false;

                for (std::size_t i = 0; i < nb; ++i)
                    if (header(g, i).count > 0) { any_block = true; break; }

                // ---- NOW -------------------------------------------------------------
                const auto room_now = [&](std::size_t i) { return diag_room(g, i, s); };

                if (diag_target_with(g, s, room_now) != SIZE_MAX) {
                    ++out.migratable;
                    out.migratable_bytes += s.size();
                } else if (!any_block) {
                    ++out.no_target;
                } else {
                    const std::size_t fc = first_candidate(g, s);

                    if (header(g, fc).count >= slots_of(g)) ++out.fail_slots;
                    else ++out.fail_bytes;
                }

                // ---- AFTER (per record, against the SAME recovered state: an upper bound)
                const auto room_after = [&](std::size_t i) {
                    return header(g, i).count < slots_of(g) &&
                           s.size() <= diag_free_after_reclaim(g, i);
                };

                if (diag_target_with(g, s, room_after) != SIZE_MAX) {
                    ++out.migratable_after;
                    out.migratable_after_bytes += s.size();
                } else if (any_block) {
                    const std::size_t fc = first_candidate(g, s);

                    if (header(g, fc).count >= slots_of(g)) ++out.fail_slots_after;
                    else ++out.fail_bytes_after;
                } else {
                    // Under the post-reclaim watermark an EMPTY block offers its whole byte
                    // region, so failing here means the suffix is longer than the region --
                    // a byte failure, not a "nowhere to put it". The variants therefore do
                    // not share the no_target counter.
                    ++out.fail_bytes_after;
                }

                // ---- GREEDY ----------------------------------------------------------
                const auto room_greedy = [&](std::size_t i) {
                    return vcount[i] < slots_of(g) && s.size() <= vfree[i];
                };

                const std::size_t gi = diag_target_with(g, s, room_greedy);

                if (gi != SIZE_MAX) {
                    ++vcount[gi];
                    vfree[gi] -= s.size();
                    ++out.greedy_migratable;
                    out.greedy_bytes += s.size();
                    greedy_take[j] = 1;
                    ++greedy_removed;

                    if (j < greedy_first) greedy_first = j;
                }
            }

            // Removal cost, on the set a mechanism would actually take. Ascending
            // one-at-a-time makes every erase walk whatever tail is left; a single
            // compaction pass moves each survivor at most once. The two differ by orders of
            // magnitude, and which applies is a design decision, so both are counted.
            if (greedy_removed) {
                std::size_t m = n;

                for (std::size_t j = 0; j < n; ++j) {
                    if (!greedy_take[j]) continue;

                    out.shift_one_by_one += m - 1 - j;
                    --m;
                }

                out.shift_compaction += (n - 1 - greedy_first) - (greedy_removed - 1);
            }
        }
    }

    SubBlockStats subblock_stats(std::uint32_t g, std::size_t i) const {
        const auto& h = header(g, i);
        const auto* es = entries(g, i);
        SubBlockStats s{};
        s.entries = h.count;
        for (std::size_t j = 0; j < h.count; ++j) s.bytes += es[j].length;
        s.free_bytes = h.bottom - start_of(g);
        s.garbage_bytes = bb_[g] - h.bottom - s.bytes;
        return s;
    }

    // E37: drain every group once, with the same budget per group. This is the entry the
    // "repair an already-degraded snapshot" experiment uses -- the insert path only drains
    // when a placement is about to fail, so a snapshot that is already full would otherwise
    // never be visited. Public because Index drives it; the placement internals it uses stay
    // private.
    std::size_t drain_all_bounded() {
        std::size_t total = 0;

        for (std::uint32_t g = 0; g < groups_; ++g)
            total += drain_overflow_bounded(g, kMigrateBudget);

        return total;
    }

    struct MemBreakdown {
        std::size_t blocks, model, prefix, overflow, tables, directory, self;
    };

    MemBreakdown memory_breakdown() const {
        MemBreakdown m{};
        // E9: bb_ describes one sub-block, so the group's bytes come from off_.
        m.blocks = off_.back();
        // The rank backend's segments are resident model memory too, so they belong in the
        // same field instead of being left out of the decomposition.
        m.model = model_.allocated_bytes() + block_rank_.owned_bytes() +
                  depth_dir_.owned_bytes();
        m.prefix = prefix_.capacity() + 1;
        m.overflow = overflow_owners_.capacity() * sizeof(void*);
        // O1: the container reports its own space in the terms this decomposition
        // uses, so the two representations are counted alike and this function does
        // not have to know which one is live. The OFF figure is exactly the
        // expression that used to be inline here, so the frozen build numbers hold.
        for (const auto& o : overflow_owners_) m.overflow += o->allocated_bytes();
        m.tables = off_.capacity() * sizeof(std::size_t) +
                   sg_.capacity() * sizeof(std::uint16_t) +
                   bb_.capacity() * sizeof(std::uint32_t);
        // E11: the boundary directory is real memory -- the outer slots for every
        // group, plus each boundary's own key buffer. It is NOT free metadata.
        m.directory = dirs_.capacity() * sizeof(std::vector<Boundary>);
        for (const auto& d : dirs_) {
            m.directory += d.capacity() * sizeof(Boundary);
            for (const auto& b : d) m.directory += b.key.capacity() + 1;
        }
        // E31H: the hysteresis tables are real memory when they are allocated.
        m.tables += reclaim_last_op_.capacity() * sizeof(std::uint64_t) +
                    reclaim_last_garbage_.capacity() * sizeof(std::size_t);
        m.self = sizeof(*this);
        return m;
    }

    const Model& model() const { return model_; }
    std::string_view residual(std::string_view key) const { return key.substr(prefix_.size()); }
    bool accepts_prefix(std::string_view key) const { return prefix_matches(key); }

    // E8-12: per-record overflow suffix lengths (E9 feasibility diagnostics).
    std::vector<std::uint32_t> overflow_lengths(std::uint32_t g) const {
        std::vector<std::uint32_t> out;
        if (auto* o = overflow(g))
            for (const auto& e : *o) out.push_back(static_cast<std::uint32_t>(e.suffix.size()));
        return out;
    }

    // E11 gate: the overflow suffixes of one group, so a test can query records
    // that live outside every primary block.
    std::vector<std::string> overflow_keys(std::uint32_t g) const {
        std::vector<std::string> out;
        if (auto* o = overflow(g))
            for (const auto& e : *o) out.emplace_back(e.suffix);
        return out;
    }

    std::size_t memory() const {
        // E8-6 fix: the offset/slot/block-size tables are now counted (they were
        // missing before; the omission is what the space-decomposition audit found).
        // E11 adds the boundary directory, which is likewise counted, not assumed away.
        const auto mb = memory_breakdown();
        return mb.self + mb.model + mb.prefix + mb.overflow + mb.blocks + mb.tables + mb.directory;
    }

    std::vector<GroupStats> group_stats() const {
        std::vector<GroupStats> out;
        out.reserve(groups_);

        for (std::uint32_t g = 0; g < groups_; ++g) {
            GroupStats s;
            s.group = g;

            // E9: aggregate the sub-blocks so the per-group totals keep the same
            // meaning they had with a single block (free/garbage are sums).
            for (std::size_t i = 0, nb = subblocks(g); i < nb; ++i) {
                const auto& h = header(g, i);
                const auto* es = entries(g, i);
                s.primary_entries += h.count;
                for (std::size_t j = 0; j < h.count; ++j) s.primary_bytes += es[j].length;
                s.free_entries += slots_of(g) - h.count;
                s.free_bytes += h.bottom - start_of(g);
                s.garbage_bytes += bb_[g] - h.bottom;
            }
            s.garbage_bytes -= s.primary_bytes;

            if (const auto* o = overflow(g)) {
                for (const auto& e : *o) {
                    ++s.overflow_entries;
                    s.overflow_bytes += e.suffix.size();
                    const bool slots = s.free_entries == 0, bytes = e.suffix.size() > s.free_bytes;
                    if (slots && bytes) ++s.blocked_both;
                    else if (slots) ++s.blocked_slots_only;
                    else if (bytes) ++s.blocked_bytes_only;
                    else ++s.fits_now;
                    if (e.suffix.size() > bb_[g] - start_of(g)) ++s.oversized;
                }
            }
            out.push_back(s);
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // E16: per-group statistics for a key range, WITHOUT constructing a Region.
    //
    // This is what the construction pipeline actually consumes. It builds a
    // "probe" Region per range purely to call group_stats(), and the two
    // capacity allocators read exactly four fields of each GroupStats
    // (primary_entries, primary_bytes, overflow_entries, overflow_bytes) --
    // E16's report carries the field-by-field audit of e9_alloc.hpp.
    //
    // The path replays the SAME acceptance test the single-block constructor runs
    // (insert_into's predicate) against the default capacity, in the same global
    // sorted order, keeping only a running (count, bottom) per group. It allocates
    // no block storage, copies no suffix into a block, and builds no overflow
    // container. The one per-key array is the group id, with the top bit recording
    // whether that key went to overflow, so the overflow-blocking partition can be
    // computed in a second pass without re-running the model.
    //
    // Every field of GroupStats is produced, not just the four the allocators read,
    // so this is a drop-in replacement and the gate can compare all of them.
    //
    // SCOPE: the single-block layout only, which is what the probe uses. A
    // multi-block config is REJECTED rather than approximated -- put_primary then
    // spans sub-blocks and the replay would have to model invariant M1 as well.
    // -----------------------------------------------------------------------
    // E21: public because Index::bulk_load_with_plan verifies a context with it.
    // E21: one FNV step per key over (length | first byte | last byte). O(1) per
    // key, so it stays far below the cost of the work it is protecting -- but it is
    // NOT free, and it is paid twice per build (once when the context is produced,
    // once when it is consumed). Hashing the full key bytes would cost more than
    // the optimisation it guards.
    static std::uint64_t key_sequence_digest(const Pair* first, const Pair* last) {
        std::uint64_t h = 1469598103934665603ULL;
        for (const Pair* it = first; it != last; ++it) {
            const auto& k = it->first;
            const std::uint64_t n = k.size();
            const std::uint64_t v =
                n | (std::uint64_t(n ? static_cast<unsigned char>(k.front()) : 0u) << 40) |
                (std::uint64_t(n ? static_cast<unsigned char>(k.back()) : 0u) << 48);
            h ^= v;
            h *= 1099511628211ULL;
        }
        return h;
    }

    static std::vector<GroupStats> demand_profile(const std::vector<Pair>& sorted, const Config& cfg) {
        return demand_profile(sorted.data(), sorted.data() + sorted.size(), cfg);
    }

    static std::vector<GroupStats> demand_profile(const Pair* first, const Pair* last, const Config& cfg) {
        // keep_ids = false: the statistics-only caller does not need the ids, and
        // building them here would make this path allocate an array per region that
        // it immediately discards -- which would also make it an unfair baseline for
        // the R2 comparison.
        //
        // std::move, not a plain member access: `demand_profile_impl(...)` is a
        // temporary, and `temp.stats` is an lvalue, so returning it by value would
        // COPY every region's statistics. That copy is invisible in the structure
        // and would only show up as a slower baseline.
        RegionDemand rd = demand_profile_impl(first, last, cfg, false);
        return std::move(rd.stats);
    }

    // E20 (R2): the same replay, but keeping the ids, the prefix and the logical
    // group count so the final construction can reuse them.
    static RegionDemand demand_profile_ctx(const Pair* first, const Pair* last, const Config& cfg,
                                           const std::shared_ptr<const lits_hpt::SharedModel>& lits_shared = nullptr) {
        return demand_profile_impl(first, last, cfg, true, lits_shared);
    }

    static RegionDemand demand_profile_impl(const Pair* first, const Pair* last, const Config& cfg,
                                            bool keep_ids,
                                            const std::shared_ptr<const lits_hpt::SharedModel>& lits_shared = nullptr) {
        if (cfg.multiblock)
            throw std::invalid_argument("E16: demand_profile covers the single-block layout only");
        if (cfg.block_bytes > 65535 || cfg.block_bytes % 64 || cfg.load <= 0 || cfg.load >= 1)
            throw std::invalid_argument("layout config");

        const std::size_t n = static_cast<std::size_t>(last - first);
        const std::string prefix = compute_prefix(first, last);
        const std::uint64_t sfx_bytes = suffix_bytes_of(first, last, prefix.size());
        const std::uint32_t groups = derive_groups(n, sfx_bytes, cfg);

        E18_ADD(group_calls, n);   // one prediction per key in this pass
        Model model;
        model.train(first, last, prefix.size(), cfg.model_kind);

        // E38: THE SAME BACKEND DECISION AND THE SAME TABLE AS THE REAL REGION.
        //
        // This pass computes the capacity plan, and the plan has to describe the layout the
        // regions will actually build. Measuring with the reference model and placing with LITS
        // is the mistake the task book's section 5.2 names -- and it would not show up as an
        // error, only as a plan that fits the wrong demand.
        bool use_lits = false;
        lits_hpt::RegionModel lits;
        // BlockLinearRank: the SAME fit the region will make, over the same key range, prefix and
        // group count. Measuring demand with one mapping and placing with another is the mistake
        // the E38 comment above names, and it would not show up as an error -- only as a plan
        // that fits the wrong demand.
        bool use_block_rank = false;
        blockrank::RegionModel block_rank;
        // EqualDepthDir: the same directory the region will build, from the same key range,
        // prefix and group count. Measuring demand with one map and placing with another is the
        // mistake the E38 comment above names.
        bool use_depth_dir = false;
        equivdir::RegionModel depth_dir;
        std::size_t max_key_bytes = 0;
        for (const Pair* it = first; it != last; ++it) {
            max_key_bytes = std::max(max_key_bytes, it->first.size());
        }

        if (cfg.model_backend == ModelBackend::LitsHpt) {
            if (!lits_shared) {
                throw std::invalid_argument("demand profile: LitsHpt needs the shared table");
            }
            lits = n == 0
                ? lits_hpt::RegionModel::boundary(groups, lits_hpt::Status::NoKeys)
                : lits_hpt::RegionModel::calibrate(*lits_shared, first[0].first,
                                                   first[n - 1].first, prefix.size(), groups, n,
                                                   max_key_bytes);
            // A region that fell back to the reference model in the real build must fall back
            // HERE too, or the plan would fund a layout the build will not produce.
            use_lits = lits.usable();
        }

        if (cfg.model_backend == ModelBackend::BlockLinearRank) {
            std::vector<std::string_view> block_keys;
            block_keys.reserve(n);
            for (const Pair* it = first; it != last; ++it) block_keys.push_back(it->first);
            block_rank = blockrank::RegionModel::fit(block_keys.data(), block_keys.size(),
                                                     prefix.size(), groups);
            use_block_rank = true;
        }

        if (is_depth_dir_backend(cfg.model_backend)) {
            std::vector<std::string_view> depth_keys;
            depth_keys.reserve(n);
            for (const Pair* it = first; it != last; ++it) depth_keys.push_back(it->first);
            depth_dir = equivdir::RegionModel::fit(
                depth_keys.data(), depth_keys.size(), prefix.size(), groups,
                cfg.model_backend == ModelBackend::EqualDepthDirTrunc);
            use_depth_dir = true;
        }

        const std::size_t slots = cfg.slots;
        const std::size_t start = sizeof(Header) + slots * sizeof(Entry);
        const std::size_t bb = cfg.block_bytes;

        std::vector<GroupStats> out(groups);
        for (std::uint32_t g = 0; g < groups; ++g) out[g].group = g;

        std::vector<std::uint32_t> count(groups, 0);
        std::vector<std::uint32_t> bottom(groups, static_cast<std::uint32_t>(bb));
        // `where` carries the group id with the top bit marking "went to overflow".
        // The overflow bit is a property of the PROBE's capacity -- it must be
        // stripped before the ids are reused for placement.
        std::vector<std::uint32_t> where(n);
        constexpr std::uint32_t OVERFLOW_BIT = 0x80000000u;

        for (std::size_t i = 0; i < n; ++i) {
            const std::string_view s = std::string_view(first[i].first).substr(prefix.size());
            Fingerprint dummy;
            // E38: the same dispatch the region uses, on the same coordinates -- full key to the
            // LITS walk, suffix to the reference model.
            const std::uint32_t g = use_depth_dir
                ? depth_dir.predict_group(first[i].first)
                : use_block_rank
                    ? block_rank.predict_group(first[i].first)
                    : use_lits
                        ? lits.predict_group(first[i].first)
                        : model.predict<false>(s, groups, cfg.canonical, dummy).group;
            where[i] = g;
            const std::size_t len = s.size();
            // insert_into's acceptance test, verbatim, for a single-block group.
            if (count[g] < slots && len <= std::size_t(bottom[g]) - start) {
                ++count[g];
                bottom[g] = static_cast<std::uint32_t>(bottom[g] - len);
                ++out[g].primary_entries;
                out[g].primary_bytes += len;
            } else {
                where[i] = g | OVERFLOW_BIT;
                ++out[g].overflow_entries;
                out[g].overflow_bytes += len;
            }
        }

        for (std::uint32_t g = 0; g < groups; ++g) {
            out[g].free_entries = slots - count[g];
            out[g].free_bytes = std::size_t(bottom[g]) - start;
            out[g].garbage_bytes = (bb - std::size_t(bottom[g])) - out[g].primary_bytes;
        }

        // Second pass: the overflow-blocking partition uses the FINAL primary
        // state, exactly as group_stats does.
        for (std::size_t i = 0; i < n; ++i) {
            if (!(where[i] & OVERFLOW_BIT)) continue;
            const std::uint32_t g = where[i] & ~OVERFLOW_BIT;
            const std::size_t len = first[i].first.size() - prefix.size();
            GroupStats& st = out[g];
            const bool no_slots = st.free_entries == 0, no_bytes = len > st.free_bytes;
            if (no_slots && no_bytes) ++st.blocked_both;
            else if (no_slots) ++st.blocked_slots_only;
            else if (no_bytes) ++st.blocked_bytes_only;
            else ++st.fits_now;
            if (len > bb - start) ++st.oversized;
        }

        RegionDemand rd;
        rd.prefix = prefix;
        rd.groups = groups;
        rd.model_digest =
            mapping_identity(model, groups, cfg.canonical, cfg.model_backend, block_rank,
                             depth_dir);
        if (keep_ids) rd.key_digest = key_sequence_digest(first, last);
        rd.stats = std::move(out);
        if (keep_ids) {
            rd.group_ids.resize(n);
            for (std::size_t i = 0; i < n; ++i) rd.group_ids[i] = where[i] & ~OVERFLOW_BIT;
        }
        return rd;
    }

    // --- E25: located records and write helpers ---------------------------------
    //
    // Where a located record lives, and what the locate learned about the key. The
    // two halves answer different questions and the review asked for them to be
    // kept apart:
    //
    //   * the CONTEXT (group, tag, "the whole group was searched and the key is not
    //     here") is what a MISS needs, so the insert that follows does not predict
    //     the group and the fingerprint a second time;
    //   * the POSITION (sub-block + slot, or overflow index) is what a HIT needs,
    //     so update/erase write in place instead of searching again.
    //
    // VALIDITY: a Located is valid only inside the public operation that produced
    // it. Compaction, descriptor moves and container growth all invalidate the
    // position -- and the context too, if the group geometry or the model changed.
    // It is deliberately not a handle any caller can hold across operations.
    //
    // NotLocated vs Absent is load bearing: the first means no valid group id was
    // ever obtained (empty index, prefix mismatch), so there is nothing to reuse;
    // only the second means the legitimate group was searched exhaustively and the
    // key is genuinely absent, which is what makes it safe to place directly.
    enum class Where : std::uint8_t { NotLocated, Absent, Primary, Overflow };

    struct Located {
        Where where = Where::NotLocated;
        std::uint32_t group = 0;   // context: valid unless NotLocated
        std::uint8_t tag = 0;      // context: Fingerprint::of(s)
        std::size_t subblock = 0;  // position: Primary only
        std::size_t slot = 0;      // position: Primary only
        std::size_t index = 0;     // position: Overflow only
    };

    // E25 V1: the mode the write paths search with.
    Mode write_mode() const {
        return Mode{config_.canonical, Prefetch::Off, true, config_.write_simd,
                    OvSearch::Hybrid, config_.ov_threshold, config_.write_locate};
    }

    // Write the value of an already-located record, in place. No search.
    void store_at(const Located& loc, std::uint64_t value) {
        if (loc.where == Where::Primary)
            entries(loc.group, loc.subblock)[loc.slot].value = value;
        else if (loc.where == Where::Overflow)
            (*overflow(loc.group))[loc.index].value = value;
    }

    // Remove an already-located record: descriptors and tags shift down, the count
    // and the size drop, and a front removal rebuilds this group's boundary
    // directory because its minimum changed.
    bool erase_at(const Located& loc) {
        if (loc.where == Where::Primary) {
            auto& h = header(loc.group, loc.subblock);
            auto* es = entries(loc.group, loc.subblock);
            E25_ADD(erase_shifts, h.count - 1 - loc.slot);
            // The suffix bytes are NOT freed here -- the entry is dropped from the
            // descriptor array and its bytes stay in the block as garbage. That is
            // why this is counted apart from the overflow case.
            E25_ADD(erase_released_primary, es[loc.slot].length);
            for (std::size_t j = loc.slot + 1; j < h.count; ++j) {
                es[j - 1] = es[j];
                h.tags[j - 1] = h.tags[j];
            }
            --h.count;
            --size_;
            if (loc.slot == 0) rebuild_dir(loc.group);
            return true;
        }
        if (loc.where == Where::Overflow) {
            auto* o = overflow(loc.group);
            E25_ADD(erase_shifts, o->size() - 1 - loc.index);
            // E31 item 3: the same movement figure for the erase. The clock below covers
            // the vector::erase AND the destruction of the erased record -- a real free()
            // here, where a primary erase leaves its bytes in the block as garbage.
            E25_ADD(ov_shifted_records_erase, o->size() - 1 - loc.index);
            E25_ADD(ov_shifted_bytes_erase, (o->size() - 1 - loc.index) * sizeof(OverflowRecord));
            E25_ADD(erase_released_overflow, (*o)[loc.index].suffix.size());
            E25_INC(string_frees);   // the OverflowRecord, and with it its string, is destroyed
#ifdef E30_TIME
            const auto ov_t0 = E30_T0();
#endif
            o->erase_at(loc.index);
#ifdef E30_TIME
            E30_ADD(ov_erase_ns, ov_t0);
            E30_INC(ov_erase_calls);
#endif
            --size_;
            --overflow_count_;
            return true;
        }
        return false;
    }

    // --- lookup ---

    template <bool Hash>
    Result find_impl(std::string_view key, Mode mode, Located* loc = nullptr) const {
        E25_INC(existence_searches);
        if (loc) *loc = Located{};   // NotLocated unless a valid group is reached below
        if (!size_ || !prefix_matches(key)) return {};
        if ((config_.canonical == Route::Residual) != (mode.route == Route::Residual))
            throw std::logic_error("incompatible mapping");

        const auto s = key.substr(prefix_.size());
        Fingerprint fp;
        const auto pred = predict_for<Hash>(key, s, mode.route, fp);
        const auto g = pred.group;
        const std::byte* address = base_of(g, 0);

        if (mode.prefetch == Prefetch::Early) __builtin_prefetch(address, 0, 3);
        std::uint8_t tag = 0;
        if constexpr (Hash) {
            // E38: `fp_consumed`, not `consumed`. On the reference path they are equal and this
            // line is what it always was; on the LITS path the model contributed nothing to the
            // hash and this hashes the whole suffix -- which is what reproduces the STORED tag.
            fp.tail(s.substr(pred.fp_consumed));
            tag = fp.tag();
        }
        if (mode.prefetch == Prefetch::Late) __builtin_prefetch(address, 0, 3);

        // E25: the context half. From here the group id is valid, so a miss found
        // by the search below is genuinely "searched and absent".
        if (loc) {
            loc->where = Where::Absent;   // provisional; only kept if nothing is found
            loc->group = g;
            loc->tag = tag;
        }

        Result result;
        result.steps = pred.consumed;

        auto exact = [&](std::size_t blk, std::size_t i) {
            const auto* es = entries(g, blk);
            if (es[i].length == s.size()) {
                ++result.str_cmps;
                E12_TICK(result.str_cmps_primary);
                if (E25_CMP_P(suffix(g, blk, es[i]), s) == 0) {
                    result.found = true;
                    result.value = es[i].value;
                    result.source = 1;
                    if (loc) { loc->where = Where::Primary; loc->subblock = blk; loc->slot = i; }
                }
            }
        };

#if defined(__SSE2__)
        constexpr bool has_simd = true;
#else
        constexpr bool has_simd = false;
#endif

        // E9: a group may span several sub-blocks; each is tag-filtered in turn.
        // The tag filter keeps the extra blocks cheap (no full-key compares).
        //
        // E11: with the boundary directory in use, M1 makes the candidate sub-block
        // unique, so at most ONE block is visited. Not finding s there says nothing
        // about the overflow container -- M1 orders the primary sub-blocks among
        // themselves and says nothing about where overflow records sit -- so the
        // overflow search below runs unchanged on both paths.
        std::size_t lo_blk = 0, hi_blk = subblocks(g);
        if (mode.locate && has_dir(g)) {
            const std::size_t cand = locate_subblock(g, s, result);
            if (cand == NO_SUBBLOCK) { lo_blk = 0; hi_blk = 0; }   // below every primary key
            else { lo_blk = cand; hi_blk = cand + 1; }
        }

        for (std::size_t blk = lo_blk; blk < hi_blk && !result.found; ++blk) {
            const auto& h = header(g, blk);
            E12_TICK(result.subblocks_visited);   // header read even if the sub-block is empty
            E12_TICK(result.candidate_blocks);

            if constexpr (Hash && has_simd) {
                if (mode.simd) {
                    // E5-2: batched tag filter. Mask bits are clamped to live entries, so
                    // stale tag bytes beyond count can never produce a false candidate.
                    const __m128i vt = _mm_set1_epi8(static_cast<char>(tag));
                    for (std::size_t c = 0; c * 16 < slots_of(g); ++c) {
                        const std::size_t lo = c * 16;
                        const std::size_t top = std::min<std::size_t>(std::min(slots_of(g), lo + 16), h.count);
                        if (top <= lo) break;
                        E12_TICK(result.simd_batches);
                        const __m128i tv = _mm_load_si128(reinterpret_cast<const __m128i*>(h.tags + lo));
                        std::uint32_t mask = static_cast<std::uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(tv, vt)));
                        mask &= (1u << unsigned(top - lo)) - 1u;
                        E12_ADD(result.tag_candidates, static_cast<std::uint32_t>(__builtin_popcount(mask)));
                        while (mask && !result.found) {
                            exact(blk, lo + unsigned(__builtin_ctz(mask)));
                            mask &= mask - 1;
                        }
                        if (result.found) break;
                    }
                    continue;
                }
            }
            if constexpr (Hash) {
                for (std::size_t i = 0; i < h.count && !result.found; ++i) {
                    if (h.tags[i] != tag) continue;
                    E12_TICK(result.tag_candidates);
                    exact(blk, i);
                }
            } else {
                for (std::size_t i = 0; i < h.count && !result.found; ++i) exact(blk, i);
            }
        }

        if (result.found) return result;

        if (auto* o = overflow(g)) {
            E12_TICK(result.overflow_entered);
            // E7: three lookup policies over the same sorted overflow vector.
            const bool use_bin = mode.ov == OvSearch::Binary ||
                (mode.ov == OvSearch::Hybrid && o->size() >= mode.ov_thr);

            if (use_bin) {
                E25_INC(ov_queries_bin);
                std::size_t lo = 0, hi = o->size();
                while (lo < hi) {
                    const auto mid = lo + (hi - lo) / 2;
                    ++result.overflow_checks;
                    ++result.str_cmps;
                    E12_TICK(result.str_cmps_overflow);
#ifdef E25_DIAG
                    // E35: what a W-byte inline prefix would have decided, measured on the
                    // record actually being compared. Read-only; compiled out entirely in
                    // the timing build.
                    {
                        const std::string& a = (*o)[mid].suffix;
                        const std::size_t lcp = ::validation::e25::ov_lcp(a, s);
                        const std::size_t shorter = a.size() < s.size() ? a.size() : s.size();
                        E25_INC(ov_bin_cmps);
                        E25_ADD(ov_bin_lcp_bytes, lcp);
                        if (shorter < 4) {
                            E25_INC(ov_bin_short4);
                        } else if (lcp < 4) {
                            E25_INC(ov_bin_decided4);
                        }
                        if (shorter >= 8 && lcp < 8) E25_INC(ov_bin_decided8);
                    }
#endif
                    if (E25_CMP_O((*o)[mid].suffix, s) < 0) lo = mid + 1;
                    else hi = mid;
                }
                if (lo < o->size()) {
                    ++result.str_cmps;
                    E12_TICK(result.str_cmps_overflow);
                    if (E25_CMP_O((*o)[lo].suffix, s) == 0) {
                        result.found = true;
                        result.value = (*o)[lo].value;
                        result.source = 2;
                        if (loc) { loc->where = Where::Overflow; loc->index = lo; }
                    }
                }
            } else {
                E25_INC(ov_queries_linear);
                // E36: the inline-tag arm gets its OWN loop. Putting a per-record test into
                // the loop below would change the code B and O were measured with, and the
                // whole point of this arm is to compare against exactly that code.
                // `if constexpr (Hash)` because without fingerprints there is no tag to
                // compare, and the arm would then have nothing to reject on.
                bool tagged = false;
                if constexpr (Hash) tagged = o->tags_are_inline();

                if (tagged) {
                    for (std::size_t oi = 0; oi < o->size(); ++oi) {
                        ++result.overflow_checks;
#ifdef E25_DIAG
                        E25_INC(ov_tag_entries);
#endif
                        // The tag comes from the entry, and the entries are contiguous in the
                        // order array, so a mismatch is settled without touching the store.
                        // THAT is the access this arm is trying to remove.
                        if (o->entry_tag(oi) != tag) {
#ifdef E25_DIAG
                            E25_INC(ov_tag_reject_pre);
#endif
                            continue;
                        }
                        // A match proves nothing: 8 bits collide, so the record is fetched
                        // and the full key compared, exactly as in the other arms.
                        const auto& e = o->record_at(oi);
#ifdef E25_DIAG
                        E25_INC(ov_linear_recs);
#endif
                        ++result.str_cmps;
                        E12_TICK(result.str_cmps_overflow);
                        if (E25_CMP_O(e.suffix, s) == 0) {
                            result.found = true;
                            result.value = e.value;
                            result.source = 2;
                            if (loc) { loc->where = Where::Overflow; loc->index = oi; }
                            break;
                        }
                    }
                } else {
                for (std::size_t oi = 0; oi < o->size(); ++oi) {
                    const auto& e = (*o)[oi];
                    ++result.overflow_checks;
#ifdef E25_DIAG
                    E25_INC(ov_tag_entries);
                    E25_INC(ov_linear_recs);
#endif
                    if constexpr (Hash) {
                        if (e.tag != tag) {
#ifdef E25_DIAG
                            E25_INC(ov_linear_tag_reject);
#endif
                            continue;
                        }
                    }
#ifdef E25_DIAG
                    // The candidate survived the tag filter and is about to be compared.
                    // What a prefix would add ON TOP OF the tag is exactly this: of the
                    // survivors, how many an 8-byte window settles as "not equal" without
                    // the read the tag already forced.
                    {
                        E25_INC(ov_linear_tag_pass);
                        const std::size_t lcp = ::validation::e25::ov_lcp(e.suffix, s);
                        E25_ADD(ov_linear_lcp_bytes, lcp);
                        const std::size_t shorter =
                            e.suffix.size() < s.size() ? e.suffix.size() : s.size();
                        if (shorter >= 8 && lcp < 8) E25_INC(ov_linear_reject8);
                    }
#endif
                    ++result.str_cmps;
                    E12_TICK(result.str_cmps_overflow);
                    if (E25_CMP_O(e.suffix, s) == 0) {
                        result.found = true;
                        result.value = e.value;
                        result.source = 2;
                        if (loc) { loc->where = Where::Overflow; loc->index = oi; }
                        break;
                    }
                }
                }
            }
        }
        return result;
    }

    Result find(std::string_view key, Mode mode) const {
        return mode.fingerprint ? find_impl<true>(key, mode) : find_impl<false>(key, mode);
    }

    // --- update paths ---

    // E9: whether any sub-block of the group could still take a record this long.
    // Used only to decide whether compaction is worth attempting before overflow.
    bool group_has_room(std::uint32_t g, std::size_t len) const {
        for (std::size_t i = 0, nb = subblocks(g); i < nb; ++i) {
            const auto& h = header(g, i);
            if (h.count < slots_of(g) && len <= h.bottom - start_of(g)) return true;
        }
        return false;
    }

    // E26 C2: un-referenced suffix bytes in this group -- the most a compaction could
    // reclaim. Occupied region minus live suffix bytes, summed over the sub-blocks.
    // O(entries), and evaluated only on a placement that ALREADY failed, so it is far
    // cheaper than the compaction it can skip.
    std::size_t group_garbage_bytes(std::uint32_t g) const {
        std::size_t total = 0;
        for (std::size_t i = 0, nb = subblocks(g); i < nb; ++i) {
            const auto& h = header(g, i);
            const auto* es = entries(g, i);
            std::size_t live = 0;
            for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
            const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
            if (occupied > live) total += occupied - live;
        }
        return total;
    }

    // E29 C3: could a full repack of this group hold one more record of length len?
    // A NECESSARY condition for compact() to succeed -- it replays every live record
    // and throws if the group cannot hold them.
    bool repack_budget_fits(std::uint32_t g, std::size_t len) const {
        std::size_t live = 0, free_slots = 0;
        const std::size_t nb = subblocks(g);
        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            const auto* es = entries(g, i);
            for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
            free_slots += slots_of(g) - h.count;
        }
        const std::size_t area = (std::size_t(bb_[g]) - start_of(g)) * nb;
        return free_slots > 0 && live + len <= area;
    }

    // E26 C2 / E29 C3: should a failed placement bother with a compaction?
    bool worth_compacting(std::uint32_t g, std::size_t len) const {
        if (group_has_room(g, len)) {                    // M0's own condition
            if (!config_.compact_needs_garbage) return true;
            return group_garbage_bytes(g) >= len;        // C2: can it free enough?
        }
        if (!config_.compact_repack_budget) return false;
        // C3: no single sub-block has both a slot and the bytes, but a repack of the
        // WHOLE group could still hold the record. Budget-guarded, because compact()
        // must not be run on a group that would not fit.
        return group_garbage_bytes(g) >= len && repack_budget_fits(g, len);
    }

    // E31H (checklist item 4.3): may the reclaim run for this group now? Off by default, in
    // which case this is a constant true and the E31 path is untouched.
    //
    // The checklist's rule is to set a minimum operation interval and not to retry while the
    // state has not materially changed. The two recorded quantities are the group's garbage at
    // its last attempt (the "layout version": a repack only helps again once new garbage has
    // accumulated) and the placement counter at that attempt (the interval). A first attempt
    // on a group is always allowed; a repeat needs BOTH a moved interval and grown garbage.
    // Each skip is counted separately, so the effect is countable rather than asserted.
    bool reclaim_gated() const {
        return config_.reclaim_hysteresis || config_.reclaim_skip_repeat;
    }

    bool reclaim_allowed(std::uint32_t g) {
        if (!reclaim_gated()) return true;
        const std::size_t garbage = group_garbage_bytes(g);
        const std::uint64_t last = reclaim_last_op_[g];
        if (last != kNoReclaimAttempt) {
            if (config_.reclaim_hysteresis &&
                placements_since_ - last < kMinReclaimGap) {
                E25_INC(reclaim_skipped_gap);
                return false;
            }
            if (garbage <= reclaim_last_garbage_[g]) {
                E25_INC(reclaim_skipped_unchanged);
                return false;
            }
        }
        reclaim_last_op_[g] = placements_since_;
        reclaim_last_garbage_[g] = garbage;
        return true;
    }

    // E25 V2: place using the group id and tag the locate already produced, instead
    // of asking the model and the fingerprint again. Duplicate detection is NOT
    // skipped -- the caller only reaches here after the locate said Absent.
    //
    // Compaction may run here and move slots, but it does not change the group or
    // the tag, so those stay valid; put_primary recomputes the physical position.
    bool place_with_ctx(std::string_view key, std::uint64_t value, const Located& loc) {
        E30_W(place);
        if (reclaim_gated()) {
            ++placements_since_;   // E31H/E31P: the interval and the state scale
        }
        const auto s = key.substr(prefix_.size());
        if (!put_primary(loc.group, s, value, loc.tag)) {
            // E31H: the same reclaim, gated by the hysteresis when that switch is on.
            if (config_.subblock_reclaim && reclaim_allowed(loc.group)) {
                E25_INC(subblock_reclaims);
                const std::size_t rank_blk = first_candidate(loc.group, s);
#ifdef E25_DIAG
                {
                    // Stage D supplementary check: would the SEARCH's own directory
                    // answer have chosen this same block? The candidate threads that
                    // answer through Located instead of recomputing it; this measures
                    // the equality on the real workload rather than assuming it.
                    if (!has_dir(loc.group)) {
                        E25_INC(hint_nodir);
                    } else {
                        Result tmp;
                        const std::size_t dir_cand = locate_subblock(loc.group, s, tmp);
                        if (dir_cand == NO_SUBBLOCK) E25_INC(hint_below);
                        else if (dir_cand == rank_blk) E25_INC(hint_eq);
                        else E25_INC(hint_ne);
                    }
                }
#endif
                // DIAGNOSTIC ONLY, and it is read before the repack: the pass is what the
                // candidate changes, and its retry is what may place the record. Guarded so
                // that a timing build -- where E25_INC is ((void)0) -- does not even read the
                // header, keeping the shipped path unchanged rather than merely equivalent.
#ifdef E25_DIAG
                const bool la_pass = header(loc.group, rank_blk).count < 2;
#endif
                repack_subblock(loc.group, rank_blk);
                if (put_primary(loc.group, s, value, loc.tag)) {
                    E25_INC(subblock_reclaim_helped);
#ifdef E25_DIAG
                    if (la_pass) E25_INC(low_activity_helped);
#endif
                    ++size_;
                    return true;
                }
                // G3/A1': the rank-selected block was not enough. Try the group's OTHER
                // blocks -- the ones the insert path never reaches on its own.
                if (config_.reclaim_widest &&
                    reclaim_widen(loc.group, s, value, loc.tag, rank_blk)) {
                    ++size_;
                    return true;
                }
            }
            // E26: compact() is reached from exactly here. Whether the retry then
            // SUCCEEDS is the direct measure of what that compaction did for the
            // operation that triggered it. A failure is recorded as "did not enable
            // this placement", NOT as "the compaction had no value".
            E25_INC(compact_entered);
            // I10: the result is DELIBERATELY not branched on. compact() returning false means
            // the plan proved the group cannot hold its own records, in which case the group
            // was left untouched -- so the put_primary below simply fails again and the record
            // goes to overflow, which is exactly what would have happened had the compaction
            // run and not helped. The two cases are indistinguishable here ON PURPOSE (neither
            // changes what the caller must do); they are told apart by compact_plan_rejected.
            if (worth_compacting(loc.group, s.size())) compact(loc.group);
            if (put_primary(loc.group, s, value, loc.tag)) {
                E25_INC(compact_helped);
            } else {
                // E37: before this key goes into overflow, try to bring records the other
                // way. The drain opens room only by moving records INTO the primary area, so
                // the retry rides on what it opened.
                bool placed = false;

                if (config_.overflow_migrate) {
                    E25_INC(migrate_calls);
                    const std::size_t moved = drain_overflow_bounded(loc.group, kMigrateBudget);

                    if (moved) placed = put_primary(loc.group, s, value, loc.tag);

                    if (placed) E25_INC(migrate_helped);
                }

                if (!placed) {
                    E25_INC(compact_no_help);
                    put_overflow(loc.group, s, value, loc.tag);
                }
            }
        }
        ++size_;
        return true;
    }

    bool insert(std::string_view key, std::uint64_t value) {
        if (!prefix_matches(key)) throw std::invalid_argument("prefix requires rebuild");

        if (config_.write_fast_locate) {
            // V1 search, V2 reuse. Without reuse the mode does not matter for
            // correctness, and holding it fixed between V1 and V2 makes the measured
            // difference "how many searches ran" and nothing else.
            Located loc;
            const Result r = find_located(key, loc);
            if (r.found) return false;
            if (config_.reuse_locate_ctx && loc.where == Where::Absent)
                return place_with_ctx(key, value, loc);
            // V1 without reuse, or a NotLocated result: keep the original flow.
            if (config_.reuse_locate_ctx && loc.where == Where::NotLocated)
                E25_INC(notlocated_fallback);
            return insert_slow(key, value);
        }

        Mode m{config_.canonical, Prefetch::Off, true, false};
        if (find(key, m).found) return false;
        return insert_slow(key, value);
    }

    // The placement half of insert, with the group id and tag computed the
    // original way. Kept separate so V1's control flow is visibly unchanged.
    bool insert_slow(std::string_view key, std::uint64_t value) {
        if (reclaim_gated()) {
            ++placements_since_;   // E31H/E31P: the interval and the state scale
        }
        const auto e30_t0 = E30_T0();
        E30_INC(insert_calls);
        struct E30Scope { std::uint64_t t0; ~E30Scope() { E30_ADD(insert_ns, t0); } } e30_scope{e30_t0};
        auto s = key.substr(prefix_.size());
        const auto g = canonical_group(key);
        auto tag = Fingerprint::of(s);

        if (!put_primary(g, s, value, tag)) {
            // E31: reclaim the rank-selected sub-block in place, then retry. Cheap
            // (6-11 ns/record) and the boundary does not change, so nothing a caller
            // holds is invalidated.
            // E31H: the same reclaim, gated by the hysteresis when that switch is on.
            if (config_.subblock_reclaim && reclaim_allowed(g)) {
                E25_INC(subblock_reclaims);
                const std::size_t rank_blk = first_candidate(g, s);
                // Same as the other call site; see the note there.
#ifdef E25_DIAG
                const bool la_pass = header(g, rank_blk).count < 2;
#endif
                repack_subblock(g, rank_blk);
                if (put_primary(g, s, value, tag)) {
                    E25_INC(subblock_reclaim_helped);
#ifdef E25_DIAG
                    if (la_pass) E25_INC(low_activity_helped);
#endif
                    ++size_;
                    return true;
                }
                // G3/A1': the rank-selected block was not enough. Try the group's OTHER
                // blocks -- the ones the insert path never reaches on its own.
                if (config_.reclaim_widest &&
                    reclaim_widen(g, s, value, tag, rank_blk)) {
                    ++size_;
                    return true;
                }
            }
            // E26: compact() is reached from exactly here. Whether the retry then
            // SUCCEEDS is the direct measure of what that compaction did for the
            // operation that triggered it. A failure is recorded as "did not enable
            // this placement", NOT as "the compaction had no value".
            E25_INC(compact_entered);
            // I10: see the note at the other call site -- the false return is handled by
            // falling through to the overflow path, and the group is provably unchanged.
            if (worth_compacting(g, s.size())) compact(g);
            if (put_primary(g, s, value, tag)) {
                E25_INC(compact_helped);
            } else {
                bool placed = false;

                if (config_.overflow_migrate) {
                    E25_INC(migrate_calls);
                    const std::size_t moved = drain_overflow_bounded(g, kMigrateBudget);

                    if (moved) placed = put_primary(g, s, value, tag);

                    if (placed) E25_INC(migrate_helped);
                }

                if (!placed) {
                    E25_INC(compact_no_help);
                    put_overflow(g, s, value, tag);
                }
            }
        }
        ++size_;
        return true;
    }

    // E25 V2: one locate, then either write or place. Returns true when an existing
    // record was written and false when a new one was placed.
    //
    // NOTE THE SIGN, because this comment used to claim the two functions shared a convention
    // and they do not. `Index::upsert` writes `if (r.upsert_one(...)) return false;`, so at the
    // INDEX level the answer is INVERTED: false means the record already existed, true means a
    // new one was placed. The contract above is this function's own.
    //
    // The behaviour is unchanged -- E25's gates pin it -- and the correction is to the TEXT.
    // check_lits_index found it by encoding the observed return value against a reference map,
    // which is the only way a wrong comment gets caught.
    //
    // Duplicate detection IS the locate, so it is not skipped.
    //
    // Region level ONLY: the caller is responsible for the region-level control flow
    // that Index::insert owns (the prefix-mismatch rebuild and the region split). If
    // this were reachable from Index::upsert directly, a "one locate" that skips
    // those would be a different structure, not a faster one.
    bool upsert_one(std::string_view key, std::uint64_t value) {
        Located loc;
        const Result r = find_located(key, loc);
        if (r.found) { store_at(loc, value); return true; }
        if (loc.where == Where::Absent) { place_with_ctx(key, value, loc); return false; }
        E25_INC(notlocated_fallback);   // no context to reuse: the old path runs
        insert_slow(key, value);
        return false;
    }

    // E25 V1: one search through the query machinery, returning the position too.
    Result find_located(std::string_view key, Located& loc) const {
        E30_W(findloc);
        const Mode m = write_mode();
        return m.fingerprint ? find_impl<true>(key, m, &loc) : find_impl<false>(key, m, &loc);
    }

    bool update(std::string_view key, std::uint64_t value) {
        if (!size_ || !prefix_matches(key)) return false;
        auto s = key.substr(prefix_.size());

        // E25 V1: locate through the tag filter / SIMD / boundary directory / overflow
        // policy, then write in place. The slow path below is the E24 behaviour and is
        // kept reachable so V0 and V1 are one binary apart.
        if (config_.write_fast_locate) {
            Located loc;
            if (!find_located(key, loc).found) return false;
            store_at(loc, value);
            return true;
        }

        const auto g = canonical_group(key);

        for (std::size_t blk = 0, nb = subblocks(g); blk < nb; ++blk) {
            E24_UP_BLOCK();
            auto* es = entries(g, blk);
            for (std::size_t i = 0; i < header(g, blk).count; ++i) {
                E24_UP_ENTRY();
                if (E25_CMP(suffix(g, blk, es[i]), s) == 0) { es[i].value = value; return true; }
            }
        }
        if (auto* o = overflow(g))
            for (auto& e : *o) {
                E24_UP_ENTRY();
                if (E25_CMP(e.suffix, s) == 0) { e.value = value; return true; }
            }
        return false;
    }

    bool erase(std::string_view key) {
        const auto e30_t0 = E30_T0();
        E30_INC(erase_calls);
        struct E30Scope { std::uint64_t t0; ~E30Scope() { E30_ADD(erase_ns, t0); } } e30_scope{e30_t0};
        if (!size_ || !prefix_matches(key)) return false;
        auto s = key.substr(prefix_.size());

        // E25 V1: same search, then the removal that the found position implies.
        if (config_.write_fast_locate) {
            Located loc;
            if (!find_located(key, loc).found) return false;
            return erase_at(loc);
        }

        const auto g = canonical_group(key);

        for (std::size_t blk = 0, nb = subblocks(g); blk < nb; ++blk) {
            E24_ERA_BLOCK();
            auto& h = header(g, blk);
            auto* es = entries(g, blk);
            for (std::size_t i = 0; i < h.count; ++i) {
                E24_ERA_ENTRY();
                if (E25_CMP(suffix(g, blk, es[i]), s) == 0) {
                    E25_ADD(erase_shifts, h.count - 1 - i);
                    E25_ADD(erase_released_primary, es[i].length);
                    for (std::size_t j = i + 1; j < h.count; ++j) {
                        es[j - 1] = es[j];
                        h.tags[j - 1] = h.tags[j];
                    }
                    --h.count;
                    --size_;
                    // Removing the first entry changes this sub-block's minimum, and
                    // emptying it removes the boundary entirely.
                    if (i == 0) rebuild_dir(g);
                    return true;
                }
            }
        }
        if (auto* o = overflow(g))
            for (auto it = o->begin(); it != o->end(); ++it) {
                E24_ERA_ENTRY();
                if (E25_CMP(it->suffix, s) == 0) {
                    E25_ADD(erase_shifts, static_cast<std::size_t>(o->end() - it) - 1);
                    E25_ADD(erase_released_overflow, it->suffix.size());
                    E25_INC(string_frees);
                    o->erase(it);
                    --size_;
                    --overflow_count_;
                    return true;
                }
            }
        return false;
    }

#ifdef E25_DIAG
    // E26 long-run diagnosis, step 2: why does a placement get refused?
    //
    // READ-ONLY. Nothing here mutates the index: every level is a question asked of a
    // snapshot, which is what the review required (trials happen on a copy or a
    // read-only simulation, never on the timed object).
    //
    //   F0  the rule as it stands -- would put_primary take this key, and if not,
    //       which of its candidate sub-blocks refused and why
    //   F1  tidy ONLY the candidate sub-block (reclaim its own garbage, keep ownership
    //       and geometry): does the key fit then? If yes, byte fragmentation in that
    //       sub-block is the direct cause; if no, F2/F3 are the places to look
    //   F2  repack the WHOLE group within its existing budget: does the budget suffice
    //       at all?
    struct TrialPlace {
        std::size_t candidates = 0;       // sub-blocks put_primary would try
        std::size_t refused_by_bytes = 0;
        std::size_t refused_by_slots = 0;
        bool f0_fits = false;
        bool f1_fits = false;             // after tidying ANY candidate put_primary tries
        // The rank-restricted version: only the FIRST candidate put_primary considers.
        // E28 reported f1_fits as if it were this one. Kept BESIDE it so the gap between
        // "some candidate" and "the rank-selected candidate" is visible instead of assumed.
        bool f1_rank_fits = false;
        std::size_t cand_subblock = 0;      // which sub-block the first candidate is
        // The rank-selected candidate's OWN refuse reason, as two booleans. The
        // refused_by_* fields above are per-CANDIDATE counters; a share computed from them
        // is a share of candidates, not of events, and this project has already paid once
        // for presenting a per-candidate counter as an event property.
        bool cand_refused_by_slots = false;
        bool cand_refused_by_bytes = false;
        std::size_t cand_live_records = 0;  // live records in the first candidate
        std::size_t cand_headroom = 0;      // area - live in the first candidate = free + garbage
        bool f2_budget_suffices = false;  // whole-group repack, same budget
        std::size_t cand_garbage = 0;     // garbage in the first candidate (bytes)
        std::size_t cand_free_bytes = 0;
        std::size_t group_garbage = 0;
        std::size_t group_free_bytes = 0;
        std::size_t group_free_slots = 0;
        std::size_t group_live_bytes = 0;
    };

    TrialPlace trial_place(std::uint32_t g, std::string_view s) const {
        TrialPlace t;
        const std::size_t nb = subblocks(g);
        const std::size_t area = std::size_t(bb_[g]) - start_of(g);
        const std::size_t len = s.size();

        // The group's own totals, for F2's budget test.
        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            const auto* es = entries(g, i);
            std::size_t live = 0;
            for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
            t.group_live_bytes += live;
            const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
            if (occupied > live) t.group_garbage += occupied - live;
            t.group_free_bytes += std::size_t(h.bottom) - start_of(g);
            t.group_free_slots += slots_of(g) - h.count;
        }
        t.f2_budget_suffices = (t.group_live_bytes + len <= area * nb) &&
                               (t.group_free_slots > 0);

        // F0: put_primary's candidate order. Replicate its two steps.
        auto consider = [&](std::size_t i) {
            const auto& h = header(g, i);
            const std::size_t freeb = std::size_t(h.bottom) - start_of(g);
            const bool slot_ok = h.count < slots_of(g);
            const bool bytes_ok = len <= freeb;
            const bool was_first = (t.candidates == 0);
            if (was_first) {
                t.cand_free_bytes = freeb;
                t.cand_refused_by_slots = !slot_ok;
                t.cand_refused_by_bytes = !bytes_ok;
            }
            ++t.candidates;
            if (slot_ok && bytes_ok) t.f0_fits = true;
            // Unchanged, and still per-candidate: kept because E28/E30 report them.
            if (!slot_ok) ++t.refused_by_slots;
            else if (!bytes_ok) ++t.refused_by_bytes;

            // F1: tidy this sub-block -- reclaim its own garbage, keep its geometry.
            const auto* es = entries(g, i);
            std::size_t live = 0;
            for (std::size_t j = 0; j < h.count; ++j) live += es[j].length;
            const std::size_t occupied = std::size_t(bb_[g]) - h.bottom;
            const std::size_t garb = occupied > live ? occupied - live : 0;
            const bool first = (t.candidates == 1);
            if (first) {
                t.cand_subblock = i;
                t.cand_garbage = garb;
                t.cand_live_records = h.count;
                t.cand_headroom = area - live;      // = free bytes + garbage
            }
            const std::size_t tidied_free = area - live;   // repacked: live then free
            if (slot_ok && len <= tidied_free) {
                t.f1_fits = true;
                if (first) t.f1_rank_fits = true;   // the rank-restricted claim
            }
        };

        std::size_t prev = SIZE_MAX;
        for (std::size_t i = 0; i < nb; ++i) {
            const auto& h = header(g, i);
            if (h.count == 0) continue;
            const auto* es = entries(g, i);
            if (compare(suffix(g, i, es[h.count - 1]), s) < 0) { prev = i; continue; }
            if (prev != SIZE_MAX && compare(s, suffix(g, i, es[0])) < 0) consider(prev);
            consider(i);
            return t;
        }
        for (std::size_t j = prev == SIZE_MAX ? 0 : prev; j < nb; ++j) consider(j);
        return t;
    }


    // A2 entry diagnosis: one row per OVERFLOW EVENT, computed read-only.
    //
    // The population is fixed by the DRIVER, not here: the caller inserts, reads the
    // result back, and calls this only when the record came out of the overflow container.
    // Construction-time placements and drain re-stores are therefore excluded BY
    // CONSTRUCTION rather than by a predicate that has to be got right.
    struct A2Audit {
        bool valid = false;
        std::uint32_t group = 0;
        std::size_t subblocks = 0;
        std::size_t blk = 0;              // the rank-selected candidate

        // F2 -- why THAT candidate refused. Booleans: this is one event's answer.
        bool refuse_slots = false;
        bool refuse_bytes = false;

        // F3 -- would the GROUP's own totals have sufficed?
        bool group_slots_ok = false;
        bool group_bytes_ok = false;

        // F4 -- can freeing room by moving ONE adjacent boundary (M1-preserving, both
        // blocks staying inside their own slot and byte budgets) make the placement fit?
        bool adjacent_move_fits = false;

        // F5 -- what that move costs. F5c is in the unit the CODE pays, not "entries
        // changed": there is no incremental directory update, so any boundary move forces
        // one full rebuild_dir on the touched group.
        std::size_t moved_records = 0;
        std::size_t copied_bytes = 0;
        std::size_t dir_entries_rebuilt = 0;
    };

    A2Audit diag_a2(std::string_view full_key) const {
        A2Audit a;
        const std::size_t pre = prefix_.size();

        if (full_key.size() < pre || full_key.substr(0, pre) != prefix_) return a;

        const std::string_view suffix_in = full_key.substr(pre);
        const std::uint32_t g = canonical_group(full_key);
        if (g >= groups_) return a;

        a.valid = true;
        a.group = g;

        const std::size_t nb = subblocks(g);
        a.subblocks = nb;

        const TrialPlace tp = trial_place(g, suffix_in);
        a.blk = tp.cand_subblock;
        a.refuse_slots = tp.cand_refused_by_slots;
        a.refuse_bytes = tp.cand_refused_by_bytes;

        const std::size_t area = std::size_t(bb_[g]) - start_of(g);
        a.group_slots_ok = tp.group_free_slots > 0;
        a.group_bytes_ok = (tp.group_live_bytes + suffix_in.size() <= area * nb);

        // What the placement needs from block B: one free slot and len free bytes.
        const std::size_t len = suffix_in.size();
        const auto& hb = header(g, a.blk);
        const std::size_t b_free_slots = slots_of(g) - hb.count;
        const std::size_t b_free_bytes = std::size_t(hb.bottom) - start_of(g);
        const std::size_t need_slots = b_free_slots == 0 ? 1 : 0;
        const std::size_t need_bytes = len > b_free_bytes ? len - b_free_bytes : 0;

        // How many records must leave B, and how many bytes. Records are held in slot
        // order, so the TAIL holds the largest keys and the HEAD the smallest.
        std::size_t got_slots = 0, got_bytes = 0, k_tail = 0;
        const auto* esb = entries(g, a.blk);

        for (std::size_t j = hb.count; j-- > 0;) {
            if (got_slots >= need_slots && got_bytes >= need_bytes) break;
            got_slots += 1;
            got_bytes += esb[j].length;
            ++k_tail;
        }

        if (got_slots < need_slots || got_bytes < need_bytes) return a;   // B alone cannot give it

        // F5c: one full rebuild_dir per touched group, sized by its NON-EMPTY sub-blocks.
        std::size_t non_empty = 0;
        for (std::size_t i = 0; i < nb; ++i) if (header(g, i).count) ++non_empty;
        a.dir_entries_rebuilt = non_empty;

        // Two directions, because either can free room in B:
        //   TAIL of B -> head of B+1   (B's largest keys are still < B+1's smallest)
        //   HEAD of B -> tail of B-1   (B's smallest keys are still > B-1's largest)
        // Either way the receiver must stay inside ITS OWN slot cap and byte budget.
        auto receiver_accepts = [&](std::size_t j, bool from_tail) {
            if (j >= nb) return false;

            const auto& hr = header(g, j);
            std::size_t slots_left = slots_of(g) - hr.count;
            std::size_t bytes_left = std::size_t(hr.bottom) - start_of(g);
            std::size_t moved = 0, bytes = 0;

            for (std::size_t n = 0; n < k_tail; ++n) {
                const std::size_t rec = from_tail ? (hb.count - 1 - n) : n;
                const std::size_t l = esb[rec].length;

                if (slots_left == 0 || l > bytes_left) return false;

                --slots_left;
                bytes_left -= l;
                ++moved;
                bytes += l;
            }

            a.moved_records = moved;
            a.copied_bytes = bytes;

            return true;
        };

        if (a.blk + 1 < nb && receiver_accepts(a.blk + 1, true)) {
            a.adjacent_move_fits = true;
            return a;
        }

        if (a.blk > 0 && receiver_accepts(a.blk - 1, false)) {
            a.adjacent_move_fits = true;
            return a;
        }

        return a;
    }

    // E26 long-run diagnosis: which logical group does a key map to, and which
    // sub-block does that group's key order select? Diagnostic only; the physical
    // group id is the thing the capacity plan was built for, so a replacement that
    // changes it is a different placement question from one that does not.
    std::size_t diag_prefix_size() const { return prefix_.size(); }
    // E31 item 3: the live overflow containers' record counts, so the size distribution
    // at a checkpoint is a measurement rather than an inference from the record total.
    // Empty containers are skipped: a group that never overflowed would otherwise pull
    // the mean toward zero and make a few large containers look harmless. Read-only, and
    // called with the clock stopped.
    void diag_overflow_sizes(std::vector<std::size_t>& out) const {
        for (std::uint32_t g = 0; g < groups_; ++g)
            if (const auto* o = overflow(g))
                if (!o->empty()) out.push_back(o->size());
    }

    // Branch C: per-container capacity state, WITH ITS UNITS. Every field's unit is in its
    // name or its comment because the review's first finding was that "element capacity" had
    // no accessor at all and the only capacity-shaped quantity in the container is in BYTES.
    //
    // EMPTY CONTAINERS ARE INCLUDED. A container that once held records keeps its vector
    // capacity, so its slack is real; whether to count it depends on the question, so the
    // program reports every existing container and the ANALYZER picks the population.
#ifdef E25_DIAG
    // One byte-failure event, as it happened. Sizes are bounded by bb_ (<= 65535) and by
    // MAX_SLOTS, so the narrow types are exact rather than a guess.
    struct HoleEvent {
        std::uint32_t group = 0;
        std::uint16_t blk = 0;
        std::uint16_t len = 0;          // the suffix that could not be placed
        std::uint16_t contig = 0;       // contiguous free bytes the failing test looked at
        std::uint16_t garbage = 0;      // dead bytes in this block
        std::uint16_t live = 0;         // bytes a repack of this block would copy
        std::uint16_t nholes = 0;       // fragments a hole table would hold for this block
        std::uint16_t best_hole = 0;
        bool fits_hole = false;
        bool fits_repack = false;
        bool slots_free = false;        // true by construction; recorded, not assumed
    };
    // A cap, stated rather than silent. It is GLOBAL, not per region: the first version
    // capped each region at 200k and a full-scale run has enough regions that the cap
    // never bound anything, so the log grew without limit while still looking bounded.
    // The driver reports the drop count, so a truncated run reads as truncated rather than
    // as a complete sample. The AGGREGATES are counters and stay exact either way.
    static constexpr std::size_t kMaxHoleEvents = 500000;
    std::vector<HoleEvent> hole_events_;
    static std::size_t& diag_hole_total() { static std::size_t n = 0; return n; }
#endif

    const std::vector<HoleEvent>& hole_events() const { return hole_events_; }

    // The vector accumulates for the region's whole life, so a run that reports per
    // checkpoint has to be able to start each segment clean; otherwise the events of the
    // first quarter are indistinguishable from the events of the last.
    //
    // THE BUDGET IS RESET TOO, and that is not a detail. It is a process-global static, so
    // without this the first segments consume it and every later segment records nothing --
    // which is exactly what happened: the pure-insert arm recorded zero rows and a
    // 2.9M-refusal checkpoint contributed none, while the structure table was published as
    // if it described the population.
    void clear_hole_events() {
        hole_events_.clear();
        diag_hole_total() = 0;
    }

    struct OvCapStat {
        std::size_t size = 0;          // elements
        std::size_t cap_elems = 0;     // elements
        std::size_t cap_bytes = 0;     // bytes (container_slot_bytes)
        std::size_t high_water = 0;    // elements
        std::uint64_t drops = 0;       // events
        std::uint64_t regrowth = 0;    // events
        std::uint64_t peaks = 0;       // events: exceeded this container's own peak
        std::uint64_t cap_changes = 0; // events: the shipped policy grew the capacity
        bool stable = false;
    };

    void diag_overflow_caps(std::vector<OvCapStat>& out) const {
        for (std::uint32_t g = 0; g < groups_; ++g)
            if (const auto* o = overflow(g)) {
                OvCapStat s;
                s.size = o->size();
                s.cap_elems = o->capacity_elems();
                s.cap_bytes = o->container_slot_bytes();
                s.high_water = o->size_high_water();
                s.drops = o->drop_events();
                s.regrowth = o->regrowth_events();
                s.peaks = o->new_peak_events();
                s.cap_changes = o->cap_change_events();
                s.stable = o->stable_representation();
                out.push_back(s);
            }
    }

    // E36: how big an order entry really is. Published here because the movement and space
    // claims must cite sizeof rather than an intended 5 bytes -- if the struct pads to 8,
    // the logical movement is 1/6 of a 48-byte record and TWICE O1's 4-byte handle, and
    // that trade is exactly what the round measures.
    static std::size_t ordered_entry_bytes() { return sizeof(OrderedEntry); }

    // E33/O1: the live overflow records' SUFFIX LENGTHS, so "what actually moves" is a
    // measurement rather than a guess. std::string keeps up to 15 bytes inline (SSO), so a
    // record's 48-byte object CONTAINS its characters when the suffix is at most 15 bytes,
    // and only points at them when it is longer. Read-only, called with the clock stopped.
    void diag_overflow_suffix_lengths(std::vector<std::size_t>& out) const {
        for (std::uint32_t g = 0; g < groups_; ++g)
            if (const auto* o = overflow(g))
                for (const auto& r : *o) out.push_back(r.suffix.size());
    }

#ifdef E25_DIAG
    // E36: sum the inline-tag pairing check over this region's containers. Only a group
    // that has a container contributes, since the other arms have no order array at all.
    std::size_t diag_tag_pairing(std::size_t& checked) const {
        std::size_t bad = 0;
        for (std::uint32_t g = 0; g < groups_; ++g)
            if (const auto* o = overflow(g)) bad += o->diag_tag_pairing(checked);
        return bad;
    }
#endif
    // The element size the movement byte counters are counted in, so a reader can turn
    // "shifted bytes" back into "shifted records" without guessing. sizeof() on the
    // private record type can only be named where the type is visible, which is here.
    static constexpr std::size_t diag_overflow_record_bytes() { return sizeof(OverflowRecord); }
    std::uint32_t diag_group(std::string_view key) const {
        if (!prefix_matches(key)) return UINT32_MAX;   // no valid group
        return canonical_group(key);
    }

#ifdef E25_DIAG
    // Scan-entry round: enumerate one group's records so the hand-check can recompute the
    // starting-position work INDEPENDENTLY of the counters the scan itself reports. The
    // primary entries come back in sub-block-major order (blk, slot, suffix copy) -- the
    // exact order the advance loop walks -- and the overflow records in container order.
    // Read-only, diagnostic build only; a timing build does not carry either function.
    struct DiagScanEntry {
        std::size_t blk = 0;
        std::size_t slot = 0;
        std::string suffix;
    };
    void diag_primary_entries(std::uint32_t g, std::vector<DiagScanEntry>& out) const {
        out.clear();
        const std::size_t nb = subblocks(g);
        for (std::size_t blk = 0; blk < nb; ++blk) {
            const std::size_t count = header(g, blk).count;
            for (std::size_t i = 0; i < count; ++i) {
                DiagScanEntry e;
                e.blk = blk;
                e.slot = i;
                e.suffix = std::string(suffix(g, blk, entries(g, blk)[i]));
                out.push_back(std::move(e));
            }
        }
    }
    void diag_ov_entries(std::uint32_t g, std::vector<std::string>& out) const {
        out.clear();
        if (const auto* o = overflow(g))
            for (std::size_t i = 0; i < o->size(); ++i) out.emplace_back((*o)[i].suffix);
    }
    // The advance loop steps over sub-blocks as whole units, EMPTY ONES INCLUDED, so an
    // entry list alone undercounts the sub-blocks it passes. The hand-check walks the
    // per-sub-block counts instead, which is the order the loop itself uses.
    std::size_t diag_subblocks(std::uint32_t g) const { return subblocks(g); }
    std::size_t diag_subblock_count(std::uint32_t g, std::size_t blk) const {
        return header(g, blk).count;
    }
    std::string diag_suffix(std::uint32_t g, std::size_t blk, std::size_t slot) const {
        return std::string(suffix(g, blk, entries(g, blk)[slot]));
    }
#endif

    // check_scan_contract's negative control, and the only thing in the tree that can
    // make key -> group non-monotone on purpose. Rotating every group id by one is
    // deterministic, in range, and a bijection, so the index stays internally
    // coherent: what stops matching key order is the group-ASCENDING traversal the
    // scan and its start cursor rely on. With the flag set, the scan must therefore
    // return records out of key order, or omit records, or both -- which is the whole
    // point, because it shows the scan's correctness rests on the mapping's
    // monotonicity and not on M1 alone.
    //
    // DIAGNOSTIC ONLY, and deliberately NOT a Config switch: the four switches.hpp
    // tables (baseline_config, pin_frozen_switches, switches, the Field table) are
    // untouched, so no frozen or shipped configuration can see this. Compiled out
    // unless -DE25_DIAG, like every other hook in this block; a timed build does not
    // even carry the branch.
    static inline bool sar_rotate_groups = false;

    // G1 FAILURE INJECTION. Diagnostic-only in the same way as the hook above: no Config
    // switch, no switches.hpp table touched, compiled out unless -DE25_DIAG.
    //
    // `inject_at < 0` is off. Otherwise the `inject_at`-th compaction that reaches the
    // named point throws std::bad_alloc:
    //
    //   point 0 -- PREPARE. After the gather has read every live record and built its
    //              scratch buffers, and BEFORE the first byte of the group is touched.
    //   point 1 -- COMMIT. During the replay, after the blocks have been reset and half
    //              the records re-placed, as if a placement refused a record the group
    //              had just accepted.
    //
    // The two ask different questions and are NOT interchangeable. Point 0 asks whether a
    // maintenance that fails while it still costs only its own scratch leaves the index
    // exactly as it was. Point 1 asks whether the in-place replay can survive a commit
    // that fails half way -- which is the case task book section 6.3's
    // prepare/verify/commit split was written to rule out, and which this design does not
    // provide, because E31 traded the temporary buffer for speed.
    inline static long long inject_at = -1;
    inline static long long inject_point = 0;
    inline static long long inject_seen = 0;
    inline static long long injected = 0;

    static bool maybe_inject(long long point) {
        if (inject_at < 0 || inject_point != point) return false;
        if (inject_seen++ != inject_at) return false;
        ++injected;
        return true;
    }
#endif

#ifdef E30_TIME
    // E30 probe: the per-record cost of an in-place sub-block tightening, measured on a
    // scratch copy so the index is never touched. mutable because the probe is const.
    mutable std::vector<std::byte> e30_scratch_;
    mutable std::vector<std::size_t> e30_order_;

    void e30_repack_probe(std::uint32_t g, std::size_t blk, std::uint64_t& copy_ns,
                          std::uint64_t& pass_ns, std::size_t& records,
                          std::uint64_t& moved_bytes, bool& pass_ok) const {
        const std::size_t bytes = bb_[g];
        if (e30_scratch_.size() < bytes) e30_scratch_.resize(bytes);
        if (e30_order_.size() < slots_of(g)) e30_order_.resize(slots_of(g));

        const auto tc0 = E30_T0();
        std::memcpy(e30_scratch_.data(), base_of(g, blk), bytes);
        copy_ns = E30_T0() - tc0;

        std::byte* buf = e30_scratch_.data();
        auto* h = reinterpret_cast<Header*>(buf);
        auto* es = reinterpret_cast<Entry*>(buf + sizeof(Header));
        const std::size_t n = h->count;
        records = n;
        moved_bytes = 0;
        pass_ok = true;
        if (n == 0) { pass_ns = 0; return; }

        // Snapshot the pre-pass view so the pass can be checked, not argued.
        std::vector<std::string> before;
        before.reserve(n);
        for (std::size_t j = 0; j < n; ++j)
            before.emplace_back(reinterpret_cast<const char*>(buf + es[j].offset), es[j].length);

        const std::size_t start = sizeof(Header) + slots_of(g) * sizeof(Entry);
        const auto tp0 = E30_T0();

        // Slot order, sorted by offset DESCENDING (insertion sort; n <= 48).
        for (std::size_t j = 0; j < n; ++j) e30_order_[j] = j;
        for (std::size_t a = 1; a < n; ++a) {
            const std::size_t key = e30_order_[a];
            const auto koff = es[key].offset;
            std::size_t b = a;
            while (b > 0 && es[e30_order_[b - 1]].offset < koff) {
                e30_order_[b] = e30_order_[b - 1];
                --b;
            }
            e30_order_[b] = key;
        }

        std::size_t top = bytes;
        for (std::size_t a = 0; a < n; ++a) {
            Entry& e = es[e30_order_[a]];
            top -= e.length;
            std::memmove(buf + top, buf + e.offset, e.length);
            moved_bytes += e.length;
            e.offset = static_cast<std::uint32_t>(top);
        }
        h->bottom = static_cast<std::uint16_t>(top);
        pass_ns = E30_T0() - tp0;

        if (top < start) { pass_ok = false; return; }
        for (std::size_t j = 0; j < n; ++j) {
            const std::string_view now(reinterpret_cast<const char*>(buf + es[j].offset), es[j].length);
            if (now != before[j]) { pass_ok = false; break; }
        }
    }
#endif

#ifdef E24_DIAG
    // E24: live per-group geometry, walked from the REAL structure -- not the
    // static GroupStats model, which describes an input rather than the current
    // layout. Appends one report per logical group.
    void e24_report(std::vector<e24::GroupReport>& out) const {
        for (std::uint32_t g = 0; g < groups_; ++g) {
            e24::GroupReport rep;
            rep.group = g;
            const std::size_t nb = subblocks(g);
            rep.subs.reserve(nb);
            for (std::size_t blk = 0; blk < nb; ++blk) {
                const auto& h = header(g, blk);
                e24::SubBlock sb;
                sb.slots = slots_of(g);
                sb.used = h.count;
                sb.suffix_cap = std::size_t(bb_[g]) - start_of(g);
                const auto* es = entries(g, blk);
                for (std::size_t i = 0; i < h.count; ++i) sb.suffix_live += es[i].length;
                sb.suffix_free = std::size_t(h.bottom) - start_of(g);
                sb.occupied = sb.suffix_cap - sb.suffix_free;
                sb.garbage = sb.occupied - sb.suffix_live;   // deleted bytes not reclaimed
                rep.subs.push_back(sb);
            }

            if (const auto* o = overflow(g)) {
                // O1: the container reports its own slot bytes, so this accounting
                // does not have to know which representation is live. The OFF value
                // is the expression that used to be here.
                rep.ov_container_cap = o->container_slot_bytes();
                rep.ov_rec_size = sizeof(OverflowRecord);
                Result tmp;
                for (const auto& e : *o) {
                    ++rep.ov_records;
                    rep.ov_live_bytes += e.suffix.size();
                    rep.ov_suffix_cap += e.suffix.capacity() + 1;
                    const std::size_t L = e.suffix.size();

                    // M1: with more than one sub-block, key order selects exactly ONE
                    // candidate. With a single sub-block the choice is vacuous, and
                    // within-block sorted insert handles the position.
                    std::size_t cand = 0;
                    if (nb == 0) { ++rep.ov_blocked_order; continue; }
                    if (nb > 1) {
                        cand = has_dir(g) ? locate_subblock(g, e.suffix, tmp) : 0;
                        if (cand == NO_SUBBLOCK) { ++rep.ov_blocked_order; continue; }
                    }

                    const auto& h = header(g, cand);
                    const bool slot_ok = h.count < slots_of(g);
                    const bool bytes_ok = L <= std::size_t(h.bottom) - start_of(g);
                    if (slot_ok && bytes_ok) {
                        ++rep.ov_fits_ordered;
                        rep.ov_fits_ordered_bytes += L;
                    } else if (!slot_ok) {
                        ++rep.ov_blocked_slots;
                    } else {
                        ++rep.ov_blocked_bytes;
                    }
                    // What the existing coarse predicate would claim instead.
                    if (group_has_room(g, L)) ++rep.ov_fits_if_any_subblock;
                }
            }
            out.push_back(std::move(rep));
        }
    }
#endif

    // Ordered merge of primary entries and overflow; the group count is monotone
    // in the key order, so scanning groups in ascending order yields sorted output.
    void scan(std::string_view start, std::size_t limit, std::vector<Pair>& out) const {
        if (!size_ || out.size() >= limit) return;

        bool matches = prefix_matches(start);
        if (!matches && compare(start, prefix_) > 0) return;
        const auto s = matches ? start.substr(prefix_.size()) : std::string_view{};
        const auto first = matches ? canonical_group(start) : 0;
#ifdef E25_DIAG
        // Counted only when it is actually evaluated: this region is the first one, so a
        // later region in the same Index::scan never re-derives the group. Whether that is
        // true is exactly what the diagnosis is for, so the counter sits on the call.
        if (matches) E25_INC(scan_canonical_group);
#endif

        for (std::uint32_t g = first; g < groups_ && out.size() < limit; ++g) {
            E25_INC(scan_groups);
            const auto* o = overflow(g);
            const std::size_t nb = subblocks(g);
            std::size_t blk = 0, a = 0, b = 0;
            E25_INC(scan_cursor_starts);
            if (o) E25_INC(scan_ov_cursor_starts);

            if (g == first && matches) {
                // Advance the primary cursor past every entry < s. Sub-blocks are
                // rank-ordered, so this skips whole sub-blocks at a time.
                //
                // Rewritten from the short-circuit form so the two things the round has
                // to price can be counted where they happen: the comparison (evaluated
                // even when it FAILS and stops the advance, so cmps is skipped, or
                // skipped plus one per stopped sub-block) and the record it consumed.
                // E25_INC is a no-op outside the diagnostic build, and the rewrite keeps
                // the original evaluation order, so timed semantics are unchanged.
                //
                // scan_ordered_start replaces the WALK with an ORDERED SEARCH that lands
                // on the same position, without reading the records in between. The
                // skipped-record counter below is computed from the landed position, so
                // "same records passed, fewer comparisons" is a measured statement.
                if (config_.scan_ordered_start) {
                    // Level 1: the first primary sub-block whose LAST record is >= s.
                    // Emptied sub-blocks sit anywhere in the rank order, so the probe at
                    // a midpoint walks left to the last non-empty block at or before it;
                    // the predicate is monotone over physical indices, which is what the
                    // binary search needs.
                    std::size_t lo = 0, hi = nb;
                    while (lo < hi) {
                        const std::size_t mid = lo + (hi - lo) / 2;
                        std::size_t j = mid;
                        while (j > 0 && header(g, j).count == 0) --j;
                        bool ge = false;
                        if (header(g, j).count > 0) {
                            E25_INC(scan_adv_p_cmps);
                            ge = !(compare(suffix(g, j, entries(g, j)[header(g, j).count - 1]),
                                           s) < 0);
                        }
                        if (ge) hi = mid; else lo = mid + 1;
                    }
                    blk = lo;
                    a = 0;
                    if (blk < nb) {
                        // Level 2: the first slot of that sub-block whose suffix is >= s.
                        std::size_t clo = 0, chi = header(g, blk).count;
                        while (clo < chi) {
                            const std::size_t mid = clo + (chi - clo) / 2;
                            E25_INC(scan_adv_p_cmps);
                            if (compare(suffix(g, blk, entries(g, blk)[mid]), s) < 0) clo = mid + 1;
                            else chi = mid;
                        }
                        a = clo;
                    }
#ifdef E25_DIAG
                    // The same set of records the linear walk would have passed, derived
                    // from where the search landed. blk counts fully-passed sub-blocks
                    // (empty ones included), exactly as the linear loop counted them.
                    {
                        std::size_t skipped = a;
                        for (std::size_t t = 0; t < blk && t < nb; ++t)
                            skipped += header(g, t).count;
                        E25_ADD(scan_adv_p_skipped, skipped);
                        E25_ADD(scan_adv_p_subblocks, blk);
                    }
#endif
                } else {
                    while (blk < nb) {
                        while (a < header(g, blk).count) {
                            E25_INC(scan_adv_p_cmps);
                            if (!(compare(suffix(g, blk, entries(g, blk)[a]), s) < 0)) break;
                            E25_INC(scan_adv_p_skipped);
                            ++a;
                        }
                        if (a < header(g, blk).count) break;
                        E25_INC(scan_adv_p_subblocks);
                        ++blk;
                        a = 0;
                    }
                }
                if (o) {
                    if (config_.scan_ordered_start) {
                        // The container's own ordered lookup; the same position the linear
                        // walk stopped at. Its internal comparisons are shared with the
                        // write path and are not individually counted -- the search itself
                        // is counted as one, and the skipped count is exact.
                        b = o->lower_bound_pos(s);
                        E25_INC(scan_adv_ov_cmps);
                        E25_ADD(scan_adv_ov_skipped, b);
                    } else {
                        while (b < o->size()) {
                            E25_INC(scan_adv_ov_cmps);
                            if (!(compare((*o)[b].suffix, s) < 0)) break;
                            E25_INC(scan_adv_ov_skipped);
                            ++b;
                        }
                    }
                }
            }

            while (out.size() < limit) {
                // Skip empty and exhausted sub-blocks before looking at the head.
                // This must happen here and not only when the last entry is taken:
                // an emptied sub-block in the middle would otherwise end the whole
                // group's scan and silently drop every later sub-block's records.
                while (blk < nb && a >= header(g, blk).count) {
                    E25_INC(scan_blocks_skipped);
                    ++blk;
                    a = 0;
                }

                const bool have_a = blk < nb;
                const bool have_b = o && b < o->size();
                if (!have_a && !have_b) {
                    if (blk == 0) E25_INC(scan_groups_empty);   // nothing here at all
                    break;
                }

                bool take_a = have_a;
                if (have_a && have_b) {
                    E25_INC(scan_merge_cmps);
                    take_a = compare(suffix(g, blk, entries(g, blk)[a]), (*o)[b].suffix) < 0;
                }
                if (take_a) {
                    const auto suf = suffix(g, blk, entries(g, blk)[a]);
                    // NOTE: without the flag this is prefix_ + std::string(suffix): a
                    // temporary string plus a concatenation. Counted as the allocations
                    // it is -- 2, not 1.
                    E25_ADD(scan_result_allocs, config_.scan_reserve_result ? 1 : 2);
                    if (config_.scan_reserve_result) {
                        std::string full;
                        full.reserve(prefix_.size() + suf.size());
                        full.append(prefix_);
                        full.append(suf);
                        out.emplace_back(std::move(full), entries(g, blk)[a].value);
                    } else {
                        out.emplace_back(prefix_ + std::string(suf),
                                         entries(g, blk)[a].value);
                    }
                    E25_INC(scan_out_primary);
                    E25_ADD(scan_out_bytes, prefix_.size() + suf.size());
                    ++a;
                } else {
                    E25_ADD(scan_result_allocs, 1);
                    out.emplace_back(prefix_ + (*o)[b].suffix, (*o)[b].value);
                    E25_INC(scan_out_overflow);
                    E25_ADD(scan_out_bytes, prefix_.size() + (*o)[b].suffix.size());
                    ++b;
                }
            }
        }
    }

#ifdef E25_DIAG
    // M4 entry diagnosis (2026-10-11): an independent SHADOW cursor that walks this region
    // through the FROZEN candidate policy and counts, per call:
    //
    //   on arriving at a non-empty primary sub-block, let r be the records from the current
    //   slot to the block end clipped by the remaining return budget. Only when r >= 4 and
    //   the overflow side still has records, compare the segment's LAST key with the
    //   overflow's CURRENT first key. Strictly less lets the whole segment be emitted row by
    //   row with no per-row cross-side compare; anything else falls back to the original
    //   row-by-row merge for the rest of THIS sub-block, which is probed exactly once. A
    //   side that is already exhausted never probes and never counts as a saving.
    //
    // The proof reads the same descriptor bytes an implementable candidate would use -- no
    // answer key is consulted. The caller checks that the shadow's rows equal the real
    // scan's rows AND the ordered reference, and that merge_cmps + seg_rows equals the real
    // scan's measured cross-side compare count, so a drifting shadow fails the run.
    void diag_scan_merge_shadow(std::string_view start, std::size_t limit,
                                std::vector<Pair>& out, MergeShadowCounts& m) const {
        if (!size_ || out.size() >= limit) return;

        bool matches = prefix_matches(start);
        if (!matches && compare(start, prefix_) > 0) return;
        const auto s = matches ? start.substr(prefix_.size()) : std::string_view{};
        const auto first = matches ? canonical_group(start) : 0;

        int last_side = 0;   // 0 none, 1 primary, 2 overflow

        for (std::uint32_t g = first; g < groups_ && out.size() < limit; ++g) {
            const auto* o = overflow(g);
            const std::size_t nb = subblocks(g);
            std::size_t blk = 0, a = 0, b = 0;
            std::size_t probed_blk = static_cast<std::size_t>(-1);

            if (g == first && matches) {
                // Same landing as the production start advance: the S2 ordered search lands
                // on this exact position (that equivalence is gated in check_scan_contract),
                // and a landing error would show up as a row mismatch below.
                while (blk < nb) {
                    while (a < header(g, blk).count &&
                           compare(suffix(g, blk, entries(g, blk)[a]), s) < 0)
                        ++a;
                    if (a < header(g, blk).count) break;
                    ++blk;
                    a = 0;
                }
                if (o)
                    while (b < o->size() && compare((*o)[b].suffix, s) < 0) ++b;
            }

            while (out.size() < limit) {
                while (blk < nb && a >= header(g, blk).count) {
                    ++blk;
                    a = 0;
                }

                const bool have_a = blk < nb;
                const bool have_b = o && b < o->size();
                if (!have_a && !have_b) break;

                if (have_a && blk != probed_blk) {
                    probed_blk = blk;
                    const std::size_t remain = limit - out.size();
                    const std::size_t left = header(g, blk).count - a;
                    const std::size_t r = std::min(left, remain);
                    if (r < 4) {
                        m.skip_short += 1;
                    } else if (!have_b) {
                        m.skip_no_ov += 1;
                    } else {
                        m.probes += 1;
                        std::uint64_t bytes = 0;
                        const int c = compare_bytes(
                            suffix(g, blk, entries(g, blk)[a + r - 1]), (*o)[b].suffix, bytes);
                        m.probe_bytes += bytes;
                        if (c < 0) {
                            m.probe_ok += 1;
                            m.seg_rows += r;
                            if (r > m.seg_rows_max) m.seg_rows_max = r;
                            m.seg_lens.push_back(static_cast<std::uint32_t>(r));
                            for (std::size_t t = 0; t < r; ++t) {
                                const auto suf = suffix(g, blk, entries(g, blk)[a]);
                                std::string full;
                                full.reserve(prefix_.size() + suf.size());
                                full.append(prefix_);
                                full.append(suf);
                                out.emplace_back(std::move(full), entries(g, blk)[a].value);
                                ++a;
                            }
                            m.outputs += r;
                            if (last_side != 1) {
                                if (last_side != 0) m.side_switches += 1;
                                last_side = 1;
                            }
                            continue;
                        }
                        m.probe_fail += 1;
                        if (c == 0) m.probe_fail_equal += 1;
                        m.probe_fail_bytes += bytes;
                    }
                }

                bool take_a = have_a;
                if (have_a && have_b) {
                    m.merge_cmps += 1;
                    take_a = compare(suffix(g, blk, entries(g, blk)[a]), (*o)[b].suffix) < 0;
                }
                if (take_a) {
                    const auto suf = suffix(g, blk, entries(g, blk)[a]);
                    std::string full;
                    full.reserve(prefix_.size() + suf.size());
                    full.append(prefix_);
                    full.append(suf);
                    out.emplace_back(std::move(full), entries(g, blk)[a].value);
                    if (last_side != 1) {
                        if (last_side != 0) m.side_switches += 1;
                        last_side = 1;
                    }
                    ++a;
                } else {
                    out.emplace_back(prefix_ + (*o)[b].suffix, (*o)[b].value);
                    if (last_side != 2) {
                        if (last_side != 0) m.side_switches += 1;
                        last_side = 2;
                    }
                    ++b;
                }
                m.outputs += 1;
            }
        }
    }
#endif

#ifdef E25_SCAN_CONSUME
    // ---------------------------------------------------------------------------
    // Scan-entry round, stage B2: the traversal WITHOUT result materialisation.
    //
    // The counters said the start-advance is not the cost (one comparison, zero
    // skipped records in every regressing cell) but the wall clock still differed,
    // so the plan's single bounded discriminator is to run the SAME traversal with
    // the result strings left unbuilt and see whether the W/B gap survives.
    //
    // This is a DIAGNOSTIC CONSUMPTION MODE, not a deployable API: the rows are
    // delivered to a sink instead of a vector<Pair>. Guarded by E25_SCAN_CONSUME,
    // which no existing target defines -- the timed targets' code does not see any
    // of it. The body below is a mirror of scan(), row for row and branch for
    // branch; the only difference is the row-production points.
    // ---------------------------------------------------------------------------
    struct ConsumeSink {
        // Timed mode: counts and hashes only. Verify mode: additionally rebuilds the
        // full key so the same rows can be checked against the reference one by one.
        std::uint64_t rows = 0;
        std::uint64_t bytes = 0;                     // suffix bytes delivered
        std::uint64_t full_bytes = 0;                // prefix+suffix bytes, verify mode
        std::uint64_t hash = 0;                      // row checksum (value + suffix bytes)
        std::uint64_t suffix_hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // <=15,<=23,<=31,<=63,<=127,<=255,<=1023,>1023
        std::vector<Pair>* capture = nullptr;        // non-null: verify mode
        const std::string* prefix = nullptr;         // set per region call, verify mode only
        // Timed mode pays for the row and its byte count and nothing else: hashing and
        // the histogram belong to the untimed verify pass, and a byte loop inside the
        // timed span would swamp the very difference the discriminator is measuring.
        bool timed_fast = false;

        static void bucket(std::uint64_t* h, std::size_t n) {
            const std::size_t b = n <= 15 ? 0 : n <= 23 ? 1 : n <= 31 ? 2 : n <= 63 ? 3
                                : n <= 127 ? 4 : n <= 255 ? 5 : n <= 1023 ? 6 : 7;
            ++h[b];
        }

        void row(std::string_view suffix, std::uint64_t value) {
            ++rows;
            bytes += suffix.size();
            if (timed_fast) return;
            hash = (hash ^ value) * 1099511628211ULL;
            for (unsigned char c : suffix) hash = (hash ^ c) * 1099511628211ULL;
            bucket(suffix_hist, suffix.size());
            if (capture) {
                std::string full;
                full.reserve(prefix->size() + suffix.size());
                full.append(*prefix);
                full.append(suffix);
                full_bytes += full.size();
                capture->emplace_back(std::move(full), value);
            }
        }
    };

    void scan_consume(std::string_view start, std::size_t limit, ConsumeSink& sink) const {
        if (!size_ || sink.rows >= limit) return;

        sink.prefix = &prefix_;
        bool matches = prefix_matches(start);
        if (!matches && compare(start, prefix_) > 0) return;
        const auto s = matches ? start.substr(prefix_.size()) : std::string_view{};
        const auto first = matches ? canonical_group(start) : 0;

        for (std::uint32_t g = first; g < groups_ && sink.rows < limit; ++g) {
            const auto* o = overflow(g);
            const std::size_t nb = subblocks(g);
            std::size_t blk = 0, a = 0, b = 0;

            if (g == first && matches) {
                while (blk < nb) {
                    while (a < header(g, blk).count) {
                        if (!(compare(suffix(g, blk, entries(g, blk)[a]), s) < 0)) break;
                        ++a;
                    }
                    if (a < header(g, blk).count) break;
                    ++blk;
                    a = 0;
                }
                if (o) {
                    while (b < o->size()) {
                        if (!(compare((*o)[b].suffix, s) < 0)) break;
                        ++b;
                    }
                }
            }

            while (sink.rows < limit) {
                while (blk < nb && a >= header(g, blk).count) {
                    ++blk;
                    a = 0;
                }

                const bool have_a = blk < nb;
                const bool have_b = o && b < o->size();
                if (!have_a && !have_b) break;

                bool take_a = have_a;
                if (have_a && have_b) {
                    take_a = compare(suffix(g, blk, entries(g, blk)[a]), (*o)[b].suffix) < 0;
                }
                if (take_a) {
                    sink.row(suffix(g, blk, entries(g, blk)[a]), entries(g, blk)[a].value);
                    ++a;
                } else {
                    sink.row((*o)[b].suffix, (*o)[b].value);
                    ++b;
                }
            }
        }
    }
#endif

    std::vector<Pair> all() const {
        const auto t0 = E14_T0();
        std::vector<Pair> out;
        out.reserve(size_);
        scan({}, size_, out);
        E14_ADD(region_all, t0);
        return out;
    }
};

// ---------------------------------------------------------------------------
// Index: outer router over ordered region fences.
// ---------------------------------------------------------------------------

// Test-only accessor, declared here and defined after the class. It exists so the
// round's gate can build counter-examples with a mismatched or stale context;
// nothing in the library or the benchmarks should go through it.
struct IndexTestAccess;

class Index {
    Config config_;
    struct Node {
        std::string fence;
        std::unique_ptr<Region> region;
    };
    std::vector<Node> nodes_;
    std::size_t size_ = 0;

    // ---- compact region routing (candidate) --------------------------------
    // MUTABLE because the rebuild is lazy and happens inside locate(), which is const. The
    // alternative -- rebuilding at each mutation site -- is a list of call sites that can
    // grow a missing entry; a version compare cannot. `routing_version_` starts at a value no
    // structure_version_ can take, so the first lookup after construction always rebuilds.
    //
    // Kept only while the switch is ON: a default build pays one integer compare per lookup
    // and allocates nothing.
    mutable std::vector<std::uint64_t> fence_digests_;
    mutable std::size_t fence_prefix_len_ = 0;
    mutable std::size_t routing_version_ = static_cast<std::size_t>(-1);

    // E38: the one shared LITS table, or null when the backend is the reference model.
    std::shared_ptr<const lits_hpt::SharedModel> lits_shared_;

    // E6-1: bumped on every fence-changing event so auxiliary routing metadata
    // (built outside the index) can detect staleness and rebuild.
    std::uint64_t structure_version_ = 0;

public:
    std::size_t rebuilds = 0, splits = 0;

    explicit Index(Config c) : config_(c) {}

    // ---- E38: the shared LITS table -------------------------------------------------------
    //
    // ONE table for the whole index, trained from the initial complete key set and read-only
    // afterwards (task book section 2.2). Every region and the statistics replay hold the same
    // shared_ptr, so "the demand pass and the final build used the same mapping" is a property
    // of the object graph rather than a consequence of two trainings happening to agree.
    //
    // Trained LAZILY and once. The alternative -- a constructor parameter -- would put the
    // training on every path that builds an Index, including the ones that never look at a key.
    void ensure_lits_shared(const std::vector<Pair>& pairs) {
        if (config_.model_backend != ModelBackend::LitsHpt || lits_shared_) return;

        // Fewer than two keys cannot be trained on -- the pinned train() reads keys[1] -- and
        // must not throw either: an index built from an empty or single-key set is a legitimate
        // object whose regions all take the boundary path. So the table is simply not built,
        // and `calibrate_lits` handles the null.
        if (pairs.size() < 2) return;

        std::vector<std::string> keys;
        keys.reserve(pairs.size());
        for (const auto& kv : pairs) keys.push_back(kv.first);

        lits_shared_ = lits_hpt::SharedModel::train_shared(keys);
    }

    // Adopt a table that was trained elsewhere. Used by bulk_load_reusing_ids, where the
    // STATISTICS INDEX would otherwise train its own copy: two identical tables are two
    // trainings, and the task book's section 5.2 is explicit that the default is one shared
    // immutable object.
    void adopt_lits_shared(std::shared_ptr<const lits_hpt::SharedModel> shared) {
        if (!lits_shared_) lits_shared_ = std::move(shared);
    }

    const std::shared_ptr<const lits_hpt::SharedModel>& lits_shared() const { return lits_shared_; }

    const std::string& fence(std::size_t i) const { return nodes_[i].fence; }
    std::uint64_t structure_version() const { return structure_version_; }

    void bulk_load(const std::vector<Pair>& pairs) {
        const auto e14_t0 = E14_T0();
        // E38: one table for the whole index, trained from the complete key set BEFORE any
        // region exists -- so every region below references the same object.
        ensure_lits_shared(pairs);
        nodes_.clear();
        size_ = pairs.size();
        const auto chunk = config_.region_keys ? config_.region_keys : std::max<std::size_t>(1, pairs.size());
        for (std::size_t p = 0; p < pairs.size(); p += chunk) {
            std::vector<Pair> part(pairs.begin() + p, pairs.begin() + std::min(p + chunk, pairs.size()));
            nodes_.push_back({part.front().first,
                             std::make_unique<Region>(part, config_, lits_shared_)});
        }
        if (nodes_.empty())
            nodes_.push_back({{}, std::make_unique<Region>(std::vector<Pair>{}, config_,
                                                          lits_shared_)});
        ++structure_version_;
        E14_ADD(bulk_load, e14_t0);
    }

    // -----------------------------------------------------------------------
    // E15: build every region directly under a supplied capacity plan.
    //
    // This replaces the pair "bulk_load under the DEFAULT capacity, then one
    // rebuild_with_caps per region". Both halves construct a full Region per
    // region, but the first half's only consumer is Region::all(), which returns
    // a contiguous slice of the sorted input -- exactly the slice this method
    // passes in directly. E14 measured that discarded half at 55.6%-66.3% of the
    // build (E14_TABLES.md §3), so it is pure cost.
    //
    // The chunking, the fences and the region count are bulk_load's, and the
    // constructor call per region is rebuild_with_caps's, so the structure this
    // produces is the same one the two-pass pipeline produces. That is not
    // asserted here -- bench_e15 compares layout digests before it reports any
    // timing, and the digest covers the partition, the prefix, all three capacity
    // tables, the derived sub-block structure, the boundary directory and every
    // stored (key, value).
    //
    // A plan that does not cover every region is rejected rather than silently
    // padded, and a per-region shape that does not match that region's group
    // count is rejected by the Region constructor itself.
    //
    // Strong exception guarantee: the plan is validated and every Region is built
    // into a LOCAL vector, which is swapped in only once all of them succeeded. A
    // rejected plan therefore leaves the index in its previous state rather than
    // half-built. (Without this, a mid-loop throw left `nodes_` holding some of
    // the new regions while `size_` already reported the full key count -- an
    // internally inconsistent object for any caller that caught the exception.
    // check_e15_plan's case A1 asserts the guarantee directly.)
    // -----------------------------------------------------------------------
    void bulk_load_with_plan(const std::vector<Pair>& pairs,
                             const std::vector<std::vector<std::uint16_t>>& sg,
                             const std::vector<std::vector<std::uint32_t>>& bb,
                             const std::vector<std::vector<std::uint16_t>>& nb_cap = {}) {
        bulk_load_with_context(pairs, sg, bb, nb_cap, nullptr);
    }

    // INTERNAL. Takes a BuildContext, which must not be held across calls or
    // replayed against other data -- see bulk_load_reusing_ids. Exposed only so the
    // round's gate can exercise the misuse cases; callers use the one-call form.
 private:
    friend struct IndexTestAccess;

    // PRIVATE. Takes a BuildContext, which must not be held across calls or replayed
    // against other data -- see bulk_load_reusing_ids. Reachable only through
    // IndexTestAccess. Ordinary callers use the one-call form and cannot obtain a
    // context at all, which is the point: the guarantee is the lifetime.
    void bulk_load_with_context(const std::vector<Pair>& pairs,
                                const std::vector<std::vector<std::uint16_t>>& sg,
                                const std::vector<std::vector<std::uint32_t>>& bb,
                                const std::vector<std::vector<std::uint16_t>>& nb_cap,
                                const BuildContext* ctx) {
        const auto e15_t0 = E14_T0();
        E14_INC(bulk_planned);
        if (!nb_cap.empty() && nb_cap.size() != sg.size())
            throw std::invalid_argument("E15: sub-block cap table size");

        // E38: same table as the statistics pass -- see bulk_load_reusing_ids.
        ensure_lits_shared(pairs);
        const auto chunk = config_.region_keys ? config_.region_keys : std::max<std::size_t>(1, pairs.size());

        // E21: a context is only usable on the very key vector it was built from.
        // Address and size are the cheap half of that; the per-region key digest is
        // the half that survives an in-place edit.
        const bool source_ok = ctx != nullptr && ctx->source == pairs.data() &&
                               ctx->source_size == pairs.size();

        std::vector<Node> fresh;
        std::size_t r = 0;
        for (std::size_t p = 0; p < pairs.size(); p += chunk, ++r) {
            if (r >= sg.size() || r >= bb.size())
                throw std::invalid_argument("E15: capacity plan does not cover every region");
            const Pair* f = pairs.data() + p;
            const Pair* l = pairs.data() + std::min(p + chunk, pairs.size());
            std::vector<Pair> part(pairs.begin() + p, pairs.begin() + std::min(p + chunk, pairs.size()));
            std::vector<std::uint16_t> cap;
            if (!nb_cap.empty()) cap = nb_cap[r];
            // E21: the context is accepted only for the key vector it was built
            // from. `source_ok` is checked once per call; the per-region key digest
            // then confirms the sequence has not been edited or reordered.
            const RegionDemand* preset = nullptr;
            if (source_ok && r < ctx->regions.size()) {
                const RegionDemand& rd = ctx->regions[r];
                if (rd.group_ids.size() == std::size_t(l - f) &&
                    rd.key_digest == Region::key_sequence_digest(f, l))
                    preset = &rd;
            }
            fresh.push_back({part.front().first,
                             std::make_unique<Region>(part, config_, sg[r], bb[r], std::move(cap),
                                                      preset, lits_shared_)});
        }
        if (fresh.empty())
            fresh.push_back({{}, std::make_unique<Region>(std::vector<Pair>{}, config_,
                                                        lits_shared_)});

        nodes_ = std::move(fresh);
        size_ = pairs.size();
        ++structure_version_;
        E14_ADD(bulk_planned, e15_t0);
    }

 public:

    // E22: the capacity plan a construction runs under, as returned by the
    // caller's callback in bulk_load_reusing_ids.
    struct CapacityPlan {
        std::vector<std::vector<std::uint16_t>> sg;
        std::vector<std::vector<std::uint32_t>> bb;
        std::vector<std::vector<std::uint16_t>> caps;
    };

    // -----------------------------------------------------------------------
    // E22 (R2 as an option): statistics, planning and construction in ONE call,
    // with the group ids reused between the first and the third.
    //
    // This is the ONLY documented way to use the reuse. The region context is an
    // ordinary local here and never escapes, so it cannot be stored and later
    // replayed against a different key array. That is the guarantee: a controlled
    // lifetime, not a hash. The per-region checks (source, key digest, prefix,
    // group count, model digest) remain as defence in depth and will refuse an
    // obvious mismatch, but they are not what makes this correct.
    //
    // `stats_cfg` is the config the statistics replay runs under (single-block,
    // which is what the pipeline probes with); this index is built under its own
    // config_. `plan_for` is called once with the per-region statistics and returns
    // the capacity plan to build under -- a callback, so this header does not have
    // to depend on the allocator.
    // -----------------------------------------------------------------------
    template <class PlanFor>
    void bulk_load_reusing_ids(const std::vector<Pair>& pairs, const Config& stats_cfg,
                               PlanFor&& plan_for) {
        // The statistics replay covers the single-block layout, which is what the
        // pipeline probes with; this index is built under its own config_. Two
        // configs, one call -- and `ctx` is an ordinary local either way.
        Index stats(stats_cfg);

        // E38: the STATISTICS index trains the table and this index ADOPTS it, so the demand
        // pass and the placement pass read one object. Training it twice would produce two
        // identical tables today and two different ones the moment anything about the training
        // or the input changes -- which is the failure the task book's section 5.2 names.
        stats.ensure_lits_shared(pairs);
        adopt_lits_shared(stats.lits_shared());

        const BuildContext ctx = stats.demand_profile_ctx(pairs);
        const CapacityPlan plan = plan_for(ctx.regions);
        bulk_load_with_context(pairs, plan.sg, plan.bb, plan.caps, &ctx);
    }

    // -----------------------------------------------------------------------
    // E16: the probe's OUTPUT -- per-region, per-group statistics -- without
    // building the probe.
    //
    // Splits `pairs` exactly as bulk_load does, and hands each chunk to
    // Region::demand_profile as a POINTER RANGE, so unlike `probe.bulk_load(keys)`
    // this materialises no per-region vector<Pair> and allocates no blocks.
    //
    // The pipeline's probe is built with the same Config this Index carries, so
    // `probe.region(i).group_stats()` and `index.demand_profile(keys)[i]` are
    // meant to be the same vector. check_e16_probe asserts that field by field.
    // -----------------------------------------------------------------------
    BuildContext demand_profile_ctx(const std::vector<Pair>& pairs) const {
        BuildContext bc;
        bc.source = pairs.data();
        bc.source_size = pairs.size();
        const auto chunk = config_.region_keys ? config_.region_keys : std::max<std::size_t>(1, pairs.size());
        for (std::size_t p = 0; p < pairs.size(); p += chunk) {
            const Pair* f = pairs.data() + p;
            const Pair* l = pairs.data() + std::min(p + chunk, pairs.size());
            bc.regions.push_back(Region::demand_profile_ctx(f, l, config_, lits_shared_));
        }
        if (bc.regions.empty())
            bc.regions.push_back(Region::demand_profile_ctx(nullptr, nullptr, config_, lits_shared_));
        return bc;
    }

    std::vector<std::vector<GroupStats>> demand_profile(const std::vector<Pair>& pairs) const {
        std::vector<std::vector<GroupStats>> out;
        const auto chunk = config_.region_keys ? config_.region_keys : std::max<std::size_t>(1, pairs.size());
        for (std::size_t p = 0; p < pairs.size(); p += chunk) {
            const Pair* f = pairs.data() + p;
            const Pair* l = pairs.data() + std::min(p + chunk, pairs.size());
            out.push_back(Region::demand_profile(f, l, config_));
        }
        if (out.empty()) out.push_back(Region::demand_profile(nullptr, nullptr, config_));
        return out;
    }


    // The 8-byte order-preserving digest. Big-endian, zero-padded past the end of the key, so
    // for any two keys digest(a) < digest(b) whenever a < b. The converse does NOT hold and
    // is not used: equal digests always fall back to comparing the full keys.
    static std::uint64_t fence_digest(std::string_view s) {
        std::uint64_t out = 0;

        for (std::size_t i = 0; i < 8; ++i) {
            out = (out << 8) | (i < s.size() ? static_cast<unsigned char>(s[i]) : 0);
        }

        return out;
    }

    // Recomputes the digest array from the CURRENT fences. O(regions), and called only when
    // structure_version_ has moved -- so it runs once after a build, a rebuild or a split,
    // and never on a steady-state lookup.
    //
    // const because locate() is const and the routing members are mutable: the digests are
    // derived state, and rebuilding them does not change what the index contains.
    void rebuild_region_routing() const {
        const std::string_view first = nodes_.front().fence;
        const std::string_view last = nodes_.back().fence;

        // The bytes every fence shares are the same for every digest, so they carry no
        // routing information. Skipping them makes the 8 bytes that DO matter land on the
        // part of the key that distinguishes regions.
        std::size_t common = 0;

        while (common < std::min(first.size(), last.size()) && first[common] == last[common]) {
            ++common;
        }

        fence_prefix_len_ = common;
        fence_digests_.resize(nodes_.size());

        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            fence_digests_[i] = fence_digest(std::string_view(nodes_[i].fence).substr(common));
        }
    }

    // The compact search. Same predicate as the original upper_bound -- the first fence
    // strictly greater than the key, minus one -- but the binary search walks fence_digests_
    // instead of stepping over Nodes.
    std::size_t locate_routed(std::string_view key) const {
        const std::size_t n = nodes_.size();

        // Keys outside the shared prefix cannot be inside any region and are answered
        // without a search at all. This is exactly equivalent to the general path: a key
        // below the shared prefix is below the first fence, and one above it is above the
        // last -- both because every fence starts with that prefix.
        if (fence_prefix_len_ != 0) {
            const std::string_view head =
                key.substr(0, std::min(key.size(), fence_prefix_len_));
            const std::string_view shared =
                std::string_view(nodes_.front().fence).substr(0, fence_prefix_len_);
            const int order = compare(head, shared);

            if (order < 0) return 0;
            if (order > 0) return n - 1;
        }

        const std::uint64_t query = fence_digest(key.substr(fence_prefix_len_));
        std::size_t lo = 0;
        std::size_t hi = n;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;

            // Equal digests mean the 8 informative bytes agree, NOT that the keys agree, so
            // they fall back to the full comparison. This is the branch that makes short
            // keys, embedded NULs and prefixes longer than 8 bytes correct.
            const bool before = (query != fence_digests_[mid])
                                    ? (query < fence_digests_[mid])
                                    : (compare(key, nodes_[mid].fence) < 0);

            if (before) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        return lo == 0 ? 0 : lo - 1;
    }

    // The ORIGINAL search, kept as its own function so the gate can run both side by side on
    // the same key. "The two implement the same predicate" is an argument; comparing their
    // answers on every key is evidence, and only the second one belongs in a verdict.
    std::size_t locate_scanned(std::string_view key) const {
#ifdef E25_DIAG
        // P1 read-path diagnosis: the same comparator, counting the boundary probes and the
        // logical bytes they walk. Compiled out of every other build.
        auto it = std::upper_bound(nodes_.begin(), nodes_.end(), key,
            [](std::string_view k, const Node& n) {
                E25_INC(locate_cmp_calls);
                const int c =
                    compare_bytes(k, n.fence, ::validation::e25::counts().locate_cmp_bytes);
                return c < 0;
            });
#else
        auto it = std::upper_bound(nodes_.begin(), nodes_.end(), key,
            [](std::string_view k, const Node& n) { return compare(k, n.fence) < 0; });
#endif
        return it == nodes_.begin() ? 0 : static_cast<std::size_t>(it - nodes_.begin() - 1);
    }

    std::size_t locate(std::string_view key) const {
        E30_W(locate);
        E25_INC(locate_calls);
        if (nodes_.empty()) throw std::logic_error("bulk_load must initialize index");

        if (!config_.compact_region_routing) {
            return locate_scanned(key);
        }

        // Any change to nodes_ bumps structure_version_, so a mismatch means the digests
        // describe a directory that no longer exists. Rebuilding here rather than at the
        // mutation sites is what makes a missed call site impossible.
        if (routing_version_ != structure_version_) {
            rebuild_region_routing();
            routing_version_ = structure_version_;
        }

        return locate_routed(key);
    }

    const Region& region(std::size_t i) const { return *nodes_[i].region; }
    Result find(std::string_view key, Mode m) const { return region(locate(key)).find(key, m); }
    Result find_known(std::size_t r, std::string_view key, Mode m) const { return region(r).find(key, m); }
    std::size_t size() const { return size_; }
    std::size_t region_count() const { return nodes_.size(); }

    // -----------------------------------------------------------------------
    // E38: the LITS coverage of the CURRENT index, walked over its live regions.
    //
    // WHY THIS IS NOT `Region::lits_stats()`. That counter accumulates over every region ever
    // CONSTRUCTED -- splits, prefix rebuilds and erases included -- so it answers "what fraction
    // of the work went through LITS", not "what fraction of the index I am about to time is
    // mapped by LITS". Those are different questions and the second is the one that explains a
    // timed result. The reviewer's point: the coverage quoted beside a measurement must describe
    // the state that was measured.
    //
    // The per-reason split exists for the same reason: "17% fell back" is not actionable until
    // it says whether they fell back because of the numerical admission bound, a degenerate
    // calibration, or a region too small to calibrate.
    // -----------------------------------------------------------------------
    struct LitsCoverage {
        std::uint64_t regions = 0;
        std::uint64_t regions_lits = 0;
        std::uint64_t records = 0;
        std::uint64_t records_lits = 0;
        std::uint64_t reason_admission = 0;    // refused by the conservative coefficient bound
        std::uint64_t reason_degenerate = 0;   // hi <= lo, non-finite coefficients, short key
        std::uint64_t reason_boundary = 0;     // fewer than two keys: no calibration needed
        std::uint64_t reason_unsupported = 0;  // outside the adapter's input domain
    };

    LitsCoverage lits_coverage() const {
        LitsCoverage c;

        // The reference-model backend has nothing to report here, and must not be described as a
        // pile of fallbacks: its regions were never offered to the LITS calibration at all, so
        // `lits_` still holds its default `NoKeys` and a naive reason breakdown would file every
        // one of them under "boundary". The counts are still reported; the reasons are not.
        const bool lits_backend = config_.model_backend == ModelBackend::LitsHpt;

        for (const auto& n : nodes_) {
            const Region& r = *n.region;
            const std::uint64_t sz = r.size();

            ++c.regions;
            c.records += sz;

            if (!lits_backend) continue;

            switch (r.lits_status()) {
                case lits_hpt::Status::Ok:
                case lits_hpt::Status::SingleGroup:
                    // G == 1 is a defined LITS answer (group 0), not a fallback: every model
                    // maps a one-group region to its only group.
                    ++c.regions_lits;
                    c.records_lits += sz;
                    break;

                case lits_hpt::Status::CoefficientTooLarge:
                    ++c.reason_admission;
                    break;

                case lits_hpt::Status::DegenerateCdf:
                case lits_hpt::Status::NonFinite:
                case lits_hpt::Status::KeyShorterThanPrefix:
                    ++c.reason_degenerate;
                    break;

                case lits_hpt::Status::NoKeys:
                case lits_hpt::Status::TooFewKeys:
                    ++c.reason_boundary;
                    break;

                default:
                    ++c.reason_unsupported;
                    break;
            }
        }

        return c;
    }

    // Which region a key routes to. Exposed so a benchmark can report the QUERY share of the
    // fallback -- the one caliber that cannot be derived from the index's own state, because it
    // depends on the query sequence that was run.
    std::size_t which_region(std::string_view key) const { return locate(key); }

    // The routing metadata's own footprint. UNCONDITIONAL, unlike the diag_* helpers below:
    // it is a space account, not a counter, and the timing build has to be able to report
    // what the candidate costs in memory without carrying diagnostic instrumentation.
    // Counted at CAPACITY, because that is what the allocator actually holds.
    std::size_t routing_bytes() const {
        return fence_digests_.capacity() * sizeof(std::uint64_t);
    }

    std::size_t routing_regions() const { return fence_digests_.size(); }

#ifdef E25_DIAG
    // Compact-routing gate helpers. Both searches are reachable regardless of the switch, so
    // the gate can compare them key by key on the SAME index state instead of running two
    // builds and hoping they agree.
    std::size_t diag_locate_scanned(std::string_view key) const { return locate_scanned(key); }

    std::size_t diag_locate_routed(std::string_view key) const {
        if (routing_version_ != structure_version_) {
            rebuild_region_routing();
            routing_version_ = structure_version_;
        }

        return locate_routed(key);
    }
#endif

    // E8-9: memory decomposition across regions plus index-level metadata.
    struct MemBreakdown {
        std::size_t blocks = 0, model = 0, prefix = 0, overflow = 0, tables = 0,
                    directory = 0, regions_self = 0, fences = 0, self = 0;
    };

    MemBreakdown memory_breakdown() const {
        MemBreakdown m;
        for (const auto& x : nodes_) {
            const auto b = x.region->memory_breakdown();
            m.blocks += b.blocks;
            m.model += b.model;
            m.prefix += b.prefix;
            m.overflow += b.overflow;
            m.tables += b.tables;
            m.directory += b.directory;
            m.regions_self += b.self;
            m.fences += x.fence.capacity() + 1;
        }

        // The compact-routing digests are index-owned metadata, so they belong in this
        // account. Reporting them ONLY in a side column would let a reader who looks at the
        // size decomposition conclude the candidate costs nothing.
        m.tables += fence_digests_.capacity() * sizeof(std::uint64_t);

        m.self = sizeof(*this) + nodes_.capacity() * sizeof(Node);
        return m;
    }

#ifdef E25_DIAG
    // E26 long-run diagnosis: the logical group a key maps to, and the region it
    // routes to. Wraps the Region accessor; diagnostic only.
    std::pair<std::size_t, std::uint32_t> diag_where(std::string_view key) const {
        const std::size_t r = locate(key);
        return {r, nodes_[r].region->diag_group(key)};
    }
    // Read-only F0/F1/F2 trial for a key, on the CURRENT state. Does not mutate.
#ifdef E30_TIME
    // E30: probe the rank-selected candidate sub-block of `key`, on a scratch copy.
    void e30_probe(std::string_view key, std::uint64_t& copy_ns, std::uint64_t& pass_ns,
                   std::size_t& records, std::uint64_t& moved_bytes, bool& ok) const {
        const std::size_t r = locate(key);
        const Region& reg = *nodes_[r].region;
        const std::uint32_t g = reg.diag_group(key);
        if (g == UINT32_MAX) { copy_ns = pass_ns = 0; records = 0; moved_bytes = 0; ok = true; return; }
        const auto tp = reg.trial_place(g, key.substr(reg.diag_prefix_size()));
        reg.e30_repack_probe(g, tp.cand_subblock, copy_ns, pass_ns, records, moved_bytes, ok);
    }
#endif

    // E31 item 3: every live overflow container's record count, over all regions, plus
    // the element size the movement counters are expressed in. Diagnostic only.
    void diag_overflow_sizes(std::vector<std::size_t>& out) const {
        for (const auto& x : nodes_) x.region->diag_overflow_sizes(out);
    }

    // Branch C: forward every region's container stats.
    void diag_overflow_caps(std::vector<Region::OvCapStat>& out) const {
        for (const auto& x : nodes_) x.region->diag_overflow_caps(out);
    }

#ifdef E25_DIAG
    // Hole-reuse: forward every region's recorded byte-failure events. Each region only
    // ever receives inserts for its own key range, so concatenating them does not mix
    // populations -- the grouping that matters (which sub-block) is preserved on the row.
    void diag_hole_events(std::vector<Region::HoleEvent>& out) const {
        for (const auto& x : nodes_) {
            const auto& ev = x.region->hole_events();
            out.insert(out.end(), ev.begin(), ev.end());
        }
    }

    void diag_clear_hole_events() {
        for (const auto& x : nodes_) x.region->clear_hole_events();
    }
#endif

    void diag_overflow_suffix_lengths(std::vector<std::size_t>& out) const {
        for (const auto& x : nodes_) x.region->diag_overflow_suffix_lengths(out);
    }

#ifdef E25_DIAG
    std::size_t diag_tag_pairing(std::size_t& checked) const {
        std::size_t bad = 0;
        for (const auto& x : nodes_) bad += x.region->diag_tag_pairing(checked);
        return bad;
    }
#endif

#ifdef E25_DIAG
    // E37: sum the migration scan over every region.
    void diag_migration_scan(Region::MigrationScan& out) const {
        for (const auto& x : nodes_) x.region->diag_migration_scan(out);
    }
#endif

    // E37: drain every region once. Used by the repair experiment, where the workload has
    // already run to a degraded state and the question is whether the layout can be
    // recovered at an acceptable cost.
    std::size_t drain_all_bounded() {
        std::size_t total = 0;

        for (const auto& x : nodes_) total += x.region->drain_all_bounded();

        return total;
    }
    std::size_t diag_overflow_record_bytes() const { return Region::diag_overflow_record_bytes(); }

    // A2 entry diagnosis: forward to the region that owns the key. The caller must have
    // just observed that this record landed in overflow -- see the note in Region::diag_a2.
    Region::A2Audit diag_a2(std::string_view key) const {
        const std::size_t i = locate(key);
        return nodes_[i].region->diag_a2(key);
    }

    Region::TrialPlace diag_trial(std::string_view key) const {
        const std::size_t r = locate(key);
        const Region& reg = *nodes_[r].region;
        const std::uint32_t g = reg.diag_group(key);
        if (g == UINT32_MAX) return {};
        return reg.trial_place(g, key.substr(reg.diag_prefix_size()));
    }
#endif

#ifdef E24_DIAG
    // E24: every group of every region, for the space decomposition and the
    // reclaim analysis. Diagnostic only.
    std::vector<e24::GroupReport> e24_report() const {
        std::vector<e24::GroupReport> out;
        for (const auto& x : nodes_) x.region->e24_report(out);
        return out;
    }
#endif

    std::size_t memory() const {
        std::size_t n = sizeof(*this) + nodes_.capacity() * sizeof(Node);
        for (const auto& x : nodes_) n += x.fence.capacity() + 1 + x.region->memory();
        return n;
    }

    std::size_t overflow_count() const {
        std::size_t n = 0;
        for (const auto& x : nodes_) n += x.region->overflow_count();
        return n;
    }

    std::size_t compaction_count() const {
        std::size_t n = 0;
        for (const auto& x : nodes_) n += x.region->compaction_count();
        return n;
    }

    // E8: replace a region with an explicit per-group capacity table (no split;
    // caller guarantees data fits the region size). Caps only change capacity,
    // never semantics.
    void rebuild_with_caps(std::size_t i, std::vector<Pair> data,
                           std::vector<std::uint16_t> sg, std::vector<std::uint32_t> bb,
                           std::vector<std::uint16_t> nb_cap = {}) {
        E30_W(rebuild);
        const auto e14_t0 = E14_T0();
        E14_INC(rebuild_calls);
        // E38: a REBUILT region references the SAME shared table. A region that trained its
        // own would map the same key to a different group than the one it was placed under,
        // and a rebuild that changed the region's prefix or group count has already re-placed
        // every survivor below.
        nodes_[i].region = std::make_unique<Region>(data, config_, std::move(sg), std::move(bb),
                                                    std::move(nb_cap), nullptr, lits_shared_);
        ++rebuilds;
        ++structure_version_;
        E14_ADD(rebuild, e14_t0);
    }

    void rebuild(std::size_t i, std::vector<Pair> data) {
        E30_W(rebuild);
        const auto e14_t0 = E14_T0();
        E14_INC(rebuild_calls);
        E25_INC(region_rebuilds);
        if (config_.region_keys && data.size() > 2 * config_.region_keys) {
            const auto mid = data.size() / 2;
            std::vector<Pair> left(data.begin(), data.begin() + mid), right(data.begin() + mid, data.end());
            // A SPLIT calibres each half SEPARATELY -- a new prefix and a new G -- while both
            // keep reading the one global table (task book section 8).
            auto a = std::make_unique<Region>(left, config_, lits_shared_);
            auto b = std::make_unique<Region>(right, config_, lits_shared_);
            const auto fence = right.front().first;
            nodes_[i].region = std::move(a);
            nodes_.insert(nodes_.begin() + i + 1, Node{fence, std::move(b)});
            ++splits;
        } else {
            auto fresh = std::make_unique<Region>(data, config_, lits_shared_);
            nodes_[i].region = std::move(fresh);
        }
        ++rebuilds;
        ++structure_version_;
        E14_ADD(rebuild, e14_t0);
    }

    bool insert(std::string_view key, std::uint64_t value) {
        E30_W(insert);
#ifdef E30_WRITE
        const auto w_l0 = E30_T0();
#endif
        const auto i = locate(key);
#ifdef E30_WRITE
        E30_ADD(w_locate_ins_ns, w_l0);
        E30_INC(w_locate_ins_calls);
#endif
        auto& r = *nodes_[i].region;
        if (!r.accepts_prefix(key)) {
            E25_INC(prefix_rebuilds);
#ifdef E30_WRITE
            const auto w_a0 = E30_T0();
#endif
            auto data = r.all();
#ifdef E30_WRITE
            E30_ADD(w_rebuild_all_ns, w_a0);
            E30_INC(w_rebuild_all_calls);
            ::validation::e30::timers().w_rebuild_all_records += data.size();
#endif
            auto it = std::lower_bound(data.begin(), data.end(), key,
                [](const Pair& a, std::string_view b) { return compare(a.first, b) < 0; });
            if (it != data.end() && compare(it->first, key) == 0) return false;
            data.insert(it, {std::string(key), value});
            rebuild(i, std::move(data));
            ++size_;
            return true;
        }
        if (!r.insert(key, value)) return false;
        ++size_;
        if (config_.region_keys && r.size() > 2 * config_.region_keys) {
            E25_INC(threshold_splits);
#ifdef E30_WRITE
            const auto w_a0 = E30_T0();
#endif
            auto w_all = r.all();
#ifdef E30_WRITE
            E30_ADD(w_rebuild_all_ns, w_a0);
            E30_INC(w_rebuild_all_calls);
            ::validation::e30::timers().w_rebuild_all_records += w_all.size();
#endif
            rebuild(i, std::move(w_all));
        }
        return true;
    }

    bool update(std::string_view key, std::uint64_t v) { return nodes_[locate(key)].region->update(key, v); }

    bool erase(std::string_view key) {
        E30_W(erase);
        // The locate is clocked ALONE. Bracketing the whole statement below would have
        // charged the region erase to it as well -- the first version of this did exactly
        // that, and locate read as 91% of Index::erase.
#ifdef E30_WRITE
        const auto w_l0 = E30_T0();
#endif
        const std::size_t w_r = locate(key);
#ifdef E30_WRITE
        E30_ADD(w_locate_era_ns, w_l0);
        E30_INC(w_locate_era_calls);
#endif
        bool ok = nodes_[w_r].region->erase(key);
        if (ok) --size_;
        return ok;
    }

    bool upsert(std::string_view key, std::uint64_t value) {
        // E25 V2: one locate decides write-or-place. V0/V1 keep the two-step form.
        if (config_.reuse_locate_ctx) {
            const auto i = locate(key);
            auto& r = *nodes_[i].region;
            // The region cannot take this key at all: no context to reuse, and the
            // existing path carries the rebuild that makes it acceptable.
            if (!r.accepts_prefix(key)) {
                if (update(key, value)) return false;
                return insert(key, value);
            }
            if (r.upsert_one(key, value)) return false;   // an existing record was written
            ++size_;
            // The region-level split Index::insert performs. Omitting it here would
            // make V2 a DIFFERENT structure rather than a faster writer.
            if (config_.region_keys && r.size() > 2 * config_.region_keys) rebuild(i, r.all());
            return true;
        }
        if (update(key, value)) return false;
        return insert(key, value);
    }

    void rebuild_all() {
        for (std::size_t i = 0; i < nodes_.size(); ++i) rebuild(i, nodes_[i].region->all());
    }

    std::vector<Pair> scan(std::string_view start, std::size_t count) const {
        std::vector<Pair> out;
        out.reserve(std::min(count, size_));
        // E27 S2 (checklist 7.1): locate(start) was re-evaluated on every iteration of
        // this loop and once more in the body -- across R regions that is R+1 outer
        // searches for one answer. The switch caches the first evaluation and uses it
        // for both the loop bound and the "is this the starting region" test, so the
        // scan pays one. scan_start_locates is incremented on EVERY evaluation in BOTH
        // arms, so the claim is measured: S0 counts R+1 per scan, S2 exactly 1.
        if (config_.scan_cache_start) {
            const std::size_t first = locate(start);
            E25_INC(scan_start_locates);
            for (std::size_t i = first; i < nodes_.size() && out.size() < count; ++i) {
                E25_INC(scan_regions);
                nodes_[i].region->scan(i == first ? start : std::string_view{}, count, out);
            }
            return out;
        }
        // Pre-E27 path. The two locate(start) expressions are exactly as they were;
        // only the no-op counter statement is new.
        std::size_t i = locate(start);
        E25_INC(scan_start_locates);
        for (; i < nodes_.size() && out.size() < count; ++i) {
            E25_INC(scan_regions);
            E25_INC(scan_start_locates);
            nodes_[i].region->scan(i == locate(start) ? start : std::string_view{}, count, out);
        }
        return out;
    }

#ifdef E25_DIAG
    // M4 entry diagnosis: the shadow mirror of scan(), same region loop and budget as the
    // shipped cached-start path the entry device runs.
    MergeShadowCounts diag_scan_merge_shadow(std::string_view start, std::size_t count,
                                             std::vector<Pair>& out) const {
        MergeShadowCounts m;
        const std::size_t first = locate(start);
        for (std::size_t i = first; i < nodes_.size() && out.size() < count; ++i) {
            nodes_[i].region->diag_scan_merge_shadow(i == first ? start : std::string_view{},
                                                     count, out, m);
        }
        return m;
    }
#endif

#ifdef E25_SCAN_CONSUME
    // The consume-mode mirror of scan(), region for region. The full limit is passed down
    // and each region call sees the remaining budget, exactly as the materialised path
    // does through out.size() -- the walk, the locate() count and the region arithmetic
    // are the same; only the row production differs (see Region::scan_consume).
    void scan_consume(std::string_view start, std::size_t count, Region::ConsumeSink& sink) const {
        std::size_t i = locate(start);
        // The region receives the GLOBAL target, not the remaining budget: the region
        // loop compares the sink's running total against it, exactly as Region::scan
        // compares out.size() against the shared vector's. Passing count - sink.rows
        // here was wrong across a region boundary (the earlier regions' rows are
        // already in sink.rows, so a remaining budget double-counts them and the walk
        // stops short) -- found by this round's verify pass on tranco/limit=4096,
        // where one start crosses regions. Same bug shape as the smoke check caught.
        for (; i < nodes_.size() && sink.rows < count; ++i) {
            nodes_[i].region->scan_consume(i == locate(start) ? start : std::string_view{}, count,
                                           sink);
        }
    }
#endif
};

// Test-only. Do not call from library code or from a benchmark: a context is
// only sound within the single call that produced it.
struct IndexTestAccess {
    static void build_with_context(Index& idx, const std::vector<Pair>& pairs,
                                   const std::vector<std::vector<std::uint16_t>>& sg,
                                   const std::vector<std::vector<std::uint32_t>>& bb,
                                   const std::vector<std::vector<std::uint16_t>>& caps,
                                   const BuildContext* ctx) {
        idx.bulk_load_with_context(pairs, sg, bb, caps, ctx);
    }
};


}  // namespace validation