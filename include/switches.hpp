#pragma once
// E25/E26/E27 switch discipline.
//
// The Config defaults used to BE the experimental baseline: every gate built a
// Config from the defaults and then flipped only the switches it was studying. That
// works until a default changes -- and at freeze time one did. A gate written that
// way would then silently compare a different baseline than the round it belongs to,
// which is exactly the kind of quiet redefinition the project's review discipline
// forbids.
//
// So every gate now builds its baseline from `baseline_config()`, which sets ALL of
// the switches this project added, explicitly. Flipping a Config default can no
// longer move a gate underneath itself.
//
// It also prints the EFFECTIVE switch values into the gate's output, and asserts the
// arms differ in exactly the switches the gate claims to be studying -- so a gate
// that accidentally compares two identical configurations fails instead of passing.
//
// Include AFTER index.hpp.

#include <initializer_list>
#include <stdexcept>
#include <string>

namespace e25sw {

// The pre-E25 configuration: everything this project added is OFF. Field-by-field
// explicit; nothing here reads a Config default.
inline validation::Config baseline_config(std::uint32_t ov_threshold) {
    validation::Config c;
    c.block_bytes = 1024;
    c.slots = 16;
    c.model_kind = validation::ModelKind::Context16;
    // E38: the model FAMILY. Explicit here even though it is the Config default, because
    // this table exists precisely so that flipping a default cannot move a baseline.
    c.model_backend = validation::ModelBackend::Context16;
    c.multiblock = true;
    c.reuse_group_ids = true;      // the frozen R1 default, unchanged since E20
    c.ov_threshold = ov_threshold;
    // ---- every switch added since E24, all explicit ----
    c.write_fast_locate = false;      // E25 V1
    c.reuse_locate_ctx = false;       // E25 V2
    c.compact_reuse_tag = false;      // E26 C1
    c.compact_needs_garbage = false;  // E26 C2
    c.scan_reserve_result = false;    // E27 S1
    c.subblock_reclaim = false;       // E31 (default off; the baseline is pre-E31)
    c.reclaim_widest = false;         // G3/A1' (default off; the E31 trigger surface)
    c.reclaim_hysteresis = false;     // E31H (default off; E31 without hysteresis)
    c.reclaim_skip_repeat = false;    // E31P (default off; the state check alone)
    c.scan_cache_start = false;       // E27 S2 (default off; the pre-E27 scan candidate)
    c.scan_ordered_start = false;     // scan-entry round S (default off; the ordered advance)
    c.overflow_stable_store = false;  // O1 (default off; the ordered overflow vector)
    c.overflow_inline_tag = false;    // E36 (default off; needs the stable store on)
    c.overflow_migrate = false;       // E37 (default off; drains overflow on pressure)
    c.compact_plan_commit = true;     // I10 (default ON: a fix, not a candidate)
    c.compact_region_routing = false;   // structural candidate, not in the frozen set
    c.reclaim_low_activity = false;     // structural candidate, not in the frozen set
    c.primary_hole_reuse = false;       // structural candidate, not in the frozen set
    c.lits_skip_reference_table = false;  // M3 space candidate (2026-10-11), not in the frozen set
    // S0 audit: these three were in neither the white list nor the equality check, so an
    // arm pair differing only in one of them asserted nothing. They are named here for the
    // same reason everything else is: a baseline may not inherit a switch it does not state.
    c.offset_directory = true;          // Config default, stated explicitly
    c.write_simd = true;                // Config default, stated explicitly
    c.write_locate = false;             // Config default (E11/E13: a selectable component, not a default)
                                      //   baseline_config sets it to the SHIPPED value
                                      //   for the same reason reuse_group_ids is set here:
                                      //   it is not an experimental arm's variable.
    return c;
}

// The FROZEN combination (2026-09-20): every switch on. Gates that were written
// before the freeze and simply built a Config from defaults now call this, so their
// experimental condition is stated in their own source instead of being inherited
// from a header that may change again.
//
// These gates test the SHIPPED configuration. That is a deliberate choice: an
// invariant that only holds on the old default is not an invariant worth keeping.
inline void pin_frozen_switches(validation::Config& c) {
    c.write_fast_locate = true;
    c.reuse_locate_ctx = true;
    c.compact_reuse_tag = true;
    c.compact_needs_garbage = true;
    c.scan_reserve_result = true;
    c.subblock_reclaim = false;   // NOT part of the frozen set; E31 is a candidate
    c.reclaim_widest = false;     // NOT part of the frozen set; G3/A1' is a candidate
    c.reclaim_hysteresis = false;   // NOT part of the frozen set either
    c.reclaim_skip_repeat = false;   // NOT part of the frozen set either
    c.scan_cache_start = false;   // NOT part of the frozen set; E27 S2 is a candidate
    c.scan_ordered_start = false; // NOT part of the frozen set; the S candidate
    c.overflow_stable_store = false;   // NOT part of the frozen set; O1 is a candidate
    c.overflow_inline_tag = false;     // NOT part of the frozen set; E36 is a candidate
    c.overflow_migrate = false;        // NOT part of the frozen set; E37 is a candidate
    c.compact_plan_commit = true;      // part of the shipped configuration as of 2026-09-23
    c.compact_region_routing = false;  // NOT part of the frozen set; the routing candidate
    c.reclaim_low_activity = false;    // NOT part of the frozen set; the n<2 reclaim candidate
    c.primary_hole_reuse = false;      // NOT part of the frozen set; candidate H (2026-10-11)
    c.lits_skip_reference_table = false;  // NOT part of the frozen set; candidate M3 (2026-10-11)
    c.offset_directory = true;         // shipped value
    c.write_simd = true;               // shipped value
    c.write_locate = false;            // shipped value (the write path searches without the directory)
    c.model_backend = validation::ModelBackend::Context16;   // NOT part of the frozen set
}

// The backend name, for gate output. A three-valued field printed as a two-valued one turns
// BlockLinearRank into "LitsHpt", and a reader comparing two arms' printed lines would then
// conclude the arms differ in nothing. The switch to the third backend is what made this a
// real defect rather than a latent one.
inline const char* backend_name(validation::ModelBackend b) {
    switch (b) {
        case validation::ModelBackend::Context16: return "Context16";
        case validation::ModelBackend::LitsHpt: return "LitsHpt";
        case validation::ModelBackend::BlockLinearRank: return "BlockLinearRank";
        case validation::ModelBackend::EqualDepthDir: return "EqualDepthDir";
        case validation::ModelBackend::EqualDepthDirTrunc: return "EqualDepthDirTrunc";
    }
    return "?";
}

// The effective switch values, for the gate's own output. A reader should never have
// to infer which configuration produced a line.
inline std::string switches(const validation::Config& c) {
    return std::string("write_fast_locate=") + (c.write_fast_locate ? "1" : "0") +
           ",reuse_locate_ctx=" + (c.reuse_locate_ctx ? "1" : "0") +
           ",compact_reuse_tag=" + (c.compact_reuse_tag ? "1" : "0") +
           ",compact_needs_garbage=" + (c.compact_needs_garbage ? "1" : "0") +
           ",compact_repack_budget=" + (c.compact_repack_budget ? "1" : "0") +
           ",subblock_reclaim=" + (c.subblock_reclaim ? "1" : "0") +
           ",reclaim_widest=" + (c.reclaim_widest ? "1" : "0") +
           ",reclaim_hysteresis=" + (c.reclaim_hysteresis ? "1" : "0") +
           ",reclaim_skip_repeat=" + (c.reclaim_skip_repeat ? "1" : "0") +
           ",scan_cache_start=" + (c.scan_cache_start ? "1" : "0") +
           ",scan_ordered_start=" + (c.scan_ordered_start ? "1" : "0") +
           ",scan_reserve_result=" + (c.scan_reserve_result ? "1" : "0") +
           ",overflow_stable_store=" + (c.overflow_stable_store ? "1" : "0") +
           ",overflow_inline_tag=" + (c.overflow_inline_tag ? "1" : "0") +
           ",overflow_migrate=" + (c.overflow_migrate ? "1" : "0") +
           ",compact_plan_commit=" + (c.compact_plan_commit ? "1" : "0") +
           ",compact_region_routing=" + (c.compact_region_routing ? "1" : "0") +
           ",reclaim_low_activity=" + (c.reclaim_low_activity ? "1" : "0") +
           ",primary_hole_reuse=" + (c.primary_hole_reuse ? "1" : "0") +
           ",lits_skip_reference_table=" + (c.lits_skip_reference_table ? "1" : "0") +
           ",offset_directory=" + (c.offset_directory ? "1" : "0") +
           ",write_simd=" + (c.write_simd ? "1" : "0") +
           ",write_locate=" + (c.write_locate ? "1" : "0") +
           ",model_backend=" + backend_name(c.model_backend);
}

// Fail unless the two configurations differ in EXACTLY the named switches, and agree
// everywhere else.
//
// THE FIELD TABLE BELOW IS A HAND-MAINTAINED WHITELIST. Every new Config switch MUST
// be added to it AND to switches(). This table fell behind once -- compact_repack_budget
// was missing, so the C3 arm-equality assertion in check_c3/bench_long passed
// vacuously and verified nothing. A switch that is absent here cannot be checked, and
// the gate that names it will silently succeed. Without this, an arm that silently became a copy of another would
// make a gate report "passed" for a comparison that never happened.
inline void require_only_switches_differ(const validation::Config& a,
                                         const validation::Config& b,
                                         std::initializer_list<const char*> which) {
    // THE TABLE HOLDS "DIFFERS", NOT A BOOL ARM VALUE. The previous shape stored each
    // configuration's value and compared them, which works only for a two-valued switch: with
    // `model_backend` a three-valued enum, encoding it as "== LitsHpt" made Context16 vs
    // BlockLinearRank look like NO difference, so a gate naming that field would have thrown
    // "should not differ" while the arms did differ. A table that cannot express a difference
    // cannot check one.
    struct Field { const char* name; bool differs; };
    const Field fields[] = {
        {"write_fast_locate", a.write_fast_locate != b.write_fast_locate},
        {"reuse_locate_ctx", a.reuse_locate_ctx != b.reuse_locate_ctx},
        {"compact_reuse_tag", a.compact_reuse_tag != b.compact_reuse_tag},
        {"compact_needs_garbage", a.compact_needs_garbage != b.compact_needs_garbage},
        {"compact_repack_budget", a.compact_repack_budget != b.compact_repack_budget},
        {"subblock_reclaim", a.subblock_reclaim != b.subblock_reclaim},
        {"reclaim_widest", a.reclaim_widest != b.reclaim_widest},
        {"reclaim_hysteresis", a.reclaim_hysteresis != b.reclaim_hysteresis},
        {"reclaim_skip_repeat", a.reclaim_skip_repeat != b.reclaim_skip_repeat},
        {"scan_cache_start", a.scan_cache_start != b.scan_cache_start},
        {"scan_ordered_start", a.scan_ordered_start != b.scan_ordered_start},
        {"scan_reserve_result", a.scan_reserve_result != b.scan_reserve_result},
        {"overflow_stable_store", a.overflow_stable_store != b.overflow_stable_store},
        {"overflow_inline_tag", a.overflow_inline_tag != b.overflow_inline_tag},
        {"overflow_migrate", a.overflow_migrate != b.overflow_migrate},
        {"compact_plan_commit", a.compact_plan_commit != b.compact_plan_commit},
        {"compact_region_routing", a.compact_region_routing != b.compact_region_routing},
        {"reclaim_low_activity", a.reclaim_low_activity != b.reclaim_low_activity},
        {"primary_hole_reuse", a.primary_hole_reuse != b.primary_hole_reuse},
        {"lits_skip_reference_table", a.lits_skip_reference_table != b.lits_skip_reference_table},
        // S0 audit: these three were missing from this table, which made an arm pair that
        // differed only in one of them pass vacuously (the exact failure the table exists to
        // catch, and the one G0 found once already in a different form).
        {"offset_directory", a.offset_directory != b.offset_directory},
        {"write_simd", a.write_simd != b.write_simd},
        {"write_locate", a.write_locate != b.write_locate},
        // Valued by the ENUM, not by a two-way test: this row was "== LitsHpt" until the
        // BlockLinearRank round, which made Context16 vs BlockLinearRank look like NO
        // difference. Comparing the enum keeps that fixed for every value the enum gains --
        // EqualDepthDir needed no change here for exactly that reason.
        {"model_backend", a.model_backend != b.model_backend},
    };
    for (const auto& f : fields) {
        bool expected = false;
        for (const char* w : which) if (std::string(w) == f.name) expected = true;
        if (f.differs != expected) {
            throw std::runtime_error(std::string("arm configuration is wrong: switch '") +
                                     f.name + "' should " + (expected ? "" : "not ") +
                                     "differ between the two arms");
        }
    }
    // The non-switch fields must agree, or the comparison is not controlled.
    if (a.block_bytes != b.block_bytes || a.slots != b.slots ||
        a.model_kind != b.model_kind || a.multiblock != b.multiblock ||
        a.reuse_group_ids != b.reuse_group_ids || a.ov_threshold != b.ov_threshold) {
        throw std::runtime_error("arm configuration is wrong: a non-switch field differs");
    }
}

}  // namespace e25sw
