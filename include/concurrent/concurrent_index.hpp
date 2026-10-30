#pragma once

// Concurrent multi-reader / multi-writer wrapper for validation::Index
// (Context16). The base index itself is single-threaded: its Region owns raw
// block storage, per-group overflow containers and a boundary directory, and
// insert/erase/compact mutate them in place. Making that safe at node level
// would mean rebuilding Region around a version word and an epoch reclaimer.
//
// This wrapper takes the coarser publication unit that the index already has:
// one immutable validation::Index per shard. A write builds a new shard
// snapshot, publishes it with release/acquire, and the old snapshot is freed
// when the last reader releases its shared_ptr. That is the RCU/COW analogue
// of ART-OLC's contract:
//
//   * readers never write shared state and never take a writer lock;
//   * a writer publishes a new version instead of mutating a visible one;
//   * old versions are reclaimed only after readers can no longer see them.
//
// Concurrency model:
//   * many concurrent readers: find / scan / all / size / memory;
//   * many writers: insert / update / erase, but writers to the SAME shard
//     serialize on that shard's mutex; writers to different shards run in
//     parallel;
//   * scan is cross-shard and uses a multi-writer epoch: writers mark
//     themselves active, publish, bump the generation, then clear active. A
//     scan retries if it overlapped a writer or the generation changed, so it
//     never returns a mix of two generations.
//
// Sharding is by the first key byte (257 shards: 0 = empty key, 1..256 =
// first byte 0x00..0xff). This preserves the unsigned-byte global order:
// shard 0 < shard 1 < ... < shard 256.
//
// The base index must be built with compact_region_routing = false: that
// candidate lazily rebuilds a mutable routing cache inside find(), which would
// be a data race under concurrent readers. The constructor enforces it.
//
// bulk_load() must complete before any reader/writer starts; it is the only
// operation that is not concurrent with the others.

#include "../index.hpp"
#include "../switches.hpp"
#include "compat.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace concurrent_index {

using Value = std::uint64_t;
using Pair = validation::Pair;

// Same unsigned-byte ordering as validation::compare(); transparent so
// std::map can find(std::string_view) without constructing a std::string.
struct UnsignedByteLess {
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return validation::compare(a, b) < 0;
    }
};

// RAII write marker. The generation is bumped after the new snapshot has been
// published and before active_writers_ is cleared, which is the order scan()
// relies on to distinguish "writer still active" from "writer finished".
class WriteScope {
public:
    WriteScope(std::atomic<std::uint32_t>& active,
               std::atomic<std::uint64_t>& generation)
        : active_(active), generation_(generation) {
        active_.fetch_add(1, std::memory_order_acq_rel);
    }

    ~WriteScope() {
        if (commit_) {
            generation_.fetch_add(1, std::memory_order_release);
        }
        active_.fetch_sub(1, std::memory_order_release);
    }

    void commit() { commit_ = true; }

    WriteScope(const WriteScope&) = delete;
    WriteScope& operator=(const WriteScope&) = delete;

private:
    std::atomic<std::uint32_t>& active_;
    std::atomic<std::uint64_t>& generation_;
    bool commit_ = false;
};

class ConcurrentIndex {
public:
    static constexpr std::size_t kShards = 257;

    ConcurrentIndex() : ConcurrentIndex(default_config()) {}

    explicit ConcurrentIndex(validation::Config config)
        : config_(std::move(config)) {
        if (config_.compact_region_routing) {
            throw std::invalid_argument(
                "ConcurrentIndex requires compact_region_routing=false: "
                "find() would lazily mutate the routing cache");
        }
        if (config_.model_backend != validation::ModelBackend::Context16) {
            throw std::invalid_argument(
                "ConcurrentIndex currently supports the Context16 backend only");
        }
        if (config_.canonical == validation::Route::Residual) {
            throw std::invalid_argument(
                "ConcurrentIndex uses Route::Interval queries and cannot wrap a "
                "Residual-built index");
        }
        mode_ = default_mode(config_.ov_threshold);
    }

    ConcurrentIndex(const ConcurrentIndex&) = delete;
    ConcurrentIndex& operator=(const ConcurrentIndex&) = delete;

    // Exactly once, before readers/writers start. Builds all shards and
    // publishes them; the input is copied into the shards' ordered maps.
    void bulk_load(const std::vector<Pair>& rows) {
        LockGuard<Mutex> load_lock(load_mu_);
        if (loaded_.load(std::memory_order_acquire)) {
            throw std::logic_error("bulk_load may be called only once");
        }

        for (auto& shard : shards_) {
            shard.data.clear();
            std::atomic_store_explicit(
                &shard.snapshot,
                std::shared_ptr<const validation::Index>{},
                std::memory_order_relaxed);
        }

        for (const auto& row : rows) {
            auto inserted =
                shards_[shard_of(row.first)].data.emplace(row.first, row.second);
            if (!inserted.second) {
                throw std::invalid_argument(
                    "bulk_load input contains a duplicate key");
            }
        }

        for (auto& shard : shards_) {
            if (!shard.data.empty()) {
                auto snapshot = build_snapshot(shard.data);
                std::atomic_store_explicit(&shard.snapshot, std::move(snapshot),
                                           std::memory_order_release);
            }
        }

        size_.store(rows.size(), std::memory_order_release);
        loaded_.store(true, std::memory_order_release);
    }

    // ---- concurrent reads -------------------------------------------------

    std::optional<Value> find(std::string_view key) const {
        if (!loaded_.load(std::memory_order_acquire)) return std::nullopt;
        const Shard& shard = shards_[shard_of(key)];
        auto snapshot = std::atomic_load_explicit(&shard.snapshot,
                                                  std::memory_order_acquire);
        if (!snapshot) return std::nullopt;
        const validation::Result result = snapshot->find(key, mode_);
        if (!result.found) return std::nullopt;
        return result.value;
    }

    // Cross-shard scan with a consistent-generation check. The result is a
    // snapshot: either the scan did not overlap a publication, or it retries.
    std::vector<Pair> scan(std::string_view lower_bound,
                           std::size_t limit) const {
        if (limit == 0 || !loaded_.load(std::memory_order_acquire)) return {};

        for (;;) {
            if (active_writers_.load(std::memory_order_acquire) != 0) {
                yield();
                continue;
            }
            const std::uint64_t generation_before =
                generation_.load(std::memory_order_acquire);

            std::vector<Pair> out;
            bool first = true;
            for (std::size_t i = shard_of(lower_bound);
                 i < kShards && out.size() < limit; ++i) {
                auto snapshot = std::atomic_load_explicit(
                    &shards_[i].snapshot, std::memory_order_acquire);
                if (snapshot) {
                    const std::string_view from =
                        first ? lower_bound : std::string_view{};
                    auto part = snapshot->scan(from, limit - out.size());
                    for (auto& row : part) {
                        out.push_back(std::move(row));
                    }
                }
                first = false;
            }

            const std::uint64_t generation_after =
                generation_.load(std::memory_order_acquire);
            if (generation_before == generation_after &&
                active_writers_.load(std::memory_order_acquire) == 0) {
                return out;
            }
        }
    }

    std::vector<Pair> all() const {
        return scan({}, static_cast<std::size_t>(-1));
    }

    std::size_t size() const {
        return size_.load(std::memory_order_acquire);
    }

    // Approximate heap cost of the shard maps (the key/value copy that the
    // snapshot also stores). The map nodes are not part of snapshot->memory(),
    // so a memory report that wants the whole wrapper should add this.
    std::size_t overhead_memory() const {
        if (!loaded_.load(std::memory_order_acquire)) return 0;
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            LockGuard<Mutex> lock(shard.mu);
            total += shard.data.size() *
                     (sizeof(std::pair<const std::string, Value>) +
                      3 * sizeof(void*));
            for (const auto& entry : shard.data) {
                total += entry.first.capacity();
            }
        }
        return total;
    }

    // Sum of the shard snapshots' own accounting. Consistent with respect to
    // the same generation rule as scan(); retries if a write overlaps.
    std::size_t memory() const {
        if (!loaded_.load(std::memory_order_acquire)) return 0;
        for (;;) {
            if (active_writers_.load(std::memory_order_acquire) != 0) {
                yield();
                continue;
            }
            const std::uint64_t generation_before =
                generation_.load(std::memory_order_acquire);

            std::size_t total = 0;
            for (const auto& shard : shards_) {
                auto snapshot = std::atomic_load_explicit(
                    &shard.snapshot, std::memory_order_acquire);
                if (snapshot) total += snapshot->memory();
            }

            const std::uint64_t generation_after =
                generation_.load(std::memory_order_acquire);
            if (generation_before == generation_after &&
                active_writers_.load(std::memory_order_acquire) == 0) {
                return total;
            }
        }
    }

    // ---- concurrent writes ------------------------------------------------

    bool insert(std::string_view key, Value value) {
        require_loaded();
        Shard& shard = shards_[shard_of(key)];
        LockGuard<Mutex> lock(shard.mu);
        if (shard.data.find(key) != shard.data.end()) return false;

        WriteScope scope(active_writers_, generation_);
        auto next = shard.data;
        next.emplace(std::string(key), value);

        publish(shard, std::move(next));
        size_.fetch_add(1, std::memory_order_relaxed);
        scope.commit();
        return true;
    }

    bool update(std::string_view key, Value value) {
        require_loaded();
        Shard& shard = shards_[shard_of(key)];
        LockGuard<Mutex> lock(shard.mu);
        auto existing = shard.data.find(key);
        if (existing == shard.data.end()) return false;
        if (existing->second == value) return true;

        WriteScope scope(active_writers_, generation_);
        auto next = shard.data;
        next.find(key)->second = value;

        publish(shard, std::move(next));
        scope.commit();
        return true;
    }

    bool erase(std::string_view key) {
        require_loaded();
        Shard& shard = shards_[shard_of(key)];
        LockGuard<Mutex> lock(shard.mu);
        if (shard.data.find(key) == shard.data.end()) return false;

        WriteScope scope(active_writers_, generation_);
        auto next = shard.data;
        next.erase(next.find(key));

        publish(shard, std::move(next));
        size_.fetch_sub(1, std::memory_order_relaxed);
        scope.commit();
        return true;
    }

private:
    struct Shard {
        mutable Mutex mu;
        std::map<std::string, Value, UnsignedByteLess> data;
        std::shared_ptr<const validation::Index> snapshot;
    };

    static std::size_t shard_of(std::string_view key) noexcept {
        if (key.empty()) return 0;
        return 1 + static_cast<std::size_t>(
                       static_cast<unsigned char>(key.front()));
    }

    static validation::Config default_config() {
        validation::Config config = e25sw::baseline_config(128);
        e25sw::pin_frozen_switches(config);
        config.model_backend = validation::ModelBackend::Context16;
        config.compact_region_routing = false;
        return config;
    }

    static validation::Mode default_mode(std::uint32_t ov_threshold) {
        validation::Mode mode;
        mode.route = validation::Route::Interval;
        mode.prefetch = validation::Prefetch::Off;
        mode.fingerprint = true;
        mode.simd = true;
        mode.ov = validation::OvSearch::Hybrid;
        mode.ov_thr = ov_threshold;
        mode.locate = false;
        return mode;
    }

    std::shared_ptr<const validation::Index> build_snapshot(
        const std::map<std::string, Value, UnsignedByteLess>& data) const {
        std::vector<Pair> rows;
        rows.reserve(data.size());
        for (const auto& entry : data) {
            rows.emplace_back(entry.first, entry.second);
        }
        auto index = std::make_shared<validation::Index>(config_);
        index->bulk_load(rows);
        return index;
    }

    // Build the new snapshot BEFORE replacing the data, so a failed build
    // leaves both the map and the published snapshot untouched.
    void publish(Shard& shard,
                 std::map<std::string, Value, UnsignedByteLess> next) {
        auto snapshot = next.empty()
                            ? std::shared_ptr<const validation::Index>{}
                            : build_snapshot(next);
        shard.data = std::move(next);
        std::atomic_store_explicit(&shard.snapshot, std::move(snapshot),
                                   std::memory_order_release);
    }

    void require_loaded() const {
        if (!loaded_.load(std::memory_order_acquire)) {
            throw std::logic_error("bulk_load must complete before writes");
        }
    }

    validation::Config config_;
    validation::Mode mode_;
    std::array<Shard, kShards> shards_;
    std::atomic<std::size_t> size_{0};
    std::atomic<bool> loaded_{false};
    Mutex load_mu_;
    std::atomic<std::uint32_t> active_writers_{0};
    std::atomic<std::uint64_t> generation_{0};
};

}  // namespace concurrent_index
