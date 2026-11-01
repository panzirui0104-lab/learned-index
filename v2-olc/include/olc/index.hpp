#pragma once

#include "common.hpp"
#include "region.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace olc {

// Region-level OLC index.
//
// 257 fixed regions: 0 = empty key, 1..256 = first byte 0x00..0xff. The region
// prefix is the first byte only, so any inserted key with that first byte has a
// matching prefix. Point operations are handled by Region OLC; cross-region
// scan uses a writer-active/generation check so it never mixes two generations.
class OlcIndex {
public:
    static constexpr std::size_t kRegions = 257;

    OlcIndex() = default;
    OlcIndex(const OlcIndex&) = delete;
    OlcIndex& operator=(const OlcIndex&) = delete;

    void bulk_load(const std::vector<Pair>& rows) {
        if (loaded_.load(std::memory_order_acquire)) {
            throw std::logic_error("bulk_load may be called only once");
        }

        std::array<std::vector<Pair>, kRegions> buckets;
        std::set<std::string, Less> seen;
        for (const auto& row : rows) {
            if (!seen.insert(row.key).second) {
                throw std::invalid_argument(
                    "bulk_load input contains a duplicate key");
            }
            buckets[region_of(row.key)].push_back(row);
        }

        for (std::size_t r = 0; r < kRegions; ++r) {
            regions_[r] = std::make_unique<Region>(std::move(buckets[r]));
        }

        size_.store(rows.size(), std::memory_order_release);
        loaded_.store(true, std::memory_order_release);
    }

    // ---- concurrent reads -------------------------------------------------

    std::optional<Value> find(std::string_view key) const {
        if (!loaded_.load(std::memory_order_acquire)) return std::nullopt;
        const std::size_t r = region_of(key);
        if (!regions_[r]) return std::nullopt;
        Value value = 0;
        if (!regions_[r]->find(key, value)) return std::nullopt;
        return value;
    }

    std::vector<Pair> scan(std::string_view lower_bound,
                           std::size_t limit) const {
        if (limit == 0 || !loaded_.load(std::memory_order_acquire)) return {};

        for (;;) {
            if (active_writers_.load(std::memory_order_acquire) != 0) {
                continue;
            }
            const std::uint64_t generation_before =
                generation_.load(std::memory_order_acquire);

            std::vector<Pair> out;
            bool first = true;
            for (std::size_t r = region_of(lower_bound);
                 r < kRegions && out.size() < limit; ++r) {
                if (!regions_[r]) {
                    first = false;
                    continue;
                }
                std::vector<Pair> part;
                const std::string_view start =
                    first ? lower_bound : std::string_view{};
                regions_[r]->scan(start, limit - out.size(), part);
                for (auto& row : part) out.push_back(std::move(row));
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

    std::size_t memory() const {
        std::size_t bytes = sizeof(*this);
        for (const auto& region : regions_) {
            if (region) bytes += region->memory();
        }
        return bytes;
    }

    // ---- concurrent writes ------------------------------------------------

    bool insert(std::string_view key, Value value) {
        require_loaded();
        WriteScope scope(*this);
        const std::size_t r = region_of(key);
        auto& region = regions_[r];
        if (!region->insert(key, value)) return false;
        size_.fetch_add(1, std::memory_order_relaxed);
        scope.commit();
        return true;
    }

    bool update(std::string_view key, Value value) {
        require_loaded();
        WriteScope scope(*this);
        const std::size_t r = region_of(key);
        if (!regions_[r] || !regions_[r]->update(key, value)) return false;
        scope.commit();
        return true;
    }

    bool erase(std::string_view key) {
        require_loaded();
        WriteScope scope(*this);
        const std::size_t r = region_of(key);
        if (!regions_[r] || !regions_[r]->erase(key)) return false;
        size_.fetch_sub(1, std::memory_order_relaxed);
        scope.commit();
        return true;
    }

private:
    class WriteScope {
    public:
        explicit WriteScope(OlcIndex& owner) : owner_(owner) {
            owner_.active_writers_.fetch_add(1, std::memory_order_acq_rel);
        }

        ~WriteScope() {
            if (commit_) {
                owner_.generation_.fetch_add(1, std::memory_order_release);
            }
            owner_.active_writers_.fetch_sub(1, std::memory_order_release);
        }

        void commit() { commit_ = true; }

        WriteScope(const WriteScope&) = delete;
        WriteScope& operator=(const WriteScope&) = delete;

    private:
        OlcIndex& owner_;
        bool commit_ = false;
    };

    static std::size_t region_of(std::string_view key) noexcept {
        if (key.empty()) return 0;
        return 1 + static_cast<std::size_t>(
                       static_cast<unsigned char>(key.front()));
    }

    void require_loaded() const {
        if (!loaded_.load(std::memory_order_acquire)) {
            throw std::logic_error("bulk_load must complete before writes");
        }
    }

    std::array<std::unique_ptr<Region>, kRegions> regions_;
    std::atomic<std::size_t> size_{0};
    std::atomic<bool> loaded_{false};
    std::atomic<std::uint32_t> active_writers_{0};
    std::atomic<std::uint64_t> generation_{0};
};

}  // namespace olc
