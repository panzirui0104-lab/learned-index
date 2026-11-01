#pragma once

#include "common.hpp"
#include "group.hpp"
#include "model.hpp"
#include "version.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace olc {

// A Region owns a prefix (the first byte of its shard), a Context16 model, and
// a fixed array of Group slots. The model maps a suffix to a group; groups are
// ordered, so group order equals key order. Readers use optimistic version
// checks on the slot and on the region change counter; writers take the slot's
// version lock, publish a new Group for insert/erase, or write a value in place
// for update.
class Region {
public:
    explicit Region(std::vector<Pair> rows) {
        if (rows.empty()) {
            group_count_ = 1;
            group_slots_.resize(1);
            slots_.resize(1);
            slots_[0] = std::make_unique<Version>();
            size_.store(0, std::memory_order_relaxed);
            return;
        }

        if (!rows.front().key.empty()) {
            prefix_.assign(1, rows.front().key.front());
        }

        model_.train(rows, prefix_.size());

        const std::size_t wanted = std::max<std::size_t>(
            1, std::min<std::size_t>(4096, rows.size() / 32));
        group_count_ = static_cast<std::uint32_t>(wanted);
        group_slots_.resize(group_count_);
        slots_.resize(group_count_);

        std::vector<std::vector<Group::Record>> buckets(group_count_);
        for (auto& pair : rows) {
            const std::string_view suffix =
                std::string_view(pair.key).substr(prefix_.size());
            const std::uint32_t g = model_.group(suffix, group_count_);
            buckets[g].emplace_back(std::string(suffix), pair.value);
        }

        for (std::uint32_t g = 0; g < group_count_; ++g) {
            slots_[g] = std::make_unique<Version>();
            if (!buckets[g].empty()) {
                group_slots_[g] = std::make_shared<Group>(std::move(buckets[g]));
            }
        }
        size_.store(rows.size(), std::memory_order_release);
    }

    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    bool find(std::string_view key, Value& out) const {
        if (!has_prefix(key)) return false;
        const std::string_view suffix = key.substr(prefix_.size());

        for (;;) {
            bool restart = false;
            const std::uint64_t region_version =
                version_.load(std::memory_order_acquire);

            const std::uint32_t g = model_.group(suffix, group_count_);
            std::uint64_t slot_version =
                slots_[g]->read_lock(restart);
            if (restart) {
                continue;
            }

            auto group = std::atomic_load_explicit(
                &group_slots_[g], std::memory_order_acquire);
            bool found = false;
            if (group && !group->obsolete()) {
                const Group::Record* record = group->find(suffix);
                if (record) {
                    out = record->value.load(std::memory_order_acquire);
                    found = true;
                }
            }

            slots_[g]->check(slot_version, restart);
            if (restart) continue;
            version_.check(region_version, restart);
            if (restart) continue;
            return found;
        }
    }

    bool insert(std::string_view key, Value value) {
        if (!has_prefix(key)) return false;
        const std::string_view suffix = key.substr(prefix_.size());
        const std::uint32_t g = model_.group(suffix, group_count_);

        for (;;) {
            bool restart = false;
            std::uint64_t slot_version = 0;
            if (!slots_[g]->write_lock_or_restart(slot_version, restart)) {
                if (restart) continue;
                return false;
            }

            auto old = std::atomic_load_explicit(
                &group_slots_[g], std::memory_order_acquire);
            if (old && old->find(suffix)) {
                slots_[g]->unlock();
                return false;
            }

            try {
                std::vector<Group::Record> records;
                if (old) records = old->records();
                records.emplace_back(std::string(suffix), value);
                auto next = std::make_shared<Group>(std::move(records));

                std::atomic_store_explicit(
                    &group_slots_[g], std::shared_ptr<const Group>(next),
                    std::memory_order_release);
                if (old) old->mark_obsolete();
                slots_[g]->unlock();
                version_.bump();
                size_.fetch_add(1, std::memory_order_relaxed);
                return true;
            } catch (...) {
                slots_[g]->unlock();
                throw;
            }
        }
    }

    bool update(std::string_view key, Value value) {
        if (!has_prefix(key)) return false;
        const std::string_view suffix = key.substr(prefix_.size());
        const std::uint32_t g = model_.group(suffix, group_count_);

        for (;;) {
            bool restart = false;
            std::uint64_t slot_version = 0;
            if (!slots_[g]->write_lock_or_restart(slot_version, restart)) {
                if (restart) continue;
                return false;
            }

            auto group = std::atomic_load_explicit(
                &group_slots_[g], std::memory_order_acquire);
            const Group::Record* record =
                group ? group->find(suffix) : nullptr;
            if (!record) {
                slots_[g]->unlock();
                return false;
            }

            record->value.store(value, std::memory_order_release);
            slots_[g]->unlock();
            version_.bump();
            return true;
        }
    }

    bool erase(std::string_view key) {
        if (!has_prefix(key)) return false;
        const std::string_view suffix = key.substr(prefix_.size());
        const std::uint32_t g = model_.group(suffix, group_count_);

        for (;;) {
            bool restart = false;
            std::uint64_t slot_version = 0;
            if (!slots_[g]->write_lock_or_restart(slot_version, restart)) {
                if (restart) continue;
                return false;
            }

            auto old = std::atomic_load_explicit(
                &group_slots_[g], std::memory_order_acquire);
            if (!old || !old->find(suffix)) {
                slots_[g]->unlock();
                return false;
            }

            try {
                std::vector<Group::Record> records;
                records.reserve(old->records().size() - 1);
                for (const auto& record : old->records()) {
                    if (compare(record.suffix, suffix) != 0) {
                        records.push_back(record);
                    }
                }

                std::shared_ptr<const Group> next;
                if (!records.empty()) {
                    next = std::make_shared<Group>(std::move(records));
                }
                std::atomic_store_explicit(
                    &group_slots_[g], std::move(next), std::memory_order_release);
                old->mark_obsolete();
                slots_[g]->unlock();
                version_.bump();
                size_.fetch_sub(1, std::memory_order_relaxed);
                return true;
            } catch (...) {
                slots_[g]->unlock();
                throw;
            }
        }
    }

    void scan(std::string_view start, std::size_t limit,
              std::vector<Pair>& out) const {
        if (limit == 0) return;

        for (;;) {
            bool restart = false;
            const std::uint64_t region_version =
                version_.load(std::memory_order_acquire);

            std::uint32_t gstart = 0;
            std::string_view suffix_start;
            if (has_prefix(start)) {
                suffix_start = start.substr(prefix_.size());
                gstart = model_.group(suffix_start, group_count_);
            }

            std::vector<Pair> local;
            for (std::uint32_t g = gstart;
                 g < group_count_ && local.size() < limit; ++g) {
                std::uint64_t slot_version =
                    slots_[g]->read_lock(restart);
                if (restart) {
                    break;
                }

                auto group = std::atomic_load_explicit(
                    &group_slots_[g], std::memory_order_acquire);
                if (group && group->obsolete()) {
                    restart = true;
                    break;
                }

                if (group) {
                    for (const auto& record : group->records()) {
                        std::string full = prefix_;
                        full += record.suffix;
                        if (compare(full, start) < 0) continue;
                        local.push_back(
                            {std::move(full),
                             record.value.load(std::memory_order_acquire)});
                        if (local.size() >= limit) break;
                    }
                }

                slots_[g]->check(slot_version, restart);
                if (restart) break;
            }

            if (restart) continue;
            version_.check(region_version, restart);
            if (restart) continue;

            for (auto& pair : local) out.push_back(std::move(pair));
            return;
        }
    }

    std::size_t size() const {
        return size_.load(std::memory_order_acquire);
    }

    std::size_t memory() const {
        std::size_t bytes = sizeof(*this) + prefix_.capacity() +
                            model_.memory() +
                            group_slots_.capacity() * sizeof(std::shared_ptr<const Group>) +
                            slots_.capacity() * sizeof(std::unique_ptr<Version>);
        for (const auto& group : group_slots_) {
            if (group) bytes += group->memory();
        }
        return bytes;
    }

private:
    bool has_prefix(std::string_view key) const {
        return key.size() >= prefix_.size() &&
               (prefix_.empty() ||
                std::memcmp(key.data(), prefix_.data(), prefix_.size()) == 0);
    }

    std::string prefix_;
    Model model_;
    std::uint32_t group_count_ = 1;
    std::vector<std::shared_ptr<const Group>> group_slots_;
    std::vector<std::unique_ptr<Version>> slots_;
    mutable Version version_;
    std::atomic<std::size_t> size_{0};
};

}  // namespace olc
