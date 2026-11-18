#pragma once

#include "compat.hpp"
#include "group.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace olc {

// Epoch-based reclamation (global epoch + reader guard).
//
// A reader constructs a Guard for the duration of find/scan. Writers publish a
// new Group and call retire(old). When no reader is inside a Guard, try_gc()
// advances the epoch and deletes retired Groups. This is the global-epoch
// variant of EBR: it is coarser than ART-OLC's per-thread epoch lists, but it
// gives the same safety property without thread registration.
class EpochReclaimer {
public:
    class Guard {
    public:
        explicit Guard(EpochReclaimer& owner) : owner_(owner) {
            // Entering under the same mutex as try_gc() closes the window in
            // which a reader could start after the "readers == 0" check but
            // still observe a Group that is being deleted.
            LockGuard<Mutex> lock(owner_.mutex_);
            owner_.readers_.fetch_add(1, std::memory_order_relaxed);
        }

        ~Guard() {
            owner_.readers_.fetch_sub(1, std::memory_order_release);
        }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        EpochReclaimer& owner_;
    };

    EpochReclaimer() = default;
    EpochReclaimer(const EpochReclaimer&) = delete;
    EpochReclaimer& operator=(const EpochReclaimer&) = delete;

    ~EpochReclaimer() { flush(); }

    void retire(Group* group) {
        if (!group) return;
        LockGuard<Mutex> lock(mutex_);
        retired_.push_back({group, epoch_.load(std::memory_order_acquire)});
    }

    void try_gc() {
        LockGuard<Mutex> lock(mutex_);
        if (readers_.load(std::memory_order_acquire) != 0) return;
        if (retired_.empty()) return;

        const std::uint64_t next =
            epoch_.load(std::memory_order_acquire) + 1;
        epoch_.store(next, std::memory_order_release);

        std::size_t keep = 0;
        for (std::size_t i = 0; i < retired_.size(); ++i) {
            if (retired_[i].epoch < next) {
                delete retired_[i].group;
            } else {
                retired_[keep++] = retired_[i];
            }
        }
        retired_.resize(keep);
    }

    void flush() {
        LockGuard<Mutex> lock(mutex_);
        for (auto& entry : retired_) {
            delete entry.group;
        }
        retired_.clear();
    }

    std::size_t pending() const {
        LockGuard<Mutex> lock(mutex_);
        return retired_.size();
    }

private:
    std::atomic<std::uint64_t> epoch_{1};
    std::atomic<std::uint32_t> readers_{0};
    mutable Mutex mutex_;
    struct Retired {
        Group* group;
        std::uint64_t epoch;
    };
    std::vector<Retired> retired_;
};

}  // namespace olc
