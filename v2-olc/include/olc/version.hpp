#pragma once

#include <atomic>
#include <cstdint>

namespace olc {

// ART-OLC style version word:
//   bit 63..62 type (unused here, kept for layout compatibility)
//   bit 61..2  version (60 bit)
//   bit 1      lock
//   bit 0      obsolete
//
// Initial value 0b100: version=1, unlocked, live.
class Version {
public:
    Version() = default;
    Version(const Version&) = delete;
    Version& operator=(const Version&) = delete;

    static bool is_locked(std::uint64_t v) noexcept { return (v & 0b10U) != 0; }
    static bool is_obsolete(std::uint64_t v) noexcept { return (v & 0b01U) != 0; }

    std::uint64_t load(std::memory_order order = std::memory_order_acquire) const noexcept {
        return value_.load(order);
    }

    // Reader side: return a version only if the node is live and unlocked.
    std::uint64_t read_lock(bool& restart) const noexcept {
        const std::uint64_t v = value_.load(std::memory_order_acquire);
        if (is_locked(v) || is_obsolete(v)) {
            restart = true;
            return 0;
        }
        return v;
    }

    void check(std::uint64_t expected, bool& restart) const noexcept {
        restart = value_.load(std::memory_order_acquire) != expected;
    }

    // Writer side: CAS the lock bit. Returns true and the locked version, or
    // false with restart=true if the node is locked/obsolete/changed.
    bool write_lock_or_restart(std::uint64_t& locked_version,
                               bool& restart) noexcept {
        std::uint64_t v = value_.load(std::memory_order_acquire);
        for (;;) {
            if (is_locked(v) || is_obsolete(v)) {
                restart = true;
                return false;
            }
            if (value_.compare_exchange_weak(v, v + 0b10U,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                locked_version = v + 0b10U;
                return true;
            }
        }
    }

    void unlock() noexcept {
        value_.fetch_add(0b10U, std::memory_order_release);
    }

    // Only when locked: clear lock, set obsolete, version+1.
    void unlock_obsolete() noexcept {
        value_.fetch_add(0b11U, std::memory_order_release);
    }

    // Mark an already-published node obsolete without taking its lock.
    void mark_obsolete() noexcept {
        value_.fetch_add(0b01U, std::memory_order_release);
    }

    bool obsolete() const noexcept {
        return is_obsolete(value_.load(std::memory_order_acquire));
    }

    // Bump the version without locking; used by Region as a change counter.
    void bump() noexcept {
        value_.fetch_add(0b10U, std::memory_order_release);
    }

private:
    std::atomic<std::uint64_t> value_{0b100U};
};

}  // namespace olc
