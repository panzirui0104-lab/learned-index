#pragma once

// Minimal thread/mutex compatibility layer.
//
// The project is normally built with GCC on Linux, where std::mutex and
// std::thread are available. The MinGW 8.1 win32-threads toolchain used for a
// local smoke build has no gthreads, so std::mutex/std::thread do not compile
// there. This header keeps the wrapper portable:
//
//   * POSIX: std::mutex / std::thread;
//   * Windows: a small atomic_flag spinlock and a CreateThread wrapper.
//
// The spinlock is only used for per-shard writer serialization and for the
// one-shot bulk_load; readers never take it. On the target Linux build the
// standard std::mutex is used.

#include <atomic>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <mutex>
#include <sched.h>
#include <thread>
#endif

namespace olc {

#if defined(_WIN32)

class Mutex {
public:
    Mutex() = default;
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;

    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            Sleep(0);
        }
    }

    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

inline void yield() noexcept { Sleep(0); }

class Thread {
public:
    Thread() = default;

    template <class Function>
    explicit Thread(Function&& function) {
        auto* task = new std::function<void()>(std::forward<Function>(function));
        handle_ = CreateThread(nullptr, 0, &Thread::run, task, 0, nullptr);
        if (handle_ == nullptr) {
            delete task;
            throw std::runtime_error("CreateThread failed");
        }
    }

    ~Thread() { join(); }

    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    Thread(Thread&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Thread& operator=(Thread&& other) noexcept {
        if (this != &other) {
            join();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void join() {
        if (handle_ != nullptr) {
            WaitForSingleObject(handle_, INFINITE);
            CloseHandle(handle_);
            handle_ = nullptr;
        }
    }

private:
    static DWORD WINAPI run(LPVOID argument) {
        auto* task = static_cast<std::function<void()>*>(argument);
        try {
            (*task)();
        } catch (...) {
            // The test's worker catches and records exceptions itself; never
            // let one escape the thread entry point.
        }
        delete task;
        return 0;
    }

    HANDLE handle_ = nullptr;
};

#else

using Mutex = std::mutex;
using Thread = std::thread;

inline void yield() noexcept { sched_yield(); }

#endif

template <class MutexType>
class LockGuard {
public:
    explicit LockGuard(MutexType& mutex) : mutex_(mutex) { mutex_.lock(); }
    ~LockGuard() { mutex_.unlock(); }

    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    MutexType& mutex_;
};

}  // namespace olc
