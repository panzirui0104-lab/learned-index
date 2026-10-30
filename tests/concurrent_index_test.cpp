// Multi-reader / multi-writer test for concurrent_index.hpp.
//
// Build (MinGW or Linux):
//   g++ -std=c++17 -O2 -pthread concurrent_index_test.cpp -o concurrent_index_test
//
// What it checks:
//   1. bulk_load + sequential insert/update/erase/scan against a std::map.
//   2. 8 writer threads on 8 disjoint first-byte shards, 8 reader threads doing
//      find/scan/size/memory while the writers run.
//   3. After the writers finish, a full scan is compared to the reference map.
//   4. Every scan is strictly increasing by unsigned-byte order and <= limit.
//
// It is a smoke/stress test, not a linearizability checker. A scan that
// overlapped a writer must retry (the wrapper's generation rule); the test
// exercises that path but cannot prove the absence of every race.

#include "concurrent_index.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>

#include <random>
#include <string>

#include <utility>
#include <vector>

namespace {

using concurrent_index::ConcurrentIndex;
using concurrent_index::Value;
using validation::Pair;

std::string key(char prefix, int index) {
    std::string out(1, prefix);
    std::string number = std::to_string(index);
    out.append(5 - number.size(), '0');
    out += number;
    return out;
}

using Reference = std::map<std::string, Value, validation::Less>;

concurrent_index::Mutex g_error_mu;
std::string g_error;
std::atomic<bool> g_failed{false};

void fail(const std::string& message) {
    concurrent_index::LockGuard<concurrent_index::Mutex> lock(g_error_mu);
    if (!g_failed.exchange(true)) g_error = message;
}

void check(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void check_order(const std::vector<Pair>& rows, std::size_t limit,
                 const std::string& where) {
    check(rows.size() <= limit, where + ": returned more than limit");
    for (std::size_t i = 1; i < rows.size(); ++i) {
        if (validation::compare(rows[i - 1].first, rows[i].first) >= 0) {
            fail(where + ": scan is not strictly increasing at row " +
                 std::to_string(i));
            return;
        }
    }
}

void test_empty_and_single() {
    ConcurrentIndex index;
    index.bulk_load({});
    check(index.size() == 0, "empty: size");
    check(!index.find("a").has_value(), "empty: find");
    check(index.scan("", 10).empty(), "empty: scan");
    check(index.insert("a", 7), "empty: insert");
    check(index.size() == 1, "after insert: size");
    check(index.find("a").value_or(0) == 7, "after insert: find");
    check(index.update("a", 9), "update");
    check(index.find("a").value_or(0) == 9, "after update: find");
    check(index.erase("a"), "erase");
    check(!index.find("a").has_value(), "after erase: find");
    check(index.scan("", 10).empty(), "after erase: scan");
}

void test_bulk_load_contract() {
    ConcurrentIndex duplicate;
    bool threw = false;
    try {
        duplicate.bulk_load({{"a", 1}, {"a", 2}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "bulk_load: duplicate key must throw");

    ConcurrentIndex twice;
    twice.bulk_load({{"a", 1}});
    threw = false;
    try {
        twice.bulk_load({{"b", 2}});
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, "bulk_load: a second call must throw");
}

// Empty key (shard 0), NUL (shard 1) and 0xff (shard 256) exercise the
// routing boundaries and the unsigned-byte order.
void test_empty_key_nul_and_high_byte() {
    std::vector<Pair> rows;
    rows.emplace_back(std::string(), 1);
    rows.emplace_back(std::string("\0", 1), 2);
    rows.emplace_back("a", 3);
    rows.emplace_back(std::string("\xff", 1), 4);

    ConcurrentIndex index;
    index.bulk_load(rows);

    check(index.find("").value_or(0) == 1, "boundary: empty key");
    check(index.find(std::string("\0", 1)).value_or(0) == 2, "boundary: NUL");
    check(index.find("a").value_or(0) == 3, "boundary: a");
    check(index.find(std::string("\xff", 1)).value_or(0) == 4,
          "boundary: 0xff");

    auto all = index.all();
    check(all.size() == 4, "boundary: all size");
    if (all.size() == 4) {
        check(all[0].first.empty(), "boundary: empty first");
        check(all[1].first == std::string("\0", 1), "boundary: NUL second");
        check(all[2].first == "a", "boundary: a third");
        check(all[3].first == std::string("\xff", 1), "boundary: 0xff last");
    }
}

void test_sequential() {
    std::vector<Pair> rows;
    Reference reference;
    for (int i = 0; i < 200; ++i) {
        rows.emplace_back(key('a', i), static_cast<Value>(i));
        reference.emplace(key('a', i), static_cast<Value>(i));
    }

    ConcurrentIndex index;
    index.bulk_load(rows);

    for (int i = 0; i < 40; ++i) {
        const std::string k = key('a', 200 + i);
        check(index.insert(k, 1000 + i), "sequential: insert");
        reference.emplace(k, 1000 + i);
    }
    for (int i = 0; i < 40; ++i) {
        const std::string k = key('a', i);
        check(index.update(k, 2000 + i), "sequential: update");
        reference[k] = 2000 + i;
    }
    for (int i = 40; i < 80; ++i) {
        const std::string k = key('a', i);
        check(index.erase(k), "sequential: erase");
        reference.erase(k);
    }

    check(!index.insert(key('a', 0), 1), "sequential: duplicate insert");
    check(!index.update(key('a', 999), 1), "sequential: missing update");
    check(!index.erase(key('a', 999)), "sequential: missing erase");
    check(index.scan("", 0).empty(), "sequential: scan limit 0");
    check(index.scan("zzz", 10).empty(), "sequential: scan beyond end");
    auto from_existing = index.scan(key('a', 100), 3);
    check(!from_existing.empty() &&
              from_existing.front().first == key('a', 100),
          "sequential: scan is inclusive at an existing key");

    check(index.size() == reference.size(), "sequential: size");
    auto all = index.all();
    check(all.size() == reference.size(), "sequential: all size");
    check_order(all, all.size(), "sequential");
    for (const auto& row : all) {
        auto found = reference.find(row.first);
        check(found != reference.end(), "sequential: extra key " + row.first);
        if (found != reference.end()) {
            check(found->second == row.second, "sequential: wrong value");
        }
    }
    for (const auto& entry : reference) {
        auto value = index.find(entry.first);
        check(value.has_value(), "sequential: missing key " + entry.first);
        if (value.has_value()) {
            check(*value == entry.second, "sequential: find wrong value");
        }
    }
}

void test_concurrent() {
    constexpr int kPrefixes = 8;
    constexpr int kInitialPerPrefix = 1000;
    constexpr int kWritesPerKind = 60;

    std::vector<Pair> rows;
    Reference reference;
    std::vector<std::string> sample_keys;
    rows.reserve(kPrefixes * kInitialPerPrefix);
    for (int p = 0; p < kPrefixes; ++p) {
        const char prefix = static_cast<char>('a' + p);
        for (int i = 0; i < kInitialPerPrefix; ++i) {
            const std::string k = key(prefix, i);
            rows.emplace_back(k, static_cast<Value>(i));
            reference.emplace(k, static_cast<Value>(i));
            sample_keys.push_back(k);
        }
    }
    for (int p = 0; p < kPrefixes; ++p) {
        const char prefix = static_cast<char>('a' + p);
        for (int i = 0; i < kWritesPerKind; ++i) {
            sample_keys.push_back(key(prefix, kInitialPerPrefix + i));
        }
    }

    ConcurrentIndex index;
    index.bulk_load(rows);
    check(index.size() == reference.size(), "concurrent: initial size");

    std::atomic<bool> stop_readers{false};
    concurrent_index::Mutex reference_mu;

    std::vector<concurrent_index::Thread> writers;
    for (int p = 0; p < kPrefixes; ++p) {
        writers.emplace_back([&, p] {
            try {
                const char prefix = static_cast<char>('a' + p);
                for (int i = 0; i < kWritesPerKind; ++i) {
                    const std::string k = key(prefix, i);
                    check(index.update(k, 1000000 + i), "concurrent: update");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(reference_mu);
                    reference[k] = 1000000 + i;
                }
                for (int i = 0; i < kWritesPerKind; ++i) {
                    const std::string k = key(prefix, kInitialPerPrefix + i);
                    check(index.insert(k, 2000000 + i), "concurrent: insert");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(reference_mu);
                    reference.emplace(k, 2000000 + i);
                }
                for (int i = 0; i < kWritesPerKind; ++i) {
                    const std::string k = key(prefix, 100 + i);
                    check(index.erase(k), "concurrent: erase");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(reference_mu);
                    reference.erase(k);
                }
            } catch (const std::exception& error) {
                fail(std::string("writer exception: ") + error.what());
            } catch (...) {
                fail("writer unknown exception");
            }
        });
    }

    std::vector<concurrent_index::Thread> readers;
    for (int r = 0; r < 8; ++r) {
        readers.emplace_back([&, r] {
            try {
                std::mt19937_64 rng(0x5EED0000u + r);
                std::uint64_t iteration = 0;
                while (!stop_readers.load(std::memory_order_relaxed)) {
                    const std::string& query =
                        sample_keys[rng() % sample_keys.size()];
                    (void)index.find(query);

                    const std::string& lower =
                        sample_keys[rng() % sample_keys.size()];
                    auto rows = index.scan(lower, 32);
                    check_order(rows, 32, "concurrent reader");
                    if (g_failed.load()) return;

                    if ((iteration++ & 63) == 0) {
                        (void)index.size();
                        (void)index.memory();
                    }
                }
            } catch (const std::exception& error) {
                fail(std::string("reader exception: ") + error.what());
            } catch (...) {
                fail("reader unknown exception");
            }
        });
    }

    for (auto& writer : writers) writer.join();
    stop_readers.store(true, std::memory_order_relaxed);
    for (auto& reader : readers) reader.join();

    check(!g_failed.load(), "concurrent: thread failure");

    check(index.size() == reference.size(), "concurrent: final size");
    auto all = index.all();
    check(all.size() == reference.size(), "concurrent: all size");
    check_order(all, all.size(), "concurrent");
    for (const auto& row : all) {
        auto found = reference.find(row.first);
        check(found != reference.end(), "concurrent: extra key " + row.first);
        if (found != reference.end()) {
            check(found->second == row.second, "concurrent: wrong value");
        }
    }
    for (const auto& entry : reference) {
        auto value = index.find(entry.first);
        check(value.has_value(), "concurrent: missing key " + entry.first);
        if (value.has_value()) {
            check(*value == entry.second, "concurrent: find wrong value");
        }
    }
}

// All writers target the SAME shard ('z'), so the per-shard lock must
// serialize them. Readers still run concurrently. This is the path that the
// disjoint-prefix test does not exercise.
void test_same_shard_writers() {
    constexpr int kInitial = 2000;
    constexpr int kWriters = 4;
    constexpr int kOps = 40;

    std::vector<Pair> rows;
    Reference reference;
    std::vector<std::string> sample_keys;
    rows.reserve(kInitial + kWriters * kOps);
    for (int i = 0; i < kInitial; ++i) {
        const std::string k = key('z', i);
        rows.emplace_back(k, static_cast<Value>(i));
        reference.emplace(k, static_cast<Value>(i));
        sample_keys.push_back(k);
    }
    for (int t = 0; t < kWriters; ++t) {
        for (int i = 0; i < kOps; ++i) {
            sample_keys.push_back(key('z', kInitial + t * kOps + i));
        }
    }

    ConcurrentIndex index;
    index.bulk_load(rows);

    std::atomic<bool> stop_readers{false};
    concurrent_index::Mutex reference_mu;

    std::vector<concurrent_index::Thread> writers;
    for (int t = 0; t < kWriters; ++t) {
        writers.emplace_back([&, t] {
            try {
                for (int i = 0; i < kOps; ++i) {
                    const std::string k = key('z', t * kOps + i);
                    check(index.update(k, 500000 + t * 1000 + i),
                          "same-shard: update");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(
                        reference_mu);
                    reference[k] = 500000 + t * 1000 + i;
                }
                for (int i = 0; i < kOps; ++i) {
                    const std::string k =
                        key('z', kInitial + t * kOps + i);
                    check(index.insert(k, 600000 + t * 1000 + i),
                          "same-shard: insert");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(
                        reference_mu);
                    reference.emplace(k, 600000 + t * 1000 + i);
                }
                for (int i = 0; i < kOps; ++i) {
                    const std::string k = key('z', 500 + t * kOps + i);
                    check(index.erase(k), "same-shard: erase");
                    concurrent_index::LockGuard<concurrent_index::Mutex> lock(
                        reference_mu);
                    reference.erase(k);
                }
            } catch (const std::exception& error) {
                fail(std::string("same-shard writer exception: ") +
                     error.what());
            } catch (...) {
                fail("same-shard writer unknown exception");
            }
        });
    }

    std::vector<concurrent_index::Thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&, r] {
            try {
                std::mt19937_64 rng(0xC0FFEE00u + r);
                while (!stop_readers.load(std::memory_order_relaxed)) {
                    const std::string& query =
                        sample_keys[rng() % sample_keys.size()];
                    (void)index.find(query);
                    auto rows = index.scan(query, 24);
                    check_order(rows, 24, "same-shard reader");
                    if (g_failed.load()) return;
                }
            } catch (const std::exception& error) {
                fail(std::string("same-shard reader exception: ") +
                     error.what());
            } catch (...) {
                fail("same-shard reader unknown exception");
            }
        });
    }

    for (auto& writer : writers) writer.join();
    stop_readers.store(true, std::memory_order_relaxed);
    for (auto& reader : readers) reader.join();

    check(!g_failed.load(), "same-shard: thread failure");
    check(index.size() == reference.size(), "same-shard: final size");
    auto all = index.all();
    check(all.size() == reference.size(), "same-shard: all size");
    check_order(all, all.size(), "same-shard");
    for (const auto& row : all) {
        auto found = reference.find(row.first);
        check(found != reference.end(), "same-shard: extra key " + row.first);
        if (found != reference.end()) {
            check(found->second == row.second, "same-shard: wrong value");
        }
    }
    for (const auto& entry : reference) {
        auto value = index.find(entry.first);
        check(value.has_value(), "same-shard: missing key " + entry.first);
        if (value.has_value()) {
            check(*value == entry.second, "same-shard: find wrong value");
        }
    }
}

}  // namespace

int main() {
    const auto start = std::chrono::steady_clock::now();

    test_empty_and_single();
    if (!g_failed.load()) test_bulk_load_contract();
    if (!g_failed.load()) test_empty_key_nul_and_high_byte();
    if (!g_failed.load()) test_sequential();
    if (!g_failed.load()) test_concurrent();
    if (!g_failed.load()) test_same_shard_writers();

    const auto end = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - start)
                        .count();

    if (g_failed.load()) {
        std::cerr << "FAILED: " << g_error << "\n";
        return 1;
    }
    std::cout << "PASSED in " << ms << " ms\n";
    return 0;
}
