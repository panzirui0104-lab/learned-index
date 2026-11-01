// Same-Group contention test: 4 keys -> Region has one Group, then 4 writers
// insert 100 keys each into that same Group while 4 readers scan.

#include "olc/index.hpp"
#include "olc/compat.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using olc::OlcIndex;
using olc::Pair;
using olc::Value;

std::atomic<bool> failed{false};
std::atomic<bool> stop_readers{false};

std::string key(int i) {
    std::string out = "a";
    std::string n = std::to_string(i);
    out.append(5 - n.size(), '0');
    out += n;
    return out;
}

void fail(const std::string& m) {
    if (!failed.exchange(true)) std::cerr << "FAILED: " << m << "\n";
}

}  // namespace

int main() {
    constexpr int kWriters = 4;
    constexpr int kPerWriter = 100;

    std::vector<Pair> rows;
    std::vector<std::string> queries;
    for (int i = 0; i < 4; ++i) {
        rows.push_back({key(i), static_cast<Value>(i)});
        queries.push_back(key(i));
    }

    OlcIndex index;
    index.bulk_load(rows);

    std::vector<olc::Thread> writers;
    for (int t = 0; t < kWriters; ++t) {
        writers.emplace_back([&index, t] {
            try {
                for (int i = 0; i < kPerWriter; ++i) {
                    const int id = 1000 + t * kPerWriter + i;
                    if (!index.insert(key(id), static_cast<Value>(id))) {
                        fail("insert");
                    }
                }
            } catch (const std::exception& e) {
                fail(std::string("writer: ") + e.what());
            } catch (...) {
                fail("writer unknown");
            }
        });
    }

    std::vector<olc::Thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&index, &queries, r] {
            try {
                std::mt19937 rng(0x77770000u + r);
                while (!stop_readers.load(std::memory_order_relaxed)) {
                    const std::string& k = queries[rng() % queries.size()];
                    auto v = index.find(k);
                    if (!v) fail("find initial");
                    auto rows = index.scan(k, 16);
                    for (std::size_t i = 1; i < rows.size(); ++i) {
                        if (!(rows[i - 1].key < rows[i].key)) {
                            fail("scan order");
                            return;
                        }
                    }
                    if (failed.load()) return;
                }
            } catch (const std::exception& e) {
                fail(std::string("reader: ") + e.what());
            } catch (...) {
                fail("reader unknown");
            }
        });
    }

    for (auto& writer : writers) writer.join();
    stop_readers.store(true, std::memory_order_relaxed);
    for (auto& reader : readers) reader.join();
    if (failed.load()) return 1;

    const std::size_t expected = 4 + kWriters * kPerWriter;
    if (index.size() != expected) {
        std::cerr << "FAILED: size " << index.size() << "\n";
        return 1;
    }
    auto all = index.all();
    if (all.size() != expected) {
        std::cerr << "FAILED: all size\n";
        return 1;
    }
    for (std::size_t i = 1; i < all.size(); ++i) {
        if (!(all[i - 1].key < all[i].key)) {
            std::cerr << "FAILED: all order\n";
            return 1;
        }
    }
    for (int t = 0; t < kWriters; ++t) {
        for (int i = 0; i < kPerWriter; ++i) {
            const int id = 1000 + t * kPerWriter + i;
            auto v = index.find(key(id));
            if (!v || *v != static_cast<Value>(id)) {
                std::cerr << "FAILED: final find\n";
                return 1;
            }
        }
    }

    std::cout << "v2 same-group contention passed (" << all.size()
              << " keys)\n";
    return 0;
}
