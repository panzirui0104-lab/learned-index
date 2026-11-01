#include "olc/index.hpp"
#include "olc/compat.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using olc::OlcIndex;
using olc::Pair;
using olc::Value;
using Expected = std::map<std::string, Value, olc::Less>;

std::atomic<bool> failed{false};
std::atomic<bool> stop_readers{false};

std::string key(char prefix, int index) {
    std::string out(1, prefix);
    std::string number = std::to_string(index);
    out.append(5 - number.size(), '0');
    out += number;
    return out;
}

void fail(const std::string& message) {
    if (!failed.exchange(true)) std::cerr << "FAILED: " << message << "\n";
}

void check_find(const OlcIndex& index, const Expected& known,
                const std::string& k) {
    auto actual = index.find(k);
    if (!actual) return;
    auto want = known.find(k);
    if (want == known.end()) {
        fail("unknown key");
    } else if (*actual != want->second) {
        fail("wrong value");
    }
}

void check_scan(const OlcIndex& index, const Expected& known,
                std::string_view start, std::size_t limit) {
    auto rows = index.scan(start, limit);
    if (rows.size() > limit) fail("scan limit");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto want = known.find(rows[i].key);
        if (want == known.end()) {
            fail("scan unknown key");
            return;
        }
        if (rows[i].value != want->second) {
            fail("scan wrong value");
            return;
        }
        if (i > 0 && !(rows[i - 1].key < rows[i].key)) {
            fail("scan order");
            return;
        }
    }
}

}  // namespace

int main() {
    constexpr int kPrefixes = 8;
    constexpr int kInitial = 1000;
    constexpr int kWrites = 200;

    std::vector<Pair> rows;
    Expected known;
    Expected final_state;
    std::vector<std::string> query_keys;

    for (int p = 0; p < kPrefixes; ++p) {
        const char prefix = static_cast<char>('a' + p);
        for (int i = 0; i < kInitial; ++i) {
            const std::string k = key(prefix, i);
            const Value value = static_cast<Value>(p * 100000 + i);
            rows.push_back({k, value});
            known.emplace(k, value);
            final_state.emplace(k, value);
            query_keys.push_back(k);
        }
        for (int i = 0; i < kWrites; ++i) {
            const std::string k = key(prefix, kInitial + i);
            const Value value = static_cast<Value>(p * 100000 + 500000 + i);
            known.emplace(k, value);
            final_state.emplace(k, value);
            query_keys.push_back(k);
        }
        for (int i = 0; i < kWrites; ++i) {
            final_state.erase(key(prefix, i));
        }
    }

    OlcIndex index;
    index.bulk_load(rows);

    std::vector<olc::Thread> writers;
    for (int p = 0; p < kPrefixes; ++p) {
        writers.emplace_back([&index, p] {
            try {
                const char prefix = static_cast<char>('a' + p);
                for (int i = 0; i < kWrites; ++i) {
                    const Value value =
                        static_cast<Value>(p * 100000 + 500000 + i);
                    if (!index.insert(key(prefix, kInitial + i), value)) {
                        fail("insert");
                    }
                }
                for (int i = 0; i < kWrites; ++i) {
                    if (!index.erase(key(prefix, i))) fail("erase");
                }
            } catch (const std::exception& e) {
                fail(std::string("writer: ") + e.what());
            } catch (...) {
                fail("writer unknown");
            }
        });
    }

    std::vector<olc::Thread> readers;
    for (int r = 0; r < 8; ++r) {
        readers.emplace_back([&index, &known, &query_keys, r] {
            try {
                std::mt19937 rng(0x12340000u + r);
                while (!stop_readers.load(std::memory_order_relaxed)) {
                    const std::string& k =
                        query_keys[rng() % query_keys.size()];
                    check_find(index, known, k);
                    check_scan(index, known, k, 12);
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

    if (index.size() != final_state.size()) {
        std::cerr << "FAILED: size " << index.size() << " != "
                  << final_state.size() << "\n";
        return 1;
    }

    for (int p = 0; p < kPrefixes; ++p) {
        const char prefix = static_cast<char>('a' + p);
        for (int i = 0; i < kInitial + kWrites; ++i) {
            const std::string k = key(prefix, i);
            auto actual = index.find(k);
            auto want = final_state.find(k);
            if (want == final_state.end()) {
                if (actual) {
                    std::cerr << "FAILED: erased key present\n";
                    return 1;
                }
            } else if (!actual || *actual != want->second) {
                std::cerr << "FAILED: missing/wrong key\n";
                return 1;
            }
        }
    }

    auto all = index.all();
    if (all.size() != final_state.size()) {
        std::cerr << "FAILED: all size\n";
        return 1;
    }
    for (const auto& row : all) {
        auto want = final_state.find(row.key);
        if (want == final_state.end() || want->second != row.value) {
            std::cerr << "FAILED: all content\n";
            return 1;
        }
    }

    std::cout << "v2 value test passed (" << all.size() << " keys)\n";
    return 0;
}
