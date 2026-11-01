#include "olc/index.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using olc::OlcIndex;
using olc::Pair;
using olc::Value;
using Reference = std::map<std::string, Value, olc::Less>;

bool same_rows(const std::vector<Pair>& a, const std::vector<Pair>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].key != b[i].key || a[i].value != b[i].value) return false;
    }
    return true;
}

std::vector<Pair> ref_scan(const Reference& ref, std::string_view start,
                           std::size_t limit) {
    std::vector<Pair> rows;
    for (auto it = ref.lower_bound(start);
         it != ref.end() && rows.size() < limit; ++it) {
        rows.push_back(Pair{it->first, it->second});
    }
    return rows;
}

std::string random_key(std::mt19937& rng) {
    static const unsigned char alphabet[] = {
        0x00, 'a', 'b', 'c', 'z', 0x7f, 0x80, 0xff};
    const std::size_t length = rng() % 7;
    std::string key;
    for (std::size_t i = 0; i < length; ++i) {
        key.push_back(static_cast<char>(alphabet[rng() % 8]));
    }
    return key;
}

Value random_value(std::mt19937& rng) {
    switch (rng() % 8) {
        case 0: return 0;
        case 1: return std::numeric_limits<Value>::max();
        default: return static_cast<Value>(rng());
    }
}

int run(unsigned seed, int operations) {
    std::mt19937 rng(seed);
    Reference ref;

    std::vector<std::string> pool;
    pool.emplace_back();
    pool.emplace_back(std::string("\0", 1));
    pool.emplace_back(std::string("\xff", 1));
    for (char prefix : {'a', 'b', 'c', 'd'}) {
        for (int i = 0; i < 20; ++i) {
            pool.push_back(std::string(1, prefix) + std::to_string(i));
        }
    }
    for (int i = 0; i < 64; ++i) pool.push_back(random_key(rng));
    for (const auto& key : pool) ref.emplace(key, random_value(rng));
    for (int i = 0; i < 128; ++i) {
        ref.emplace(random_key(rng), random_value(rng));
    }

    std::vector<Pair> rows;
    for (const auto& entry : ref) rows.push_back({entry.first, entry.second});

    OlcIndex index;
    index.bulk_load(rows);
    if (index.size() != ref.size()) {
        std::cerr << "seed " << seed << ": initial size\n";
        return 1;
    }

    for (int op = 0; op < operations; ++op) {
        const std::string key = random_key(rng);
        const unsigned kind = rng() % 100;

        if (kind < 35) {
            const Value value = random_value(rng);
            const auto expected = ref.emplace(key, value);
            if (index.insert(key, value) != expected.second) {
                std::cerr << "seed " << seed << " op " << op
                          << ": insert mismatch\n";
                return 1;
            }
            if (!expected.second) {
                auto actual = index.find(key);
                if (!actual || *actual != expected.first->second) {
                    std::cerr << "seed " << seed << " op " << op
                              << ": duplicate insert changed value\n";
                    return 1;
                }
            }
        } else if (kind < 55) {
            const Value value = random_value(rng);
            auto found = ref.find(key);
            const bool present = found != ref.end();
            if (present) found->second = value;
            if (index.update(key, value) != present) {
                std::cerr << "seed " << seed << " op " << op
                          << ": update mismatch\n";
                return 1;
            }
            auto actual = index.find(key);
            if (present) {
                if (!actual || *actual != value) {
                    std::cerr << "seed " << seed << " op " << op
                              << ": update value\n";
                    return 1;
                }
            } else if (actual) {
                std::cerr << "seed " << seed << " op " << op
                          << ": update inserted\n";
                return 1;
            }
        } else if (kind < 75) {
            const bool present = ref.erase(key) != 0;
            if (index.erase(key) != present) {
                std::cerr << "seed " << seed << " op " << op
                          << ": erase mismatch\n";
                return 1;
            }
            if (!present && index.find(key)) {
                std::cerr << "seed " << seed << " op " << op
                          << ": erase produced value\n";
                return 1;
            }
        } else {
            auto expected = ref.find(key);
            auto actual = index.find(key);
            if (expected == ref.end()) {
                if (actual) {
                    std::cerr << "seed " << seed << " op " << op
                              << ": false hit\n";
                    return 1;
                }
            } else if (!actual || *actual != expected->second) {
                std::cerr << "seed " << seed << " op " << op
                          << ": false miss/wrong value\n";
                return 1;
            }
        }

        if (index.size() != ref.size()) {
            std::cerr << "seed " << seed << " op " << op << ": size\n";
            return 1;
        }

        if ((op % 13) == 0) {
            const std::string start = random_key(rng);
            const std::size_t limit = rng() % 26;
            if (!same_rows(index.scan(start, limit),
                           ref_scan(ref, start, limit))) {
                std::cerr << "seed " << seed << " op " << op
                          << ": scan mismatch\n";
                return 1;
            }
        }
        if ((op % 127) == 0) {
            if (!same_rows(index.all(),
                           ref_scan(ref, std::string_view(), ref.size()))) {
                std::cerr << "seed " << seed << " op " << op
                          << ": all mismatch\n";
                return 1;
            }
        }
    }

    if (!same_rows(index.all(),
                   ref_scan(ref, std::string_view(), ref.size()))) {
        std::cerr << "seed " << seed << ": final all\n";
        return 1;
    }
    if (index.size() != ref.size()) {
        std::cerr << "seed " << seed << ": final size\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    for (unsigned seed : {1u, 2u, 3u, 42u}) {
        if (run(seed, 2500) != 0) return 1;
        std::cout << "seed " << seed << " passed\n";
    }
    std::cout << "v2 randomized differential passed\n";
    return 0;
}
