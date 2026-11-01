#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace olc {

using Value = std::uint64_t;

struct Pair {
    std::string key;
    Value value;

    Pair() = default;
    Pair(std::string k, Value v) : key(std::move(k)), value(v) {}
};

// Unsigned-byte lexicographic comparison, the same contract as the base index.
inline int compare(std::string_view a, std::string_view b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    const int c = n ? std::memcmp(a.data(), b.data(), n) : 0;
    return c ? c : (a.size() < b.size() ? -1 : (a.size() > b.size() ? 1 : 0));
}

struct Less {
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const noexcept {
        return compare(a, b) < 0;
    }
};

}  // namespace olc
