#pragma once

#include "common.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <vector>

namespace olc {

// Context16 model: 16 rows x 258 cumulative probabilities, Q=2^47, MASS=2^16.
// Integer interval arithmetic keeps bucket() monotone, so group order equals
// byte order. The model only proposes a group; exact key comparison follows.
class Model {
public:
    static constexpr std::uint64_t Q = 1ULL << 47;
    static constexpr std::uint32_t MASS = 1U << 16;
    static constexpr std::size_t ROWS = 16;

    using Row = std::array<std::uint32_t, 258>;

    void train(const std::vector<Pair>& rows, std::size_t prefix) {
        cumulative_.assign(ROWS, Row{});

        std::vector<std::array<std::uint64_t, 257>> counts(ROWS);
        for (auto& row : counts) row.fill(1);  // smoothing for EOS and every byte

        for (const auto& pair : rows) {
            const std::string_view suffix =
                std::string_view(pair.key).substr(prefix);
            unsigned previous = 0;
            const std::size_t limit = std::min<std::size_t>(16, suffix.size());
            for (std::size_t p = 0; p < limit; ++p) {
                const unsigned d =
                    static_cast<unsigned char>(suffix[p]) + 1U;
                ++counts[row_for(p, previous)][d];
                previous = d;
            }
            if (suffix.size() <= 16) {
                ++counts[row_for(suffix.size(), previous)][0];
            }
        }

        for (std::size_t r = 0; r < ROWS; ++r) {
            std::uint64_t total = 0;
            for (auto count : counts[r]) total += count;

            std::array<std::uint32_t, 257> frequency{};
            std::uint32_t sum = 0;
            for (std::size_t d = 0; d < 257; ++d) {
                frequency[d] = static_cast<std::uint32_t>(
                    counts[r][d] * MASS / total);
                sum += frequency[d];
            }
            for (std::uint32_t d = 0; d < MASS - sum; ++d) {
                ++frequency[d];
            }

            cumulative_[r][0] = 0;
            for (std::size_t d = 0; d < 257; ++d) {
                cumulative_[r][d + 1] =
                    cumulative_[r][d] + frequency[d];
            }
            assert(cumulative_[r][257] == MASS);
        }
    }

    std::uint32_t group(std::string_view suffix,
                        std::uint32_t groups) const {
        assert(groups > 0);
        if (groups == 1) return 0;

        std::uint64_t lo = 0;
        std::uint64_t hi = Q;
        unsigned previous = 0;

        for (std::size_t p = 0; p < suffix.size(); ++p) {
            const unsigned ch = static_cast<unsigned char>(suffix[p]);
            step(cumulative_[row_for(p, previous)], ch + 1U, lo, hi);
            previous = ch + 1U;
            if (lo == hi || bucket(lo, groups) == bucket(hi, groups)) {
                break;
            }
        }
        return bucket(lo, groups);
    }

    std::size_t memory() const {
        return cumulative_.capacity() * sizeof(Row);
    }

private:
    static std::size_t row_for(std::size_t position,
                               unsigned previous) noexcept {
        return ((position & 3U) << 2U) | (previous & 3U);
    }

    static void step(const Row& cumulative, unsigned d,
                     std::uint64_t& lo, std::uint64_t& hi) noexcept {
        const std::uint64_t origin = lo;
        const std::uint64_t width = hi - lo;
        lo = origin + ((width * cumulative[d]) >> 16);
        hi = origin + ((width * cumulative[d + 1]) >> 16);
    }

    static std::uint32_t bucket(std::uint64_t x,
                                std::uint32_t groups) noexcept {
        const auto g = static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(x) * groups) >> 47);
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(groups - 1, g));
    }

    std::vector<Row> cumulative_;
};

}  // namespace olc
