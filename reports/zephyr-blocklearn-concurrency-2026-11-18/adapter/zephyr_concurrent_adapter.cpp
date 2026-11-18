#include "zephyr_concurrent_adapter.h"

#include "benchmark/factories.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

using string_index_bench::v1::Descriptor;
using string_index_bench::v1::EraseResult;
using string_index_bench::v1::Index;
using string_index_bench::v1::InsertResult;
using string_index_bench::v1::MemoryUsage;
using string_index_bench::v1::Parameters;
using string_index_bench::v1::Row;
using string_index_bench::v1::UpdateResult;
using string_index_bench::v1::Value;

// The upstream BlockLearn adapter describes a single-threaded index. Its
// default find path (instrument == false) only reads immutable build state; the
// mutable Statistics counters are touched only when instrument is enabled.
// This wrapper exposes that read-only property to the benchmark and rejects
// instrument=1 so concurrent find cannot race on the counters.
class ConcurrentReadIndex final : public Index {
public:
    explicit ConcurrentReadIndex(std::unique_ptr<Index> inner)
        : inner_(std::move(inner)) {}

    Descriptor descriptor() const override {
        Descriptor result = inner_->descriptor();
        result.implementation += "-concurrent-read";
        result.capabilities.concurrent_find = true;
        result.adaptation_notes +=
            " Adapter-level read-only concurrency: the benchmark's parallel"
            " workload calls only const find(); bulk_load completes before"
            " threads start; writes are not declared concurrent-safe;"
            " instrument is forced off so mutable statistics are not touched.";
        return result;
    }

    void bulk_load(const std::vector<Row>& rows) override {
        inner_->bulk_load(rows);
    }

    std::optional<Value> find(std::string_view key) const override {
        return inner_->find(key);
    }

    InsertResult insert(std::string_view key, Value value) override {
        return inner_->insert(key, value);
    }

    UpdateResult update(std::string_view key, Value value) override {
        return inner_->update(key, value);
    }

    EraseResult erase(std::string_view key) override {
        return inner_->erase(key);
    }

    std::vector<Row> scan(std::string_view lower_bound,
                          std::size_t limit) const override {
        return inner_->scan(lower_bound, limit);
    }

    Value range_sum(std::string_view lower_bound,
                    std::string_view upper_bound) const override {
        return inner_->range_sum(lower_bound, upper_bound);
    }

    MemoryUsage memory_usage() const override {
        return inner_->memory_usage();
    }

private:
    std::unique_ptr<Index> inner_;
};

}  // namespace

std::unique_ptr<string_index_bench::v1::Index> make_zephyr_concurrent(
    const Parameters& parameters) {
    for (const auto& [name, value] : parameters) {
        if (name == "instrument" && value != "0") {
            throw std::invalid_argument(
                "instrument must be 0 for the read-only concurrent adapter");
        }
    }
    auto inner = make_block_adaptive_delta_shortdir_first(parameters);
    if (!inner) {
        throw std::runtime_error("Zephyr factory returned null");
    }
    return std::make_unique<ConcurrentReadIndex>(std::move(inner));
}
