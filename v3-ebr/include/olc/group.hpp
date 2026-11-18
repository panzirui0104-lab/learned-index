#pragma once

#include "common.hpp"
#include "version.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace olc {

// A Group is an immutable sorted suffix list; only each record's value is
// mutable (atomic). Insert/erase publish a new Group, updates write the value
// in place. This is the block-level "grow/shrink publishes a new node" part of
// ART-OLC.
class Group {
public:
    struct Record {
        std::string suffix;
        mutable std::atomic<Value> value;

        Record(std::string s, Value v)
            : suffix(std::move(s)), value(v) {}

        Record(const Record& other)
            : suffix(other.suffix),
              value(other.value.load(std::memory_order_acquire)) {}

        Record& operator=(const Record& other) {
            suffix = other.suffix;
            value.store(other.value.load(std::memory_order_acquire),
                        std::memory_order_release);
            return *this;
        }

        Record(Record&& other) noexcept
            : suffix(std::move(other.suffix)),
              value(other.value.load(std::memory_order_acquire)) {}

        Record& operator=(Record&& other) noexcept {
            suffix = std::move(other.suffix);
            value.store(other.value.load(std::memory_order_acquire),
                        std::memory_order_release);
            return *this;
        }
    };

    explicit Group(std::vector<Record> records)
        : records_(std::move(records)) {
        std::sort(records_.begin(), records_.end(),
                  [](const Record& a, const Record& b) {
                      return compare(a.suffix, b.suffix) < 0;
                  });
    }

    Group(const Group&) = delete;
    Group& operator=(const Group&) = delete;

    const Record* find(std::string_view suffix) const {
        const auto it = std::lower_bound(
            records_.begin(), records_.end(), suffix,
            [](const Record& record, std::string_view key) {
                return compare(record.suffix, key) < 0;
            });
        if (it == records_.end() || compare(it->suffix, suffix) != 0) {
            return nullptr;
        }
        return &*it;
    }

    const std::vector<Record>& records() const { return records_; }

    bool obsolete() const { return version_.obsolete(); }
    void mark_obsolete() const { version_.mark_obsolete(); }

    std::size_t memory() const {
        std::size_t bytes = sizeof(*this) +
                            records_.capacity() * sizeof(Record);
        for (const auto& record : records_) {
            bytes += record.suffix.capacity();
        }
        return bytes;
    }

private:
    std::vector<Record> records_;
    mutable Version version_;
};

}  // namespace olc
