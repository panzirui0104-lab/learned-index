// EBR primitive test: retire/try_gc with and without an active reader guard.

#include "olc/epoch.hpp"
#include "olc/group.hpp"
#include "olc/index.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main() {
    olc::EpochReclaimer epoch;

    auto make_group = [] {
        return new olc::Group(std::vector<olc::Group::Record>{});
    };

    olc::Group* first = make_group();
    epoch.retire(first);
    assert(epoch.pending() == 1);
    epoch.try_gc();
    assert(epoch.pending() == 0);

    olc::Group* second = make_group();
    epoch.retire(second);
    assert(epoch.pending() == 1);
    {
        olc::EpochReclaimer::Guard guard(epoch);
        epoch.try_gc();
        assert(epoch.pending() == 1);  // reader is active: no reclamation
    }
    epoch.try_gc();
    assert(epoch.pending() == 0);

    olc::Group* third = make_group();
    epoch.retire(third);
    epoch.flush();
    assert(epoch.pending() == 0);

    // Index-level smoke: writes retire Groups; reclaim() drains them when no
    // reader guard is active.
    olc::OlcIndex index;
    std::vector<olc::Pair> rows;
    for (int i = 0; i < 128; ++i) {
        rows.push_back({"a" + std::to_string(i), static_cast<olc::Value>(i)});
    }
    index.bulk_load(rows);
    for (int i = 0; i < 64; ++i) {
        index.insert("b" + std::to_string(i), static_cast<olc::Value>(i));
    }
    for (int i = 0; i < 32; ++i) {
        index.erase("a" + std::to_string(i));
    }
    index.reclaim();
    assert(index.pending_retired() == 0);
    assert(index.size() == 128 + 64 - 32);

    std::cout << "v3 EBR test passed\n";
    return 0;
}
