#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
[[noreturn]] void fail(const char* message) {
    throw std::runtime_error(message);
}
#include "exact_exp.hpp"

int main() {
    std::uint64_t gap, offset, fraction;
    unsigned expected;
    unsigned rows = 0, fallbacks = 0, above = 0, below = 0;
    while (std::cin >> gap >> offset >> fraction >> expected) {
        // The two comparison cells are already disjoint from the threshold at
        // 64 bits according to the high-precision independent oracle.
        exact_exp::Bits a(0x0123456789abcdefULL), b(a.state);
        bool lazy_fallback = false, forced_fallback = false;
        bool got = exact_exp::accept(gap, offset, fraction, a, lazy_fallback);
        bool forced = exact_exp::accept(gap, offset, fraction, b, forced_fallback, true);
        if (got != bool(expected) || forced != bool(expected) || !forced_fallback)
            fail("native rejection comparison differs from boundary oracle");
        ++rows;
        fallbacks += lazy_fallback;
        below += bool(expected);
        above += !expected;
    }
    if (!rows || fallbacks != rows || above != below)
        fail("threshold test missed arithmetic path");
    std::cout << "{\"native_threshold_cases\":" << rows << ",\"natural_slow_paths\":" << fallbacks
              << ",\"below\":" << below << ",\"above\":" << above << "}\n";
}
