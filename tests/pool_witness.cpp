#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
constexpr unsigned PAGE = 4096;
[[noreturn]] void fail(const char* message) {
    throw std::runtime_error(message);
}
#include "sector_pool.hpp"

std::uint64_t check(PagePolicy policy, unsigned delay) {
    SectorPool pool(2, PoolConfig{policy, 0, false});
    std::array<std::uint16_t, 2> views{SectorPool::NONE, SectorPool::NONE};
    std::uint64_t charged = 0;
    auto request = [&](unsigned lane, std::uint16_t key) {
        pool.fill(lane, key, views[lane], charged, [&](auto& out, auto& syscalls) {
            out.fill(static_cast<std::uint8_t>(key));
            ++syscalls;
        });
        const auto* bytes = pool.view(views[lane]);
        if (!std::all_of(bytes, bytes + PAGE, [&](auto v) { return v == key; }))
            fail("witness wrong page bytes");
        pool.release(views[lane]);
    };
    request(0, 7);
    request(1, 9);
    pool.park(0, 7);
    request(1, 11);
    for (unsigned i = 0; i < delay; ++i)
        request(1, i % 2 ? 11 : 9);
    request(0, 7);
    if (!pool.clean() || charged != 4 + delay || charged != pool.logical ||
        pool.reads + pool.hit_idle + pool.hit_pinned != charged || pool.reads != pool.syscalls)
        fail("witness counter mismatch");
    return pool.reads;
}
int main() {
    std::cout << "{\"cases\":[";
    for (unsigned i = 0; i < 33; ++i) {
        auto lru = check(PagePolicy::Lease, i), dp = check(PagePolicy::Ticket, i);
        if (lru != 4 || dp != 3 + i)
            fail("unexpected counterexample");
        if (i)
            std::cout << ',';
        std::cout << "{\"delay\":" << i << ",\"logical\":" << 4 + i << ",\"lru\":" << lru
                  << ",\"drawpin\":" << dp << '}';
    }
    std::cout << "]}\n";
}
