#pragma once
#include <gmpxx.h>
#include "exp_grid.hpp"

namespace exact_exp {
using wide = __uint128_t;
constexpr unsigned MASS_BITS = 48, SHIFT = GRID_PRECISION - MASS_BITS;
constexpr uint64_t F = uint64_t(1) << GRID_PRECISION, MASS_ONE = uint64_t(1) << MASS_BITS;
static_assert(GRID_PRECISION == 62 && sizeof(unsigned long) == 8);
struct Enclosure {
    uint64_t lo, hi;
};
inline Enclosure fast(uint64_t d) {
    const auto b = d / GRID_STEP, r = d % GRID_STEP;
    const auto a = b < EXP_BLOCK.size() ? EXP_BLOCK[b] : 0, c = EXP_REMAINDER[r];
    return {uint64_t(wide(a) * c >> GRID_PRECISION),
            uint64_t((wide(a + (b != 0)) * (c + (r != 0)) + F - 1) >> GRID_PRECISION)};
}
inline uint64_t envelope(uint64_t d) {
    return (fast(d).hi + (1ULL << SHIFT) - 1) >> SHIFT;
}
struct Bits {
    uint64_t state;
    explicit Bits(uint64_t s) : state(s) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) {
        if (!n)
            fail("empty ticket envelope");
        const uint64_t cut = (uint64_t(0) - n) % n;
        for (;;) {
            const auto x = next();
            if (x >= cut)
                return x % n;
        }
    }
};
inline mpz_class ceil_div(const mpz_class& x, const mpz_class& y) {
    mpz_class z;
    mpz_cdiv_q(z.get_mpz_t(), x.get_mpz_t(), y.get_mpz_t());
    return z;
}
// Alternate terms enclose exp(-x) for x<=1/2. Squaring uses outward rounding.
inline std::pair<mpz_class, mpz_class> precise(uint64_t d, unsigned long p) {
    static const mpz_class bn(BETA_NUM), bd(BETA_DEN);
    const mpz_class unit = mpz_class(1) << p;
    if (!d)
        return {unit, unit};
    const mpz_class num = bn * static_cast<unsigned long>(d);
    mpz_class den = bd;
    unsigned n = 0;
    while (2 * num > den) {
        den <<= 1;
        ++n;
    }
    mpz_class lo = unit, hi = unit, tl = unit, th = unit;
    for (unsigned long k = 1;; ++k) {
        const mpz_class div = den * k;
        tl = (tl * num) / div;
        th = ceil_div(th * num, div);
        if (k & 1u) {
            lo -= th;
            hi -= tl;
        } else {
            lo += tl;
            hi += th;
        }
        if (th <= 1) {
            lo -= 1;
            hi += 1;
            break;
        }
    }
    if (lo < 0)
        lo = 0;
    if (hi > unit)
        hi = unit;
    while (n--) {
        lo = (lo * lo) >> p;
        hi = ceil_div(hi * hi, unit);
    }
    return {lo, hi};
}
inline bool accept(uint64_t d, uint64_t j, uint64_t u, Bits& rng, bool& fallback,
                   bool force = false) {
    if (j >= MASS_ONE)
        fail("ticket offset outside grid");
    fallback = false;
    if (!force) {
        const auto a = fast(d);
        const auto k = (wide(j) << 64) | wide(u);
        constexpr unsigned up = 64 - SHIFT;
        if (k + 1 <= (wide(a.lo) << up))
            return true;
        if (k >= (wide(a.hi) << up))
            return false;
    }
    fallback = true;
    mpz_class ticket =
        (mpz_class(static_cast<unsigned long>(j)) << 64) + static_cast<unsigned long>(u);
    for (unsigned long n = 64;; n += 64) {
        const auto [lo, hi] = precise(d, n + 64);
        const mpz_class t = ticket << (64 - MASS_BITS);
        if (t + (mpz_class(1) << (64 - MASS_BITS)) <= lo)
            return true;
        if (t >= hi)
            return false;
        ticket <<= 64;
        ticket += static_cast<unsigned long>(rng.next());
    }
}
} // namespace exact_exp
