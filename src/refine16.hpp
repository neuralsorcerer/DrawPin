// Exact resident high-bit prefixes for a bias-free paged head.
#include <bit>

inline int32_t plane_dot(const uint8_t* bits, const Query& x) {
#ifdef __AVX2__
    const auto sel = _mm256_setr_epi32(0, 0, 0x01010101, 0x01010101, 0x02020202, 0x02020202,
                                       0x03030303, 0x03030303);
    const auto mask = _mm256_set1_epi64x(0x8040201008040201ULL);
    const auto one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    auto acc = _mm256_setzero_si256();
    for (size_t j = 0; j < D; j += 32) {
        uint32_t packed;
        std::memcpy(&packed, bits + j / 8, sizeof(packed));
        const auto z = _mm256_shuffle_epi8(_mm256_set1_epi32(std::bit_cast<int32_t>(packed)), sel);
        const auto keep = _mm256_cmpeq_epi8(_mm256_and_si256(z, mask), mask);
        const auto q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.raw + j));
        const auto lanes = _mm256_maddubs_epi16(one8, _mm256_and_si256(q, keep));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lanes, one16));
    }
    return reduce32(acc);
#else
    int32_t a = 0;
    for (size_t j = 0; j < D; ++j)
        if ((bits[j / 8] >> (j % 8)) & 1u)
            a += x.x[j];
    return a;
#endif
}

inline void verify_zero_bias(const fs::path& p, size_t count) {
    if (fs::file_size(p) != count * sizeof(int32_t))
        fail("invalid frozen bias length");
    std::ifstream f(p, std::ios::binary);
    std::array<int32_t, 256> z{};
    for (size_t k = 0; k < count;) {
        const auto n = std::min(count - k, z.size());
        if (!f.read(reinterpret_cast<char*>(z.data()), n * sizeof(int32_t)))
            fail("invalid frozen bias");
        if (std::any_of(z.begin(), z.begin() + n, [](int32_t a) { return a != 0; }))
            fail("refinement requires zero output bias");
        k += n;
    }
}

class Refinement {
    static constexpr uint32_t PLANE_MASK = 4095;
    std::vector<uint32_t> norm;
    std::vector<uint8_t> norm8;
    size_t norm_count() const {
        return norm.empty() ? norm8.size() : norm.size();
    }
    unsigned suffix_norm(size_t k, unsigned r) const {
        return norm.empty() ? decode_norm<255>(norm8[k], r) : norm[k];
    }
    std::vector<uint16_t> mask, base, entry;
    std::vector<uint8_t> bits;
    std::vector<uint64_t> present;
    std::vector<uint16_t> prefix_count;
    size_t dir_page(size_t page) const {
        if (present.empty())
            return page;
        const size_t word = page / 64;
        const uint64_t bit = 1ULL << (page % 64), v = present[word];
        if (!(v & bit))
            return mask.size();
        return size_t(prefix_count[word]) + std::popcount(v & (bit - 1));
    }

  public:
    Refinement(const fs::path& p, const RowMeta& meta, size_t pages) {
        if (p.empty())
            return;
        size_t slots = pages;
        if (fs::exists(p / "present.u64")) {
            if (pages != 1024)
                fail("unexpected sparse refinement geometry");
            present = read_file<uint64_t>(p / "present.u64", pages / 64);
            prefix_count = read_file<uint16_t>(p / "prefix.u16", pages / 64);
            slots = 0;
            for (size_t j = 0; j < present.size(); ++j) {
                if (prefix_count[j] != slots)
                    fail("invalid sparse prefix counts");
                slots += std::popcount(present[j]);
            }
            if (!slots)
                fail("empty sparse refinement");
        }
        mask = read_file<uint16_t>(p / "mask.u16", slots);
        base = read_file<uint16_t>(p / "base.u16", slots);
        if (!present.empty() && std::find(mask.begin(), mask.end(), 0) != mask.end())
            fail("empty active sparse tile");
        entry = read_file<uint16_t>(p / "entry.u16");
        if (fs::exists(p / "norm.u8"))
            norm8 = read_file<uint8_t>(p / "norm.u8");
        else
            norm = read_file<uint32_t>(p / "norm.u32");
        bits = read_file<uint8_t>(p / "planes.u8", norm_count() * (D / 8));
        if (bytes() > meta.size() * 4)
            fail("resident refinement exceeds replaced table");
        size_t a = 0, b = 0;
        for (size_t page = 0; page < pages; ++page) {
            const auto slot = dir_page(page);
            if (slot == mask.size())
                continue;
            if (base[slot] != a)
                fail("invalid refinement page directory");
            for (size_t k = 0; k < ROWS; ++k)
                if ((mask[slot] >> k) & 1u) {
                    if (a >= entry.size())
                        fail("invalid refinement row directory");
                    const auto c = entry[a++];
                    const unsigned h = levels(c), e = meta.exponent(page * ROWS + k);
                    if (!h || e > 8 || h > e || first(c) != b || b + h > norm_count())
                        fail("invalid refinement extent");
                    for (unsigned j = 1; j <= h; ++j) {
                        const unsigned r = (1u << (e - j)) - 1;
                        if (suffix_norm(b++, r) > D * r * r)
                            fail("invalid refinement norm");
                    }
                }
        }
        if (a != entry.size() || b != norm_count())
            fail("extra refinement data");
    }
    size_t bytes() const {
        return present.size() * 8 + prefix_count.size() * 2 + mask.size() * 2 + base.size() * 2 +
               entry.size() * 2 + norm.size() * 4 + norm8.size() + bits.size();
    }
    uint16_t code(uint32_t page, size_t k) const {
        if (mask.empty())
            return 0;
        const auto slot = dir_page(page);
        if (slot == mask.size())
            return 0;
        const uint32_t m = mask[slot], b = uint32_t(1) << k;
        return (m & b) ? entry[base[slot] + std::popcount(m & (b - 1))] : 0;
    }
    static unsigned levels(uint16_t c) {
        return c >> 12;
    }
    static unsigned first(uint16_t c) {
        return c & PLANE_MASK;
    }
    const uint8_t* plane(unsigned k) const {
        return bits.data() + k * (D / 8);
    }
    int64_t exact_dot(uint16_t c, const Query& x) const {
        int64_t out = 0;
        for (unsigned j = 0; j < levels(c); ++j)
            out = 2 * out + plane_dot(plane(first(c) + j), x);
        return out;
    }
    int64_t upper(uint16_t c, unsigned exponent, const Query& x, int64_t old, int64_t stop,
                  size_t& evaluated) const {
        int64_t partial = 0, high = old;
        for (unsigned h = 1; h <= levels(c); ++h) {
            const unsigned k = first(c) + h - 1;
            const int64_t b = int64_t(1) << (exponent - h), r = b - 1;
            partial += b * plane_dot(plane(k), x);
            ++evaluated;
            const auto box = r * x.pos, a = r * x.sum;
            const auto n = uint64_t(suffix_norm(k, unsigned(r))) * uint64_t(x.sq);
            const auto u = std::min<int64_t>(box, half_floor(a + int64_t(isqrt(n))));
            high = std::min(high, partial + u);
            // Bounds below a proven-losing cutoff are never used for promotion.
            if (high < stop)
                break;
        }
        return high;
    }
    void unpack_exact(uint16_t c, std::array<uint8_t, D>& u) const {
        u.fill(0);
        for (unsigned h = 0; h < levels(c); ++h) {
            const auto* m = plane(first(c) + h);
            for (size_t j = 0; j < D; ++j)
                u[j] = uint8_t(2 * u[j] + ((m[j / 8] >> (j % 8)) & 1u));
        }
    }
};
