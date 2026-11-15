// Same row widths as Slices. 32-lane tiles use a nibble stream and bit planes.
#include <cstring>

struct FusedDots {
    int32_t quotient, residual;
};

#ifdef __AVX2__
inline __m256i spread_nibbles(const uint8_t* in) {
    const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
    const auto m = _mm_set1_epi8(15);
    const auto a = _mm_and_si128(v, m), b = _mm_and_si128(_mm_srli_epi16(v, 4), m);
    return _mm256_set_m128i(_mm_unpackhi_epi8(a, b), _mm_unpacklo_epi8(a, b));
}
inline __m256i spread_plane(const uint8_t* in) {
    int32_t word;
    std::memcpy(&word, in, 4);
    const auto s = _mm256_shuffle_epi8(
        _mm256_set1_epi32(word), _mm256_setr_epi64x(0x0000000000000000LL, 0x0101010101010101LL,
                                                    0x0202020202020202LL, 0x0303030303030303LL));
    const auto mask = _mm256_set1_epi64x(-9223372036854775807LL + 0x0040201008040200LL);
    // mask bytes from low to high: 01,02,04,08,10,20,40,80.
    return _mm256_cmpeq_epi8(_mm256_and_si256(s, mask), mask);
}
template <unsigned E> inline __m256i read_tile(const uint8_t* in) {
    if constexpr (E == 8)
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in));
    else {
        auto v = _mm256_setzero_si256();
        constexpr unsigned start = E >= 4 ? 4 : 0;
        if constexpr (E >= 4)
            v = spread_nibbles(in);
        for (unsigned k = start; k < E; ++k) {
            const auto p = spread_plane(in + 4 * k);
            v = _mm256_or_si256(v, _mm256_and_si256(p, _mm256_set1_epi8(char(1u << k))));
        }
        return v;
    }
}
inline __m256i dot8_groups(__m256i a, __m256i b) {
    // Call only for unsigned a <= 127. The two-product sum cannot saturate.
    return _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1));
}
template <unsigned E>
inline FusedDots fused_pair_impl(const uint8_t* q, const uint8_t* r, const Query& x) {
    auto qa = _mm256_setzero_si256(), ra = _mm256_setzero_si256();
    for (size_t j = 0; j < D; j += 32) {
        const auto x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.raw + j));
        qa = _mm256_add_epi32(qa, dot8_groups(spread_nibbles(q + j / 2), x8));
        if constexpr (E == 8) {
            const auto r0 =
                _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r + j)));
            const auto r1 =
                _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r + j + 16)));
            const auto x0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(x.x.data() + j));
            const auto x1 =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(x.x.data() + j + 16));
            ra = _mm256_add_epi32(ra, _mm256_madd_epi16(r0, x0));
            ra = _mm256_add_epi32(ra, _mm256_madd_epi16(r1, x1));
        } else if constexpr (E > 0) {
            ra = _mm256_add_epi32(ra, dot8_groups(read_tile<E>(r + (j / 8) * E), x8));
        }
    }
    return {int32_t(reduce32(qa) - 8 * x.sum), reduce32(ra)};
}

template <unsigned E> inline void put_tile(uint8_t* out, __m256i row) {
    if constexpr (E < 8) {
        const auto forbidden = _mm256_and_si256(row, _mm256_set1_epi8(char(~((1u << E) - 1u))));
        if (!_mm256_testz_si256(forbidden, forbidden))
            fail("residual outside fused width");
    }
    if constexpr (E == 8)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), row);
    else {
        constexpr unsigned start = E >= 4 ? 4 : 0;
        if constexpr (E >= 4) {
            const auto low = _mm256_and_si256(row, _mm256_set1_epi8(15));
            const auto words = _mm256_maddubs_epi16(low, _mm256_set1_epi16(0x1001));
            const auto bytes =
                _mm_packus_epi16(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out), bytes);
        }
        for (unsigned k = start; k < E; ++k) {
            const auto p = _mm256_sll_epi16(row, _mm_cvtsi32_si128(int(7 - k)));
            const uint32_t bits = uint32_t(_mm256_movemask_epi8(p));
            std::memcpy(out + 4 * k, &bits, 4);
        }
    }
}
template <unsigned E> inline void fused_pack_impl(uint8_t* out, const uint8_t* in) {
    for (size_t j = 0; j < D; j += 32)
        put_tile<E>(out + (j / 8) * E,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + j)));
}
template <unsigned E> inline void fused_unpack_impl(const uint8_t* in, uint8_t* out) {
    for (size_t j = 0; j < D; j += 32)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), read_tile<E>(in + (j / 8) * E));
}
#else
template <unsigned E> inline unsigned fused_cell(const uint8_t* in, size_t j) {
    if constexpr (E == 8)
        return in[j];
    const auto tile = in + (j / 32) * 4 * E;
    const unsigned n = j % 32;
    unsigned a = 0;
    constexpr unsigned start = E >= 4 ? 4 : 0;
    if constexpr (E >= 4)
        a = (tile[n / 2] >> (4 * (n % 2))) & 15;
    for (unsigned k = start; k < E; ++k)
        a |= ((tile[4 * k + n / 8] >> (n % 8)) & 1u) << k;
    return a;
}
template <unsigned E>
inline FusedDots fused_pair_impl(const uint8_t* q, const uint8_t* r, const Query& x) {
    int32_t a = 0, b = 0;
    for (size_t j = 0; j < D; ++j) {
        a += (int((q[j / 2] >> (4 * (j % 2))) & 15) - 8) * x.x[j];
        b += int(fused_cell<E>(r, j)) * x.x[j];
    }
    return {a, b};
}
template <unsigned E> inline void fused_pack_impl(uint8_t* out, const uint8_t* in) {
    if constexpr (E == 8)
        std::copy_n(in, D, out);
    else {
        std::fill_n(out, D * E / 8, 0);
        for (size_t j = 0; j < D; ++j) {
            if (in[j] >= (1u << E))
                fail("residual outside fused width");
            auto* tile = out + (j / 32) * 4 * E;
            const unsigned n = j % 32;
            constexpr unsigned start = E >= 4 ? 4 : 0;
            if constexpr (E >= 4)
                tile[n / 2] |= uint8_t((in[j] & 15) << (4 * (n % 2)));
            for (unsigned k = start; k < E; ++k)
                tile[4 * k + n / 8] |= uint8_t(((in[j] >> k) & 1u) << (n % 8));
        }
    }
}
template <unsigned E> inline void fused_unpack_impl(const uint8_t* in, uint8_t* out) {
    for (size_t j = 0; j < D; ++j)
        out[j] = uint8_t(fused_cell<E>(in, j));
}
#endif

#define QBF_WIDTHS(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8)
inline FusedDots fused_pair(const uint8_t* q, const uint8_t* r, const Query& x, unsigned e) {
#define QBF_SCORE(E)                                                                               \
    case E:                                                                                        \
        return fused_pair_impl<E>(q, r, x);
    switch (e) { QBF_WIDTHS(QBF_SCORE) default : fail("invalid fused width"); }
#undef QBF_SCORE
}
inline uint8_t* fused_pack(uint8_t* out, const uint8_t* in, unsigned e) {
#define QBF_PACK(E)                                                                                \
    case E:                                                                                        \
        fused_pack_impl<E>(out, in);                                                               \
        break;
    switch (e) { QBF_WIDTHS(QBF_PACK) default : fail("invalid fused width"); }
#undef QBF_PACK
    return out + e * (D / 8);
}
inline const uint8_t* fused_unpack(const uint8_t* in, uint8_t* out, unsigned e) {
#define QBF_UNPACK(E)                                                                              \
    case E:                                                                                        \
        fused_unpack_impl<E>(in, out);                                                             \
        break;
    switch (e) { QBF_WIDTHS(QBF_UNPACK) default : fail("invalid fused width"); }
#undef QBF_UNPACK
    return in + e * (D / 8);
}
#undef QBF_WIDTHS
