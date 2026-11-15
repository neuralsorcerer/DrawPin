// Frozen CPU body and sector-backed output head for DrawPin.
#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace fs = std::filesystem;
static_assert(std::endian::native == std::endian::little,
              "DrawPin's frozen model and prompts are little endian");
static_assert(sizeof(size_t) == 8 && sizeof(float) == 4 && sizeof(double) == 8 &&
                  std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "DrawPin requires a 64-bit IEEE-754 host");
constexpr size_t D = 128, PAGE = 4096, ROWS = 16;
constexpr uint32_t N_BITS = 23, N_MASK = (1u << N_BITS) - 1, NO = UINT32_MAX;

[[noreturn]] void fail(const std::string& s) {
    throw std::runtime_error(s);
}
[[noreturn]] void io_fail(const std::string& s) {
    const int e = errno;
    fail(s + ": " + std::strerror(e));
}

size_t number(const char* s) {
    const std::string t(s);
    if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos)
        fail("expected nonnegative integer");
    return std::stoull(t);
}
template <typename T> std::vector<T> read_file(const fs::path& p, size_t count = 0) {
    auto n = fs::file_size(p);
    if (!n || n % sizeof(T) || (count && n / sizeof(T) != count))
        fail("invalid length: " + p.string());
    std::vector<T> a(n / sizeof(T));
    std::ifstream in(p, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(a.data()), n))
        fail("cannot read: " + p.string());
    return a;
}
struct Rank {
    int64_t score = std::numeric_limits<int64_t>::min();
    uint32_t row = std::numeric_limits<uint32_t>::max();
};
using Clock = std::chrono::steady_clock;
uint64_t ns(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}
uint64_t isqrt(uint64_t n) {
    // The row-norm geometry bounds n well below the 64-bit limit.
    auto r = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
    while (r * r > n)
        --r;
    while ((r + 1) * (r + 1) <= n)
        ++r;
    return r;
}
int64_t half_floor(int64_t n) {
    return n >= 0 ? n / 2 : -((-n + 1) / 2);
}

struct Query {
    const int8_t* raw;
    alignas(32) std::array<int16_t, D> x;
    int64_t sum = 0, pos = 0, sq = 0;
    explicit Query(const int8_t* b) : raw(b) {
        for (size_t j = 0; j < D; ++j) {
            x[j] = b[j];
            sum += x[j];
            pos += std::max<int>(x[j], 0);
            sq += x[j] * x[j];
        }
    }
};
enum class Dot { I16, I8, U8, Q4 };
#ifdef __AVX2__
int32_t reduce32(__m256i a) {
    auto p = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    p = _mm_hadd_epi32(p, p);
    p = _mm_hadd_epi32(p, p);
    return _mm_cvtsi128_si32(p);
}
#endif
template <Dot K> int32_t dot(const void* p, const Query& x) {
#ifdef __AVX2__
    auto acc = _mm256_setzero_si256();
    if constexpr (K == Dot::Q4) {
        const auto mask = _mm_set1_epi8(15);
        const auto ones = _mm256_set1_epi16(1);
        for (size_t j = 0; j < D; j += 32) {
            const auto z = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(static_cast<const uint8_t*>(p) + j / 2));
            const auto low = _mm_and_si128(z, mask),
                       high = _mm_and_si128(_mm_srli_epi16(z, 4), mask);
            const auto a =
                _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high));
            const auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.raw + j));
            // No int16 saturation: two products of u4 and s8 have magnitude <= 3840.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
        }
        return reduce32(acc) - 8 * x.sum;
    } else {
        for (size_t j = 0; j < D; j += 16) {
            __m256i a;
            if constexpr (K == Dot::I16)
                a = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(static_cast<const int16_t*>(p) + j));
            else if constexpr (K == Dot::I8)
                a = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(static_cast<const int8_t*>(p) + j)));
            else
                a = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(static_cast<const uint8_t*>(p) + j)));
            const auto b = _mm256_load_si256(reinterpret_cast<const __m256i*>(x.x.data() + j));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
        }
        return reduce32(acc);
    }
#else
    int32_t acc = 0;
    for (size_t j = 0; j < D; ++j) {
        int a;
        if constexpr (K == Dot::I16)
            a = static_cast<const int16_t*>(p)[j];
        else if constexpr (K == Dot::I8)
            a = static_cast<const int8_t*>(p)[j];
        else if constexpr (K == Dot::U8)
            a = static_cast<const uint8_t*>(p)[j];
        else
            a = ((static_cast<const uint8_t*>(p)[j / 2] >> (4 * (j % 2))) & 15) - 8;
        acc += a * x.x[j];
    }
    return acc;
#endif
}

uint64_t spool_preads = 0;
class PageFile {
    int fd = -1;

  public:
    PageFile(const fs::path& path, size_t pages, bool direct) {
        if (fs::file_size(path) != pages * PAGE)
            fail("invalid residual file length");
        int flags = O_CLOEXEC | O_RDONLY;
#ifdef O_DIRECT
        if (direct)
            flags |= O_DIRECT;
#else
        if (direct)
            fail("O_DIRECT unavailable");
#endif
        fd = ::open(path.c_str(), flags);
        if (fd < 0)
            io_fail("open " + path.string());
    }
    ~PageFile() {
        if (fd >= 0)
            ::close(fd);
    }
    PageFile(const PageFile&) = delete;
    PageFile& operator=(const PageFile&) = delete;
    void get(uint32_t page, uint8_t* dest, uint64_t& calls) const {
        size_t at = 0;
        while (at < PAGE) {
            ++calls;
            ++spool_preads;
            auto n = ::pread(fd, dest + at, PAGE - at, off_t(page) * PAGE + at);
            if (n < 0 && errno == EINTR)
                continue;
            if (n < 0)
                io_fail("pread");
            if (n == 0)
                fail("short residual page");
            at += n;
        }
    }
};

struct WinnerRow {
    std::array<int16_t, D> values{};
    size_t position = 0;
};

constexpr size_t CONTEXT = 6;
using Embedding = std::array<int16_t, D>;
using Prompt = std::array<Embedding, CONTEXT>;
struct KnownRows {
    const Prompt& rows;
    const std::array<size_t, CONTEXT>& pos;
    bool exclude;
    bool forbid_context = false;
    uint32_t mask(uint32_t page) const {
        uint32_t z = 0;
        if (exclude)
            for (size_t i : pos)
                if (i / ROWS == page)
                    z |= 1u << (i % ROWS);
        return z;
    }
};

using QueryBytes = std::array<int8_t, D>;
#include "fused_dot.hpp"
#include "row_meta.hpp"
#include "refine16.hpp"
#include "bundle.hpp"

class Head {
    template <size_t> friend class Draw;
    size_t n, pages;
    std::vector<uint8_t> q;
    std::vector<int8_t> shift;
    RowMeta meta;
    std::vector<uint32_t> rows;
    Refinement refine;
    SectorFile residual;
    alignas(PAGE) std::array<uint8_t, PAGE> buf{};

    static int32_t bias_at(size_t) {
        return 0;
    }
    int64_t g(size_t i, const Query& x) const {
        return int64_t(dot<Dot::Q4>(q.data() + i * (D / 2), x)) * (1LL << (meta.exponent(i))) +
               int64_t(shift[i]) * x.sum + bias_at(i);
    }

    void materialize_row(size_t i, std::array<int16_t, D>& dst, const uint8_t* residual_row) const {
        const auto B = 1 << (meta.exponent(i));
        for (size_t j = 0; j < D; ++j) {
            const auto code = (q[i * (D / 2) + j / 2] >> (4 * (j % 2))) & 15;
            dst[j] = int16_t(B * (int(code) - 8) + shift[i] + residual_row[j]);
        }
    }

  public:
    size_t sectors() const {
        return residual.sectors();
    }
    size_t locate(uint32_t id) const {
        auto p = std::find(rows.begin(), rows.end(), id);
        if (p == rows.end())
            fail("invalid prompt token");
        return size_t(p - rows.begin());
    }

    template <size_t N>
    size_t prefill(const uint32_t* ids, std::array<std::array<int16_t, D>, N>& out, uint64_t& calls,
                   std::array<size_t, N>* keep = nullptr) {
        std::array<size_t, N> positions{};
        std::array<bool, N> done{};
        for (size_t j = 0; j < N; ++j)
            positions[j] = locate(ids[j]);
        if (keep)
            *keep = positions;
        size_t reads = 0;
        for (size_t j = 0; j < N; ++j)
            if (!done[j]) {
                const auto sector = residual.sector(positions[j] / ROWS);
                const auto got = residual.get(sector, buf, calls);
                ++reads;
                for (size_t m = 0; m < got.size; ++m) {
                    const auto a = got.members[m];
                    for (size_t k = j; k < N; ++k)
                        if (!done[k] && positions[k] / ROWS == a.tile) {
                            const auto i = positions[k];
                            const uint8_t* p = buf.data() + a.offset;
                            for (size_t t = size_t(a.tile) * ROWS; t < i; ++t)
                                p += (meta.exponent(t)) * (D / 8);
                            std::array<uint8_t, D> r{};
                            fused_unpack(p, r.data(), meta.exponent(i));
                            materialize_row(i, out[k], r.data());
                            done[k] = true;
                        }
                }
                if (!done[j])
                    fail("missing prefill tile");
            }
        return reads;
    }

    Head(size_t v, const fs::path& p, bool direct, const fs::path& extra)
        : n(v), pages(v / ROWS), q(read_file<uint8_t>(p / "quotient.bin", n * (D / 2))),
          shift(read_file<int8_t>(p / "shift.i8", n)), meta(p, n),
          rows(read_file<uint32_t>(p / "rows.u32", n)), refine(extra, meta, pages),
          residual(p, pages, direct, meta) {
        verify_zero_bias(p / "bias.i32", n);
        constexpr size_t vocabulary = 16384;
        if (n != vocabulary)
            fail("unexpected frozen vocabulary size");
        std::bitset<vocabulary> seen;
        for (auto token : rows) {
            if (token >= n || seen.test(token))
                fail("invalid frozen vocabulary permutation");
            seen.set(token);
        }
        for (size_t j = 0; j < meta.size(); ++j) {
            const auto u = meta[j];
            const auto e = u >> N_BITS;
            if (e > 8)
                fail("invalid radix");
            const auto r = (1u << e) - 1;
            if ((u & N_MASK) > D * r * r)
                fail("invalid residual norm");
        }
    }
};

class Body {
    std::vector<float> rec, offset;
    double scale;

  public:
    explicit Body(const fs::path& m)
        : rec(read_file<float>(m / "recurrent.f32", D * D)),
          offset(read_file<float>(m / "hidden_bias.f32", D)),
          scale(read_file<double>(m / "weight_scale.f64", 1)[0]) {
        if (!std::isfinite(scale) || scale <= 0)
            fail("invalid weight scale");
    }
    double weight_scale() const {
        return scale;
    }
    std::array<int8_t, D> compute(const Prompt& p) const {
        std::array<double, D> h{};
        for (const auto& token : p) {
            std::array<double, D> next{};
            for (size_t i = 0; i < D; ++i) {
                const auto a = h[i];
                for (size_t j = 0; j < D; ++j)
                    next[j] += a * double(rec[i * D + j]);
            }
            for (size_t j = 0; j < D; ++j)
                h[j] = std::tanh(3.0 * (scale * token[j]) + next[j] + offset[j]);
        }
        std::array<int8_t, D> q{};
        for (size_t j = 0; j < D; ++j)
            q[j] = int8_t(std::nearbyint(127 * h[j]));
        return q;
    }
};

#include "ledger.hpp"

#include "fold_config.hpp"
#include "stride_config.hpp"
#include "sector_pool.hpp"
#include "lease_draw.hpp"
#include "lease_receipts.hpp"
#include "lease_driver.hpp"
