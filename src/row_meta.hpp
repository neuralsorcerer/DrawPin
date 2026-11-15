#pragma once
// This file is included after the inherited dimension constants and file loader.
constexpr unsigned norm_ceiling(unsigned r) {
    return unsigned(D) * r * r;
}
template <unsigned Bins> uint32_t decode_norm(unsigned k, unsigned r) {
    // Every centered residual is odd when r>0. For D=128 its squared
    // norm is a multiple of eight, allowing a conservative rounded endpoint.
    static_assert(D % 8 == 0 && uint64_t(Bins) * norm_ceiling(255) / 8 <= UINT32_MAX);
    return 8u * ((k * (norm_ceiling(r) / 8u)) / Bins);
}
class RowMeta {
    std::vector<uint32_t> wide;
    std::vector<uint8_t> byte;

  public:
    RowMeta(const fs::path& p, size_t n) {
        if (fs::exists(p / "metadata.u8")) {
            byte = read_file<uint8_t>(p / "metadata.u8", n);
            for (uint8_t b : byte)
                if ((b >> 5) > 4)
                    fail("invalid byte metadata exponent");
        } else
            wide = read_file<uint32_t>(p / "metadata.u32", n);
    }
    bool compact() const {
        return !byte.empty();
    }
    uint8_t code8(size_t i) const {
        return byte[i];
    }
    size_t size() const {
        return byte.empty() ? wide.size() : byte.size();
    }
    size_t bytes() const {
        return wide.size() * sizeof(uint32_t) + byte.size();
    }
    unsigned exponent(size_t i) const {
        return byte.empty() ? wide[i] >> N_BITS : 4u + (byte[i] >> 5);
    }
    unsigned norm(size_t i) const {
        if (byte.empty())
            return wide[i] & N_MASK;
        const unsigned b = byte[i], r = (1u << (4u + (b >> 5))) - 1u;
        return decode_norm<31>(b & 31u, r);
    }
    uint32_t operator[](size_t i) const {
        if (byte.empty())
            return wide[i];
        const unsigned b = byte[i], e = 4u + (b >> 5), r = (1u << e) - 1;
        return (e << N_BITS) | decode_norm<31>(b & 31u, r);
    }
};
