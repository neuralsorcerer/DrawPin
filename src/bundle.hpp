// One read supplies a validated group of complete, bit-packed logical tiles.
#include <bitset>

class SectorFile {
    static constexpr size_t header = 16, max_members = 3;
    static size_t count(const fs::path& p, size_t tiles) {
        const auto bytes = fs::file_size(p / "residual.bundle");
        if (!bytes || bytes % PAGE || bytes / PAGE > tiles)
            fail("invalid bundle file length");
        return bytes / PAGE;
    }
    PageFile file;
    std::vector<uint16_t> directory;
    std::vector<uint8_t> expected;
    const RowMeta& meta;
    static uint16_t u16(const uint8_t* p) {
        return uint16_t(p[0]) | uint16_t(p[1]) << 8;
    }

  public:
    struct Member {
        uint16_t tile, offset;
    };
    struct Read {
        std::array<Member, max_members> members{};
        size_t size = 0, used = 0;
    };
    SectorFile(const fs::path& p, size_t tiles, bool direct, const RowMeta& m)
        : file(p / "residual.bundle", count(p, tiles), direct),
          directory(read_file<uint16_t>(p / "tile_to_sector.u16", tiles)),
          expected(count(p, tiles)), meta(m) {
        if (tiles != 1024)
            fail("unsupported tile geometry");
        for (const auto id : directory) {
            if (id >= expected.size() || ++expected[id] > max_members)
                fail("invalid tile directory");
        }
        if (std::any_of(expected.begin(), expected.end(), [](uint8_t n) { return !n; }))
            fail("empty sector");
    }
    uint16_t sector(size_t tile) const {
        return directory[tile];
    }
    size_t bytes() const {
        return 2 * directory.size() + expected.size();
    }
    size_t sectors() const {
        return expected.size();
    }
    size_t width(size_t tile) const {
        size_t n = 0;
        for (size_t i = tile * ROWS; i < (tile + 1) * ROWS; ++i)
            n += meta.exponent(i);
        return n * (D / 8);
    }
    Read describe(uint16_t id, const uint8_t* data, size_t bytes) const {
        if (id >= expected.size() || bytes < header - 4 || bytes > PAGE - 4)
            fail("invalid sector extent");
        const auto n = u16(data);
        if (n != expected[id] || data[8] || data[9] || data[10] || data[11])
            fail("invalid sector header");
        Read out;
        out.size = n;
        size_t at = header - 4;
        for (size_t k = 0; k < max_members; ++k) {
            const auto tile = u16(data + 2 + 2 * k);
            if (k >= n) {
                if (tile != 65535)
                    fail("invalid empty tile slot");
                continue;
            }
            if (tile >= directory.size() || directory[tile] != id)
                fail("tile/sector mismatch");
            for (size_t j = 0; j < k; ++j)
                if (out.members[j].tile == tile)
                    fail("duplicate tile");
            const auto len = width(tile);
            if (!len || at + len > bytes)
                fail("invalid packed tile extent");
            out.members[k] = {tile, uint16_t(at)};
            at += len;
        }
        out.used = at;
        return out;
    }
    Read get(uint16_t id, std::array<uint8_t, PAGE>& buf, uint64_t& calls) const {
        if (id >= expected.size())
            fail("sector outside file");
        file.get(id, buf.data(), calls);
        if (std::memcmp(buf.data(), "QBT1", 4))
            fail("invalid sector magic");
        auto out = describe(id, buf.data() + 4, PAGE - 4);
        for (size_t i = 0; i < out.size; ++i)
            out.members[i].offset += 4;
        return out;
    }
};
