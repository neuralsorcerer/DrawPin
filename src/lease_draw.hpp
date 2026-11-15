#pragma once
#include <bit>
#include "exact_exp.hpp"

constexpr size_t V = 16384, MAX_PROPOSALS = 64;
enum class CarryPolicy { Sector, Recent, CurrentGap, NextMass, NextGap };
CarryPolicy carry_policy(std::string_view s) {
    if (s == "sector")
        return CarryPolicy::Sector;
    if (s == "recent")
        return CarryPolicy::Recent;
    if (s == "current_gap")
        return CarryPolicy::CurrentGap;
    if (s == "next_mass")
        return CarryPolicy::NextMass;
    if (s == "next_gap")
        return CarryPolicy::NextGap;
    fail("invalid retention policy");
}
struct Trial {
    uint32_t id = NO, sector = NO, before = 0, after = 0;
    uint64_t ticket = 0, offset = 0, u = 0, mass = 0;
    int32_t bound = 0, score = 0;
    // 0=read, 1=current read buffer, 2=pinned, 3=context,
    // 4=remembered exact score, 5=retained packed bytes.
    unsigned source = 0;
    bool accepted = false, fallback = false;
};
struct Attempt {
    WinnerRow token;
    Rank rank;
    std::array<Trial, MAX_PROPOSALS> trial{};
    std::array<uint16_t, MAX_PROPOSALS> path{};
    size_t n = 0, reads = 0, learned = 0;
    int32_t anchor = 0;
    uint64_t mass = 0, calls = 0;
};
struct Admission {
    bool ran = false;
    int32_t anchor = 0;
    uint64_t value = 0;
    size_t candidates = 0, used = 0;
};
using Positions = std::array<size_t, CONTEXT>;

template <size_t Bytes> class Draw {
    static_assert(Bytes == 4096 || Bytes == 8192);
    static constexpr size_t K = Bytes / PAGE, CAP = Bytes / 2;
    static constexpr size_t ITEMS = Bytes / (2 + 4 * (D / 8)) + 3 * ROWS;
    static constexpr uint16_t EMPTY = 65535;
    Head& h;
    const CarryPolicy p;
    const bool legacy;
    const Layout layout;
    struct Sector {
        std::array<uint16_t, 3> tile{};
        uint16_t size = 0;
    };
    std::vector<Sector> sectors;
    SectorPool& pool;
    const uint16_t lane;
    uint16_t slot = SectorPool::NONE;
    const uint8_t* page() const {
        return pool.view(slot);
    }
    std::vector<int32_t> bounds;
    std::vector<uint64_t> tickets, sector_mass;
    std::bitset<1024> exact_sector;
    // Sector mode uses two bytes per slot ID. Row mode uses a two-byte
    // payload length. Both reserve precisely the same number of bytes.
    struct Bank {
        std::array<uint8_t, Bytes> raw{};
        std::array<uint16_t, 2> info{};
    } bank;
    static_assert(sizeof(Bank) == Bytes + 4);
    uint16_t buffer = EMPTY;
    int32_t anchor = 0;
    uint64_t mass = 0;
#include "fold_fields.inc"

    struct Item {
        const uint8_t* raw = nullptr;
        uint64_t value = 0;
        int32_t upper = 0, exact = 0;
        uint16_t row = 0, cost = 0;
        bool fresh = false;
    };
    using Dp = std::array<uint64_t, CAP + 1>;
    using Marks = std::array<std::bitset<CAP + 1>, ITEMS>;

    uint16_t prefix(size_t i) const {
        return h.refine.code(i / ROWS, i % ROWS);
    }
    unsigned exponent(size_t i) const {
        return h.meta.exponent(i);
    }
    uint16_t sector(size_t i) const {
        return h.residual.sector(i / ROWS);
    }
    size_t width(size_t i) const {
        return exponent(i) * (D / 8);
    }
    static int32_t narrow(int64_t s) {
        if (s < INT32_MIN || s > INT32_MAX)
            fail("score outside int32");
        return int32_t(s);
    }
    static uint64_t gap(int32_t m, int32_t s) {
        if (s > m)
            fail("score exceeded the certified anchor");
        return uint64_t(int64_t(m) - s);
    }
    int32_t upper(size_t i, const Query& x, int32_t* crude = nullptr) const {
        const auto e = exponent(i);
        const int64_t y = h.g(i, x), r = (1LL << e) - 1;
        const auto n = uint64_t(x.sq) * h.meta.norm(i);
        auto u = std::min(r * x.pos, half_floor(r * x.sum + int64_t(isqrt(n))));
        if (crude)
            *crude = narrow(y + u);
        const auto c = prefix(i);
        size_t unused = 0;
        if (c)
            u = h.refine.upper(c, e, x, u, INT64_MIN, unused);
        return narrow(y + u);
    }
    int32_t score(size_t i, const uint8_t* b, const Query& x) const {
        const auto e = exponent(i);
        const auto d = fused_pair(h.q.data() + i * (D / 2), b, x, e);
        return narrow(int64_t(d.quotient) * (1LL << e) + int64_t(h.shift[i]) * x.sum +
                      h.bias_at(i) + d.residual);
    }
    static uint16_t id(const uint8_t* b) {
        return uint16_t(b[0]) | uint16_t(b[1]) << 8;
    }
    template <class F> void walk_sector(uint16_t sid, const uint8_t* raw, F visit) const {
        const auto z = h.residual.describe(sid, raw + 4, PAGE - 4);
        for (size_t j = 0; j < z.size; ++j) {
            auto b = raw + 4 + z.members[j].offset;
            const size_t start = size_t(z.members[j].tile) * ROWS;
            for (size_t i = start; i < start + ROWS; ++i) {
                visit(i, b);
                b += width(i);
            }
        }
    }
    template <class F> void walk_rows(F visit) const {
        for (size_t at = 0; at < bank.info[0];) {
            const auto* b = bank.raw.data() + at;
            const auto i = id(b);
            if (i >= V || at + 2 + width(i) > bank.info[0])
                fail("invalid packed row record");
            visit(i, b + 2);
            at += 2 + width(i);
        }
    }
    const uint8_t* from_sector(size_t i, uint16_t sid, const uint8_t* raw) const {
        const auto z = h.residual.describe(sid, raw + 4, PAGE - 4);
        for (size_t j = 0; j < z.size; ++j)
            if (z.members[j].tile == i / ROWS) {
                auto b = raw + 4 + z.members[j].offset;
                for (size_t k = i / ROWS * ROWS; k < i; ++k)
                    b += width(k);
                return b;
            }
        fail("missing buffered row");
    }
    const uint8_t* from_bank(size_t i) const {
        if (legacy && buffer != EMPTY)
            return nullptr; // archived single-buffer semantics
        if (p == CarryPolicy::Sector) {
            for (size_t j = 0; j < K; ++j)
                if (bank.info[j] == sector(i))
                    return from_sector(i, bank.info[j], bank.raw.data() + j * PAGE);
            return nullptr;
        }
        const uint8_t* ans = nullptr;
        walk_rows([&](size_t k, const uint8_t* b) {
            if (i == k)
                ans = b;
        });
        return ans;
    }
    void read(uint16_t sid, Attempt& o) {
        if (o.reads >= o.path.size())
            fail("read path capacity");
        pool.fill(lane, sid, slot, o.calls,
                  [&](auto& bytes, uint64_t& calls) { h.residual.get(sid, bytes, calls); });
        buffer = sid;
        o.path[o.reads++] = sid;
    }
#include "lease_index.inc"
  public:
    Draw(Head& head, CarryPolicy mode, bool old, SectorPool& supply, uint16_t index)
        : h(head), p(mode), legacy(old), layout(fold_layout()), sectors(h.sectors()), pool(supply),
          lane(index), bounds(layout == Layout::Sector ? 0 : h.n),
          tickets(layout == Layout::Dense ? h.n : 0), sector_mass(h.sectors()) {
        if (h.n != V || p != CarryPolicy::NextGap || legacy)
            fail("unsupported Fold geometry or cache policy");
        verify_ticket_grid();
        init_fold();
        for (size_t i = 0; i < V; ++i)
            if (exponent(i) < 4)
                fail("unsupported short residual");
        for (size_t t = 0; t < h.pages; ++t) {
            auto& s = sectors[h.residual.sector(t)];
            if (s.size == s.tile.size())
                fail("sector overflow");
            s.tile[s.size++] = t;
        }
    }
    static constexpr size_t persistent_bytes() {
        return sizeof(Bank);
    }
    static constexpr size_t dp_bytes() {
        return sizeof(Dp) + sizeof(Marks);
    }
    static constexpr size_t choice_bytes() {
        return sizeof(std::array<Item, ITEMS>) + sizeof(Bank) + sizeof(std::bitset<ITEMS>);
    }
    ~Draw() {
        pool.release(slot);
    }
    void park(uint32_t sid) {
        if (sid >= h.sectors())
            fail("invalid parked sector");
        pool.park(lane, uint16_t(sid));
    }
    Draw(const Draw&) = delete;
    Draw& operator=(const Draw&) = delete;
    void reset() {
        pool.release(slot);
        buffer = EMPTY;
        if (p == CarryPolicy::Sector)
            bank.info.fill(EMPTY);
        else
            bank.info.fill(0);
    }
#include "stride_scan.inc"
#include "lease_prepare.inc"
#include "lease_trial.inc"
    template <class F> void stable_sector_rows(size_t sid, F visit) const {
        const auto& s = sectors[sid];
        for (size_t j = 0; j < s.size; ++j) {
            const size_t a = size_t(s.tile[j]) * ROWS;
            for (size_t i = a; i < a + ROWS; ++i)
                visit(i);
        }
    }

    Admission retain(const Query& current, const Query& next, const Positions& positions) {
        Admission a;
        a.ran = true;
        if (p == CarryPolicy::Sector) {
            if (buffer != EMPTY) {
                Bank save;
                save.info.fill(EMPTY);
                save.info[0] = buffer;
                std::memcpy(save.raw.data(), page(), PAGE);
                size_t used = 1;
                for (size_t j = 0; j < K && used < K; ++j) {
                    if (bank.info[j] == EMPTY || bank.info[j] == buffer)
                        continue;
                    save.info[used] = bank.info[j];
                    std::memcpy(save.raw.data() + used * PAGE, bank.raw.data() + j * PAGE, PAGE);
                    ++used;
                }
                bank = save;
            }
            a.used = PAGE * size_t(std::count_if(bank.info.begin(), bank.info.begin() + K,
                                                 [](auto s) { return s != EMPTY; }));
            return a;
        }
        std::array<Item, ITEMS> items{};
        size_t n = 0;
        const auto collect = [&](size_t i, const uint8_t* b, bool fresh) {
            if (std::find(positions.begin(), positions.end(), i) != positions.end() ||
                Refinement::levels(prefix(i)) == exponent(i))
                return;
            for (size_t j = 0; j < n; ++j)
                if (items[j].row == i)
                    return;
            if (n >= items.size())
                fail("retention pool overflow");
            items[n++] = {b, 0, 0, 0, uint16_t(i), uint16_t(2 + width(i)), fresh};
        };
        if (buffer != EMPTY)
            walk_sector(buffer, page(), [&](size_t i, const uint8_t* b) { collect(i, b, true); });
        walk_rows([&](size_t i, const uint8_t* b) { collect(i, b, false); });
        a.candidates = n;
        std::sort(items.begin(), items.begin() + n, [&](const Item& x, const Item& y) {
            if (p == CarryPolicy::Recent && x.fresh != y.fresh)
                return x.fresh > y.fresh;
            return h.rows[x.row] < h.rows[y.row];
        });
        std::bitset<ITEMS> chosen;
        if (p == CarryPolicy::Recent) {
            size_t left = Bytes;
            for (size_t j = 0; j < n; ++j)
                if (items[j].cost <= left) {
                    left -= items[j].cost;
                    chosen.set(j);
                }
        } else if (n) {
            const Query& q = p == CarryPolicy::CurrentGap ? current : next;
            a.anchor = INT32_MIN;
            for (size_t j = 0; j < n; ++j) {
                auto& it = items[j];
                it.upper = upper(it.row, q);
                it.exact = score(it.row, it.raw, q);
                if (it.exact > it.upper)
                    fail("admission score exceeds bound");
                a.anchor = std::max(a.anchor, it.upper);
            }
            Dp dp{};
            Marks take{};
            for (size_t j = 0; j < n; ++j) {
                auto& it = items[j];
                const auto u = exact_exp::envelope(gap(a.anchor, it.upper));
                const auto w = std::min(u, exact_exp::envelope(gap(a.anchor, it.exact)));
                it.value = p == CarryPolicy::NextMass ? u : u - w;
                const size_t c = it.cost / 2;
                if (!it.value)
                    continue;
                for (size_t k = CAP; k >= c; --k) {
                    const auto z = dp[k - c] + it.value;
                    if (z > dp[k]) {
                        dp[k] = z;
                        take[j].set(k);
                    }
                }
            }
            a.value = dp[CAP];
            size_t k = CAP;
            for (size_t j = n; j-- > 0;)
                if (take[j][k]) {
                    chosen.set(j);
                    k -= items[j].cost / 2;
                }
        }
        Bank save;
        for (size_t j = 0; j < n; ++j)
            if (chosen[j]) {
                const auto& it = items[j];
                const auto k = save.info[0];
                if (k + it.cost > Bytes)
                    fail("cache overflow");
                auto b = save.raw.data() + k;
                b[0] = uint8_t(it.row);
                b[1] = uint8_t(it.row >> 8);
                std::memcpy(b + 2, it.raw, it.cost - 2);
                save.info[0] += it.cost;
            }
        a.used = save.info[0];
        bank = save;
        return a;
    }
    void emit_bank() const {
        bool first = true;
        const auto emit = [&](uint32_t i) {
            if (!first)
                std::cout << ';';
            first = false;
            std::cout << i;
        };
        if (p == CarryPolicy::Sector) {
            for (size_t j = 0; j < K; ++j)
                if (bank.info[j] != EMPTY)
                    emit(bank.info[j]);
        } else
            walk_rows([&](size_t i, const uint8_t*) { emit(h.rows[i]); });
    }
};
uint64_t limit_data() {
    rlimit lim{};
    if (::getrlimit(RLIMIT_DATA, &lim))
        io_fail("getrlimit");
    if (const auto* s = std::getenv("QB_DATA_LIMIT_BYTES")) {
        lim.rlim_cur = number(s);
        if (::setrlimit(RLIMIT_DATA, &lim))
            io_fail("setrlimit");
    }
    return lim.rlim_cur;
}
uint64_t step_seed(uint64_t seed, size_t prompt, size_t step) {
    exact_exp::Bits hash(seed ^ (uint64_t(prompt + 1) * 0xd1b54a32d192ed03ULL) ^
                         (uint64_t(step + 1) * 0x94d049bb133111ebULL));
    return hash.next();
}
