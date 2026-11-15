private:
static constexpr size_t coarse_codes = 5 * 32;
using CoarseTable = std::array<int32_t, coarse_codes>;
static int32_t suffix_upper(int64_t r, uint64_t norm, const Query& x) {
    return narrow(
        std::min(r * x.pos, half_floor(r * x.sum + int64_t(isqrt(norm * uint64_t(x.sq))))));
}

template <bool L>
int32_t coarse_residual(size_t i, unsigned e, const Query& x, const CoarseTable& tab) const {
    if constexpr (L)
        return tab[h.meta.code8(i)];
    else
        return suffix_upper((1LL << e) - 1, h.meta.norm(i), x);
}

int64_t quotient(size_t i, unsigned e, const Query& x) const {
    return int64_t(dot<Dot::Q4>(h.q.data() + i * (D / 2), x)) * (1LL << e) +
           int64_t(h.shift[i]) * x.sum + h.bias_at(i);
}

void fill_coarse(CoarseTable& tab, const Query& x) const {
    if (!h.meta.compact())
        fail("coarse lookup requires the Reclaim byte metadata");
    for (size_t code = 0; code < tab.size(); ++code) {
        const unsigned r = (1u << (4 + (code >> 5))) - 1;
        tab[code] = suffix_upper(r, decode_norm<31>(code & 31, r), x);
    }
    ++stride_stats.table_fills;
    stride_stats.root_checks += tab.size();
}

void audit_prepared(const Query& x, const KnownRows& k) const {
    if (h.sectors() > 1024)
        fail("mass audit geometry");
    int32_t legacy_anchor = INT32_MIN;
    for (size_t t = 0; t < h.pages; ++t) {
        const auto omit = k.forbid_context ? k.mask(t) : 0;
        for (size_t j = 0; j < ROWS; ++j)
            if (!((omit >> j) & 1u)) {
                const auto i = t * ROWS + j;
                int32_t c = 0;
                (void)upper(i, x, &c);
                legacy_anchor = std::max(c, legacy_anchor);
            }
    }
    if (legacy_anchor != anchor)
        fail("guarded anchor differed from full Fold scan");
    for (uint16_t sid = 0; sid < h.sectors(); ++sid) {
        uint64_t sum = 0;
        stable_sector_rows(sid, [&](size_t i) { sum += weight(initial_bound(i, x, k)); });
        if (sum != sector_mass[sid])
            fail("guarded preparation differed from full Fold scan");
    }
    ++stride_stats.audited_prepares;
    stride_stats.audited_sectors += h.sectors();
}

template <bool L, bool G, bool Forced> void scan_prepare(const Query& x, const KnownRows& k) {
    static_assert(!Forced || G);
    CoarseTable tab;
    if constexpr (L)
        fill_coarse(tab, x);
    uint64_t work = 0, roots = 0;
    if constexpr (G) {
        for (const auto i : stride_anchor_seeds) {
            if (k.forbid_context && context_index(i, k) < CONTEXT)
                continue;
            const auto e = exponent(i);
            anchor = std::max(anchor, narrow(quotient(i, e, x) + coarse_residual<L>(i, e, x, tab)));
            ++work;
            if constexpr (!L)
                ++roots;
        }
        if (anchor == INT32_MIN)
            fail("no available anchor seed");
        if constexpr (Forced)
            --anchor; // guaranteed miss, tested only in verification
        ++stride_stats.predictions;
    }

    // Confirm a candidate maximum using only coarse bounds. In a failed
    // guarded pass only the still-unseen rows can improve the maximum.
    const auto finish_anchor = [&](size_t start) {
        for (size_t t = start / ROWS; t < h.pages; ++t) {
            const auto omit = k.forbid_context ? k.mask(t) : 0;
            for (size_t j = (t == start / ROWS ? start % ROWS : 0); j < ROWS; ++j) {
                if ((omit >> j) & 1u)
                    continue;
                const size_t i = t * ROWS + j;
                const auto e = exponent(i);
                const auto y = quotient(i, e, x);
                ++work;
                if (y + ((1LL << e) - 1) * x.pos <= anchor)
                    continue;
                anchor = std::max(anchor, narrow(y + coarse_residual<L>(i, e, x, tab)));
                if constexpr (!L)
                    ++roots;
                ++fold_stats.anchor_roots;
            }
        }
    };
    if constexpr (!G)
        finish_anchor(0);

    // No entropy or weight-sector I/O is used in either pass. Tentative
    // totals under a wrong anchor are discarded, never rescaled.
    const auto stream = [&]<bool Check>() -> size_t {
        uint64_t counted = 0;
        for (size_t t = 0; t < h.pages; ++t) {
            const auto omit = k.forbid_context ? k.mask(t) : 0;
            uint64_t sum = 0;
            for (size_t j = 0; j < ROWS; ++j) {
                if ((omit >> j) & 1u)
                    continue;
                const auto i = t * ROWS + j;
                const auto e = exponent(i);
                const auto y = quotient(i, e, x);
                auto u = coarse_residual<L>(i, e, x, tab);
                ++work;
                if constexpr (!L)
                    ++roots;
                const auto c = narrow(y + u);
                if constexpr (Check)
                    if (c > anchor) {
                        stride_stats.discarded_rows += counted;
                        anchor = c;
                        return i;
                    }
                const auto key = prefix(i);
                size_t unused = 0;
                if (key)
                    u = narrow(h.refine.upper(key, e, x, u, INT64_MIN, unused));
                sum += weight(narrow(y + u));
                if constexpr (Check)
                    ++counted;
            }
            sector_mass[h.residual.sector(t)] += sum;
            mass += sum;
        }
        return V;
    };
    if constexpr (G) {
        const auto miss = stream.template operator()<true>();
        if (miss != V) {
            ++stride_stats.misses;
            finish_anchor(miss + 1);
            mass = 0;
            std::fill(sector_mass.begin(), sector_mass.end(), 0);
            stream.template operator()<false>();
        }
    } else
        stream.template operator()<false>();
    stride_stats.scan_rows += work;
    stride_stats.root_checks += roots;
}

public:
