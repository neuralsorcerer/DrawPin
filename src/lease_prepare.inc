size_t prepare(const Query& x, const KnownRows& k) {
    ++fold_stats.prepares;
    pool.release(slot);
    buffer = EMPTY;
    exact_sector.reset();
    anchor = INT32_MIN;
    mass = 0;
    std::fill(sector_mass.begin(), sector_mass.end(), 0);
    if (layout == Layout::Sector) {
        memo.front().sid = EMPTY;
        std::fill(offsets.begin(), offsets.end(), EMPTY);
        evidence.clear();
        index_bank();
        switch (stride_options()) {
        case ScanMode::Fold: {
            // Preserve the exact legacy anchor. Refine no row whose box
            // ceiling cannot improve it, and do not evaluate any prefix bits.
            for (size_t t = 0; t < h.pages; ++t) {
                const auto omit = k.forbid_context ? k.mask(t) : 0;
                for (size_t j = 0; j < ROWS; ++j)
                    if (!((omit >> j) & 1u)) {
                        const auto i = t * ROWS + j, e = uint64_t(exponent(i));
                        const int64_t r = (1LL << e) - 1, y = h.g(i, x);
                        if (y + r * x.pos <= anchor)
                            continue;
                        const auto n = uint64_t(x.sq) * h.meta.norm(i);
                        const auto u =
                            std::min(r * x.pos, half_floor(r * x.sum + int64_t(isqrt(n))));
                        anchor = std::max(anchor, narrow(y + u));
                        ++fold_stats.anchor_roots;
                    }
            }
            for (size_t t = 0; t < h.pages; ++t) {
                const auto omit = k.forbid_context ? k.mask(t) : 0;
                uint64_t sum = 0;
                for (size_t j = 0; j < ROWS; ++j)
                    if (!((omit >> j) & 1u))
                        sum += weight(upper(t * ROWS + j, x));
                sector_mass[h.residual.sector(t)] += sum;
                mass += sum;
            }
            break;
        }
        case ScanMode::Guard:
            scan_prepare<false, true, false>(x, k);
            break;
        case ScanMode::Lookup:
            scan_prepare<true, false, false>(x, k);
            break;
        case ScanMode::Stride:
            scan_prepare<true, true, false>(x, k);
            break;
        case ScanMode::Forced:
            scan_prepare<true, true, true>(x, k);
            break;
        }
        if (!k.forbid_context)
            for (size_t j = 0; j < CONTEXT; ++j) {
                const auto i = k.pos[j];
                if (context_index(i, k) != j)
                    continue;
                const auto old = upper(i, x), z = context_score(i, j, x, k);
                if (z > old)
                    fail("context bound violation");
                change(sector(i), weight(old), weight(z));
            }
        size_t n = 0;
        walk_rows([&](size_t i, const uint8_t* b) {
            const auto j = context_index(i, k);
            if (j < CONTEXT && k.forbid_context)
                return;
            const auto old = j < CONTEXT ? context_score(i, j, x, k) : upper(i, x);
            const auto z = score(i, b, x);
            if (z > old)
                fail("carry bound violation");
            change(sector(i), weight(old), weight(z));
            ++n;
        });
        if (!mass || mass > (1ULL << 62))
            fail("invalid mass");
        static const bool audit = std::getenv("QB_STRIDE_MASS_AUDIT") != nullptr;
        if (audit)
            audit_prepared(x, k);
        return n;
    }
    CoarseTable dense_table;
    const bool reuse = stride_options() == ScanMode::Lookup;
    if (reuse)
        fill_coarse(dense_table, x);
    for (size_t t = 0; t < h.pages; ++t) {
        const auto omit = k.forbid_context ? k.mask(t) : 0;
        for (size_t j = 0; j < ROWS; ++j) {
            const auto i = t * ROWS + j;
            if ((omit >> j) & 1u) {
                bounds[i] = INT32_MIN;
                continue;
            }
            int32_t crude = 0;
            if (reuse) {
                const auto e = exponent(i);
                const auto y = quotient(i, e, x), u = int64_t(dense_table[h.meta.code8(i)]);
                crude = narrow(y + u);
                const auto key = prefix(i);
                size_t unused = 0;
                bounds[i] =
                    key ? narrow(y + h.refine.upper(key, e, x, u, INT64_MIN, unused)) : crude;
            } else
                bounds[i] = upper(i, x, &crude);
            anchor = std::max(anchor, crude);
        }
    }
    if (!k.forbid_context)
        for (size_t j = 0; j < CONTEXT; ++j) {
            const auto i = k.pos[j];
            const auto z = context_score(i, j, x, k);
            if (z > bounds[i])
                fail("context bound violation");
            bounds[i] = z;
        }
    for (size_t t = 0; t < h.pages; ++t) {
        uint64_t sum = 0;
        for (size_t j = 0; j < ROWS; ++j) {
            const auto i = t * ROWS + j, m = weight(bounds[i]);
            if (layout == Layout::Dense)
                tickets[i] = m;
            sum += m;
        }
        sector_mass[h.residual.sector(t)] += sum;
        mass += sum;
    }
    if (!mass || mass > (1ULL << 62))
        fail("invalid mass");
    size_t n = 0;
    walk_rows([&](size_t i, const uint8_t* b) { n += learn_dense(i, b, x); });
    return n;
}
