static void verify_ticket_grid() {
    static const bool valid = []() {
        if (EXP_REMAINDER.size() != GRID_STEP || EXP_REMAINDER.front() != exact_exp::F)
            fail("invalid remainder table geometry");
        for (size_t j = 1; j < EXP_REMAINDER.size(); ++j)
            if (EXP_REMAINDER[j] + 1 > EXP_REMAINDER[j - 1] + (j > 1))
                fail("nonmonotone remainder table");
        for (size_t b = 1; b <= EXP_BLOCK.size(); ++b) {
            const uint64_t d = b * GRID_STEP;
            if (exact_exp::envelope(d) > exact_exp::envelope(d - 1))
                fail("nonmonotone ticket boundary");
        }
        if (std::getenv("QB_FOLD_GRID_AUDIT")) {
            const uint64_t limit = (EXP_BLOCK.size() + 1) * GRID_STEP;
            auto last = exact_exp::envelope(0);
            for (uint64_t d = 1; d < limit; ++d) {
                const auto now = exact_exp::envelope(d);
                if (now > last)
                    fail("nonmonotone integer ticket grid");
                last = now;
                ++fold_stats.grid_pairs;
            }
        }
        return true;
    }();
    (void)valid;
}
uint64_t weight(int32_t b) const {
    return b == INT32_MIN ? 0 : exact_exp::envelope(gap(anchor, b));
}
void change(size_t sid, uint64_t old, uint64_t next) {
    if (next > old || old - next > sector_mass[sid] || old - next > mass)
        fail("invalid ticket tightening");
    const auto d = old - next;
    sector_mass[sid] -= d;
    mass -= d;
}
size_t update_dense(size_t i, int32_t b) {
    if (bounds[i] == INT32_MIN)
        return 0;
    if (b > bounds[i] || b == INT32_MIN)
        fail("row bound violation");
    const auto old = layout == Layout::Dense ? tickets[i] : weight(bounds[i]);
    bounds[i] = b;
    const auto next = weight(b);
    if (layout == Layout::Dense)
        tickets[i] = next;
    change(sector(i), old, next);
    return 1;
}
size_t learn_dense(size_t i, const uint8_t* b, const Query& x) {
    return bounds[i] == INT32_MIN ? 0 : update_dense(i, score(i, b, x));
}
size_t context_index(size_t i, const KnownRows& k) const {
    return std::find(k.pos.begin(), k.pos.end(), i) - k.pos.begin();
}
int32_t context_score(size_t i, size_t j, const Query& x, const KnownRows& k) const {
    return narrow(int64_t(dot<Dot::I16>(k.rows[j].data(), x)) + h.bias_at(i));
}
void index_bank() {
    bank_lookup.clear();
    walk_rows([&](size_t i, const uint8_t* raw) {
        if (bank_lookup.size() == bank_lookup.capacity())
            fail("bank index capacity");
        bank_lookup.push_back({uint16_t(i), uint16_t(raw - bank.raw.data())});
    });
    std::sort(bank_lookup.begin(), bank_lookup.end(),
              [](const auto& a, const auto& b) { return a.i < b.i; });
}
const uint8_t* bank_exact(size_t i) const {
    const auto j = std::lower_bound(bank_lookup.begin(), bank_lookup.end(), i,
                                    [](const BankEntry& a, size_t x) { return a.i < x; });
    return j != bank_lookup.end() && j->i == i ? bank.raw.data() + j->at : nullptr;
}
int32_t initial_bound(size_t i, const Query& x, const KnownRows& k) const {
    const auto j = context_index(i, k);
    if (j < CONTEXT)
        return k.forbid_context ? INT32_MIN : context_score(i, j, x, k);
    if (const auto* b = bank_exact(i))
        return score(i, b, x);
    return upper(i, x);
}
size_t local_row(uint16_t sid, size_t i) const {
    const auto& s = sectors[sid];
    for (size_t j = 0; j < s.size; ++j)
        if (i / ROWS == s.tile[j])
            return j * ROWS + i % ROWS;
    fail("row not in sector index");
}
size_t evidence_index(uint16_t sid, size_t i) const {
    if (offsets[sid] == EMPTY)
        fail("missing evicted exact scores");
    return offsets[sid] + local_row(sid, i);
}
void reserve_evidence(uint16_t s) {
    if (offsets[s] != EMPTY)
        fail("duplicate exact-score evidence");
    const size_t need = evidence.size() + size_t(sectors[s].size) * ROWS;
    if (need > V)
        fail("exact-score arena exceeds vocabulary");
    if (need > evidence.capacity()) {
        const size_t old = evidence.capacity();
        evidence.reserve(std::min(V, std::max({size_t(64), 2 * old, need})));
        fold_stats.peak_index_bytes =
            std::max(fold_stats.peak_index_bytes,
                     index_bytes() + sizeof(int32_t) * (old + evidence.capacity()));
    }
    offsets[s] = uint16_t(evidence.size());
    evidence.resize(need);
    fold_stats.peak_evidence_rows = std::max(fold_stats.peak_evidence_rows, need);
}
void park_buffer(const Query& x, const KnownRows& k) {
    if (buffer == EMPTY || !exact_sector[buffer] || offsets[buffer] != EMPTY)
        return;
    reserve_evidence(buffer);
    walk_sector(buffer, page(), [&](size_t i, const uint8_t* b) {
        evidence[evidence_index(buffer, i)] =
            (k.forbid_context && context_index(i, k) < CONTEXT) ? INT32_MIN : score(i, b, x);
    });
}
void fetch(uint16_t s, Attempt& o, const Query& x, const KnownRows& k) {
    if (layout == Layout::Sector)
        park_buffer(x, k);
    read(s, o);
}
size_t tighten(uint16_t s, const uint8_t* bytes, const Query& x, const KnownRows& k) {
    if (exact_sector[s])
        return 0;
    size_t n = 0;
    if (layout != Layout::Sector) {
        walk_sector(s, bytes, [&](size_t i, const uint8_t* b) { n += learn_dense(i, b, x); });
    } else {
        auto& a = memo.front();
        const bool cached = a.sid == s;
        walk_sector(s, bytes, [&](size_t i, const uint8_t* b) {
            const auto old = cached ? a.b[local_row(s, i)] : initial_bound(i, x, k);
            if (old != INT32_MIN) {
                const auto z = score(i, b, x);
                if (z > old)
                    fail("row bound violation");
                change(s, weight(old), weight(z));
                ++n;
            }
        });
        if (cached)
            a.sid = EMPTY;
    }
    exact_sector.set(s);
    return n;
}
const RowCache& materialize_sector(uint16_t sid, const Query& x, const KnownRows& k) {
    auto& a = memo.front();
    if (a.sid == sid) {
        ++fold_stats.memo_hits;
        return a;
    }
    ++fold_stats.rebuild_sectors;
    const auto& s = sectors[sid];
    if (exact_sector[sid]) {
        if (sid == buffer) {
            walk_sector(sid, page(), [&](size_t i, const uint8_t* b) {
                a.b[local_row(sid, i)] = (k.forbid_context && context_index(i, k) < CONTEXT)
                                             ? INT32_MIN
                                             : score(i, b, x);
            });
        } else {
            if (offsets[sid] == EMPTY)
                fail("no evicted sector scores");
            std::copy_n(evidence.begin() + offsets[sid], size_t(s.size) * ROWS, a.b.begin());
        }
    } else {
        fold_stats.cold_rows += size_t(s.size) * ROWS;
        for (size_t j = 0; j < s.size; ++j)
            for (size_t k_ = 0; k_ < ROWS; ++k_)
                a.b[j * ROWS + k_] = initial_bound(size_t(s.tile[j]) * ROWS + k_, x, k);
    }
    uint64_t sum = 0;
    for (size_t j = 0; j < size_t(s.size) * ROWS; ++j) {
        a.w[j] = weight(a.b[j]);
        sum += a.w[j];
    }
    if (sum != sector_mass[sid])
        fail("sector did not unfold to its exact ticket mass");
    a.sid = sid;
    return a;
}
std::pair<size_t, int32_t> locate_ticket(uint64_t& ticket, uint16_t& sid, const Query& x,
                                         const KnownRows& k) {
    for (sid = 0; sid < sectors.size(); ++sid) {
        if (ticket < sector_mass[sid])
            break;
        ticket -= sector_mass[sid];
    }
    if (sid == sectors.size())
        fail("ticket outside sector envelope");
    const auto* cache = layout == Layout::Sector ? &materialize_sector(sid, x, k) : nullptr;
    const auto& g = sectors[sid];
    for (size_t j = 0; j < g.size; ++j)
        for (size_t p_ = 0; p_ < ROWS; ++p_) {
            const auto i = size_t(g.tile[j]) * ROWS + p_;
            const auto b = cache ? cache->b[j * ROWS + p_] : bounds[i];
            const auto n = cache                     ? cache->w[j * ROWS + p_]
                           : layout == Layout::Dense ? tickets[i]
                                                     : weight(b);
            if (ticket < n)
                return {i, b};
            ticket -= n;
        }
    fail("ticket outside row envelope");
}
int32_t saved_bound(size_t i, uint16_t sid) const {
    const auto z = layout == Layout::Sector ? evidence[evidence_index(sid, i)] : bounds[i];
    if (z == INT32_MIN)
        fail("missing eligible score");
    return z;
}
