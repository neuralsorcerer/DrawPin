// The cursor is only a rank in the current envelope. Waiting neither
// changes that envelope nor commits any acceptance bits.
static constexpr uint64_t no_ticket = UINT64_MAX;
bool advance(const Query& x, const KnownRows& k, bool credit, exact_exp::Bits& rng,
             uint64_t& pending, Attempt& o) {
    if (o.n || o.reads || o.calls || o.learned || o.rank.row != NO)
        fail("dirty relay scratch");
    if (pending == no_ticket)
        pending = rng.below(mass);
    auto& a = o.trial[0];
    a.mass = mass;
    a.offset = a.ticket = pending;
    uint16_t sid = 0;
    const auto [i, bound] = locate_ticket(a.offset, sid, x, k);
    a.sector = sid;
    a.id = h.rows[i];
    a.bound = bound;
    const auto j = size_t(std::find(k.pos.begin(), k.pos.end(), i) - k.pos.begin());
    const auto c = prefix(i);
    const auto e = exponent(i);
    const uint8_t* raw = nullptr;
    if (j < CONTEXT) {
        a.source = 3;
        a.score = bound;
    } else if (Refinement::levels(c) == e) {
        a.source = 2;
        a.score = bound;
    } else if ((raw = from_bank(i)) != nullptr) {
        a.source = 5;
        a.score = bound;
    } else if (exact_sector[sid]) {
        a.source = 4;
        a.score = bound;
    } else {
        if (buffer == sid)
            a.source = 1;
        else if (!credit)
            return false;
        else
            fetch(sid, o, x, k);
        raw = from_sector(i, sid, page());
        a.score = score(i, raw, x);
    }
    if (a.score > a.bound)
        fail("proposal exceeds row bound");
    // A known exact score may outlive its sector buffer. Probe on a
    // copy; only commit if an accepted row can also be materialized.
    auto probe = rng;
    a.u = probe.next();
    a.accepted = exact_exp::accept(gap(anchor, a.score), a.offset, a.u, probe, a.fallback);
    if (a.accepted) {
        const bool needs_bytes = j == CONTEXT && a.source != 2 && !raw && buffer != sid;
        if (needs_bytes && !credit) {
            if (o.reads)
                fail("unpayable second read");
            return false;
        }
        o.rank = {a.score, a.id};
        o.token.position = i;
        if (j < CONTEXT)
            o.token.values = k.rows[j];
        else {
            std::array<uint8_t, D> r{};
            if (a.source == 2)
                h.refine.unpack_exact(c, r);
            else {
                if (!raw) {
                    if (buffer != sid)
                        fetch(sid, o, x, k);
                    raw = from_sector(i, sid, page());
                }
                fused_unpack(raw, r.data(), e);
            }
            h.materialize_row(i, o.token.values, r.data());
        }
    } else if (buffer == sid)
        o.learned += tighten(sid, page(), x, k);
    if (o.reads > size_t(credit) || o.calls != o.reads)
        fail("relay read reservation violated");
    a.after = o.reads;
    o.n = 1;
    pending = no_ticket;
    rng = probe;
    return true;
}
