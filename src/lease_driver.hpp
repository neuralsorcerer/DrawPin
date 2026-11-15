#pragma once

void serve_tick(const Config& c, const Body& body, Sampler& draw, Progress& a, QueryBytes& q,
                Totals& z, bool trace) {
    const auto offer = a.offer(c.budget);
    const auto t0 = a.t;
    uint32_t spent = 0;
    size_t trials = 0;
    unsigned reason = 0;
    while (a.t < c.horizon) {
        const Query x(q.data());
        const KnownRows k{a.cx, a.pos, true, c.masked};
        if (!a.active) {
            a.token_learned = draw.prepare(x, k);
            a.active = true;
        }
        Attempt o;
        exact_exp::Bits rng(a.rng);
        if (!draw.advance(x, k, spent < offer.available, rng, a.pending, o)) {
            draw.park(o.trial[0].sector);
            a.rng = rng.state;
            reason = 2;
            break;
        }
        if (o.n != 1 || o.reads > offer.available - spent || o.calls != o.reads)
            fail("invalid committed proposal");
        a.rng = rng.state;
        if (trace)
            emit_trial(a.stream, a.t, a.j, o.trial[0]);
        ++a.j;
        ++trials;
        spent += o.reads;
        a.token_reads += o.reads;
        a.token_learned += o.learned;
        z.calls += o.calls;
        if (o.rank.row == NO)
            continue;
        QueryBytes successor{};
        Admission admission;
        if (a.t + 1 < c.horizon) {
            std::move(a.cx.begin() + 1, a.cx.end(), a.cx.begin());
            a.cx.back() = o.token.values;
            std::move(a.pos.begin() + 1, a.pos.end(), a.pos.begin());
            a.pos.back() = o.token.position;
            successor = body.compute(a.cx);
            const Query nx(successor.data());
            admission = draw.retain(x, nx, a.pos);
        }
        if (trace)
            emit_token(a, o, q, admission, draw);
        ++a.t;
        ++z.tokens;
        a.j = a.token_reads = a.token_learned = 0;
        a.active = false;
        q = successor;
        a.rng = step_seed(c.seed, a.stream, a.t);
    }
    a.charge(c.budget, spent);
    z.proposals += trials;
    z.reads += spent;
    ++z.ticks;
    if (trace)
        std::cout << "K," << a.stream << ',' << a.ticks << ',' << offer.available << ','
                  << offer.balance << ',' << offer.window_before << ',' << spent << ',' << a.balance
                  << ',' << a.reads << ',' << t0 << ',' << a.t << ',' << trials << ',' << reason
                  << '\n';
}

struct LeaseTotals {
    uint64_t backing = 0, syscalls = 0, hit_lru = 0, hit_pinned = 0, hit_idle = 0, hit_last = 0,
             copied = 0;
    uint64_t page_bytes = 0, metadata_bytes = 0, slot_pins = 0, peak_valid = 0, request_bytes = 0;
    uint64_t load_ns = 0, prefill_ns = 0, decode_ns = 0, waves = 0;
    uint64_t overrides = 0, announced_reads = 0, announced_hits = 0;
};

PagePolicy page_policy(std::string_view s) {
    if (s == "private")
        return PagePolicy::Private;
    if (s == "lru")
        return PagePolicy::SeparateLRU;
    if (s == "last")
        return PagePolicy::Last;
    if (s == "alias")
        return PagePolicy::Alias;
    if (s == "lease")
        return PagePolicy::Lease;
    if (s == "ticket")
        return PagePolicy::Ticket;
    fail("invalid page policy");
}

void wave(const Config& c, const Body& body, Head& head, const uint32_t* prompts, uint16_t width,
          size_t first, PoolConfig cfg, Totals& z, LeaseTotals& w, bool trace) {
    struct Lane {
        Progress p;
        QueryBytes q{};
        std::optional<Sampler> d;
    };
    auto clock = Clock::now();
    SectorPool pool(width, cfg);
    std::vector<Lane> a(width);
    const auto before = z.reads;
    const auto account = [&] {
        size_t bytes = a.capacity() * sizeof(Lane);
        for (const auto& r : a)
            if (r.d)
                bytes += r.d->heap_bytes();
        w.request_bytes = std::max<uint64_t>(bytes, w.request_bytes);
    };
    for (uint16_t i = 0; i < width; ++i) {
        auto& r = a[i];
        r.d.emplace(head, CarryPolicy::NextGap, false, pool, i);
        r.p.stream = first + i;
        r.p.rng = step_seed(c.seed, r.p.stream, 0);
        z.prefill +=
            head.prefill<CONTEXT>(prompts + size_t(i) * CONTEXT, r.p.cx, z.prefill_calls, &r.p.pos);
        r.q = body.compute(r.p.cx);
        ++z.requests;
    }
    account();
    w.prefill_ns += ns(clock);
    clock = Clock::now();
    size_t live = width;
    while (live) {
        for (auto& r : a)
            if (r.d) {
                serve_tick(c, body, *r.d, r.p, r.q, z, trace);
                account();
                if (r.p.t == c.horizon) {
                    r.d.reset();
                    --live;
                }
            }
    }
    w.decode_ns += ns(clock);
    if (!pool.clean() || pool.logical != z.reads - before ||
        pool.logical != pool.reads + pool.hit_idle + pool.hit_pinned + pool.hit_lru + pool.hit_last)
        fail("invalid backing ledger");
    ++w.waves;
    w.backing += pool.reads;
    w.syscalls += pool.syscalls;
    w.hit_idle += pool.hit_idle;
    w.hit_pinned += pool.hit_pinned;
    w.hit_lru += pool.hit_lru;
    w.hit_last += pool.hit_last;
    w.copied += pool.copied_bytes;
    w.overrides += pool.overrides;
    w.announced_reads += pool.announced_reads;
    w.announced_hits += pool.announced_hits;
    w.page_bytes = std::max<uint64_t>(pool.page_bytes(), w.page_bytes);
    w.metadata_bytes = std::max<uint64_t>(pool.metadata_bytes(), w.metadata_bytes);
    w.slot_pins = std::max(w.slot_pins, pool.peak_slot_pins);
    w.peak_valid = std::max(w.peak_valid, pool.peak_valid);
}

int main(int argc, char** argv) try {
    if (argc != 18)
        fail("usage: drawpin HEAD REFINE MODEL PROMPTS HORIZON GRANT BANK PEAK WINDOW SEED MASK "
             "WIDTH POLICY SPARE buffered|direct shared|isolated START");
    auto start = Clock::now();
    std::ios::sync_with_stdio(false);
    const auto limit = limit_data();
    Config c;
    const auto h = number(argv[5]);
    if (!h || h > 4096)
        fail("invalid horizon");
    c.horizon = h;
    std::array<uint32_t, 4> v{};
    for (size_t j = 0; j < v.size(); ++j) {
        const auto n = number(argv[6 + j]);
        if (n > UINT32_MAX)
            fail("contract overflow");
        v[j] = uint32_t(n);
    }
    c.budget = {v[0], v[1], v[2], v[3]};
    c.budget.validate();
    if (!v[0] || !v[2])
        fail("invalid service grant");
    c.seed = number(argv[10]);
    const std::string_view mask = argv[11], io = argv[15], domain = argv[16];
    if (mask != "all" && mask != "masked")
        fail("invalid mask");
    c.masked = mask == "masked";
    if (io != "buffered" && io != "direct")
        fail("invalid backing mode");
    if (domain != "shared" && domain != "isolated")
        fail("invalid domain");
    const size_t width = number(argv[12]), spare = number(argv[14]), first = number(argv[17]);
    if (!width || width > 128 || spare > 128)
        fail("invalid page counts");
    PoolConfig cfg{page_policy(argv[13]), uint16_t(spare), domain == "isolated"};
    const bool trace = std::getenv("QB_SPOOL_QUIET") == nullptr;
    Head head(V, argv[1], io == "direct", argv[2]);
    const Body body(argv[3]);
    if (std::bit_cast<uint64_t>(body.weight_scale()) != MODEL_SCALE_BITS)
        fail("unexpected model scale");
    const auto prompts = read_file<uint32_t>(argv[4]);
    if (prompts.size() % CONTEXT || first > UINT32_MAX)
        fail("invalid prompts");
    const auto count = prompts.size() / CONTEXT;
    Totals z;
    LeaseTotals w;
    w.load_ns = ns(start);
    for (size_t i = 0; i < count; i += width)
        wave(c, body, head, prompts.data() + i * CONTEXT, uint16_t(std::min(width, count - i)),
             first + i, cfg, z, w, trace);
    if (z.calls != z.reads || z.prefill != z.prefill_calls ||
        spool_preads != z.prefill_calls + w.syscalls)
        fail("invalid process I/O");
    if (trace)
        std::cout.flush();
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage))
        io_fail("getrusage");
    std::cerr << "{\"requests\":" << z.requests << ",\"tokens\":" << z.tokens
              << ",\"ticks\":" << z.ticks << ",\"waves\":" << w.waves << ",\"logical\":" << z.reads
              << ",\"proposals\":" << z.proposals << ",\"prefill\":" << z.prefill
              << ",\"backing\":" << w.backing << ",\"decode_preads\":" << w.syscalls
              << ",\"all_preads\":" << spool_preads << ",\"hit_idle\":" << w.hit_idle
              << ",\"hit_pinned\":" << w.hit_pinned << ",\"hit_lru\":" << w.hit_lru
              << ",\"hit_last\":" << w.hit_last << ",\"copied_bytes\":" << w.copied
              << ",\"overrides\":" << w.overrides << ",\"announced_reads\":" << w.announced_reads
              << ",\"announced_hits\":" << w.announced_hits << ",\"page_bytes\":" << w.page_bytes
              << ",\"peak_valid_pages\":" << w.peak_valid
              << ",\"max_pins_on_one_page\":" << w.slot_pins
              << ",\"metadata_bytes\":" << w.metadata_bytes
              << ",\"request_bytes_without_pages\":" << w.request_bytes
              << ",\"peak_index_bytes\":" << fold_stats.peak_index_bytes
              << ",\"data_limit_bytes\":" << limit << ",\"max_rss_kib\":" << usage.ru_maxrss
              << ",\"load_ns\":" << w.load_ns << ",\"prefill_ns\":" << w.prefill_ns
              << ",\"decode_ns\":" << w.decode_ns << ",\"wall_in_process_ns\":" << ns(start)
              << ",\"prepares\":" << fold_stats.prepares << ",\"sectors\":" << head.sectors()
              << "}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "drawpin: " << e.what() << '\n';
    return 2;
}
