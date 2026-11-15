#pragma once

struct Config {
    ReadContract budget;
    size_t horizon = 0;
    bool masked = false;
    uint64_t seed = 0;
};

struct Progress {
    Prompt cx{};
    Positions pos{};
    uint64_t stream = 0, t = 0, j = 0, token_reads = 0, token_learned = 0;
    uint64_t rng = 0, pending = Draw<PAGE>::no_ticket;
    uint64_t ticks = 0, reads = 0, discarded = 0;
    uint32_t balance = 0;
    bool active = false;
    std::array<uint16_t, 63> tail{}; // oldest to newest, zero padded

    StepBudget offer(const ReadContract& cfg) const {
        uint32_t z = 0;
        for (size_t k = 0; k + 1 < cfg.window; ++k)
            z += tail[k];
        auto a = std::min(cfg.peak, cfg.rate + balance);
        if (cfg.window) {
            if (z > cfg.rate * cfg.window)
                fail("invalid ledger history");
            a = std::min(a, cfg.rate * cfg.window - z);
        }
        return {a, balance, z};
    }
    void charge(const ReadContract& cfg, uint32_t spent) {
        if (spent > offer(cfg).available)
            fail("service budget violated");
        const auto remaining = balance + cfg.rate - spent;
        balance = std::min(cfg.bank, remaining);
        discarded += remaining - balance;
        reads += spent;
        ++ticks;
        if (cfg.window > 1) {
            const auto len = cfg.window - 1;
            std::move(tail.begin() + 1, tail.begin() + len, tail.begin());
            tail[len - 1] = uint16_t(spent);
        }
    }
};

struct Totals {
    size_t reads = 0, calls = 0, prefill = 0, prefill_calls = 0, ticks = 0, tokens = 0,
           proposals = 0, requests = 0;
};

using Sampler = Draw<PAGE>;

void emit_trial(size_t s, size_t t, uint64_t j, const Trial& a) {
    std::cout << "P," << s << ',' << t << ',' << j << ',' << a.id << ',' << a.sector << ','
              << a.after - a.before << ',' << a.ticket << ',' << a.offset << ',' << a.u << ','
              << a.mass << ',' << a.bound << ',' << a.score << ',' << a.source << ',' << a.accepted
              << ',' << a.fallback << '\n';
}

void emit_query(const QueryBytes& q) {
    constexpr char hex[] = "0123456789abcdef";
    for (uint8_t v : q)
        std::cout << hex[v >> 4] << hex[v & 15];
}

void emit_token(const Progress& a, const Attempt& o, const QueryBytes& q, const Admission& v,
                const Sampler& draw) {
    std::cout << "T," << a.stream << ',' << a.t << ',' << o.rank.row << ',' << o.rank.score << ','
              << a.j << ',' << a.token_reads << ',' << a.token_learned << ',';
    emit_query(q);
    std::cout << ',' << v.ran << ',' << v.candidates << ',' << v.anchor << ',' << v.value << ','
              << v.used << ',';
    if (v.ran)
        draw.emit_bank();
    std::cout << '\n';
}
