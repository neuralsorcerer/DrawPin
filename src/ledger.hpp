#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

struct ReadContract {
    uint32_t rate, bank, peak, window;

    void validate() const {
        if (rate > 512 || bank > 65536 || peak > 512 || window > 64)
            throw std::runtime_error("contract exceeds integer/trace limits");
    }
};

struct StepBudget {
    uint32_t available, balance, window_before;
};

class ReadLedger {
    ReadContract cfg;
    uint32_t saved = 0, tail = 0, at = 0;
    std::array<uint16_t, 63> ring{};

  public:
    uint64_t reads = 0, steps = 0, discarded = 0;

    explicit ReadLedger(ReadContract c) : cfg(c) {
        cfg.validate();
    }
    StepBudget offer() const {
        auto a = std::min(cfg.peak, cfg.rate + saved);
        if (cfg.window) {
            const uint32_t window_limit = cfg.rate * cfg.window;
            if (tail > window_limit)
                throw std::logic_error("invalid rolling ledger");
            a = std::min(a, window_limit - tail);
        }
        return {a, saved, tail};
    }
    void charge(uint32_t cost) {
        if (cost > offer().available)
            throw std::logic_error("read contract violated");
        const auto rem = cfg.rate + saved - cost;
        saved = std::min(rem, cfg.bank);
        discarded += rem - saved;
        reads += cost;
        ++steps;
        // Charge the failing attempt too. The caller stops after that attempt.
        if (cfg.window > 1) {
            tail = tail - ring[at] + cost;
            ring[at] = uint16_t(cost);
            at = (at + 1) % (cfg.window - 1);
        }
    }
    uint32_t balance() const {
        return saved;
    }
};
