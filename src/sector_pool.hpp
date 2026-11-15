// Shared immutable residual pages. Logical accounting is owned by the sampler.
#pragma once

enum class PagePolicy { Private, SeparateLRU, Last, Alias, Lease, Ticket };

struct PoolConfig {
    PagePolicy policy = PagePolicy::Private;
    uint16_t spare = 0;
    bool isolate = false;
};

class SectorPool {
  public:
    static constexpr uint16_t NONE = UINT16_MAX;

  private:
    struct alignas(PAGE) Storage {
        std::array<uint8_t, PAGE> bytes;
    };
    static_assert(sizeof(Storage) == PAGE);
    struct Key {
        uint16_t sid = NONE;
        uint16_t domain = 0;
        bool operator==(const Key&) const = default;
    };
    struct Tag {
        Key key{};
        uint16_t pins = 0;
        uint64_t last = 0;
    };
    static_assert(sizeof(Tag) == 16);
    std::vector<Storage> payload;
    std::vector<Tag> tags;
    std::vector<uint16_t> promises; // exact next charged sector when parked
    const uint16_t width;
    const PoolConfig cfg;
    uint64_t remaining_pins = 0, valid = 0;
    void label(uint16_t slot, Key k) {
        if (tags[slot].key.sid == NONE) {
            ++valid;
            peak_valid = std::max(peak_valid, valid);
        }
        tags[slot].key = k;
        tags[slot].last = logical;
    }
    uint16_t find(Key k, uint16_t first, uint16_t end) const {
        for (uint16_t i = first; i < end; ++i)
            if (tags[i].key == k)
                return i;
        return NONE;
    }
    size_t announced(Key k) const {
        if (cfg.isolate)
            return k.domain < promises.size() && promises[k.domain] == k.sid;
        return std::count(promises.begin(), promises.end(), k.sid);
    }
    uint16_t victim(uint16_t first, uint16_t end, bool ticket = false) {
        uint16_t best = NONE, plain = NONE;
        size_t debt = 0;
        for (uint16_t i = first; i < end; ++i) {
            if (tags[i].pins)
                continue;
            if (tags[i].key.sid == NONE)
                return i;
            if (plain == NONE || tags[i].last < tags[plain].last)
                plain = i;
            const size_t n = ticket ? announced(tags[i].key) : 0;
            if (best == NONE || n < debt || (n == debt && tags[i].last < tags[best].last)) {
                best = i;
                debt = n;
            }
        }
        if (best == NONE)
            fail("no unpinned sector (invariant violated)");
        overrides += best != plain;
        return best;
    }
    void pin(uint16_t where, uint16_t& slot) {
        if (slot != NONE)
            fail("second live lease");
        auto& t = tags[where];
        if (t.key.sid == NONE)
            fail("pin of garbage");
        slot = where;
        ++t.pins;
        ++remaining_pins;
        peak_slot_pins = std::max<uint64_t>(peak_slot_pins, t.pins);
        t.last = logical;
        if (remaining_pins > width)
            fail("too many live leases");
    }

  public:
    uint64_t logical = 0, reads = 0, syscalls = 0, hit_idle = 0, hit_pinned = 0, hit_lru = 0,
             hit_last = 0;
    uint64_t copied_bytes = 0, peak_slot_pins = 0, peak_valid = 0;
    uint64_t overrides = 0, announced_reads = 0, announced_hits = 0;
    SectorPool(uint16_t w, const PoolConfig c)
        : payload(size_t(w) + c.spare), tags(payload.size()),
          promises(c.policy == PagePolicy::Ticket ? w : 0, NONE), width(w), cfg(c) {
        if (!w)
            fail("zero pool width");
        if ((c.policy == PagePolicy::SeparateLRU) != (c.spare != 0))
            fail("inconsistent cache");
    }
    size_t page_bytes() const {
        return payload.capacity() * sizeof(Storage);
    }
    size_t metadata_bytes() const {
        return sizeof(*this) + tags.capacity() * sizeof(Tag) +
               promises.capacity() * sizeof(uint16_t);
    }
    const uint8_t* view(uint16_t slot) const {
        if (slot >= payload.size() || !tags[slot].pins)
            fail("no live sector view");
        return payload[slot].bytes.data();
    }
    void release(uint16_t& slot) {
        if (slot == NONE)
            return;
        if (slot >= payload.size() || !tags[slot].pins || !remaining_pins)
            fail("invalid lease release");
        auto& t = tags[slot];
        --t.pins;
        --remaining_pins;
        slot = NONE;
        if (!t.pins && cfg.policy == PagePolicy::Alias) {
            t.key = Key{};
            --valid;
        }
    }
    bool clean() const {
        return !remaining_pins &&
               std::all_of(promises.begin(), promises.end(), [](uint16_t v) { return v == NONE; });
    }
    void park(uint16_t lane, uint16_t sid) {
        if (promises.empty())
            return;
        if (lane >= width || sid == NONE)
            fail("invalid parked sector");
        auto& target = promises[lane];
        if (target != NONE && target != sid)
            fail("parked sector drift");
        target = sid;
    }
    template <class Load>
    void fill(uint16_t lane, uint16_t sid, uint16_t& slot, uint64_t& logically_charged, Load load) {
        if (lane >= width || sid == NONE)
            fail("invalid sector acquire");
        release(slot);
        // A domain is an immutable model identity in a deployment. Isolate is
        // a benchmark control that deliberately prevents all cross-lane reuse.
        const Key k{sid, uint16_t(cfg.isolate ? lane : 0)};
        bool promised = false;
        if (!promises.empty() && promises[lane] != NONE) {
            if (promises[lane] != sid)
                fail("parked sector drift at commit");
            ++announced_reads;
            promised = true;
            promises[lane] = NONE;
        }
        ++logical;
        ++logically_charged;
        const auto read = [&](uint16_t to) {
            if (tags[to].pins)
                fail("overwrite of a live lease");
            load(payload[to].bytes, syscalls);
            ++reads;
        };
        const auto copy = [&](uint16_t to, uint16_t from) {
            if (to == from)
                return;
            if (tags[to].pins)
                fail("overwrite of a live lease");
            payload[to].bytes = payload[from].bytes;
            copied_bytes += PAGE;
        };
        switch (cfg.policy) {
        case PagePolicy::Private:
            read(lane);
            label(lane, k);
            pin(lane, slot);
            break;
        case PagePolicy::SeparateLRU: {
            const auto end = uint16_t(payload.size());
            auto where = find(k, width, end);
            if (where == NONE) {
                where = victim(width, end);
                read(lane);
                copy(where, lane);
            } else {
                ++hit_lru;
                copy(lane, where);
            }
            label(where, k);
            label(lane, k);
            pin(lane, slot);
            break;
        }
        case PagePolicy::Last: {
            const auto where = find(k, 0, width);
            if (where == NONE)
                read(lane);
            else {
                ++hit_last;
                copy(lane, where);
            }
            label(lane, k);
            pin(lane, slot);
            break;
        }
        case PagePolicy::Alias:
        case PagePolicy::Lease:
        case PagePolicy::Ticket: {
            auto where = find(k, 0, width);
            if (where == NONE) {
                where = victim(0, width, cfg.policy == PagePolicy::Ticket);
                read(where);
                label(where, k);
            } else {
                announced_hits += promised;
                if (tags[where].pins)
                    ++hit_pinned;
                else
                    ++hit_idle;
            }
            pin(where, slot);
            break;
        }
        }
    }
};
