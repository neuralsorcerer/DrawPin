static constexpr size_t MAX_SECTOR_ROWS = 3 * ROWS;
struct RowCache {
    std::array<int32_t, MAX_SECTOR_ROWS> b{};
    std::array<uint64_t, MAX_SECTOR_ROWS> w{};
    uint16_t sid = EMPTY;
};
struct BankEntry {
    uint16_t i = 0, at = 0;
};
static_assert(sizeof(BankEntry) == 4);
std::vector<RowCache> memo;
std::vector<BankEntry> bank_lookup;
std::vector<uint16_t> offsets;
std::vector<int32_t> evidence;

size_t index_bytes() const {
    return bounds.capacity() * sizeof(int32_t) + tickets.capacity() * sizeof(uint64_t) +
           sector_mass.capacity() * sizeof(uint64_t) + memo.capacity() * sizeof(RowCache) +
           bank_lookup.capacity() * sizeof(BankEntry) + offsets.capacity() * sizeof(uint16_t);
}
void init_fold() {
    if (layout == Layout::Sector) {
        memo.resize(1);
        offsets.resize(sectors.size(), EMPTY);
        bank_lookup.reserve(Bytes / (2 + 4 * (D / 8)));
    }
    fold_stats.index_bytes = index_bytes();
    fold_stats.peak_index_bytes = std::max(fold_stats.peak_index_bytes, index_bytes());
}

public:
size_t heap_bytes() const {
    return index_bytes() + evidence.capacity() * sizeof(int32_t) +
           sectors.capacity() * sizeof(Sector);
}

private:
