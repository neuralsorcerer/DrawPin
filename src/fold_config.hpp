#pragma once

enum class Layout { Dense, Score, Sector };

Layout fold_layout() {
    const auto* s = std::getenv("QB_FOLD_LAYOUT");
    if (!s || std::string_view(s) == "sector")
        return Layout::Sector;
    if (std::string_view(s) == "dense")
        return Layout::Dense;
    if (std::string_view(s) == "score")
        return Layout::Score;
    fail("unknown Fold layout");
}

struct FoldStats {
    size_t index_bytes = 0, peak_index_bytes = 0, peak_evidence_rows = 0;
    size_t peak_request_bytes = 0;
    uint64_t rebuild_sectors = 0, memo_hits = 0, prepares = 0, cold_rows = 0, anchor_roots = 0,
             grid_pairs = 0;
} fold_stats;
