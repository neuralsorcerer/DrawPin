#pragma once
#include "anchor_seeds.hpp"

enum class ScanMode { Fold, Guard, Lookup, Stride, Forced };
ScanMode stride_options() {
    static const auto mode = []() {
        const auto* p = std::getenv("QB_STRIDE_SCAN");
        if (!p || std::string_view(p) == "stride")
            return ScanMode::Stride;
        if (std::string_view(p) == "fold")
            return ScanMode::Fold;
        if (std::string_view(p) == "guard")
            return ScanMode::Guard;
        if (std::string_view(p) == "lookup")
            return ScanMode::Lookup;
        if (std::string_view(p) == "forced")
            return ScanMode::Forced;
        fail("invalid Stride scan mode");
    }();
    return mode;
}

struct StrideStats {
    uint64_t predictions = 0, misses = 0, discarded_rows = 0, table_fills = 0, scan_rows = 0,
             root_checks = 0;
    uint64_t audited_prepares = 0, audited_sectors = 0;
} stride_stats;
