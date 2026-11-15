CXX ?= g++
PYTHON ?= python3
CLANG_FORMAT ?= clang-format
RUFF ?= ruff
CPU ?= auto
JOBS ?= 2
RESULTS ?= build/results
CONFIG := $(CPU)
ifeq ($(CPU),auto)
CONFIG := $(shell if [ "$$(uname -m)" = x86_64 ] && grep -qw avx2 /proc/cpuinfo 2>/dev/null; then echo avx2; else echo scalar; fi)
endif
ifeq ($(CONFIG),avx2)
SIMD := -mavx2
else ifneq ($(CONFIG),scalar)
$(error CPU must be auto, scalar, or avx2)
endif

BUILD := build/$(CONFIG)
DECODER := $(BUILD)/drawpin
BASE_FLAGS := -std=c++20 -Wall -Wextra -O3 $(SIMD)
EXACT_FLAGS := -ffp-contract=off -fno-fast-math
HEADERS := $(wildcard src/*.hpp) $(wildcard src/*.inc)
CPP_SOURCES := $(sort $(wildcard src/*.cpp) $(HEADERS) $(wildcard tests/*.cpp))
PYTHON_DIRS := tools benchmarks tests

.PHONY: all build test benchmark capacity timing verify format lint
all: build
build: $(DECODER)
$(BUILD):
	mkdir -p "$@"
$(DECODER): src/drawpin.cpp $(HEADERS) Makefile | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BASE_FLAGS) $(EXACT_FLAGS) -Isrc "$<" -o "$@" $(LDFLAGS) -lgmpxx -lgmp
$(BUILD)/pool_witness: tests/pool_witness.cpp src/sector_pool.hpp Makefile | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BASE_FLAGS) $(EXACT_FLAGS) -Isrc "$<" -o "$@" $(LDFLAGS)
$(BUILD)/threshold_witness: tests/threshold_witness.cpp src/exact_exp.hpp src/exp_grid.hpp Makefile | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BASE_FLAGS) $(EXACT_FLAGS) -Isrc "$<" -o "$@" $(LDFLAGS) -lgmpxx -lgmp
test: $(DECODER) $(BUILD)/pool_witness $(BUILD)/threshold_witness
	"$(BUILD)/pool_witness" > "$(BUILD)/pool-witness.json"
	$(PYTHON) tests/threshold_cases.py --grid src/exp_grid.hpp --exe "$(BUILD)/threshold_witness" --out "$(BUILD)/threshold.json"
	$(PYTHON) tests/check_oracle.py --exe "$(BUILD)/threshold_witness"
	$(PYTHON) tests/check_cli.py --binary "$(DECODER)"
	$(PYTHON) benchmarks/run.py quick --binary "$(DECODER)" --output "$(RESULTS)/quick-$(CONFIG).json"
benchmark: $(DECODER)
	$(PYTHON) benchmarks/run.py full --binary "$(DECODER)" --jobs $(JOBS) --output "$(RESULTS)/benchmark-$(CONFIG).json"
capacity: $(DECODER)
	$(PYTHON) benchmarks/run.py capacity --binary "$(DECODER)" --jobs $(JOBS) --output "$(RESULTS)/capacity-$(CONFIG).json"
timing: $(DECODER)
	$(PYTHON) benchmarks/run.py timing --binary "$(DECODER)" --output "$(RESULTS)/timing-$(CONFIG).json"
verify:
	sha256sum --quiet --check SHA256SUMS
format:
	$(CLANG_FORMAT) -i $(CPP_SOURCES)
	$(RUFF) format $(PYTHON_DIRS)
lint:
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_SOURCES)
	$(RUFF) format --check $(PYTHON_DIRS)
	$(RUFF) check $(PYTHON_DIRS)
