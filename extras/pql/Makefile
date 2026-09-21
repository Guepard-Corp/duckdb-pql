# Standalone tests of the PQL language. They need no DuckDB build: each file
# under tests/ is one program against ../../extension/core_functions/pql/pql.hpp.
#
#   make test        every suite except test_memory
#   make sanitize    the fuzzer, the degenerate-data harness and the link oracle
#                    under ASan/UBSan, which is how most of what they guard was found
#   make test_memory allocation discipline (section A is a known failure: training
#                    still allocates a handful of vectors per step)

CXX      ?= c++
PQL_DIR  := ../../extension/core_functions/pql
BUILD    := build
SUITES   := parse model categorical links degenerate fuzz

.PHONY: test sanitize clean test_memory $(SUITES:%=test_%)

test: $(SUITES:%=test_%)

$(SUITES:%=test_%) test_memory: test_%: $(BUILD)/test_%
	@echo "== $@"
	@$<

$(BUILD)/test_%: tests/test_%.cpp $(PQL_DIR)/pql.hpp | $(BUILD)
	$(CXX) -std=c++17 -O2 -I $(PQL_DIR) -o $@ $<

sanitize: | $(BUILD)
	@set -e; for t in fuzz degenerate links; do \
		echo "== test_$$t (asan+ubsan)"; \
		$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
			-I $(PQL_DIR) -o $(BUILD)/san_$$t tests/test_$$t.cpp; \
		$(BUILD)/san_$$t; \
	done

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
