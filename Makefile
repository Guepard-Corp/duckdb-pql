PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=pql
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

#### Standalone tests of the language itself (no DuckDB build needed)
CPP_TESTS=parse model categorical links degenerate fuzz
CPP_TEST_DIR=build/cpp_tests
CXX ?= c++

$(CPP_TEST_DIR):
	mkdir -p $(CPP_TEST_DIR)

test_cpp: | $(CPP_TEST_DIR)
	@set -e; for t in $(CPP_TESTS); do \
		echo "== test_$$t"; \
		$(CXX) -std=c++17 -O2 -I src/include -o $(CPP_TEST_DIR)/test_$$t test/cpp/test_$$t.cpp; \
		$(CPP_TEST_DIR)/test_$$t; \
	done

test_cpp_sanitize: | $(CPP_TEST_DIR)
	@set -e; for t in fuzz degenerate links; do \
		echo "== test_$$t (asan+ubsan)"; \
		$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I src/include \
			-o $(CPP_TEST_DIR)/san_$$t test/cpp/test_$$t.cpp; \
		$(CPP_TEST_DIR)/san_$$t; \
	done

test_memory: | $(CPP_TEST_DIR)
	$(CXX) -std=c++17 -O2 -I src/include -o $(CPP_TEST_DIR)/test_memory test/cpp/test_memory.cpp
	$(CPP_TEST_DIR)/test_memory

.PHONY: test_cpp test_cpp_sanitize test_memory
