# Plain add_library (non-unity): pql.hpp is a self-contained DSL translation unit.
add_library(duckdb_core_functions_pql OBJECT pql_functions.cpp)
set(CORE_FUNCTION_FILES
    ${CORE_FUNCTION_FILES} $<TARGET_OBJECTS:duckdb_core_functions_pql>
    PARENT_SCOPE)
