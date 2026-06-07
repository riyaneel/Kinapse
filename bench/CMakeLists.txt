include(FetchContent)
FetchContent_Declare(
		benchmark
		GIT_REPOSITORY https://github.com/google/benchmark.git
		GIT_TAG v1.9.5
		GIT_SHALLOW TRUE
		GIT_PROGRESS FALSE
		SYSTEM
		OVERRIDE_FIND_PACKAGE
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE INTERNAL "")
FetchContent_MakeAvailable(benchmark)

if (TARGET benchmark)
	target_compile_options(benchmark PRIVATE -w)
endif ()

foreach (bench_name IN ITEMS tick layer predict)
	add_executable(bench_${bench_name} src/bench_${bench_name}.cpp)
	target_include_directories(bench_${bench_name} PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/../snn/include
	)

	target_link_libraries(bench_${bench_name} PRIVATE
			core_compiler_flags
			core_engine_lib
			benchmark::benchmark
			Threads::Threads
	)
endforeach ()
