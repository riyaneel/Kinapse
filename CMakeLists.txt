cmake_minimum_required(VERSION 3.31.0 FATAL_ERROR)

project(Kinapse
		LANGUAGES CXX
		DESCRIPTION "Kinapse SNN"
		VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif ()

set(ENGINE_SANITIZER "none" CACHE STRING "Sanitizer to use (asan, asan_ubsan, tsan)")

add_library(core_compiler_flags INTERFACE)

target_compile_options(core_compiler_flags INTERFACE
		-Wall -Wextra -Werror -Wpedantic -Wshadow
		-Wconversion -Wsign-conversion -Wno-old-style-cast
		-fno-exceptions -fno-rtti -fstrict-aliasing
		-mno-avx512f
		$<$<CONFIG:Release>:-O3 -march=native -mtune=native -flto -fno-math-errno>
		$<$<CONFIG:Debug>:-O1 -g3 -fno-omit-frame-pointer>
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT ENGINE_SANITIZER STREQUAL "none")
	if (ENGINE_SANITIZER STREQUAL "asan")
		target_compile_options(core_compiler_flags INTERFACE -fsanitize=address)
		target_link_options(core_compiler_flags INTERFACE -fsanitize=address)
	elseif (ENGINE_SANITIZER STREQUAL "asan_ubsan")
		target_compile_options(core_compiler_flags INTERFACE -fsanitize=address,undefined)
		target_link_options(core_compiler_flags INTERFACE -fsanitize=address,undefined)
	elseif (ENGINE_SANITIZER STREQUAL "tsan")
		target_compile_options(core_compiler_flags INTERFACE -fsanitize=thread)
		target_link_options(core_compiler_flags INTERFACE -fsanitize=thread)
	endif ()
endif ()

target_link_options(core_compiler_flags INTERFACE
		$<$<CXX_COMPILER_ID:Clang>:-fuse-ld=lld>
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if (COMPILER_SUPPORTS_AVX2)
	target_compile_options(core_compiler_flags INTERFACE -mavx2 -mfma)
else ()
	message(WARNING "AVX2 is not supported by the target architecture. SNN vectorization will degrade.")
endif ()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(snn)
add_subdirectory(trainer)
add_subdirectory(bench)
