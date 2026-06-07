#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <sys/mman.h>

#include <engine.hpp>

#ifndef ENGINE_CACHE_LINE_SIZE
#define ENGINE_CACHE_LINE_SIZE 128
#endif // #ifndef ENGINE_CACHE_LIE_SIZE

namespace engine::memory {
	class alignas(ENGINE_CACHE_LINE_SIZE) ArenaAllocator {
		uint8_t	   *buffer_{nullptr};
		std::size_t capacity_{0};
		std::size_t offset_{0};

		[[noreturn]] ENGINE_COLD_PATH static void handle_oom() noexcept {
			std::fprintf(stderr, "[FATAL]: ArenaAllocator: Out of Memory\n");
			__builtin_trap();
		}

	public:
		[[nodiscard]] explicit ArenaAllocator(const std::size_t capacity) noexcept {
			const auto page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
			capacity_			 = (capacity + page_size - 1) & ~(page_size - 1);

			void *ptr = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
			if (ptr == MAP_FAILED) [[unlikely]] {
				std::fprintf(stderr, "[FATAL]: ArenaAllocator: mmap failed to allocate memory\n");
				__builtin_trap();
			}

			buffer_ = static_cast<uint8_t *>(ptr);
			offset_ = 0;
		}

		~ArenaAllocator() noexcept {
			if (buffer_) [[likely]] {
				munmap(buffer_, capacity_);
			}
		}

		ArenaAllocator() = delete;

		ArenaAllocator(const ArenaAllocator &) = delete;

		ArenaAllocator &operator=(const ArenaAllocator &) = delete;

		ArenaAllocator(ArenaAllocator &&) = delete;

		ArenaAllocator &operator=(ArenaAllocator &&) = delete;

		template <typename T> [[nodiscard]] ENGINE_HOT_PATH T *allocate(const std::size_t count = 1) noexcept {
			constexpr std::size_t alignment		  = alignof(T) > ENGINE_CACHE_LINE_SIZE ? alignof(T) : ENGINE_CACHE_LINE_SIZE;
			constexpr auto		  cache_line_size = static_cast<std::size_t>(ENGINE_CACHE_LINE_SIZE);
			const std::size_t	  requested_bytes = count * sizeof(T);

			const std::size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
			const std::size_t padded_bytes	 = (requested_bytes + cache_line_size - 1) & ~(cache_line_size - 1);
			const std::size_t new_offset	 = aligned_offset + padded_bytes;
			if (new_offset > capacity_) [[unlikely]] {
				handle_oom();
			}

			T *ptr	= reinterpret_cast<T *>(buffer_ + aligned_offset);
			offset_ = new_offset;

#if defined(__clang__) || defined(__GNUC__)
			return static_cast<T *>(__builtin_assume_aligned(ptr, alignment));
#else  // #if defined(__clang__) || defined(__GNUC__)
			return ptr;
#endif // #if defined(__clang__) || defined(__GNUC__) # else
		}

		void reset() noexcept {
			offset_ = 0;
		}

		[[nodiscard]] ALWAYS_INLINE std::size_t used() const noexcept {
			return offset_;
		}

		[[nodiscard]] ALWAYS_INLINE std::size_t capacity() const noexcept {
			return capacity_;
		}
	};
} // namespace engine::memory
