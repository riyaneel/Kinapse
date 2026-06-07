#pragma once

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <memory/IntrusiveNode.hpp>

namespace engine::memory {
	class alignas(ENGINE_CACHE_LINE_SIZE) FreeList {
		uint32_t *indices_{nullptr};
		uint32_t  capacity_{0};
		uint32_t  top_{0};

	public:
		FreeList() noexcept = default;

		~FreeList() noexcept = default;

		FreeList(const FreeList &) = delete;

		FreeList &operator=(const FreeList &) = delete;

		FreeList(FreeList &&) = delete;

		FreeList &operator=(FreeList &&) = delete;

		void initialize(uint32_t *pre_allocated_buffer, const std::size_t capacity) noexcept {
			if (pre_allocated_buffer == nullptr) [[unlikely]] {
				std::fprintf(stderr, "[FATAL]: FreeList: initialization failed with null buffer\n");
				__builtin_trap();
			}

			indices_  = pre_allocated_buffer;
			capacity_ = static_cast<uint32_t>(capacity);
			top_	  = static_cast<uint32_t>(capacity);

			for (uint32_t i = 0; i < capacity_; ++i) {
				indices_[i] = capacity_ - 1 - i;
			}
		}

		[[nodiscard]] ENGINE_HOT_PATH uint32_t pop() noexcept {
			if (top_ == 0) [[unlikely]] {
				return NULL_NODE_IDX;
			}

			return indices_[--top_];
		}

		ENGINE_HOT_PATH void push(const uint32_t idx) noexcept {
			if (top_ >= capacity_) [[unlikely]] {
				__builtin_trap();
			}

			[[assume(top_ < capacity_)]];
			indices_[top_++] = idx;
		}

		[[nodiscard]] ALWAYS_INLINE std::size_t available() const noexcept {
			return top_;
		}

		[[nodiscard]] ALWAYS_INLINE std::size_t capacity() const noexcept {
			return capacity_;
		}
	};
} // namespace engine::memory
