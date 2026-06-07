#pragma once

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <memory/FreeList.hpp>
#include <memory/IntrusiveNode.hpp>

namespace engine::memory {
	class alignas(ENGINE_CACHE_LINE_SIZE) OrderPool {
		IntrusiveOrderNode *nodes_{nullptr};
		FreeList			free_list_{};
		uint32_t			capacity_{0};

	public:
		OrderPool() noexcept = default;

		~OrderPool() noexcept = default;

		OrderPool(const OrderPool &) = delete;

		OrderPool &operator=(const OrderPool &) = delete;

		OrderPool(OrderPool &&) = delete;

		OrderPool &operator=(OrderPool &&) = delete;

		void initialize(ArenaAllocator &arena, const uint32_t capacity) noexcept {
			capacity_ = capacity;
			nodes_ = arena.allocate<IntrusiveOrderNode>(capacity_);
			auto *index_buffer = arena.allocate<uint32_t>(capacity_);
			free_list_.initialize(index_buffer, capacity_);

			for (uint32_t i = 0; i < capacity_; ++i) {
				new (&nodes_[i]) IntrusiveOrderNode();
			}
		}

		[[nodiscard]] ENGINE_HOT_PATH uint32_t allocate() noexcept {
			return free_list_.pop();
		}

		ENGINE_HOT_PATH void free(const uint32_t idx) noexcept {
			free_list_.push(idx);
		}

		[[nodiscard]] ALWAYS_INLINE IntrusiveOrderNode *get_node(const uint32_t idx) const noexcept {
			[[assume(idx < capacity_)]];
			return &nodes_[idx];
		}

		[[nodiscard]] ALWAYS_INLINE uint32_t available() const noexcept {
			return static_cast<uint32_t>(free_list_.available());
		}
	};
} // namespace engine::memory
