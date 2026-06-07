#pragma once

#include <cstdint>

#include <engine.hpp>
#include <memory/IntrusiveNode.hpp>
#include <memory/OrderPool.hpp>

namespace engine::matching {
	struct alignas(16) PriceLevel {
		uint32_t head_idx_{memory::NULL_NODE_IDX};
		uint32_t tail_idx_{memory::NULL_NODE_IDX};
		uint64_t total_volume_{0};

		[[nodiscard]] ALWAYS_INLINE bool is_empty() const noexcept {
			return head_idx_ == memory::NULL_NODE_IDX;
		}

		ENGINE_HOT_PATH void push_back(const uint32_t order_idx, const memory::OrderPool &pool) noexcept {
			memory::IntrusiveOrderNode *new_node = pool.get_node(order_idx);
			new_node->next_idx					 = memory::NULL_NODE_IDX;
			new_node->prev_idx					 = tail_idx_;

			if (is_empty()) [[unlikely]] {
				head_idx_ = order_idx;
			} else [[likely]] {
				pool.get_node(tail_idx_)->next_idx = order_idx;
			}

			tail_idx_ = order_idx;
			total_volume_ += new_node->qty_ticks;
		}

		ENGINE_HOT_PATH void remove(const uint32_t order_idx, const memory::OrderPool &pool) noexcept {
			memory::IntrusiveOrderNode *target = pool.get_node(order_idx);

			if (target->prev_idx != memory::NULL_NODE_IDX) [[likely]] {
				pool.get_node(target->prev_idx)->next_idx = target->next_idx;
			} else [[unlikely]] {
				head_idx_ = target->next_idx;
			}

			if (target->next_idx != memory::NULL_NODE_IDX) [[likely]] {
				pool.get_node(target->next_idx)->prev_idx = target->prev_idx;
			} else [[unlikely]] {
				tail_idx_ = target->prev_idx;
			}

			total_volume_ -= target->qty_ticks;
			target->prev_idx = memory::NULL_NODE_IDX;
			target->next_idx = memory::NULL_NODE_IDX;
		}
	};
} // namespace engine::matching
