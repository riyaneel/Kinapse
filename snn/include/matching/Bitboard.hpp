#pragma once

#include <bit>
#include <cstdint>

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <memory/IntrusiveNode.hpp>

namespace engine::matching {
	class alignas(ENGINE_CACHE_LINE_SIZE) Bitboard {
		uint64_t root_{0};	  /* Level 1: 1 block of 64 bits (262 144 prices) */
		uint64_t l2_[64]{};	  /* Level 2: 64 blocks of 64 bits (4096 subgroups) */
		uint64_t l3_[4096]{}; /* Level 3: 4096 blocks of 64 bits (262 144 exact prices) */

	public:
		Bitboard() noexcept = default;

		ENGINE_HOT_PATH void set_price(const uint32_t price_tick) noexcept {
			const uint32_t l2_idx = price_tick >> 6;
			const uint32_t l1_idx = price_tick >> 12;

			l3_[l2_idx] |= 1ULL << (price_tick & 0x3F);
			l2_[l1_idx] |= 1ULL << (l2_idx & 0x3F);
			root_ |= 1ULL << l1_idx;
		}

		ENGINE_HOT_PATH void clear_price(const uint32_t price_tick) noexcept {
			const uint32_t l2_idx = price_tick >> 6;
			const uint32_t l1_idx = price_tick >> 12;

			l3_[l2_idx] &= ~(1ULL << (price_tick & 0x3F));
			l2_[l1_idx] &= ~(static_cast<uint64_t>(l3_[l2_idx] == 0) << (l2_idx & 0x3F));
			root_ &= ~(static_cast<uint64_t>(l2_[l1_idx] == 0) << l1_idx);
		}

		[[nodiscard]] ENGINE_HOT_PATH uint32_t get_best_ask() const noexcept {
			if (root_ == 0) [[unlikely]] {
				return memory::NULL_NODE_IDX;
			}

			const auto l1 = static_cast<uint32_t>(std::countr_zero(root_));
			const auto l2 = static_cast<uint32_t>(std::countr_zero(l2_[l1]));
			const auto l3 = static_cast<uint32_t>(std::countr_zero(l3_[l1 << 6 | l2]));

			return l1 << 12 | l2 << 6 | l3;
		}

		[[nodiscard]] ENGINE_HOT_PATH uint32_t get_best_bid() const noexcept {
			if (root_ == 0) [[unlikely]] {
				return memory::NULL_NODE_IDX;
			}

			const auto l1 = static_cast<uint32_t>(63 - std::countl_zero(root_));
			const auto l2 = static_cast<uint32_t>(63 - std::countl_zero(l2_[l1]));
			const auto l3 = static_cast<uint32_t>(63 - std::countl_zero(l3_[l1 << 6 | l2]));

			return l1 << 12 | l2 << 6 | l3;
		}
	};
} // namespace engine::matching
