#pragma once

#include <algorithm>
#include <cstdint>

#include <matching/Bitboard.hpp>
#include <matching/PriceLevel.hpp>
#include <matching/Trade.hpp>
#include <memory/Arena.hpp>
#include <memory/OrderPool.hpp>

namespace engine::matching {
	class alignas(ENGINE_CACHE_LINE_SIZE) OrderEngine {
		memory::OrderPool &pool_;
		PriceLevel		  *bid_levels_{nullptr};
		PriceLevel		  *ask_levels_{nullptr};
		uint32_t		  *order_map_{nullptr};
		Trade			  *trade_ring_{nullptr};
		uint32_t		   trade_ring_mask_{0};
		uint32_t		   trade_tx_idx_{0};
		Bitboard		   bid_board_{};
		Bitboard		   ask_board_{};

		ENGINE_COLD_PATH void reject_fok(const uint32_t taker_idx) const noexcept {
			const auto *taker = pool_.get_node(taker_idx);

			order_map_[taker->order_id] = memory::NULL_NODE_IDX;
			pool_.free(taker_idx);
		}

		ENGINE_COLD_PATH static void handle_invalid_cancel() noexcept {}

		[[nodiscard]] ENGINE_COLD_PATH bool
		evaluate_fok_bid(const uint32_t limit_price, const uint64_t required_qty) const noexcept {
			uint64_t accumulated = 0;
			Bitboard temp_board	 = ask_board_;
			uint32_t best_ask	 = temp_board.get_best_ask();

			while (best_ask != memory::NULL_NODE_IDX && best_ask <= limit_price) {
				accumulated += ask_levels_[best_ask].total_volume_;
				if (accumulated >= required_qty) [[likely]] {
					return true;
				}

				temp_board.clear_price(best_ask);
				best_ask = temp_board.get_best_ask();
			}

			return false;
		}

		[[nodiscard]] ENGINE_COLD_PATH bool
		evaluate_fok_ask(const uint32_t limit_price, const uint64_t required_qty) const noexcept {
			uint64_t accumulated = 0;
			Bitboard temp_board	 = bid_board_;
			uint32_t best_bid	 = temp_board.get_best_bid();

			while (best_bid != memory::NULL_NODE_IDX && best_bid >= limit_price) {
				accumulated += bid_levels_[best_bid].total_volume_;
				if (accumulated >= required_qty) [[likely]] {
					return true;
				}

				temp_board.clear_price(best_bid);
				best_bid = temp_board.get_best_bid();
			}

			return false;
		}

		ALWAYS_INLINE void emit_trade(
			const uint64_t maker_id, const uint64_t taker_id, const uint32_t price, const uint32_t qty, const uint64_t ts
		) noexcept {
			const uint32_t slot = trade_tx_idx_ & trade_ring_mask_;
			Trade		  &t	= trade_ring_[slot];
			t.maker_id			= maker_id;
			t.taker_id			= taker_id;
			t.price_ticks		= price;
			t.qty_ticks			= qty;
			t.timestamp_ns		= ts;
			++trade_tx_idx_;
		}

	public:
		explicit OrderEngine(
			memory::ArenaAllocator &arena,
			memory::OrderPool	   &pool,
			const uint32_t			max_prices,
			const uint32_t			max_orders,
			const uint32_t			trade_ring_capacity
		) noexcept
			: pool_(pool) {
			bid_levels_		 = arena.allocate<PriceLevel>(max_prices);
			ask_levels_		 = arena.allocate<PriceLevel>(max_prices);
			order_map_		 = arena.allocate<uint32_t>(max_orders);
			trade_ring_		 = arena.allocate<Trade>(trade_ring_capacity);
			trade_ring_mask_ = trade_ring_capacity - 1;

			for (uint32_t i = 0; i < max_prices; ++i) {
				bid_levels_[i] = PriceLevel{};
				ask_levels_[i] = PriceLevel{};
			}

			for (uint32_t i = 0; i < max_orders; ++i) {
				order_map_[i] = memory::NULL_NODE_IDX;
			}
		}

		ENGINE_HOT_PATH void submit_bid(const uint32_t taker_idx) noexcept {
			auto *taker					= pool_.get_node(taker_idx);
			order_map_[taker->order_id] = taker_idx;

			if (taker->tif == 2) [[unlikely]] {
				if (!evaluate_fok_bid(static_cast<uint32_t>(taker->price_ticks), taker->qty_ticks)) {
					reject_fok(taker_idx);
					return;
				}
			}

			while (taker->qty_ticks > 0) {
				const uint32_t best_ask = ask_board_.get_best_ask();
				if (best_ask == memory::NULL_NODE_IDX || best_ask > taker->price_ticks) {
					break;
				}

				PriceLevel	  &level	 = ask_levels_[best_ask];
				const uint32_t maker_idx = level.head_idx_;
				auto		  *maker	 = pool_.get_node(maker_idx);
				const auto	   trade_qty = static_cast<uint32_t>(std::min(taker->qty_ticks, maker->qty_ticks));
				taker->qty_ticks -= trade_qty;
				maker->qty_ticks -= trade_qty;

				level.total_volume_ -= trade_qty;
				emit_trade(maker->order_id, taker->order_id, best_ask, trade_qty, taker->timestamp_ns);
				if (maker->qty_ticks == 0) {
					level.remove(maker_idx, pool_);
					order_map_[maker->order_id] = memory::NULL_NODE_IDX;
					pool_.free(maker_idx);

					if (level.is_empty()) [[unlikely]] {
						ask_board_.clear_price(best_ask);
					}
				}
			}

			if (taker->qty_ticks > 0) [[unlikely]] {
				if (taker->tif == 0) [[likely]] {
					bid_levels_[taker->price_ticks].push_back(taker_idx, pool_);
					bid_board_.set_price(static_cast<uint32_t>(taker->price_ticks));
				} else {
					pool_.free(taker_idx);
					order_map_[taker->order_id] = memory::NULL_NODE_IDX;
				}
			} else [[likely]] {
				pool_.free(taker_idx);
				order_map_[taker->order_id] = memory::NULL_NODE_IDX;
			}
		}

		ENGINE_HOT_PATH void submit_ask(const uint32_t taker_idx) noexcept {
			auto *taker					= pool_.get_node(taker_idx);
			order_map_[taker->order_id] = taker_idx;

			if (taker->tif == 2) [[unlikely]] {
				if (!evaluate_fok_ask(static_cast<uint32_t>(taker->price_ticks), taker->qty_ticks)) {
					reject_fok(taker_idx);
					return;
				}
			}

			uint32_t best_bid = bid_board_.get_best_bid();
			while (taker->qty_ticks > 0 && best_bid != memory::NULL_NODE_IDX && best_bid >= taker->price_ticks) {
				PriceLevel &level	  = bid_levels_[best_bid];
				uint32_t	maker_idx = level.head_idx_;

				while (maker_idx != memory::NULL_NODE_IDX && taker->qty_ticks > 0) {
					auto	  *maker	 = pool_.get_node(maker_idx);
					const auto match_qty = static_cast<uint32_t>(std::min(taker->qty_ticks, maker->qty_ticks));

					taker->qty_ticks -= match_qty;
					maker->qty_ticks -= match_qty;

					level.total_volume_ -= match_qty;
					emit_trade(maker->order_id, taker->order_id, best_bid, match_qty, taker->timestamp_ns);
					const uint32_t next_maker_idx = maker->next_idx;
					if (maker->qty_ticks == 0) [[likely]] {
						level.remove(maker_idx, pool_);
						order_map_[maker->order_id] = memory::NULL_NODE_IDX;
						pool_.free(maker_idx);
					}

					maker_idx = next_maker_idx;
				}

				if (level.is_empty()) [[unlikely]] {
					bid_board_.clear_price(best_bid);
				}

				best_bid = bid_board_.get_best_bid();
			}

			if (taker->qty_ticks > 0) [[unlikely]] {
				if (taker->tif == 0) [[likely]] {
					// GTC
					ask_levels_[taker->price_ticks].push_back(taker_idx, pool_);
					ask_board_.set_price(static_cast<uint32_t>(taker->price_ticks));
				} else {
					// IOC
					pool_.free(taker_idx);
					order_map_[taker->order_id] = memory::NULL_NODE_IDX;
				}
			} else [[likely]] {
				pool_.free(taker_idx);
				order_map_[taker->order_id] = memory::NULL_NODE_IDX;
			}
		}

		ENGINE_HOT_PATH void cancel(const uint64_t order_id) noexcept {
			const uint32_t node_idx = order_map_[order_id];
			if (node_idx == memory::NULL_NODE_IDX) [[unlikely]] {
				handle_invalid_cancel();
				return;
			}

			if (const auto *order = pool_.get_node(node_idx); order->side == 0) {
				bid_levels_[order->price_ticks].remove(node_idx, pool_);
				if (bid_levels_[order->price_ticks].is_empty()) [[unlikely]] {
					bid_board_.clear_price(static_cast<uint32_t>(order->price_ticks));
				}
			} else {
				ask_levels_[order->price_ticks].remove(node_idx, pool_);
				if (ask_levels_[order->price_ticks].is_empty()) [[unlikely]] {
					ask_board_.clear_price(static_cast<uint32_t>(order->price_ticks));
				}
			}

			order_map_[order_id] = memory::NULL_NODE_IDX;
			pool_.free(node_idx);
		}

		[[nodiscard]] ALWAYS_INLINE const Trade *get_trade_ring() const noexcept {
			return trade_ring_;
		}

		[[nodiscard]] ALWAYS_INLINE uint32_t get_trade_count() const noexcept {
			return trade_tx_idx_;
		}

		[[nodiscard]] ALWAYS_INLINE uint32_t get_best_bid() const noexcept {
			return bid_board_.get_best_bid();
		}

		[[nodiscard]] ALWAYS_INLINE uint32_t get_best_ask() const noexcept {
			return ask_board_.get_best_ask();
		}

		[[nodiscard]] ALWAYS_INLINE uint64_t get_bid_volume(const uint32_t price) const noexcept {
			return bid_levels_[price].total_volume_;
		}

		[[nodiscard]] ALWAYS_INLINE uint64_t get_ask_volume(const uint32_t price) const noexcept {
			return ask_levels_[price].total_volume_;
		}
	};
} // namespace engine::matching
