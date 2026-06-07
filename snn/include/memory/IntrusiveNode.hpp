#pragma once

#include <cstdint>

namespace engine::memory {
	constexpr uint32_t NULL_NODE_IDX = 0xFFFFFFFF; /* Sentinel value replacing nullptr for 32-bit indices */

#pragma pack(push, 1)
	struct alignas(64) IntrusiveOrderNode {
		// Intrusive List Topology (8 bytes)
		uint32_t prev_idx{NULL_NODE_IDX}; /* Previous index */
		uint32_t next_idx{NULL_NODE_IDX}; /* Next index */
		// Trading Payload (44 bytes)
		uint64_t timestamp_ns{0};
		uint64_t order_id{0};
		uint64_t user_id{0};
		uint64_t price_ticks{0};
		uint64_t qty_ticks{0};
		uint16_t asset_pair_id{0};
		uint8_t	 side{0};		 // 0 = Bid, 1 = Ask
		uint8_t	 tif{0};		 // 0 = GTC, 1 = IOC, 2 = FOK
		uint8_t	 padding_[12]{}; /* Explicit padding to satisfy the 64-byte cache line boundary */
	};
#pragma pack(pop)

	static_assert(sizeof(IntrusiveOrderNode) == 64, "IntrusiveOrderNode size violates strict 64-byte boundary");
	static_assert(alignof(IntrusiveOrderNode) == 64, "IntrusiveOrderNode alignment violates 64-byte boundary");
} // namespace engine::memory
