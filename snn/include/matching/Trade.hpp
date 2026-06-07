#pragma once

#include <cstdint>

namespace engine::matching {
#pragma pack(push, 1)
	struct alignas(64) Trade {
		uint64_t taker_id{0};
		uint64_t maker_id{0};
		uint32_t price_ticks{0};
		uint32_t qty_ticks{0};
		uint64_t timestamp_ns{0};
	};
#pragma pack(pop)

	static_assert(sizeof(Trade) == 64, "Trade need to be cache-line aligned.");

#pragma pack(push, 1)
	struct alignas(64) WireOrder {
		uint64_t timestamp_ns;
		int32_t	 side;
		int32_t	 price;
		int32_t	 qty;
		uint64_t user_id;
	};
#pragma pack(pop)

	static_assert(sizeof(WireOrder) == 64, "WireOrder size mismatch with Java gateway MemorySegment");
} // namespace engine::matching
