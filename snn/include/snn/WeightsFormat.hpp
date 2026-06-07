#pragma once

#include <cstdint>

namespace engine::snn {

	/// Magic number written at the start of every weight's file. Equal to "SNNR_W1\0"
	static constexpr uint64_t kWeightsMagic = 0x534E4E525F573100ULL;

	/// Fixed seed for deterministic recurrent mask generation, produce ~10% sparse mask
	static constexpr uint64_t kRecMaskSeed = 0xDEADBEEFCAFEBABEULL;
} // namespace engine::snn
