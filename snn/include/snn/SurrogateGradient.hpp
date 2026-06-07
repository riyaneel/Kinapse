#pragma once

#include <cstdint>

namespace engine::snn {

	/// Surrogate gradient estimator for the spike discontinuity.
	enum class SurrogateType : uint8_t {
		TRIANGULAR,	  /* h'(t) = max(0, 1 - beta * |V - theta|). Bellec et al. 2020 */
		RECTANGULAR,  /* h'(t) = 1_{|V - theta| < 1/(2 * beta)} */
		FAST_SIGMOID, /* h'(t) = 1 / (1 + beta * |V - theta|)^2 */
	};
} // namespace engine::snn
