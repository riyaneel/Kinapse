#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <snn/SurrogateGradient.hpp>

namespace engine::snn {

	template <uint32_t N> class alignas(ENGINE_CACHE_LINE_SIZE) ReadoutPopulation {
		static_assert(N > 0 && (N % 8) == 0, "N must be a nonzero multiple of 8.");

		alignas(32) float V_soma_[N];
		alignas(32) float V_soma_pre_spike_[N];
		float theta_stored_{1.0f};

	public:
		static constexpr uint32_t SIZE = N;

		explicit ReadoutPopulation() noexcept {
			reset_state();
		}

		ENGINE_COLD_PATH void reset_state() noexcept {
			std::memset(V_soma_, 0, sizeof(V_soma_));
			std::memset(V_soma_pre_spike_, 0, sizeof(V_soma_pre_spike_));
		}

		/// V = rho * V + I. Saves V_pre_spike, resets on threshold crossing.
		///  out_v_pre receives the continuous pre-spike voltage (pass it to softplus)
		///
		///  @param I_in Input currents[N] need to be 32-byte aligned
		///  @param out_v_pre Output voltages [N] need to be 32-byte aligned
		///  @param rho Somatic leak per tick.
		///  @param theta Firing threshold
		ENGINE_HOT_PATH void
		tick(const float *__restrict I_in, float *__restrict out_v_pre, const float rho, const float theta) noexcept {
			const __m256 rho_v	 = _mm256_set1_ps(rho);
			const __m256 theta_v = _mm256_set1_ps(theta);
			const __m256 zero_v	 = _mm256_setzero_ps();
			theta_stored_		 = theta;

			for (uint32_t i = 0; i < N; i += 8) {
				__m256 v = _mm256_load_ps(V_soma_ + i);
				v		 = _mm256_fmadd_ps(rho_v, v, _mm256_load_ps(I_in + i));
				_mm256_store_ps(V_soma_pre_spike_ + i, v);
				_mm256_store_ps(out_v_pre + i, v);
				v = _mm256_blendv_ps(v, zero_v, _mm256_cmp_ps(v, theta_v, _CMP_GE_OS));
				_mm256_store_ps(V_soma_ + i, v);
			}
		}

		/// Computes the surrogate pseudo-derivative h'(t) for all N neurons.
		/// Dispatches on params.surrogate.
		/// Call immediately after tick() and before any state mutation.
		ENGINE_HOT_PATH void
		compute_h_prime(float *__restrict__ h_prime_out, const float beta_sg, const SurrogateType type) const noexcept {
			const __m256 beta_vec	   = _mm256_set1_ps(beta_sg);
			const __m256 theta_v	   = _mm256_set1_ps(theta_stored_);
			const __m256 one_vec	   = _mm256_set1_ps(1.0f);
			const __m256 two_vec	   = _mm256_set1_ps(2.0f);
			const __m256 zero_vec	   = _mm256_setzero_ps();
			const __m256 sign_mask	   = _mm256_set1_ps(-0.0f);
			const __m256 half_inv_beta = _mm256_set1_ps(0.5f / beta_sg);

			for (uint32_t i = 0; i < N; i += 8) {
				const __m256 v_pre	   = _mm256_load_ps(V_soma_pre_spike_ + i);
				const __m256 delta	   = _mm256_sub_ps(v_pre, theta_v);
				const __m256 abs_delta = _mm256_andnot_ps(sign_mask, delta);

				__m256 h_prime;
				switch (type) {
				case SurrogateType::TRIANGULAR:
					h_prime = _mm256_max_ps(zero_vec, _mm256_fnmadd_ps(beta_vec, abs_delta, one_vec));
					break;
				case SurrogateType::RECTANGULAR:
					h_prime = _mm256_blendv_ps(zero_vec, one_vec, _mm256_cmp_ps(abs_delta, half_inv_beta, _CMP_LT_OS));
					break;
				case SurrogateType::FAST_SIGMOID:
				default: {
					const __m256 denom	= _mm256_fmadd_ps(beta_vec, abs_delta, one_vec);
					const __m256 denom2 = _mm256_mul_ps(denom, denom);
					const __m256 r0		= _mm256_rcp_ps(denom2);
					h_prime				= _mm256_mul_ps(r0, _mm256_fnmadd_ps(denom2, r0, two_vec));
					break;
				}
				}

				_mm256_store_ps(h_prime_out + i, h_prime);
			}
		}

		[[nodiscard]] ENGINE_COLD_PATH float get_soma_v(const uint32_t i) const noexcept {
			return V_soma_[i];
		}
	};
} // namespace engine::snn
