#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include <engine.hpp>
#include <memory/Arena.hpp>

namespace engine::snn {

	struct alignas(8) SparseBlock {
		uint32_t w_offset; /* Flat index into W_: j * N_OUT + b * 8 */
		uint16_t x_idx;	   /* Presynaptic index for broadcast */
		uint16_t i_out;	   /* Output buffer offset (b * 8) */
	};

	/// Fully connected synaptic layer with online eligibility traces for e-prop.
	/// Weight layout: column-major, W_[j * N_OUT + i].
	/// Enables broadcast-fmadd GEMV without a gather: one scalar x[j] fans out across an N_OUT-wide AVX2 column.
	///
	/// @tparam N_IN   Presynaptic count.
	/// @tparam N_OUT  Postsynaptic count. Must satisfy N_OUT % 8 == 0.
	template <uint32_t N_IN, uint32_t N_OUT> class alignas(ENGINE_CACHE_LINE_SIZE) SynapticLayer {
		static_assert(N_IN > 0, "N_IN must be positive.");
		static_assert(N_OUT > 0 && (N_OUT % 8) == 0, "N_OUT must be a nonzero multiple of 8.");
		static_assert((N_IN * N_OUT) % 8 == 0, "N_IN * N_OUT must be a multiple of 8.");

		alignas(32) float W_[N_IN * N_OUT];
		alignas(32) float bias_[N_OUT];
		alignas(32) float E_trace_[N_IN * N_OUT];  /* Same column-major layout as W_ */
		alignas(32) float E2_trace_[N_IN * N_OUT]; /* EMA of E^2 for RMSProp normalization (Zenke & Ganguli 2018). */
		SparseBlock active_blocks_[N_IN * (N_OUT / 8)]{};
		uint32_t	n_active_blocks_{0};

		[[nodiscard]] ALWAYS_INLINE static uint64_t xorshift64(uint64_t s) noexcept {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			return s;
		}

	public:
		SynapticLayer() noexcept {
			std::memset(W_, 0, sizeof(W_));
			std::memset(bias_, 0, sizeof(bias_));
			std::memset(E_trace_, 0, sizeof(E_trace_));
			std::fill_n(E2_trace_, N_IN * N_OUT, 1.0f);
		}

		/// Scans W_ and records (j, b) pairs where the 8-element block is non-zero.
		/// Call once after weights are loaded or apply_mask() is done. Cold path only.
		ENGINE_COLD_PATH void build_sparse_index() noexcept {
			const __m256 zero_v = _mm256_setzero_ps();
			n_active_blocks_	= 0;

			for (uint32_t j = 0; j < N_IN; ++j) {
				const float *col = W_ + j * N_OUT;
				for (uint32_t b = 0; b < N_OUT / 8; ++b) {
					if (_mm256_movemask_ps(_mm256_cmp_ps(_mm256_load_ps(col + b * 8), zero_v, _CMP_NEQ_OS)))
						active_blocks_[n_active_blocks_++] = {
							.w_offset = j * N_OUT + b * 8,
							.x_idx	  = static_cast<uint16_t>(j),
							.i_out	  = static_cast<uint16_t>(b * 8)
						};
				}
			}
		}

		/// Sparse GEMV: skips zero blocks, identical output to forward().
		/// Requires build_sparse_index() to have been called after the last weight mutation.
		ENGINE_HOT_PATH void forward_sparse(const float *__restrict__ x, float *__restrict__ I_out) const noexcept {
			for (uint32_t i = 0; i < N_OUT; i += 8) {
				_mm256_store_ps(I_out + i, _mm256_load_ps(bias_ + i));
			}

			for (uint32_t k = 0; k < n_active_blocks_; ++k) {
				if (k + 8 < n_active_blocks_) [[likely]] {
					__builtin_prefetch(W_ + active_blocks_[k + 8].w_offset, 0, 3); // Prefetch 8 blocks to hide L2 latency
				}

				const SparseBlock &blk = active_blocks_[k];
				const __m256	   x_j = _mm256_broadcast_ss(x + blk.x_idx);
				__m256			   acc = _mm256_load_ps(I_out + blk.i_out);
				acc					   = _mm256_fmadd_ps(_mm256_load_ps(W_ + blk.w_offset), x_j, acc);
				_mm256_store_ps(I_out + blk.i_out, acc);
			}
		}

		/// Forward pass: I_out = W @ x + bias.
		ENGINE_HOT_PATH void forward(const float *__restrict__ x, float *__restrict__ I_out) const noexcept {
			for (uint32_t i = 0; i < N_OUT; i += 8) {
				_mm256_store_ps(I_out + i, _mm256_load_ps(bias_ + i));
			}

			for (uint32_t j = 0; j < N_IN; ++j) {
				const __m256 x_j	 = _mm256_broadcast_ss(x + j);
				const float *W_col_j = W_ + j * N_OUT;

				for (uint32_t i = 0; i < N_OUT; i += 8) {
					__m256 I_vec = _mm256_load_ps(I_out + i);
					I_vec		 = _mm256_fmadd_ps(_mm256_load_ps(W_col_j + i), x_j, I_vec);
					_mm256_store_ps(I_out + i, I_vec);
				}
			}
		}

		/// E_ji = rho_e * E_ji + x_j * h'_i  (Bellec et al. 2020, pure forward e-prop).
		/// @param x_in     Presynaptic activity [N_IN], 32-byte aligned.
		/// @param h_prime  Surrogate gradients [N_OUT] from DendriticPopulation::compute_h_prime().
		/// @param rho_e    Trace decay factor, typically matched to somatic rho_s.
		ENGINE_HOT_PATH void
		update_traces(const float *__restrict__ x_in, const float *__restrict__ h_prime, const float rho_e) noexcept {
			const __m256 rho_vec  = _mm256_set1_ps(rho_e);
			const __m256 beta2	  = _mm256_set1_ps(0.999f);
			const __m256 one_m_b2 = _mm256_set1_ps(1.0f - 0.999f);

			for (uint32_t j = 0; j < N_IN; ++j) {
				const __m256 x_j	  = _mm256_broadcast_ss(x_in + j);
				float		*E_col_j  = E_trace_ + j * N_OUT;
				float		*E2_col_j = E2_trace_ + j * N_OUT;

				for (uint32_t i = 0; i < N_OUT; i += 8) {
					// E = rho_e * E + x_j * h'_i
					__m256 e = _mm256_load_ps(E_col_j + i);
					e		 = _mm256_fmadd_ps(rho_vec, e, _mm256_mul_ps(x_j, _mm256_load_ps(h_prime + i)));
					_mm256_store_ps(E_col_j + i, e);
					// E2 = beta2 * E2 + (1 - beta2) * E^2 (RMSProp denominator)
					__m256 e2 = _mm256_load_ps(E2_col_j + i);
					e2		  = _mm256_fmadd_ps(beta2, e2, _mm256_mul_ps(one_m_b2, _mm256_mul_ps(e, e)));
					_mm256_store_ps(E2_col_j + i, e2);
				}
			}
		}

		/// W = W*(1-decay) + eta * L * E/sqrt(E2+eps). RMSProp-normalized e-prop (Zenke & Ganguli 2018).
		/// @param L_t          Global learning signal (advantage or TD error).
		/// @param eta          Learning rate.
		/// @param weight_decay L2 regularization coefficient applied per tick.
		ENGINE_HOT_PATH void apply_e_prop(const float L_t, const float eta, const float weight_decay) noexcept {
			const __m256	   factor  = _mm256_set1_ps(L_t * eta);
			const __m256	   keep	   = _mm256_set1_ps(1.0f - weight_decay);
			const __m256	   w_min   = _mm256_set1_ps(-10.0f);
			const __m256	   w_max   = _mm256_set1_ps(10.0f);
			const __m256	   eps_v   = _mm256_set1_ps(1e-8f);
			const __m256	   half_v  = _mm256_set1_ps(0.5f);
			const __m256	   three_v = _mm256_set1_ps(3.0f);
			constexpr uint32_t TOTAL   = N_IN * N_OUT;

			for (uint32_t k = 0; k < TOTAL; k += 8) {
				const __m256 e		= _mm256_load_ps(E_trace_ + k);
				const __m256 e2_eps = _mm256_add_ps(_mm256_load_ps(E2_trace_ + k), eps_v);
				// e_norm = e / sqrt(e2+eps) = e * rsqrt(e2 + eps); NR-refined
				const __m256 r0 = _mm256_rsqrt_ps(e2_eps);
				const __m256 r1 =
					_mm256_mul_ps(_mm256_mul_ps(half_v, r0), _mm256_fnmadd_ps(e2_eps, _mm256_mul_ps(r0, r0), three_v));
				const __m256 e_norm = _mm256_mul_ps(e, r1);
				__m256		 w		= _mm256_mul_ps(_mm256_load_ps(W_ + k), keep);
				w					= _mm256_fmadd_ps(factor, e_norm, w);
				_mm256_store_ps(W_ + k, _mm256_max_ps(w_min, _mm256_min_ps(w, w_max)));
			}
		}

		/// W_ji = W_ji*(1-decay) + eta * delta_i * E_ji/sqrt(E2_ji+eps).
		/// delta_i is the per-output-neuron learning signal: L(t) * d_act_i/dV_i * h'_i(t).
		ENGINE_HOT_PATH void
		apply_e_prop_signal(const float *__restrict__ signal, const float eta, const float weight_decay) noexcept {
			const __m256 keep	 = _mm256_set1_ps(1.0f - weight_decay);
			const __m256 eta_v	 = _mm256_set1_ps(eta);
			const __m256 w_min	 = _mm256_set1_ps(-10.0f);
			const __m256 w_max	 = _mm256_set1_ps(10.0f);
			const __m256 eps_v	 = _mm256_set1_ps(1e-8f);
			const __m256 half_v	 = _mm256_set1_ps(0.5f);
			const __m256 three_v = _mm256_set1_ps(3.0f);

			for (uint32_t j = 0; j < N_IN; ++j) {
				const float *E_col_j  = E_trace_ + j * N_OUT;
				const float *E2_col_j = E2_trace_ + j * N_OUT;
				float		*W_col_j  = W_ + j * N_OUT;

				for (uint32_t i = 0; i < N_OUT; i += 8) {
					const __m256 delta_i = _mm256_load_ps(signal + i);
					const __m256 e		 = _mm256_load_ps(E_col_j + i);
					const __m256 e2_eps	 = _mm256_add_ps(_mm256_load_ps(E2_col_j + i), eps_v);
					const __m256 r0		 = _mm256_rsqrt_ps(e2_eps);
					const __m256 r1		 = _mm256_mul_ps(
						 _mm256_mul_ps(half_v, r0), _mm256_fnmadd_ps(e2_eps, _mm256_mul_ps(r0, r0), three_v)
					 );
					const __m256 e_norm = _mm256_mul_ps(e, r1);
					__m256		 w		= _mm256_mul_ps(_mm256_load_ps(W_col_j + i), keep);
					w					= _mm256_fmadd_ps(_mm256_mul_ps(eta_v, delta_i), e_norm, w);
					_mm256_store_ps(W_col_j + i, _mm256_max_ps(w_min, _mm256_min_ps(w, w_max)));
				}
			}
		}

		/// Zeroes weights at positions where mask[k] == 0.
		/// Must be called after every apply_e_prop() on masked layers to enforce sparsity.
		/// @param mask must be size N_IN * N_OUT, each byte is 0 (pruned) or 1 (active).
		ENGINE_HOT_PATH void apply_mask(const uint8_t *__restrict__ mask) noexcept {
			constexpr uint32_t TOTAL = N_IN * N_OUT;

			for (uint32_t k = 0; k < TOTAL; k += 8) {
				// Expand 8 uint8 (0 or 1) to float (0.0 or 1.0) via int32.
				const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mask + k));
				const __m256  fmask = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
				_mm256_store_ps(W_ + k, _mm256_mul_ps(_mm256_load_ps(W_ + k), fmask));
			}
		}

		/// Zeroes eligibility traces. Call on episode or regime boundary.
		ENGINE_COLD_PATH void reset_traces() noexcept {
			std::memset(E_trace_, 0, sizeof(E_trace_));
			std::fill_n(E2_trace_, N_IN * N_OUT, 1.0f);
		}

		/// Xavier uniform init: W ~ U(-sqrt(6/(N_IN+N_OUT)), +sqrt(6/(N_IN+N_OUT))). Biases left at zero.
		ENGINE_COLD_PATH void xavier_init(uint64_t &rng_state) noexcept {
			const float limit = std::sqrt(6.0f / static_cast<float>(N_IN + N_OUT));
			const float scale = 2.0f * limit;

			for (uint32_t k = 0; k < N_IN * N_OUT; ++k) {
				rng_state = xorshift64(rng_state);
				// Mix both halves before extracting to remove linear bias in the high word.
				const uint64_t mixed = rng_state ^ (rng_state >> 32);
				const float	   u	 = static_cast<float>(mixed & 0xFFFFFFFFULL) * (1.0f / 4294967296.0f);
				W_[k]				 = u * scale - limit;
			}
		}

		[[nodiscard]] ALWAYS_INLINE float get_weight(const uint32_t j, const uint32_t i) const noexcept {
			return W_[j * N_OUT + i];
		}

		[[nodiscard]] ALWAYS_INLINE float *get_weights() noexcept {
			return W_;
		}

		[[nodiscard]] ALWAYS_INLINE const float *get_weights() const noexcept {
			return W_;
		}

		[[nodiscard]] ALWAYS_INLINE float *get_biases() noexcept {
			return bias_;
		}

		[[nodiscard]] ALWAYS_INLINE const float *get_biases() const noexcept {
			return bias_;
		}
	};
} // namespace engine::snn
