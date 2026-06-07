#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <snn/SurrogateGradient.hpp>

namespace engine::snn {

	/// Hyperparameters for the ALIF dendritic population with AMPA/NMDA dual kinetics.
	struct alignas(ENGINE_CACHE_LINE_SIZE) DendriticParams {
		float		  rho_ampa{0.80f};	/* AMPA decay per tick: tau_AMPA ~= 5ms at 1ms/tick -> exp(-1/5) ~= 0.82 */
		float		  rho_nmda{0.98f};	/* NMDA decay per tick: tau_NMDA ~= 50ms at 1ms/tick -> exp(-1/50) ~= 0.98 */
		float		  g_nmda{0.50f};	/* NMDA conductance weight relative to AMPA */
		float		  mg_conc{1.00f};	/* Mg2+ extracellular concentration (mM) */
		float		  k_unblock{0.05f}; /* Mg2⁺ unblocking rate per tick: tau_unblock ~= 20ms at rest */
		float		  k_reblock{0.08f}; /* Mg2⁺ re-blocking rate per tick per mM: tau_reblock ~= 12ms at rest */
		float		  delta_mg{0.80f};	/* Electrical distance of Mg²⁺ site (Jahr & Stevens 1990): asymmetry source */
		float		  rho_s{0.95f};		/* Somatic leak per tick */
		float		  alpha_a{0.99f};	/* Adaptation trace decay: A = alpha_a * A + beta_a * z */
		float		  beta_a{0.10f};	/* Adaptation increment per graded spike unit */
		float		  theta_0{1.00f};	/* Base firing threshold. Effective threshold = theta_0 + A. */
		float		  E_K{-0.15f};		/* K+ reversal potential (normalized). Post-spike reset target. */
		float		  rho_ahp_fast{0.90f};	   /* Fast AHP decay: tau_fast ~= 10ms -> exp(-1/10) ~= 0.90  */
		float		  rho_ahp_slow{0.995f};	   /* Slow AHP decay: tau_slow ~= 200ms -> exp(-1/200) ~= 0.995 */
		float		  delta_g_ahp_fast{0.30f}; /* Fast K+ conductance increment per spike */
		float		  delta_g_ahp_slow{0.10f}; /* Slow K+ conductance increment per spike */
		SurrogateType surrogate{SurrogateType::TRIANGULAR}; /* Surrogate gradient estimator for spike discontinuity. */
	};

	/// Centralized dendritic params to prevent values divergence
	static constexpr DendriticParams kDefaultDendriticParams = {
		.rho_ampa		  = 0.80f,
		.rho_nmda		  = 0.98f,
		.g_nmda			  = 0.50f,
		.mg_conc		  = 1.00f,
		.k_unblock		  = 0.05f,
		.k_reblock		  = 0.08f,
		.delta_mg		  = 0.80f,
		.rho_s			  = 0.50f,
		.alpha_a		  = 0.95f,
		.beta_a			  = 0.10f,
		.theta_0		  = 1.00f,
		.E_K			  = -0.15f,
		.rho_ahp_fast	  = 0.90f,
		.rho_ahp_slow	  = 0.995f,
		.delta_g_ahp_fast = 0.30f,
		.delta_g_ahp_slow = 0.10f,
		.surrogate		  = SurrogateType::TRIANGULAR
	};

	/// Multi-compartment ALIF population with 4 dendritic branches and graded spike output.
	/// SoA layout, all arrays 32-byte aligned. N must be a nonzero multiple of 8.
	///
	/// Each branch maintains separate AMPA (fast, rho_ampa) and NMDA (slow, rho_nmda) compartments.
	/// NMDA current is voltage-gated via the Mg2+ unblocking factor (Jahr & Stevens 1990):
	///  - B(V) = 1 / (1 + [Mg2+]/K_mg * exp(-eta * V))
	///
	/// making NMDA self-limiting and depolarization-dependent.
	///
	/// Post-spike AHP is modeled as two K+ conductance components (fast mAHP + slow sAHP):
	///  - I_ahp(t) = (g_fast(t) + g_slow(t)) * (V - E_K)
	///
	/// Both are incremented on each spike and decay exponentially between spikes.
	/// The soma resets to E_K rather than 0V, matching the K+ reversal potential.
	template <uint32_t N> class alignas(ENGINE_CACHE_LINE_SIZE) DendriticPopulation {
		static_assert(N > 0 && (N % 8) == 0, "N must be a nonzero multiple of 8 (AVX2 lane width).");

		alignas(32) float V_ampa_0_[N];
		alignas(32) float V_nmda_0_[N];
		alignas(32) float V_ampa_1_[N];
		alignas(32) float V_nmda_1_[N];
		alignas(32) float V_ampa_2_[N];
		alignas(32) float V_nmda_2_[N];
		alignas(32) float V_ampa_3_[N];
		alignas(32) float V_nmda_3_[N];

		alignas(32) float V_soma_[N];
		alignas(32) float A_adaptation_[N];
		alignas(32) float V_soma_pre_spike_[N]; /* Soma voltage before reset, required for e-prop h'(t). */
		alignas(32) float g_ahp_fast_[N];		/* Fast K+ conductance per neuron. Bumped on spike, decays exp. */
		alignas(32) float g_ahp_slow_[N];		/* Slow K+ conductance per neuron. Bumped on spike, decays exp. */
		alignas(32) float B_nmda_[N];			/* Dynamic Mg2⁺ unblocking factor per neuron. B ∈ [0,1]. */

		float theta_stored_{1.0f}; /* theta_0 saved from last tick() call, used by compute_h_prime. */

	public:
		static constexpr uint32_t SIZE = N;

		explicit DendriticPopulation() noexcept {
			reset_state();
		}

		/// Zeros all membrane potentials and adaptation traces.
		ENGINE_COLD_PATH void reset_state() noexcept {
			std::memset(V_ampa_0_, 0, sizeof(V_ampa_0_));
			std::memset(V_nmda_0_, 0, sizeof(V_nmda_0_));
			std::memset(V_ampa_1_, 0, sizeof(V_ampa_1_));
			std::memset(V_nmda_1_, 0, sizeof(V_nmda_1_));
			std::memset(V_ampa_2_, 0, sizeof(V_ampa_2_));
			std::memset(V_nmda_2_, 0, sizeof(V_nmda_2_));
			std::memset(V_ampa_3_, 0, sizeof(V_ampa_3_));
			std::memset(V_nmda_3_, 0, sizeof(V_nmda_3_));
			std::memset(V_soma_, 0, sizeof(V_soma_));
			std::memset(A_adaptation_, 0, sizeof(A_adaptation_));
			std::memset(V_soma_pre_spike_, 0, sizeof(V_soma_pre_spike_));
			std::memset(g_ahp_fast_, 0, sizeof(g_ahp_fast_));
			std::memset(g_ahp_slow_, 0, sizeof(g_ahp_slow_));

			const __m256 b_init = _mm256_set1_ps(0.4f);
			for (uint32_t i = 0; i < N; i += 8) {
				_mm256_store_ps(B_nmda_ + i, b_init);
			}
		}

		/// Advances the population by one time step.
		///
		/// Per-tick order:
		///   1. SFA:      A = alpha_a * A + beta_a * z_prev
		///                theta_dynamic = theta_0 + A
		///   2. Mg2+:     alpha(V) = k_unblock * exp(delta_mg * V_soma_prev)
		///                beta(V)  = k_reblock * [Mg2+] * exp(-(1-delta_mg) * V_soma_prev)
		///                B[t+1]   = (B[t] + alpha) / (1 + alpha + beta)  (implicit Euler, Kampa et al. 2004)
		///                B clamped to [0, 1]
		///   3. Dendritic (per branch, k = 0..3):
		///                V_ampa_k = rho_ampa * V_ampa_k + I_k
		///                V_nmda_k = rho_nmda * V_nmda_k + I_k
		///                d_k      = V_ampa_k + g_nmda * B * V_nmda_k
		///   4. AHP:      g_fast = rho_ahp_fast * g_fast
		///                g_slow = rho_ahp_slow * g_slow
		///                I_ahp  = (g_fast + g_slow) * (V_soma_prev - E_K)
		///   5. Somatic:  V_s = rho_s * V_s + sum(d0..d3) + I_rec - I_ahp
		///                saved to V_soma_pre_spike_ before reset (required for e-prop h'(t))
		///   6. Spike:    fire if V_s >= theta_dynamic
		///                emit graded payload max(V_s - theta_dynamic, 1.0)
		///                reset V_s to E_K; increment g_fast and g_slow
		///
		/// B(V) uses kinetic two-state gating (Kampa et al. 2004), not the static Jahr & Stevens equilibrium formula.
		/// B, I_ahp and dendritic currents are all evaluated on V_soma from the previous tick.
		/// All pointer arguments must be 32-byte aligned.
		ENGINE_HOT_PATH void tick(
			const float *__restrict__ I_d0,
			const float *__restrict__ I_d1,
			const float *__restrict__ I_d2,
			const float *__restrict__ I_d3,
			const float *__restrict__ I_rec,
			const float *__restrict__ prev_spikes,
			float *__restrict__ out_graded_spikes,
			const DendriticParams &params
		) noexcept {
			const __m256 rho_ampa		   = _mm256_set1_ps(params.rho_ampa);
			const __m256 rho_nmda		   = _mm256_set1_ps(params.rho_nmda);
			const __m256 g_nmda			   = _mm256_set1_ps(params.g_nmda);
			const __m256 k_unblock_v	   = _mm256_set1_ps(params.k_unblock);
			const __m256 k_reblock_mg_v	   = _mm256_set1_ps(params.k_reblock * params.mg_conc);
			const __m256 delta_mg_v		   = _mm256_set1_ps(params.delta_mg);
			const __m256 neg_one_m_delta_v = _mm256_set1_ps(-(1.0f - params.delta_mg));
			const __m256 rho_s			   = _mm256_set1_ps(params.rho_s);
			const __m256 alpha_a		   = _mm256_set1_ps(params.alpha_a);
			const __m256 beta_a			   = _mm256_set1_ps(params.beta_a);
			const __m256 theta_0		   = _mm256_set1_ps(params.theta_0);
			const __m256 E_K_v			   = _mm256_set1_ps(params.E_K);
			const __m256 rho_ahp_fast	   = _mm256_set1_ps(params.rho_ahp_fast);
			const __m256 rho_ahp_slow	   = _mm256_set1_ps(params.rho_ahp_slow);
			const __m256 dg_fast		   = _mm256_set1_ps(params.delta_g_ahp_fast);
			const __m256 dg_slow		   = _mm256_set1_ps(params.delta_g_ahp_slow);
			const __m256 one_v			   = _mm256_set1_ps(1.0f);
			const __m256 two_v			   = _mm256_set1_ps(2.0f);
			const __m256 zero_v			   = _mm256_setzero_ps();
			// Taylor coefficients for exp(x) ~= 1 + x*(1 + x*(1/2 + x*(1/6 + x/24)))
			// Valid for |x| < 0.4 with < 0.02% error over the normalized voltage range.
			const __m256 c2	 = _mm256_set1_ps(0.5f);
			const __m256 c6	 = _mm256_set1_ps(1.0f / 6.0f);
			const __m256 c24 = _mm256_set1_ps(1.0f / 24.0f);

			theta_stored_ = params.theta_0;

			for (uint32_t i = 0; i < N; i += 8) {
				// Slow Adaptation (SFA)
				__m256 a = _mm256_load_ps(A_adaptation_ + i);
				a		 = _mm256_fmadd_ps(alpha_a, a, _mm256_mul_ps(beta_a, _mm256_load_ps(prev_spikes + i)));
				_mm256_store_ps(A_adaptation_ + i, a);
				const __m256 theta_dynamic = _mm256_add_ps(theta_0, a);

				/// Kinetic Mg2+ asymmetric unblocking: fast at depolarized V, slow at rest (Kampa et al. 2004)
				/// alpha(V) = k_unblock * exp(delta * V; unblocking, increases with depolarization
				/// beta(V)  = k_reblock * [Mg] * exp(-(1-delta) * V); re-blocking, decreases with depolarization
				const __m256 soma_prev = _mm256_load_ps(V_soma_ + i);
				const __m256 alpha_x   = _mm256_mul_ps(delta_mg_v, soma_prev);		  // delta * V
				const __m256 beta_x	   = _mm256_mul_ps(neg_one_m_delta_v, soma_prev); // -(1-delta) * V

				/// Two independent Estrin chains evaluated
				/// 4th-order Taylor valid to ~2.5% at max normalized voltage

				// Level 0
				const __m256 ax2  = _mm256_mul_ps(alpha_x, alpha_x);
				const __m256 ae01 = _mm256_add_ps(alpha_x, one_v);
				const __m256 ae23 = _mm256_fmadd_ps(c6, alpha_x, c2);
				const __m256 bx2  = _mm256_mul_ps(beta_x, beta_x);
				const __m256 be01 = _mm256_add_ps(beta_x, one_v);
				const __m256 be23 = _mm256_fmadd_ps(c6, beta_x, c2);

				// Level 1
				const __m256 ae0123 = _mm256_fmadd_ps(ax2, ae23, ae01);
				const __m256 ax4	= _mm256_mul_ps(ax2, ax2);
				const __m256 be0123 = _mm256_fmadd_ps(bx2, be23, be01);
				const __m256 bx4	= _mm256_mul_ps(bx2, bx2);

				// Level 2
				const __m256 alpha_mg = _mm256_mul_ps(k_unblock_v, _mm256_fmadd_ps(c24, ax4, ae0123));
				const __m256 beta_mg  = _mm256_mul_ps(k_reblock_mg_v, _mm256_fmadd_ps(c24, bx4, be0123));

				__m256		 B		 = _mm256_load_ps(B_nmda_ + i);
				const __m256 B_numer = _mm256_add_ps(B, alpha_mg);
				const __m256 B_denom = _mm256_add_ps(one_v, _mm256_add_ps(alpha_mg, beta_mg));

				// Newton Raphson
				const __m256 r0_B = _mm256_rcp_ps(B_denom);
				B				  = _mm256_mul_ps(B_numer, _mm256_mul_ps(r0_B, _mm256_fnmadd_ps(B_denom, r0_B, two_v)));
				B				  = _mm256_min_ps(one_v, _mm256_max_ps(zero_v, B)); // numerical safety clamp
				_mm256_store_ps(B_nmda_ + i, B);

				/// Lambda for branchless dendritic computation
				/// Dendritic branch: AMPA (fast) + NMDA (slow, voltage-gated)
				/// I_ampa = rho_ampa * V_ampa + I (linear, fast decay)
				/// I_nmda = g_nmda * B(V_soma) * V_nmda (saturating, voltage-dependent, slow decay)
				/// Branch output to soma = I_ampa + I_nmda
				auto process_dendrite = [&](float *ampa_ptr, float *nmda_ptr, const float *in_ptr) -> __m256 {
					const __m256 in = _mm256_load_ps(in_ptr + i);

					__m256 ampa = _mm256_load_ps(ampa_ptr + i);
					ampa		= _mm256_fmadd_ps(rho_ampa, ampa, in);
					_mm256_store_ps(ampa_ptr + i, ampa);

					__m256 nmda = _mm256_load_ps(nmda_ptr + i);
					nmda		= _mm256_fmadd_ps(rho_nmda, nmda, in);
					_mm256_store_ps(nmda_ptr + i, nmda);

					// I_nmda = g_nmda * B(V_soma) * V_nmda  (E_nmda = 0 -> V_nmda - E_nmda = V_nmda)
					return _mm256_fmadd_ps(g_nmda, _mm256_mul_ps(B, nmda), ampa);
				};

				// Active Dendritic Processing
				const __m256 d0 = process_dendrite(V_ampa_0_, V_nmda_0_, I_d0);
				const __m256 d1 = process_dendrite(V_ampa_1_, V_nmda_1_, I_d1);
				const __m256 d2 = process_dendrite(V_ampa_2_, V_nmda_2_, I_d2);
				const __m256 d3 = process_dendrite(V_ampa_3_, V_nmda_3_, I_d3);

				// AHP: decay K+ conductance (forward Euler, and uses state from the previous tick)
				__m256 g_fast = _mm256_mul_ps(rho_ahp_fast, _mm256_load_ps(g_ahp_fast_ + i));
				__m256 g_slow = _mm256_mul_ps(rho_ahp_slow, _mm256_load_ps(g_ahp_slow_ + i));

				// I_ahp(t) = (g_fast + g_slow) * (V_soma_prev - E_K)
				// Pulls membrane toward E_K; two-component model captures mAHP + sAHP observed in vivo
				const __m256 I_ahp = _mm256_mul_ps(_mm256_add_ps(g_fast, g_slow), _mm256_sub_ps(soma_prev, E_K_v));

				// Somatic Integration with AHP subtracted & Spiking
				const __m256 dend_sum = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
				const __m256 rec	  = _mm256_load_ps(I_rec + i);
				__m256		 soma	  = _mm256_fmadd_ps(rho_s, soma_prev, _mm256_add_ps(dend_sum, rec));
				soma				  = _mm256_sub_ps(soma, I_ahp);
				_mm256_store_ps(V_soma_pre_spike_ + i, soma); // save pre-reset for h'(t)

				const __m256 spike_mask = _mm256_cmp_ps(soma, theta_dynamic, _CMP_GE_OS);

				// Graded spike payload: max(V - theta, 1.0), zero if no spike.
				__m256 graded = _mm256_max_ps(_mm256_sub_ps(soma, theta_dynamic), one_v);
				graded		  = _mm256_blendv_ps(zero_v, graded, spike_mask);
				_mm256_store_ps(out_graded_spikes + i, graded);

				// Post-spike: reset soma to E_K (K+ reversal), and bump both AHP conductance.
				soma   = _mm256_blendv_ps(soma, E_K_v, spike_mask);
				g_fast = _mm256_blendv_ps(g_fast, _mm256_add_ps(g_fast, dg_fast), spike_mask);
				g_slow = _mm256_blendv_ps(g_slow, _mm256_add_ps(g_slow, dg_slow), spike_mask);
				_mm256_store_ps(V_soma_ + i, soma);
				_mm256_store_ps(g_ahp_fast_ + i, g_fast);
				_mm256_store_ps(g_ahp_slow_ + i, g_slow);
			}
		}

		/// Single-branch tick: branches 1-3 and I_rec = 0 permanently (pop_slow_ in inference).
		/// Skips 3 dead process_dendrite calls and the I_rec load.
		/// Functionally identical to tick(I_d0, zero, zero, zero, zero, prev, out, params).
		ENGINE_HOT_PATH void tick_d0(
			const float *__restrict__ I_d0,
			const float *__restrict__ prev_spikes,
			float *__restrict__ out_graded_spikes,
			const DendriticParams &params
		) noexcept {
			const __m256 rho_ampa		   = _mm256_set1_ps(params.rho_ampa);
			const __m256 rho_nmda		   = _mm256_set1_ps(params.rho_nmda);
			const __m256 g_nmda			   = _mm256_set1_ps(params.g_nmda);
			const __m256 k_unblock_v	   = _mm256_set1_ps(params.k_unblock);
			const __m256 k_reblock_mg_v	   = _mm256_set1_ps(params.k_reblock * params.mg_conc);
			const __m256 delta_mg_v		   = _mm256_set1_ps(params.delta_mg);
			const __m256 neg_one_m_delta_v = _mm256_set1_ps(-(1.0f - params.delta_mg));
			const __m256 rho_s			   = _mm256_set1_ps(params.rho_s);
			const __m256 alpha_a		   = _mm256_set1_ps(params.alpha_a);
			const __m256 beta_a			   = _mm256_set1_ps(params.beta_a);
			const __m256 theta_0		   = _mm256_set1_ps(params.theta_0);
			const __m256 E_K_v			   = _mm256_set1_ps(params.E_K);
			const __m256 rho_ahp_fast	   = _mm256_set1_ps(params.rho_ahp_fast);
			const __m256 rho_ahp_slow	   = _mm256_set1_ps(params.rho_ahp_slow);
			const __m256 dg_fast		   = _mm256_set1_ps(params.delta_g_ahp_fast);
			const __m256 dg_slow		   = _mm256_set1_ps(params.delta_g_ahp_slow);
			const __m256 one_v			   = _mm256_set1_ps(1.0f);
			const __m256 two_v			   = _mm256_set1_ps(2.0f);
			const __m256 zero_v			   = _mm256_setzero_ps();
			const __m256 c2				   = _mm256_set1_ps(0.5f);
			const __m256 c6				   = _mm256_set1_ps(1.0f / 6.0f);
			const __m256 c24			   = _mm256_set1_ps(1.0f / 24.0f);
			theta_stored_				   = params.theta_0;

			for (uint32_t i = 0; i < N; i += 8) {
				// SFA
				__m256 a = _mm256_load_ps(A_adaptation_ + i);
				a		 = _mm256_fmadd_ps(alpha_a, a, _mm256_mul_ps(beta_a, _mm256_load_ps(prev_spikes + i)));
				_mm256_store_ps(A_adaptation_ + i, a);
				const __m256 theta_dynamic = _mm256_add_ps(theta_0, a);

				const __m256 soma_prev = _mm256_load_ps(V_soma_ + i);

				// Kinetic Mg2+: two independent Estrin chains (ILP), implicit Euler B update.
				const __m256 ax2 =
					_mm256_mul_ps(_mm256_mul_ps(delta_mg_v, soma_prev), _mm256_mul_ps(delta_mg_v, soma_prev));
				const __m256 alpha_x  = _mm256_mul_ps(delta_mg_v, soma_prev);
				const __m256 beta_x	  = _mm256_mul_ps(neg_one_m_delta_v, soma_prev);
				const __m256 bx2	  = _mm256_mul_ps(beta_x, beta_x);
				const __m256 ae01	  = _mm256_add_ps(alpha_x, one_v);
				const __m256 ae23	  = _mm256_fmadd_ps(c6, alpha_x, c2);
				const __m256 be01	  = _mm256_add_ps(beta_x, one_v);
				const __m256 be23	  = _mm256_fmadd_ps(c6, beta_x, c2);
				const __m256 ae0123	  = _mm256_fmadd_ps(ax2, ae23, ae01);
				const __m256 ax4	  = _mm256_mul_ps(ax2, ax2);
				const __m256 be0123	  = _mm256_fmadd_ps(bx2, be23, be01);
				const __m256 bx4	  = _mm256_mul_ps(bx2, bx2);
				const __m256 alpha_mg = _mm256_mul_ps(k_unblock_v, _mm256_fmadd_ps(c24, ax4, ae0123));
				const __m256 beta_mg  = _mm256_mul_ps(k_reblock_mg_v, _mm256_fmadd_ps(c24, bx4, be0123));
				__m256		 B		  = _mm256_load_ps(B_nmda_ + i);
				const __m256 B_numer  = _mm256_add_ps(B, alpha_mg);
				const __m256 B_denom  = _mm256_add_ps(one_v, _mm256_add_ps(alpha_mg, beta_mg));
				const __m256 r0_B	  = _mm256_rcp_ps(B_denom);
				B = _mm256_mul_ps(B_numer, _mm256_mul_ps(r0_B, _mm256_fnmadd_ps(B_denom, r0_B, two_v)));
				B = _mm256_min_ps(one_v, _mm256_max_ps(zero_v, B));
				_mm256_store_ps(B_nmda_ + i, B);

				// Branch 0 only (d1=d2=d3=0 permanently -> V_ampa/nmda_1..3 stay 0, no-op to skip).
				const __m256 in0   = _mm256_load_ps(I_d0 + i);
				__m256		 ampa0 = _mm256_fmadd_ps(rho_ampa, _mm256_load_ps(V_ampa_0_ + i), in0);
				__m256		 nmda0 = _mm256_fmadd_ps(rho_nmda, _mm256_load_ps(V_nmda_0_ + i), in0);
				_mm256_store_ps(V_ampa_0_ + i, ampa0);
				_mm256_store_ps(V_nmda_0_ + i, nmda0);
				const __m256 d0 = _mm256_fmadd_ps(g_nmda, _mm256_mul_ps(B, nmda0), ampa0);

				// AHP: decay then I_ahp = (g_fast + g_slow) * (V_prev - E_K)
				__m256		 g_f   = _mm256_mul_ps(rho_ahp_fast, _mm256_load_ps(g_ahp_fast_ + i));
				__m256		 g_s   = _mm256_mul_ps(rho_ahp_slow, _mm256_load_ps(g_ahp_slow_ + i));
				const __m256 I_ahp = _mm256_mul_ps(_mm256_add_ps(g_f, g_s), _mm256_sub_ps(soma_prev, E_K_v));

				// Somatic: d0 only, no I_rec
				__m256 soma = _mm256_fmadd_ps(rho_s, soma_prev, _mm256_sub_ps(d0, I_ahp));
				_mm256_store_ps(V_soma_pre_spike_ + i, soma);

				const __m256 spike_mask = _mm256_cmp_ps(soma, theta_dynamic, _CMP_GE_OS);
				__m256		 graded		= _mm256_max_ps(_mm256_sub_ps(soma, theta_dynamic), one_v);
				graded					= _mm256_blendv_ps(zero_v, graded, spike_mask);
				_mm256_store_ps(out_graded_spikes + i, graded);

				soma = _mm256_blendv_ps(soma, E_K_v, spike_mask);
				g_f	 = _mm256_blendv_ps(g_f, _mm256_add_ps(g_f, dg_fast), spike_mask);
				g_s	 = _mm256_blendv_ps(g_s, _mm256_add_ps(g_s, dg_slow), spike_mask);
				_mm256_store_ps(V_soma_ + i, soma);
				_mm256_store_ps(g_ahp_fast_ + i, g_f);
				_mm256_store_ps(g_ahp_slow_ + i, g_s);
			}
		}

		/// Computes the surrogate pseudo-derivative h'(t) for all N neurons.
		/// Dispatches on params.surrogate.
		/// Call immediately after tick(), before any state mutation.
		/// h_prime_out must be 32-byte aligned, size N.
		ENGINE_HOT_PATH void
		compute_h_prime(float *__restrict__ h_prime_out, const float beta_sg, const SurrogateType type) const noexcept {
			const __m256 beta_vec	   = _mm256_set1_ps(beta_sg);
			const __m256 theta_0	   = _mm256_set1_ps(theta_stored_);
			const __m256 one_vec	   = _mm256_set1_ps(1.0f);
			const __m256 two_vec	   = _mm256_set1_ps(2.0f);
			const __m256 sign_mask	   = _mm256_set1_ps(-0.0f);
			const __m256 half_inv_beta = _mm256_set1_ps(0.5f / beta_sg);
			const __m256 zero_vec	   = _mm256_setzero_ps();

			for (uint32_t i = 0; i < N; i += 8) {
				const __m256 v_pre	   = _mm256_load_ps(V_soma_pre_spike_ + i);
				const __m256 a		   = _mm256_load_ps(A_adaptation_ + i);
				const __m256 delta	   = _mm256_sub_ps(v_pre, _mm256_add_ps(theta_0, a));
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

		// Post-reset soma voltage. Returns 0.0 for neurons that spiked this tick.
		// Do NOT use for e-prop. Use get_soma_v_pre_spike() or get_soma_v_pre_spike_buf().
		[[nodiscard]] ENGINE_COLD_PATH float get_soma_v(uint32_t i) const noexcept {
			return V_soma_[i];
		}

		// Soma voltage before the spike reset. Use this for h'(t) computation.
		[[nodiscard]] ALWAYS_INLINE float get_soma_v_pre_spike(uint32_t i) const noexcept {
			return V_soma_pre_spike_[i];
		}

		// Direct buffer access to V_soma_pre_spike_ for batch operations. Size N, 32-byte aligned.
		[[nodiscard]] ALWAYS_INLINE const float *get_soma_v_pre_spike_buf() const noexcept {
			return V_soma_pre_spike_;
		}

		// Direct buffer access to A_adaptation_ for external theta_dynamic reconstruction.
		[[nodiscard]] ALWAYS_INLINE const float *get_adaptation_buf() const noexcept {
			return A_adaptation_;
		}

		[[nodiscard]] ENGINE_COLD_PATH float get_adaptation(uint32_t i) const noexcept {
			return A_adaptation_[i];
		}

		// Returns the stored pre-boost dendritic voltage for branch [0..3], neuron i.
		[[nodiscard]] ENGINE_COLD_PATH float get_dendrite_v(const uint32_t branch, uint32_t i) const noexcept {
			switch (branch) {
			case 0:
				return V_ampa_0_[i];
			case 1:
				return V_ampa_1_[i];
			case 2:
				return V_ampa_2_[i];
			case 3:
				return V_ampa_3_[i];
			default:
				return 0.0f;
			}
		}
	};
} // namespace engine::snn
