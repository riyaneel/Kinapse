#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <engine.hpp>
#include <memory/Arena.hpp>
#include <snn/DendriticPopulation.hpp>
#include <snn/ReadoutPopulation.hpp>
#include <snn/SynapticLayer.hpp>
#include <snn/WeightsFormat.hpp>

namespace engine::snn {

	/// Output structure representing the market maker's intended actions.
	struct alignas(16) MarketMakerIntent {
		float bid_size;	   /* Target buy quantity in lots. */
		float ask_size;	   /* Target sell quantity in lots. */
		float bid_urgency; /* Aggressiveness on the bid side (spread-crossing willingness). */
		float ask_urgency; /* Aggressiveness on the ask side. */
	};

	/// SNN inference engine for market-making.
	///
	/// Feature routing (one pair per dendritic branch):
	///   - [0,1] branch 0: momentum + velocity (trend)
	///   - [2,3] branch 1: imbalance + volume spike (flow)
	///   - [4,5] branch 2: inventory skew + trade direction (risk)
	///   - [6,7] branch 3: momentum*imbalance interaction + bias (state)
	///
	/// Weight file layout (float32 LE):
	///   - For each of {d0, d1, d2, d3}: weights (N_FEATURES_PER_DEND * N_HIDDEN) + biases (N_HIDDEN)
	///   - Then layer_hidden_slow_: weights (N_HIDDEN * N_SLOW) + biases (N_SLOW)
	///   - Then layer_slow_out_: weights (N_SLOW * N_ACTIONS) + biases (N_ACTIONS)
	///   - Then layer_rec_: weights (N_HIDDEN * N_HIDDEN) + biases (N_HIDDEN)
	class alignas(ENGINE_CACHE_LINE_SIZE) SnnInferencer {
		static constexpr uint32_t N_FEATURES		  = 8;
		static constexpr uint32_t N_FEATURES_PER_DEND = 2;
		static constexpr uint32_t N_HIDDEN			  = 256;
		static constexpr uint32_t N_ACTIONS			  = 8;
		static constexpr uint32_t N_SLOW			  = 64;

		SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d0_;
		SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d1_;
		SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d2_;
		SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d3_;
		DendriticPopulation<N_HIDDEN>				 pop_hidden_;
		SynapticLayer<N_HIDDEN, N_SLOW>				 layer_hidden_slow_;
		DendriticPopulation<N_SLOW>					 pop_slow_;
		SynapticLayer<N_SLOW, N_ACTIONS>			 layer_slow_out_;
		DendriticParams								 slow_dendritic_params_ = snn::kDefaultDendriticParams;
		ReadoutPopulation<N_ACTIONS>				 pop_out_;
		SynapticLayer<N_HIDDEN, N_HIDDEN>			 layer_rec_;

		DendriticParams dendritic_params_ = kDefaultDendriticParams;

		alignas(32) float hidden_graded_spikes_[N_HIDDEN]{};
		alignas(32) float out_graded_spikes_[N_ACTIONS]{};
		alignas(32) float I_d0_[N_HIDDEN]{};
		alignas(32) float I_d1_[N_HIDDEN]{};
		alignas(32) float I_d2_[N_HIDDEN]{};
		alignas(32) float I_d3_[N_HIDDEN]{};
		alignas(32) float I_out_[N_ACTIONS]{};
		alignas(32) float I_rec_[N_HIDDEN]{};
		alignas(32) float slow_graded_spikes_[N_SLOW]{};
		alignas(32) float cur_hidden_[N_HIDDEN]{};
		alignas(32) float cur_slow_[N_SLOW]{};
		alignas(32) float I_slow_[N_SLOW]{};
		uint8_t mask_rec_[N_HIDDEN * N_HIDDEN]{};

		template <uint32_t NI, uint32_t NO>
		static ENGINE_COLD_PATH bool load_layer(std::ifstream &ifs, SynapticLayer<NI, NO> &layer) noexcept {
			ifs.read(reinterpret_cast<char *>(layer.get_weights()), sizeof(float) * NI * NO);
			ifs.read(reinterpret_cast<char *>(layer.get_biases()), sizeof(float) * NO);
			return ifs.good();
		}

	public:
		/// Constructs the inferencer and loads weights from a binary file.
		/// @param filepath Absolute or relative path to the pre-trained weights binary.
		explicit SnnInferencer(const std::string &filepath) noexcept {
			std::ifstream ifs(filepath, std::ios::binary);
			if (!ifs.is_open()) [[unlikely]] {
				std::cerr << "[FATAL]: SnnInferencer: Cannot open weights file: " << filepath << '\n';
				return;
			}

			uint64_t magic = 0;
			ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
			if (!ifs.good() || magic != kWeightsMagic) [[unlikely]] {
				std::cerr << "[FATAL]: SnnInferencer: Bad magic number in weights file: " << filepath << " (got 0x" << std::hex
						  << magic << ", expected 0x" << kWeightsMagic << std::dec << ")\n";
				return;
			}

			const bool ok = load_layer(ifs, layer_d0_) && load_layer(ifs, layer_d1_) && load_layer(ifs, layer_d2_) &&
							load_layer(ifs, layer_d3_) && load_layer(ifs, layer_hidden_slow_) &&
							load_layer(ifs, layer_slow_out_) && load_layer(ifs, layer_rec_);

			if (!ok) [[unlikely]] {
				std::cerr << "[FATAL]: SnnInferencer: Weights file truncated or wrong format: " << filepath << '\n';
				return;
			}

			slow_dendritic_params_.rho_s = 0.9f;

			/// Regenerate the sparse mask deterministically and apply it to guarantee the sparsity matches what was enforced
			/// during the training
			{
				constexpr auto threshold = static_cast<uint32_t>(0xFFFFFFFFULL * 0.10 + 0.5); // ~10%
				uint64_t	   s		 = snn::kRecMaskSeed;

				for (uint32_t k = 0; k < N_HIDDEN * N_HIDDEN; ++k) {
					s ^= s << 13;
					s ^= s >> 7;
					s ^= s << 17;
					const uint64_t m = s ^ s >> 32;
					mask_rec_[k]	 = static_cast<uint8_t>((m & 0xFFFFFFFFULL) < threshold ? 1 : 0);
				}

				layer_rec_.apply_mask(mask_rec_);
				layer_rec_.build_sparse_index();
			}
		}

		/// Forward pass. Maintains SFA and recurrent state between calls; pass consecutive market snapshots in order.
		///
		/// @note layer_hidden_slow_ receives hidden_graded_spikes_ = z[t-1] from the previous call, not z[t] from
		/// the current tick. This one-tick offset avoids a store-to-load forwarding hazard on the 1024-byte hidden
		/// buffer. The impact is negligible given the slow population half-life of ~7 ticks.
		///
		/// @param features    Normalized input [N_FEATURES], 32-byte aligned.
		/// @param out_actions Output activations [N_ACTIONS], 32-byte aligned.
		ENGINE_HOT_PATH void predict(const float *__restrict__ features, float *__restrict__ out_actions) noexcept {
			layer_d0_.forward(&features[0], I_d0_);
			layer_d1_.forward(&features[2], I_d1_);
			layer_d2_.forward(&features[4], I_d2_);
			layer_d3_.forward(&features[6], I_d3_);

			layer_rec_.forward_sparse(hidden_graded_spikes_, I_rec_);

			pop_hidden_.tick(I_d0_, I_d1_, I_d2_, I_d3_, I_rec_, hidden_graded_spikes_, cur_hidden_, dendritic_params_);

			layer_hidden_slow_.forward(hidden_graded_spikes_, I_slow_);
			pop_slow_.tick_d0(I_slow_, slow_graded_spikes_, cur_slow_, slow_dendritic_params_);
			layer_slow_out_.forward(slow_graded_spikes_, I_out_);

			pop_out_.tick(I_out_, out_graded_spikes_, dendritic_params_.rho_s, dendritic_params_.theta_0);

			for (uint32_t i = 0; i < N_ACTIONS; i += 8) {
				_mm256_store_ps(out_actions + i, _mm256_load_ps(out_graded_spikes_ + i));
			}
		}

		/// Translates raw activations into market intent via population coding.
		/// Bid/ask sizing from primary neurons [0] and [4]; urgency from ensemble pairs [1,2] and [5,6].
		[[nodiscard]] ENGINE_HOT_PATH MarketMakerIntent predict_intent(const float *__restrict__ features) noexcept {
			alignas(32) float actions[N_ACTIONS]{};
			predict(features, actions);

			return MarketMakerIntent{
				.bid_size	 = actions[0],
				.ask_size	 = actions[4],
				.bid_urgency = actions[1] + actions[2],
				.ask_urgency = actions[5] + actions[6],
			};
		}

		/// Zeroes SFA traces and spike buffers. Call on market disconnect or regime shift.
		ENGINE_COLD_PATH void reset_state() noexcept {
			pop_hidden_.reset_state();
			pop_out_.reset_state();
			pop_slow_.reset_state();
			std::memset(out_graded_spikes_, 0, sizeof(out_graded_spikes_));
			std::memset(I_rec_, 0, sizeof(I_rec_));
			std::memcpy(hidden_graded_spikes_, cur_hidden_, sizeof(cur_hidden_));
			std::memcpy(slow_graded_spikes_, cur_slow_, sizeof(cur_slow_));
		}
	};
} // namespace engine::snn
