#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include <engine.hpp>
#include <matching/Trade.hpp>
#include <snn/DendriticPopulation.hpp>
#include <snn/ReadoutPopulation.hpp>
#include <snn/SynapticLayer.hpp>

namespace engine::snn_train {
	enum class MarketMakerProfile : uint32_t {
		SCALPER	  = 1,
		MOMENTUM  = 2,
		DEFENSIVE = 3,
	};

	struct SnnTrainerConfig {
		MarketMakerProfile profile{MarketMakerProfile::SCALPER};
		float			   trace_decay_e{0.90f};
		float			   learning_rate_eta{0.005f};
		float			   weight_decay{1e-5f};
		float			   risk_penalty_lambda{0.005f};
		float			   astrocyte_threshold{0.05f};
		float			   astrocyte_kappa{5.0f};
		float			   surrogate_beta{5.0f};
		int32_t			   inventory_limit{100};
		double			   max_lot_size{5.0};
	};

	class alignas(ENGINE_CACHE_LINE_SIZE) SnnTrainer {
		static constexpr uint32_t N_FEATURES		  = 8;
		static constexpr uint32_t N_FEATURES_PER_DEND = 2;
		static constexpr uint32_t N_HIDDEN			  = 256;
		static constexpr uint32_t N_ACTIONS			  = 8;
		static constexpr uint32_t N_SLOW			  = 64;

		SnnTrainerConfig								  config_;
		snn::DendriticParams							  dendritic_params_ = snn::kDefaultDendriticParams;
		snn::SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d0_;
		snn::SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d1_;
		snn::SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d2_;
		snn::SynapticLayer<N_FEATURES_PER_DEND, N_HIDDEN> layer_d3_;
		snn::DendriticPopulation<N_HIDDEN>				  pop_hidden_;
		snn::SynapticLayer<N_HIDDEN, N_SLOW>			  layer_hidden_slow_;
		snn::DendriticPopulation<N_SLOW>				  pop_slow_;
		snn::SynapticLayer<N_SLOW, N_ACTIONS>			  layer_slow_out_;
		snn::DendriticParams							  slow_dendritic_params_ = snn::kDefaultDendriticParams;
		snn::ReadoutPopulation<N_ACTIONS>				  pop_out_;
		snn::SynapticLayer<N_HIDDEN, N_HIDDEN>			  layer_rec_;

		alignas(32) float hidden_graded_spikes_[N_HIDDEN]{};
		alignas(32) float out_graded_spikes_[N_ACTIONS]{};
		alignas(32) float I_rec_[N_HIDDEN]{};
		alignas(32) float I_d0_[N_HIDDEN]{};
		alignas(32) float I_d1_[N_HIDDEN]{};
		alignas(32) float I_d2_[N_HIDDEN]{};
		alignas(32) float I_d3_[N_HIDDEN]{};
		alignas(32) float I_out_[N_ACTIONS]{};
		alignas(32) float cur_hidden_[N_HIDDEN]{};
		alignas(32) float cur_out_[N_ACTIONS]{};
		alignas(32) float slow_graded_spikes_[N_SLOW]{};
		alignas(32) float I_slow_[N_SLOW]{};
		alignas(32) float cur_slow_[N_SLOW]{};
		alignas(32) float ema_reward_[N_ACTIONS]{};

		float	 ema_crisis_{0.0f};
		float	 ema_advantage_sq_{0.0f};
		float	 ema_fill_pnl_{0.0f};
		float	 last_normalized_L_{0.0f};
		int32_t	 inventory_btc_{0};
		double	 cash_{0.0};
		double	 last_mid_price_{0.0};
		float	 ema_price_fast_{0.0f};
		float	 ema_price_slow_{0.0f};
		float	 ema_imbalance_{0.0f};
		float	 ema_velocity_{0.0f};
		float	 ema_volume_{0.0f};
		uint64_t last_timestamp_ns_{0};
		uint64_t rng_state_{0x1234567890ABCDEFULL};

		uint8_t mask_rec_[N_HIDDEN * N_HIDDEN];

		static void init_rec_mask(uint8_t *mask) noexcept;

		ENGINE_HOT_PATH void step(const matching::WireOrder &order) noexcept;

		ENGINE_HOT_PATH void extract_features(const matching::WireOrder &order, float *features_out) noexcept;

		[[nodiscard]] ENGINE_HOT_PATH double execute_actions(const float *graded_spikes) noexcept;

		template <uint32_t NI, uint32_t NO>
		static void write_layer(std::ofstream &ofs, const snn::SynapticLayer<NI, NO> &layer) {
			ofs.write(reinterpret_cast<const char *>(layer.get_weights()), sizeof(float) * NI * NO);
			ofs.write(reinterpret_cast<const char *>(layer.get_biases()), sizeof(float) * NO);
		}

	public:
		explicit SnnTrainer(const SnnTrainerConfig &config) noexcept;

		void train(const matching::WireOrder *dataset, size_t num_orders);

		void export_weights(const std::string &filepath) const;

		ENGINE_COLD_PATH void reset_episode() noexcept;
	};
} // namespace engine::snn_train
