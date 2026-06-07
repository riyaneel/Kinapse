#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>

#include <SnnTrainer.hpp>
#include <snn/WeightsFormat.hpp>

namespace engine::snn_train {

	namespace {
		[[nodiscard]] ALWAYS_INLINE uint64_t xorshift64(uint64_t s) noexcept {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			return s;
		}

		[[nodiscard]] ALWAYS_INLINE double sigmoid(const float x) noexcept {
			return 1.0 / (1.0 + std::exp(static_cast<double>(-x)));
		}
	} // namespace

	SnnTrainer::SnnTrainer(const SnnTrainerConfig &config) noexcept : config_(config) {
		layer_d0_.xavier_init(rng_state_);
		layer_d1_.xavier_init(rng_state_);
		layer_d2_.xavier_init(rng_state_);
		layer_d3_.xavier_init(rng_state_);
		layer_hidden_slow_.xavier_init(rng_state_);
		layer_slow_out_.xavier_init(rng_state_);
		slow_dendritic_params_.rho_s = 0.9f; // Integrates over ~10x longer timescale than pop_hidden
		layer_rec_.xavier_init(rng_state_);

		init_rec_mask(mask_rec_);
		layer_rec_.apply_mask(mask_rec_); // enforce sparsity on the initial xavier weights
		layer_rec_.build_sparse_index();

		switch (config_.profile) {
		case MarketMakerProfile::SCALPER:
			config_.risk_penalty_lambda = 0.010f;
			config_.max_lot_size		= 2.0;
			config_.astrocyte_threshold = 0.02f;
			break;
		case MarketMakerProfile::MOMENTUM:
			config_.risk_penalty_lambda = 0.001f;
			config_.max_lot_size		= 15.0;
			config_.astrocyte_threshold = 0.08f;
			break;
		case MarketMakerProfile::DEFENSIVE:
			config_.risk_penalty_lambda = 0.005f;
			config_.max_lot_size		= 5.0;
			config_.astrocyte_threshold = 0.05f;
			break;
		}
	}

	void SnnTrainer::init_rec_mask(uint8_t *mask) noexcept {
		constexpr auto threshold = static_cast<uint32_t>(0xFFFFFFFFULL * 0.10 + 0.5); // ~10%
		uint64_t	   s		 = snn::kRecMaskSeed;

		for (uint32_t k = 0; k < N_HIDDEN * N_HIDDEN; ++k) {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			const uint64_t m = s ^ s >> 32;
			mask[k]			 = static_cast<uint8_t>((m & 0xFFFFFFFFULL) < threshold ? 1 : 0);
		}
	}

	void SnnTrainer::train(const matching::WireOrder *dataset, const size_t num_orders) {
		std::cout << "[TRAIN] Starting Neuromorphic pipeline...\n";
		std::cout << "[TRAIN] Profile: " << static_cast<uint32_t>(config_.profile) << " | Dataset size: " << num_orders
				  << '\n';

		for (size_t i = 0; i < num_orders; ++i) {
			step(dataset[i]);
			if (i % 1'000'000 == 0 && i > 0) {
				const double mtm_pnl = cash_ + static_cast<double>(inventory_btc_) * last_mid_price_;
				const double pct	 = static_cast<double>(i) / static_cast<double>(num_orders) * 100.0;
				std::cout << "[TRAIN] " << i << " / " << num_orders << " (" << std::fixed << std::setprecision(2) << pct
						  << "%)"
						  << " | MTM PnL: " << std::setprecision(4) << mtm_pnl << " | Inventory: " << inventory_btc_
						  << " | Fill PnL/tick: " << std::setprecision(6) << ema_fill_pnl_
						  << " | L(t): " << last_normalized_L_ << '\n';
			}
		}

		std::cout << "[TRAIN] Done. Final PnL: " << cash_ + static_cast<double>(inventory_btc_) * last_mid_price_
				  << '\n';
	}

	ENGINE_HOT_PATH void SnnTrainer::step(const matching::WireOrder &order) noexcept {
		alignas(32) float features[N_FEATURES]{};
		extract_features(order, features);

		// Feature routing: pair [0,1] -> D0 (trend), [2,3] -> D1 (flow), [4,5] -> D2 (risk), [6,7] -> D3 (state).
		layer_d0_.forward(&features[0], I_d0_);
		layer_d1_.forward(&features[2], I_d1_);
		layer_d2_.forward(&features[4], I_d2_);
		layer_d3_.forward(&features[6], I_d3_);

		// Recurrent current from previous tick's hidden spikes.
		layer_rec_.forward_sparse(hidden_graded_spikes_, I_rec_);

		pop_hidden_.tick(I_d0_, I_d1_, I_d2_, I_d3_, I_rec_, hidden_graded_spikes_, cur_hidden_, dendritic_params_);

		layer_hidden_slow_.forward(cur_hidden_, I_slow_);
		// Single input on branch 0; branches 1-3 and I_rec are zero (no split input, no slow recurrence).
		pop_slow_.tick_d0(I_slow_, slow_graded_spikes_, cur_slow_, slow_dendritic_params_);
		layer_slow_out_.forward(cur_slow_, I_out_);

		// ReadoutPopulation: single input, no branches, no SFA.
		pop_out_.tick(I_out_, cur_out_, dendritic_params_.rho_s, dendritic_params_.theta_0);

		// h'(t): surrogate gradient. Must be called before any state mutation.
		alignas(32) float h_prime_hid[N_HIDDEN]{};
		alignas(32) float h_prime_slow[N_SLOW]{};
		alignas(32) float h_prime_out[N_ACTIONS]{};
		pop_hidden_.compute_h_prime(h_prime_hid, config_.surrogate_beta, dendritic_params_.surrogate);
		pop_slow_.compute_h_prime(h_prime_slow, config_.surrogate_beta, slow_dendritic_params_.surrogate);
		pop_out_.compute_h_prime(h_prime_out, config_.surrogate_beta, dendritic_params_.surrogate);

		// rho_e = rho_s: eligibility trace must decay at the same timescale as the membrane.
		// A mismatch breaks causal alignment between pre-synaptic activity and post-synaptic state.
		layer_d0_.update_traces(&features[0], h_prime_hid, dendritic_params_.rho_s);
		layer_d1_.update_traces(&features[2], h_prime_hid, dendritic_params_.rho_s);
		layer_d2_.update_traces(&features[4], h_prime_hid, dendritic_params_.rho_s);
		layer_d3_.update_traces(&features[6], h_prime_hid, dendritic_params_.rho_s);
		layer_hidden_slow_.update_traces(cur_hidden_, h_prime_slow, slow_dendritic_params_.rho_s);
		layer_slow_out_.update_traces(cur_slow_, h_prime_out, slow_dendritic_params_.rho_s);
		// Presynaptic = z(t-1) = hidden_graded_spikes_ (before memcpy).
		layer_rec_.update_traces(hidden_graded_spikes_, h_prime_hid, dendritic_params_.rho_s);

		double fill_pnl = 0.0;
		if (last_mid_price_ > 0.0) [[likely]]
			fill_pnl = execute_actions(cur_out_);
		last_mid_price_ = static_cast<double>(order.price);

		ema_fill_pnl_ = ema_fill_pnl_ * 0.999f + static_cast<float>(fill_pnl) * 0.001f;

		const double vol =
			ema_price_slow_ > 0.0f ? std::abs(ema_price_fast_ - ema_price_slow_) / ema_price_slow_ : 0.0001;
		const float market_crisis = static_cast<float>(vol * ema_volume_);
		ema_crisis_				  = ema_crisis_ * 0.99f + market_crisis * 0.01f;

		// Astrocyte neuromodulation: scales learning rate with detected market stress.
		const float neuromod   = std::max(0.0f, ema_crisis_ - config_.astrocyte_threshold);
		const float dynamic_lr = config_.learning_rate_eta * (1.0f + config_.astrocyte_kappa * neuromod);

		const float inv_risk   = config_.risk_penalty_lambda * static_cast<float>(inventory_btc_ * inventory_btc_);
		const float raw_reward = static_cast<float>(fill_pnl) - inv_risk;

		// Per-action critic: V_i(t) tracks the reward independently per output neuron.
		// Faster update during low-stress periods.
		const float		  critic_alpha = 0.01f + 0.009f * (1.0f - std::min(neuromod, 1.0f));
		alignas(32) float advantage_out[N_ACTIONS]{};
		for (uint32_t idx = 0; idx < N_ACTIONS; ++idx) {
			ema_reward_[idx]   = ema_reward_[idx] * (1.0f - critic_alpha) + raw_reward * critic_alpha;
			advantage_out[idx] = raw_reward - ema_reward_[idx];
		}

		// Global signal for hidden layers: mean of per-action advantages.
		float advantage_L_t = 0.0f;
		for (uint32_t idx = 0; idx < N_ACTIONS; ++idx)
			advantage_L_t += advantage_out[idx];
		advantage_L_t /= static_cast<float>(N_ACTIONS);

		// Advantage signal L(t) = r(t) - V(t), RMS-normalised, clamped to [-3, 3].
		ema_advantage_sq_  = ema_advantage_sq_ * 0.999f + advantage_L_t * advantage_L_t * 0.001f;
		const float rms_L  = std::sqrt(ema_advantage_sq_ + 1e-8f);
		last_normalized_L_ = std::clamp(advantage_L_t / rms_L, -3.0f, 3.0f);

		// Neuron-specific output signal: delta_i = advantage_i * d_act_i/dV_i * h'_i(t).
		// d_act/dV: sigmoid for softplus neurons (sizing: 0,4), sigmoid*(1-sig) for urgency (1,2,5,6).
		alignas(32) float delta_out[N_ACTIONS]{};
		for (uint32_t idx = 0; idx < N_ACTIONS; ++idx) {
			const float sig	  = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(cur_out_[idx]))));
			const float d_act = (idx == 0 || idx == 4) ? sig : sig * (1.0f - sig);
			delta_out[idx]	  = advantage_out[idx] * d_act * h_prime_out[idx];
		}

		// Gate weight update on non-trivial advantage to suppress noise-driven drift.
		if (std::abs(advantage_L_t) > 1e-6f) {
			layer_d0_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			layer_d1_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			layer_d2_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			layer_d3_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			layer_hidden_slow_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			// Output layer uses neuron-specific delta_i instead of global L(t).
			layer_slow_out_.apply_e_prop_signal(delta_out, dynamic_lr, config_.weight_decay);
			layer_rec_.apply_e_prop(advantage_L_t, dynamic_lr, config_.weight_decay);
			layer_rec_.apply_mask(mask_rec_); // re-enforce sparsity after each weight update
		}

		std::memcpy(hidden_graded_spikes_, cur_hidden_, sizeof(cur_hidden_));
		std::memcpy(out_graded_spikes_, cur_out_, sizeof(cur_out_));
		std::memcpy(slow_graded_spikes_, cur_slow_, sizeof(cur_slow_));
	}

	[[nodiscard]] ENGINE_HOT_PATH double SnnTrainer::execute_actions(const float *graded_spikes) noexcept {
		/// softplus: log(1 + exp(x)). Numerically stable above x=20 where exp overflows.
		/// Unbounded positive output is appropriate for continuous lot sizing.
		auto softplus = [](const float x) -> double {
			return x > 20.0f ? static_cast<double>(x) : std::log1p(std::exp(static_cast<double>(x)));
		};

		const bool can_buy	= inventory_btc_ < config_.inventory_limit;
		const bool can_sell = inventory_btc_ > -config_.inventory_limit;

		const int32_t bid_headroom = config_.inventory_limit - inventory_btc_;
		const int32_t ask_headroom = inventory_btc_ + config_.inventory_limit;

		const auto raw_bid =
			can_buy ? static_cast<int32_t>(std::round(softplus(graded_spikes[0]) * config_.max_lot_size)) : 0;
		const auto raw_ask =
			can_sell ? static_cast<int32_t>(std::round(softplus(graded_spikes[4]) * config_.max_lot_size)) : 0;

		const auto Q_bid_i = can_buy ? std::min(raw_bid, bid_headroom) : 0;
		const auto Q_ask_i = can_sell ? std::min(raw_ask, ask_headroom) : 0;

		if (Q_bid_i == 0 && Q_ask_i == 0)
			return 0.0;

		const double Q_bid = Q_bid_i;
		const double Q_ask = Q_ask_i;

		const double bid_urgency = sigmoid(graded_spikes[1]) + sigmoid(graded_spikes[2]);
		const double ask_urgency = sigmoid(graded_spikes[5]) + sigmoid(graded_spikes[6]);

		const double vol =
			ema_price_slow_ > 0.0f ? std::abs(ema_price_fast_ - ema_price_slow_) / ema_price_slow_ : 0.0001;

		const double gamma			   = static_cast<double>(config_.risk_penalty_lambda) * 20.0;
		const double inventory_clamped = std::clamp(
			static_cast<double>(inventory_btc_),
			-static_cast<double>(config_.inventory_limit),
			+static_cast<double>(config_.inventory_limit)
		);
		const double reservation_price = last_mid_price_ - inventory_clamped * gamma * vol * last_mid_price_;
		const double spread_half	   = last_mid_price_ * std::max(0.0001, vol * 0.5) / 2.0;

		const double quote_bid = reservation_price - spread_half * (1.0 - bid_urgency * 0.5);
		const double quote_ask = reservation_price + spread_half * (1.0 - ask_urgency * 0.5);

		// Exponential fill probability (Avellaneda-Stoikov approximation).
		constexpr double kappa		= 1.0;
		const double	 p_fill_bid = std::exp(-kappa * std::abs(last_mid_price_ - quote_bid) / spread_half);
		const double	 p_fill_ask = std::exp(-kappa * std::abs(quote_ask - last_mid_price_) / spread_half);

		constexpr double lambda_liq = 0.05;
		const double	 V_mkt		= std::max(1.0, static_cast<double>(ema_volume_));

		rng_state_			 = xorshift64(rng_state_);
		const uint64_t m_bid = rng_state_ ^ rng_state_ >> 32;
		const double   r_bid = static_cast<double>(m_bid & 0xFFFFFFFFULL) * (1.0 / 4294967296.0);

		rng_state_			 = xorshift64(rng_state_);
		const uint64_t m_ask = rng_state_ ^ rng_state_ >> 32;
		const double   r_ask = static_cast<double>(m_ask & 0xFFFFFFFFULL) * (1.0 / 4294967296.0);

		double fill_pnl = 0.0;

		if (Q_bid_i > 0 && r_bid < p_fill_bid) {
			const double slip = lambda_liq * (last_mid_price_ * vol) * std::sqrt(Q_bid / V_mkt);
			const double exec = quote_bid + slip;
			fill_pnl += Q_bid * (last_mid_price_ - exec);
			inventory_btc_ += Q_bid_i;
			cash_ -= Q_bid * exec;
		}

		if (Q_ask_i > 0 && r_ask < p_fill_ask) {
			const double slip = lambda_liq * (last_mid_price_ * vol) * std::sqrt(Q_ask / V_mkt);
			const double exec = quote_ask - slip;
			fill_pnl += Q_ask * (exec - last_mid_price_);
			inventory_btc_ -= Q_ask_i;
			cash_ += Q_ask * exec;
		}

		return fill_pnl;
	}

	ENGINE_HOT_PATH void SnnTrainer::extract_features(const matching::WireOrder &order, float *features_out) noexcept {
		const float price  = static_cast<float>(order.price);
		const float volume = static_cast<float>(order.qty);

		if (ema_price_fast_ == 0.0f) {
			ema_price_fast_ = price;
			ema_price_slow_ = price;
			ema_volume_		= volume;
		}

		ema_price_fast_ = ema_price_fast_ * 0.9f + price * 0.1f;
		ema_price_slow_ = ema_price_slow_ * 0.99f + price * 0.01f;
		ema_volume_		= ema_volume_ * 0.95f + volume * 0.05f;

		const float delta_price = price - ema_price_fast_;
		ema_velocity_			= ema_velocity_ * 0.9f + delta_price * 0.1f;
		ema_imbalance_			= ema_imbalance_ * 0.9f + (order.side == 1 ? 0.1f : -0.1f);

		last_timestamp_ns_ = order.timestamp_ns;

		// D0 [0,1]: trend
		features_out[0] = (ema_price_fast_ - ema_price_slow_) / ema_price_slow_ * 1000.0f; // momentum
		features_out[1] = ema_velocity_ * 100.0f;										   // velocity
		// D1 [2,3]: flow
		features_out[2] = ema_imbalance_;						// order flow imbalance
		features_out[3] = (volume - ema_volume_) / ema_volume_; // volume spike
		// D2 [4,5]: risk
		features_out[4] = std::clamp(static_cast<float>(inventory_btc_) / 50.0f, -1.0f, 1.0f); // inventory skew
		features_out[5] = order.side == 1 ? 1.0f : -1.0f;									   // trade direction
		// D3 [6,7]: state
		features_out[6] = features_out[0] * features_out[2]; // momentum x imbalance
		features_out[7] = 1.0f;								 // bias
	}

	void SnnTrainer::export_weights(const std::string &filepath) const {
		std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
		if (!ofs.is_open()) {
			std::cerr << "[TRAIN] Failed to open export file: " << filepath << '\n';
			return;
		}

		constexpr uint64_t magic = snn::kWeightsMagic;
		ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

		write_layer(ofs, layer_d0_);
		write_layer(ofs, layer_d1_);
		write_layer(ofs, layer_d2_);
		write_layer(ofs, layer_d3_);
		write_layer(ofs, layer_hidden_slow_);
		write_layer(ofs, layer_slow_out_);
		write_layer(ofs, layer_rec_);

		if (!ofs.good()) {
			std::cerr << "[TRAIN] Write error during export to: " << filepath << '\n';
		} else {
			std::cout << "[TRAIN] Weights exported to: " << filepath << '\n';
		}
	}

	void SnnTrainer::reset_episode() noexcept {
		// Reset eligibility traces: stale cross-episode gradients can corrupt the next episode's
		layer_d0_.reset_traces();
		layer_d1_.reset_traces();
		layer_d2_.reset_traces();
		layer_d3_.reset_traces();
		layer_hidden_slow_.reset_traces();
		layer_slow_out_.reset_traces();
		layer_rec_.reset_traces();

		// Network temporal state
		// SFA adaptation and dendritic voltages from the previous episode would bias the first ticks of next episode
		pop_hidden_.reset_state();
		pop_out_.reset_state();
		pop_slow_.reset_state();
		std::memset(hidden_graded_spikes_, 0, sizeof(hidden_graded_spikes_));
		std::memset(out_graded_spikes_, 0, sizeof(out_graded_spikes_));
		std::memset(slow_graded_spikes_, 0, sizeof(slow_graded_spikes_));
		std::memset(I_rec_, 0, sizeof(I_rec_));

		// Trading state: inventory, cash, and price EMAs are episode-specific
		inventory_btc_	= 0;
		cash_			= 0.0;
		last_mid_price_ = 0.0;
		ema_price_fast_ = 0.0f;
		ema_price_slow_ = 0.0f;
		ema_imbalance_	= 0.0f;
		ema_velocity_	= 0.0f;
		ema_volume_		= 0.0f;

		// Reset critic and advantage estimators to avoid cross-episode baseline contamination
		std::memset(ema_reward_, 0, sizeof(ema_reward_));
		ema_crisis_		   = 0.0f;
		ema_advantage_sq_  = 0.0f;
		ema_fill_pnl_	   = 0.0f;
		last_normalized_L_ = 0.0f;
	}
} // namespace engine::snn_train
