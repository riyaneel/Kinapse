#include <cstdio>
#include <cstdlib>
#include <string>

#include <benchmark/benchmark.h>

#include <snn/SnnInferencer.hpp>

using namespace engine::snn;

static const std::string kWeightsPath = "/tmp/snn/config/weights_defensive.bin";

struct InferencerState {
	SnnInferencer inferencer;

	alignas(32) float features[8]{};
	alignas(32) float out[8]{};

	InferencerState() : inferencer(kWeightsPath) {
		features[0] = 0.30f;  // momentum
		features[1] = 0.10f;  // velocity
		features[2] = -0.15f; // imbalance
		features[3] = 0.05f;  // volume spike
		features[4] = 0.20f;  // inventory skew
		features[5] = 1.00f;  // trade direction
		features[6] = -0.05f; // momentum x imbalance
		features[7] = 1.00f;  // bias

		for (int t = 0; t < 200; ++t) {
			inferencer.predict(features, out);
		}
	}
};

static InferencerState g_infer;

static void BM_Predict(benchmark::State &state) {
	for (auto _ : state) {
		g_infer.inferencer.predict(g_infer.features, g_infer.out);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

static void BM_PredictIntent(benchmark::State &state) {
	for (auto _ : state) {
		MarketMakerIntent intent = g_infer.inferencer.predict_intent(g_infer.features);
		benchmark::DoNotOptimize(intent);
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_Predict)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_PredictIntent)->MinTime(2.0)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
