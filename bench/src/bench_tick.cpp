#include <cstring>

#include <benchmark/benchmark.h>

#include <snn/DendriticPopulation.hpp>

using namespace engine::snn;

struct HiddenPopState {
	DendriticPopulation<256> pop;
	DendriticParams			 params = kDefaultDendriticParams;

	alignas(32) float I_d0[256]{};
	alignas(32) float I_d1[256]{};
	alignas(32) float I_d2[256]{};
	alignas(32) float I_d3[256]{};
	alignas(32) float I_rec[256]{};
	alignas(32) float prev[256]{};
	alignas(32) float out[256]{};

	HiddenPopState() {
		for (int i = 0; i < 256; ++i) {
			I_d0[i] = 0.15f;
			I_d1[i] = 0.10f;
			I_d2[i] = 0.06f;
			I_d3[i] = 0.04f;
		}

		for (int t = 0; t < 200; ++t) {
			pop.tick(I_d0, I_d1, I_d2, I_d3, I_rec, prev, out, params);
			std::memcpy(prev, out, sizeof(out));
		}
	}
};

struct SlowPopState {
	DendriticPopulation<64> pop;
	DendriticParams			params = kDefaultDendriticParams;

	alignas(32) float I_d0[64]{};
	alignas(32) float prev[64]{};
	alignas(32) float out[64]{};

	SlowPopState() {
		params.rho_s = 0.9f;
		for (int i = 0; i < 64; ++i) {
			I_d0[i] = 0.12f;
		}

		for (int t = 0; t < 200; ++t) {
			pop.tick_d0(I_d0, prev, out, params);
			std::memcpy(prev, out, sizeof(out));
		}
	}
};

static HiddenPopState g_hidden;
static SlowPopState	  g_slow;

static void BM_Tick_Hidden(benchmark::State &state) {
	for (auto _ : state) {
		g_hidden.pop.tick(
			g_hidden.I_d0,
			g_hidden.I_d1,
			g_hidden.I_d2,
			g_hidden.I_d3,
			g_hidden.I_rec,
			g_hidden.prev,
			g_hidden.out,
			g_hidden.params
		);

		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 256);
}

static void BM_Tick_Slow(benchmark::State &state) {
	for (auto _ : state) {
		g_slow.pop.tick_d0(g_slow.I_d0, g_slow.prev, g_slow.out, g_slow.params);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 64);
}

static void BM_Hprime_Triangular(benchmark::State &state) {
	alignas(32) float h_prime[256]{};
	for (auto _ : state) {
		g_hidden.pop.compute_h_prime(h_prime, 5.0f, SurrogateType::TRIANGULAR);
		benchmark::DoNotOptimize(h_prime[0]);
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 256);
}

static void BM_Hprime_FastSigmoid(benchmark::State &state) {
	alignas(32) float h_prime[256]{};
	for (auto _ : state) {
		g_hidden.pop.compute_h_prime(h_prime, 5.0f, SurrogateType::FAST_SIGMOID);
		benchmark::DoNotOptimize(h_prime[0]);
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 256);
}

BENCHMARK(BM_Tick_Hidden)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Tick_Slow)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Hprime_Triangular)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Hprime_FastSigmoid)->MinTime(2.0)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
