#include <cstdint>

#include <benchmark/benchmark.h>

#include <snn/SynapticLayer.hpp>
#include <snn/WeightsFormat.hpp>

using namespace engine::snn;

static SynapticLayer<2, 256> make_input_layer() {
	SynapticLayer<2, 256> l;
	uint64_t			  rng = 0xABCDEF1234567890ULL;
	l.xavier_init(rng);
	return l;
}

static SynapticLayer<256, 256> make_rec_layer() {
	SynapticLayer<256, 256> layer;

	uint64_t rng = 0xCAFEBABEDEADBEEFULL;
	layer.xavier_init(rng);

	uint8_t			   mask[256 * 256];
	constexpr uint32_t kThreshold = static_cast<uint32_t>(0xFFFFFFFFULL * 0.10 + 0.5);
	uint64_t		   s		  = kRecMaskSeed;
	for (auto &m : mask) {
		s ^= s << 13;
		s ^= s >> 7;
		s ^= s << 17;
		const uint64_t mixed = s ^ (s >> 32);
		m					 = static_cast<uint8_t>((mixed & 0xFFFFFFFFULL) < kThreshold ? 1 : 0);
	}

	layer.apply_mask(mask);
	layer.build_sparse_index();
	return layer;
}

static SynapticLayer<256, 256> g_rec_layer	 = make_rec_layer();
static SynapticLayer<2, 256>   g_input_layer = make_input_layer();

static void BM_Forward_Dense(benchmark::State &state) {
	alignas(32) float x[256]{};
	alignas(32) float out[256]{};
	for (int i = 0; i < 256; ++i) {
		x[i] = static_cast<float>(i) * 0.001f;
	}

	for (auto _ : state) {
		g_rec_layer.forward(x, out);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 256 * 256);
}

static void BM_Forward_Sparse(benchmark::State &state) {
	alignas(32) float x[256]{};
	alignas(32) float out[256]{};
	for (int i = 0; i < 256; ++i) {
		x[i] = static_cast<float>(i) * 0.001f;
	}

	for (auto _ : state) {
		g_rec_layer.forward_sparse(x, out);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 256 * 256);
}

static void BM_Forward_InputLayer(benchmark::State &state) {
	alignas(32) float x[2]{0.5f, 0.3f};
	alignas(32) float out[256]{};

	for (auto _ : state) {
		g_input_layer.forward(x, out);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 2 * 256);
}

BENCHMARK(BM_Forward_Dense)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Forward_Sparse)->MinTime(2.0)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Forward_InputLayer)->MinTime(2.0)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
