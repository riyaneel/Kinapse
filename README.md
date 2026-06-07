# Kinapse (MTR-RSNN)

Source code for **MTR-RSNN: Multi-Timescale Recurrent SNNs with Kinetic NMDA Gating and Online Learning.**

---

## What this is

A bare-metal C++23/AVX2 implementation of a recurrent spiking neural network designed for low-latency sequential
decision-making. The network runs a complete forward pass in **4,845 ns** on a single x86-64 P-core.

Key properties:

- Kinetic Mg2+ gating (Kampa et al. 2004) via implicit Euler discretization, per-neuron persistent state B[t] in [0,1]
  evolved each tick
- Three-population architecture: fast hidden (256N, rho_s=0.50, 4 dendritic branches, 10% sparse recurrent), slow
  integrator (64N, rho_s=0.90, single branch), readout LIF (8N)
- Online learning via forward e-prop eligibility traces with per-synapse RMSProp normalization and per-action advantage
  baseline
- Structure-of-arrays memory layout, dual Estrin ILP for kinetic rate functions, branchless AVX2 spike logic

---

## Performance

Measured on Intel i7-12650H, single P-core, 4.1 GHz, performance governor (`taskset -c 8`):

| Kernel                                  | Cycles | ns    | Cycles/unit  |
|-----------------------------------------|--------|-------|--------------|
| `tick()` (256N, 4 branches, NMDA + AHP) | 1,082  | 264   | 4.23/neuron  |
| `tick_d0()` (64N, 1 branch)             | 1,603  | 391   | 25.05/neuron |
| `forward_dense()` (256x256)             | 11,295 | 2,755 |              |
| `forward_sparse()` (256x256, 10%)       | 15,350 | 3,744 |              |
| `predict()` (full pipeline)             | 19,864 | 4,845 |              |

IPC: 3.5 (`tick()`), 4.2 (`predict()`). Branch misprediction: 0.00%.

---

## Repository layout

```
asm/                   - annotated assembly listings
  tick_full.s          - tick() inner loop (0x405ff0-0x406254)
  layer_full.s         - forward_dense() inner loop
  predict_full.s       - full inlined predict()
  annotated_asm.md     - line-by-line annotation with port assignments and TMA analysis

bench/                 - Google Benchmark harness
  bench_tick.cpp       - tick() / tick_d0() / h_prime
  bench_layer.cpp      - forward_dense() / forward_sparse()
  bench_predict.cpp    - full predict() pipeline

data/                  - training data

script/
  data_compiler.py     - downloads and compiles Binance public trade data to binary format

snn/include/snn/       - inference engine (header-only)
  DendriticPopulation  - tick() / tick_d0(), kinetic NMDA, AHP, SFA, graded spikes
  SynapticLayer        - column-major GEMV, sparse block index
  SnnInferencer        - full predict() pipeline
  SurrogateGradient    - triangular / fast-sigmoid / rectangular surrogates
  WeightsFormat        - binary weight file layout

trainer/               - online training (SnnTrainer + e-prop)
  SnnTrainer.hpp/.cpp  - apply_e_prop(), astrocyte neuromodulation, per-action advantage
```

---

## Build

Requires: GCC or Clang with C++23 support, CMake >= 3.31, AVX2-capable x86-64 CPU.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run benchmarks (pin to a single core for reproducibility):

```bash
sudo cpupower frequency-set -g performance
taskset -c 8 ./build/bench/bench_tick    --benchmark_min_time=2.0
taskset -c 8 ./build/bench/bench_layer   --benchmark_min_time=2.0
taskset -c 8 ./build/bench/bench_predict --benchmark_min_time=2.0
```

---

## Training data

`data_compiler.py` downloads public Binance trade data and compiles it to the binary format expected by `SnnTrainer`. No
proprietary data is included.

```bash
pip install numpy pandas requests
python script/data_compiler.py --symbol <symbol> --year <year> --month <month> --out <out_file.bin>
```

---

## Weight file format

Binary, float32 little-endian. Layout (in order):

1. Magic header: `kWeightsMagic`
2. For each of {d0, d1, d2, d3}: weights `(2 x 256)` + biases `(256)`
3. `layer_hidden_slow_`: weights `(256 x 64)` + biases `(64)`
4. `layer_slow_out_`: weights `(64 x 8)` + biases `(8)`
5. `layer_rec_`: weights `(256 x 256)` + biases `(256)`

The 10% sparse recurrent mask is regenerated deterministically at load time from `kRecMaskSeed`, it is not stored in the
file.

Pre-trained weights are available on request.

---

## License

See [LICENSE](LICENSE).
