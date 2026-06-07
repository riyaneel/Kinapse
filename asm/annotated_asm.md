# ASM annotation of benchmark kernels

---

## 0.1 tick() inner loop

- **Symbol:** `BM_Tick_Hidden(benchmark::State&)` at `0x405cc0`
- **Hot loop:** `0x405ff0` to `0x406254`, back-edge `jne 405ff0`
- **Configuration measured:** 256 neurons, 4 dendritic branches, kinetic NMDA gating, AHP.

### Loop structure

The benchmark produces two nested loops. The outer loop runs from `0x405eae` to `0x406264` with back-edge `jne 405ed0`.
Each outer iteration corresponds to one `state.KeepRunning()` call, i.e., one complete tick over all 256 neurons. At the
top of each outer iteration the compiler emits a prologue of `vbroadcastss` and `vmovss` instructions that broadcast
scalar neuron parameters into YMM registers. This prologue runs once per tick call, not once per 8-neuron block.

The inner loop runs from `0x405ff0` to `0x406254` with back-edge `jne 405ff0`. The stride is `add $0x20,%rax` (32
bytes = one AVX2 register of 8 floats). The bounds are set by `mov $0x474b00,%eax` at the top of the outer iteration and
`mov $0x474f00,%edx` for the comparison. The iteration count is `(0x474f00 - 0x474b00) / 0x20 = 32`. Each inner
iteration processes 8 neurons in parallel via 256-bit AVX2 packed-float arithmetic. Total: 32 x 8 = 256 neurons.

### Constant hoisting

All scalar neuron parameters are broadcast to YMM registers in the outer loop prologue at addresses `0x405eae` through
`0x405fe1`. The registers ymm13, ymm12, and ymm11 receive constants from `g_hidden`. The registers ymm8, ymm7, ymm6,
ymm9, and ymm10 receive scalar fields. ymm5 and ymm1 are computed from those scalars. Eleven additional YMM constants
are spilled to the stack in slots `0x80(%rsp)` through `0x140(%rsp)`, then reloaded from those fixed stack addresses
inside the inner loop.

There is zero `vbroadcastss` inside the inner loop. The 11 `vmovaps` instructions that load from `%rsp`-relative
addresses inside the inner loop are: `0xa0(%rsp)`, `0xc0(%rsp)`, `0x80(%rsp)`, `0x100(%rsp)`, `0x120(%rsp)`,
`0x140(%rsp)`, `0x40(%rsp)`, `0x60(%rsp)`, `0xe0(%rsp)`, `0x20(%rsp)`, `(%rsp)`. All of these read values are written in
the outer prologue. The compiler chose to keep 11 YMM constants in the stack frame rather than keep them in registers
because the inner loop already uses 16 YMM registers simultaneously.

### Instruction count: inner loop, 98 instructions total

| Mnemonic                                                    | Count  | Execution port (Alder Lake P-core) |
|-------------------------------------------------------------|--------|------------------------------------|
| `vmovaps` reg to mem, stores to SoA arrays via `%rax`       | 15     | port 4 or 7                        |
| `vmovaps` mem to reg, loads from SoA arrays via `%rax`      | 7      | port 2 or 3                        |
| `vmovaps` mem to reg, loads from stack constants via `%rsp` | 9      | port 2 or 3                        |
| `vmovaps` reg to reg, register copies                       | 7      | port 5 or register rename          |
| `vfmadd132ps`                                               | 13     | port 0 or 1                        |
| `vfmadd231ps`                                               | 8      | port 0 or 1                        |
| `vfmadd213ps`                                               | 1      | port 0 or 1                        |
| `vfnmadd132ps`                                              | 1      | port 0 or 1                        |
| `vfnmadd213ps`                                              | 1      | port 0 or 1                        |
| `vmulps`                                                    | 16     | port 0 or 1                        |
| `vaddps`                                                    | 11     | port 0 or 1                        |
| `vsubps`                                                    | 2      | port 0 or 1                        |
| `vmaxps`                                                    | 2      | port 0 or 1                        |
| `vminps`                                                    | 1      | port 0 or 1                        |
| `vblendvps`                                                 | 3      | port 5                             |
| `vrcpps`                                                    | 1      | port 0                             |
| `vcmpgeps`                                                  | 1      | port 0 or 1                        |
| `vpsrad`                                                    | 1      | port 0 or 1                        |
| `vandps`                                                    | 1      | port 0 or 1                        |
| `vxorps`                                                    | 1      | port 0 or 1                        |
| `add`                                                       | 1      | port 0, 1, 5, or 6                 |
| `cmp`                                                       | 1      | port 0, 1, 5, or 6                 |
| `jne`                                                       | 1      | port 6                             |
| **Total**                                                   | **98** |                                    |

- FMA group: 13 + 8 + 1 + 1 + 1 = 24 fused multiply-add instructions.
- Multiply group: 16 `vmulps`.
- Additive group: 11 + 2 + 2 + 1 = 16 (`vaddps`, `vsubps`, `vmaxps`, `vminps`).
- Memory group: 31 `vmovaps` (15 stores + 7 array loads + 9 stack loads) plus 7 reg-reg copies.
- Blend and logic group: 3 `vblendvps` + 1 `vpsrad` + 1 `vandps` + 1 `vxorps` + 1 `vcmpgeps` = 8.
- Control group: 1 `add` + 1 `cmp` + 1 `jne` = 3.

### Throughput analysis

The benchmark measures 1,082 cycles for 32 inner iterations, giving 1,082 / 32 = **33.8 cycles per inner iteration**.

- Port 0 and 1 carry all FMA and multiply work. With 24 FMA and 16 `vmulps` instructions, each with reciprocal
  throughput of 0.5 cycles, the port-0/1 bound is (24 + 16) x 0.5 = **20 cycles per iteration**.
- Port 4 and 7 carry all stores. With 15 stores at 1 cycle each, the store bound is **15 cycles per iteration**.
- Port 2 and 3 carry all loads. With 7 + 9 = 16 memory loads at 0.5 cycles each, the load bound is **8 cycles per
  iteration**.
- Port 5 carries `vblendvps` and register copies. With 3 `vblendvps` plus 7 reg-reg `vmovaps` the port-5 bound is
  roughly **10 cycles per iteration**.

The theoretical throughput floor is max(20, 15, 8, 10) = 20 cycles. The measured 33.8 cycles exceed this floor by 13.8
cycles. The TMA metric `tma_backend_bound` is 45.3%, which points to memory latency rather than port saturation. With 21
distinct SoA array offsets touched per inner iteration, the store-forwarding and L1D reuse pattern creates latency
stalls that are not captured by the pure throughput model.

### Memory access pattern

The inner loop accesses 21 distinct offsets relative to `%rax`:
`-0x20`, `0x3e0`, `0x7e0`, `0xbe0`, `0xfe0`, `0x13e0`, `0x17e0`, `0x1be0`, `0x1fe0`, `0x23e0`, `0x27e0`, `0x2be0`,
`0x2fe0`, `0x33e0`, `0x38e0`, `0x3ce0`, `0x40e0`, `0x44e0`, `0x48e0`, `0x4ce0`, `0x50e0`.

From `-0x20` to `0x33e0`, consecutive offsets differ by exactly `0x400` bytes (256 floats, one complete SoA array).
Between `0x33e0` and `0x38e0` the gap is `0x500` bytes (320 floats), which indicates either a padding region or an
additional field between two arrays in the struct layout. The offsets from `0x38e0` onward resume at `0x400` spacing.
This pattern touches 17 distinct SoA arrays per 8-neuron block. The total working set across all 32 inner iterations is
17 x 256 x 4 = **17,408 bytes**, which fits within the 32 KB L1D cache assuming no eviction from unrelated accesses.

---

## 0.2 forward_dense() inner loop

- **Symbol:** `BM_Forward_Dense(benchmark::State&)` at `0x4059d0`
- **Hot inner loop:** `0x405c20` to `0x405c39`, back-edge `jne 405c20`

### Loop structure

The forward pass over a dense 256x256 weight matrix produces two nested loops.

The outer loop runs from `0x405c10` to `0x405c54` with back-edge `jne 405c54`. Each outer iteration processes one input
element `x[j]` for j from 0 to 255. At the start of each outer iteration, `vbroadcastss (%rsi),%ymm1` broadcasts the
scalar `x[j]` to all 8 lanes of ymm1. The pointer `%rsi` advances by 4 bytes per outer iteration (`add $0x4,%rsi`),
traversing the input vector sequentially. The weight column pointer `%rdi` advances by `0x400` bytes (1024 bytes = 256
floats = one full weight column) per outer iteration via `add $0x400,%rdi`. The counter in `%rcx` advances by `0x100`
and stops at `0x10000`, giving `0x10000 / 0x100 = 256` outer iterations.

The inner loop runs from `0x405c20` to `0x405c39` with back-edge `jne 405c20`. Both `%rax` (accumulator pointer) and
`%rdx` (weight pointer) advance by `0x20` bytes per iteration. The accumulator buffer starts at `0x40(%rsp)` and the end
bound is at `rbx = 0x440(%rsp)`, giving `0x400 / 0x20 = 32` inner iterations. Each inner iteration covers 8 output
neurons.

* **Total:** 256 outer x 32 inner = 8,192 `vfmadd213ps` calls, each processing 8 lanes, = 65,536 scalar MACs, consistent
  with a 256x256 fully connected layer.

### Inner loop: 7 instructions

```asm
405c20:  vmovaps (%rdx),%ymm0           ; load 8 weights W[j*256+i .. j*256+i+7]
405c24:  vfmadd213ps (%rax),%ymm1,%ymm0 ; ymm0 = ymm0*ymm1 + mem[rax], i.e. W*x[j] + acc[i..i+7]
405c29:  add $0x20,%rax                 ; advance accumulator pointer by 32 bytes
405c2d:  add $0x20,%rdx                 ; advance weight pointer by 32 bytes
405c31:  vmovaps %ymm0,-0x20(%rax)      ; store result back, -0x20 because rax already advanced
405c36:  cmp %rax,%rbx                  ; check end of output block
405c39:  jne 405c20                     ; branch back if more output neurons remain
```

| Mnemonic                                              | Count |
|-------------------------------------------------------|-------|
| `vmovaps` load                                        | 1     |
| `vfmadd213ps` with fused accumulator load from `%rax` | 1     |
| `vmovaps` store                                       | 1     |
| `add`                                                 | 2     |
| `cmp`                                                 | 1     |
| `jne`                                                 | 1     |
| **Total**                                             | **7** |

`ymm1` holds the broadcast value of `x[j]` and is **never reloaded inside the inner loop**. The `vbroadcastss` that
produced `ymm1` is at `0x405c10` in the outer loop. There is zero register spill of `ymm1` in the inner loop body. The
compiler correctly identified that `x[j]` is loop-invariant with respect to the inner loop and kept it in a register for
all 32 inner iterations.

The load-to-FMA ratio is 2 loads per FMA: one explicit `vmovaps` from `%rdx` (the weight block) and one fused load
inside`vfmadd213ps` from `%rax` (the accumulator). The weight layout is column-major: for a given `j`, the 256 weights
`W[j*256+0]` through `W[j*256+255]` are contiguous in memory, which the inner loop traverses sequentially.

### Throughput analysis

The benchmark measures 11,295 cycles for 256 outer iterations. Cycles per outer iteration: 11,295 / 256 = **44.1 cycles
per j-iteration** (expected from the benchmark statement).

Each outer iteration runs 32 inner iterations. The FMA throughput bound for 32 `vfmadd213ps` at 0.5 cycles each is 16
cycles. The store throughput bound for 32 `vmovaps` stores at 1 cycle each is 32 cycles. The load throughput for 32 +
32 = 64 loads at 0.5 cycles each is 32 cycles. The dominant bound from stores and loads combined is approximately 32
cycles of useful compute per outer iteration. The measured 44.1 cycles include the outer loop overhead: 1
`vbroadcastss`, 3 `add`, 1 `cmp`, and pointer resets, accounting for the remaining ~12 cycles.

---

## 0.3 tick_d0() compared to tick()

- **Symbol:** `BM_Tick_Slow(benchmark::State&)` at `0x4062b0`
- **Hot loop:** `0x406610` to `0x4067b1`, back-edge `jne 406610`
- **Configuration measured:** 64 neurons, 1 dendritic branch (D0 only), no I_rec.

### Loop structure

The structure mirrors `BM_Tick_Hidden` exactly: an outer benchmark iteration loop with a constant-broadcasting prologue,
and an inner loop that processes 8 neurons per AVX2 iteration. The inner loop stride is `add $0x20,%rax` with bounds
`mov $0x473900,%eax` and `mov $0x473a00,%edx`.

Iterations: `(0x473a00 - 0x473900) / 0x20 = 8`. Total neurons: 8 x 8 = 64.

### Instruction count: inner loop, 69 instructions total

| Mnemonic                                                      | Count  |
|---------------------------------------------------------------|--------|
| `vmovaps` (all: stores, array loads, stack loads, reg copies) | 18     |
| `vmulps`                                                      | 13     |
| `vfmadd132ps`                                                 | 7      |
| `vfmadd231ps`                                                 | 5      |
| `vfmadd213ps`                                                 | 1      |
| `vfnmadd132ps`                                                | 1      |
| `vfnmadd213ps`                                                | 1      |
| `vaddps`                                                      | 7      |
| `vsubps`                                                      | 2      |
| `vmaxps`                                                      | 2      |
| `vminps`                                                      | 1      |
| `vblendvps`                                                   | 3      |
| `vrcpps`                                                      | 1      |
| `vcmpgeps`                                                    | 1      |
| `vpsrad`                                                      | 1      |
| `vandps`                                                      | 1      |
| `vxorps`                                                      | 1      |
| `add` + `cmp` + `jne`                                         | 3      |
| **Total**                                                     | **69** |

- FMA total: 7 + 5 + 1 + 1 + 1 = 15.

### Comparison with tick()

| Metric                                          | tick() hidden (256N, 4 branches) | tick_d0() slow (64N, 1 branch) |
|-------------------------------------------------|----------------------------------|--------------------------------|
| Inner iterations                                | 32                               | 8                              |
| Instructions per iteration                      | 98                               | 69                             |
| FMA per iteration                               | 24                               | 15                             |
| vmulps per iteration                            | 16                               | 13                             |
| vaddps + vsubps + vmaxps + vminps per iteration | 16                               | 12                             |
| Measured total cycles                           | 1,082                            | 1,603                          |
| Cycles per inner iteration                      | 33.8                             | 200.4                          |
| Cycles per neuron                               | 4.23                             | 25.05                          |

**Instruction count effect.** The reduction from 98 to 69 instructions per inner iteration represents a 29.6% decrease,
explained by three eliminated branches. Each dendritic branch contributes a `vmulps` chain for AMPA and NMDA kinetics
plus `vfmadd` for the branch output. Removing branches D1, D2, and D3 eliminates 9 FMAs (from 24 to 15) and 3 `vmulps`
(from 16 to 13). The AHP computation (the two-conductance `g_fast + g_slow` path), the somatic integration, the graded
spike logic, and the ALIF threshold update are **identical** in both loops: the same `vcmpgeps`, `vpsrad`, `vblendvps`,
`vandps`, `vrcpps`, and `vmaxps` sequence appears in both at equivalent positions in the instruction stream.

**I_rec elimination.** The `vfmadd231ps` instruction that accumulates the recurrent input into the somatic current is
absent in `tick_d0()`. This confirms that`tick_d0()` is called without a preceding `forward_sparse` pass: there is no
recurrent current to add.

**The cycles-per-iteration ratio is 200.4 / 33.8 = 5.93.** This is not explained by instruction count alone. If
instruction count were the only factor, eliminating 29.6% of instructions would predict 33.8 x 0.704 = 23.8 cycles per
iteration, not 200.4. The large ratio arises from two independent effects:

- First, the outer benchmark prologue (approximately 15 `vbroadcastss` and `vmovaps` instructions) runs once per tick
  call regardless of neuron count. For `tick()` with 32 inner iterations this prologue cost is amortized over 32 useful
  iterations. For `tick_d0()` with 8 inner iterations the same prologue cost amortizes over only 8 iterations,
  quadrupling its relative contribution.
- Second, pipeline startup and drain costs (branch predictor warmup, instruction fetch initialization, ROB fill latency)
  are fixed per benchmark outer iteration and affect `tick_d0()` proportionally more due to the 4x smaller inner loop
  count.

---

## 0.4 compute_h_prime: triangular vs. fast-sigmoid

### Triangular surrogate

- **Symbol:** `BM_Hprime_Triangular(benchmark::State&)` at `0x4067f0`
- **Hot loop:** `0x406a40` to `0x406a75`, back-edge `jne 406a40`
- **Iterations:** `0x400 / 0x20 = 32` (256 neurons, 8 per AVX2 step).

The loop computes `h'(t) = max(0, 1 - beta * |V_soma_pre_spike - theta(t)|)` for all 256 neurons in 32 AVX2 iterations.

Before the loop, at `0x4069dd` through `0x4069f2`, the constants `theta` (broadcast into ymm2), the sign-mask for
absolute value (into ymm5), `beta` (into ymm4), and `1.0` (into ymm3) are broadcast once. No constant load occurs inside
the loop.

**Inner loop: 11 instructions**

```asm
406a40:  vaddps  0x476f00(%rax),%ymm2,%ymm1  ; V_soma_pre_spike[i] + theta = shift
406a48:  vmovaps 0x477300(%rax),%ymm0         ; load V_soma_pre_spike[i]
406a50:  vsubps  %ymm1,%ymm0,%ymm0           ; V - (V + theta) = produces signed distance
406a54:  vxorps  %xmm1,%xmm1,%xmm1          ; zero register for max comparison
406a58:  vandnps %ymm0,%ymm5,%ymm0           ; clear sign bit: |V - theta|
406a5c:  vfnmadd132ps %ymm4,%ymm3,%ymm0      ; ymm0 = 1 - beta * |V - theta|
406a61:  vmaxps  %ymm0,%ymm1,%ymm0           ; max(0, 1 - beta*|V - theta|)
406a65:  vmovaps %ymm0,0x40(%rsp,%rax,1)     ; store h'[i]
406a6b:  add $0x20,%rax
406a6f:  cmp $0x400,%rax
406a75:  jne 406a40
```

| Mnemonic                  | Count  | Port        |
|---------------------------|--------|-------------|
| `vaddps`                  | 1      | port 0 or 1 |
| `vmovaps` load from array | 1      | port 2 or 3 |
| `vsubps`                  | 1      | port 0 or 1 |
| `vxorps`                  | 1      | port 0 or 1 |
| `vandnps`                 | 1      | port 0 or 1 |
| `vfnmadd132ps`            | 1      | port 0 or 1 |
| `vmaxps`                  | 1      | port 0 or 1 |
| `vmovaps` store           | 1      | port 4 or 7 |
| `add` + `cmp` + `jne`     | 3      | various     |
| **Total**                 | **11** |             |

### Fast-sigmoid surrogate

- **Symbol:** `BM_Hprime_FastSigmoid(benchmark::State&)` at `0x406ac0`
- **Hot loop:** `0x406d00` to `0x406d3e`, back-edge `jne 406d00`
- **Iterations:** `0x400 / 0x20 = 32` (256 neurons, 8 per AVX2 step, identical to triangular).

Before the loop, at `0x406cae` through `0x406ccc`, constants are broadcast once: the sign-mask into ymm6, `beta` into
ymm5, the constant `1.0` into ymm4, and the Newton-Raphson correction constant into ymm3.

**Inner loop: 13 instructions**

```asm
406d00:  vaddps  0x476f00(%rax),%ymm2,%ymm1  ; V + theta
406d08:  vmovaps 0x477300(%rax),%ymm0         ; load V
406d10:  vsubps  %ymm1,%ymm0,%ymm0           ; signed distance
406d14:  vandnps %ymm0,%ymm6,%ymm0           ; |V - theta|
406d18:  vfmadd132ps %ymm5,%ymm4,%ymm0       ; beta*|V - theta| + 1.0
406d1d:  vmulps  %ymm0,%ymm0,%ymm0           ; (1 + beta*|V - theta|)^2
406d21:  vrcpps  %ymm0,%ymm1                 ; approximate reciprocal, 12-bit precision
406d25:  vfnmadd132ps %ymm1,%ymm3,%ymm0      ; Newton-Raphson step 1: 2 - x*est
406d2a:  vmulps  %ymm0,%ymm1,%ymm1           ; Newton-Raphson step 2: est * correction
406d2e:  vmovaps %ymm1,0x40(%rsp,%rax,1)     ; store h'[i]
406d34:  add $0x20,%rax
406d38:  cmp $0x400,%rax
406d3e:  jne 406d00
```

| Mnemonic              | Count  | Port        |
|-----------------------|--------|-------------|
| `vaddps`              | 1      | port 0 or 1 |
| `vmovaps` load        | 1      | port 2 or 3 |
| `vsubps`              | 1      | port 0 or 1 |
| `vandnps`             | 1      | port 0 or 1 |
| `vfmadd132ps`         | 1      | port 0 or 1 |
| `vmulps`              | 2      | port 0 or 1 |
| `vrcpps`              | 1      | port 0 only |
| `vfnmadd132ps`        | 1      | port 0 or 1 |
| `vmovaps` store       | 1      | port 4 or 7 |
| `add` + `cmp` + `jne` | 3      | various     |
| **Total**             | **13** |             |

### Diff between triangular and fast-sigmoid

Both loops access the exact same two memory arrays: `0x476f00(%rax)` and `0x477300(%rax)`. The stride, iteration count,
output buffer address, and the set of constants broadcast before the loop are identical. The memory access pattern is
therefore byte-for-byte identical between the two variants.

The instruction count is 11 for triangular and 13 for fast-sigmoid. The two extra instructions in fast-sigmoid are the
two `vmulps` operations: one that squares the argument before the reciprocal, and one that applies the Newton-Raphson
correction. Triangular replaces the full denominator computation with a single `vfnmadd132ps`, which has no data
dependency on a prior `vrcpps`.

The 30% wall-clock speedup (16.7 ns triangular vs. 23.7 ns fast-sigmoid) has two sources. The first source is the
instruction count: 11 vs. 13 = 84.6% of the instructions, predicting roughly 15% speedup from raw instruction reduction
alone. The second source is the `vrcpps` dependency chain. On Alder Lake P-cores, `vrcpps` issues to port 0 only with
latency of 4 cycles. The `vfnmadd132ps` at `406d25` cannot begin until `vrcpps` at `406d21` completes, and the final
`vmulps` at `406d2a` cannot begin until `vfnmadd132ps` completes.
This creates a sequential dependency chain of `4 + 4 + 4 = 12 cycles` of minimum latency that cannot be hidden by
out-of-order execution since the next loop iteration depends on the store completing. Triangular has no equivalent
dependency chain: `vfnmadd132ps`at `406a5c` reads from `vandnps` (1 cycle latency) and `vmaxps` can follow immediately.
The combination of 2 extra instructions and the 12-cycle recip dependency chain is consistent with the observed 30%
difference.

---

## 0.5 forward_sparse() SparseBlock loop

- **Symbol:** `BM_Forward_Sparse(benchmark::State&)` at `0x4056c0`
- **Hot loop:** `0x405950` to `0x4059a0`, back-edge `jb 405950`
- **Iteration count:** the total number of active SparseBlocks in the 256x256 weight matrix at 10% density. This count
  is loaded from `g_rec_layer+0xd0400` into `%ecx` at `0x405944`. At 10% density in a 256x256 matrix (8 neurons per
  block, 256/8 = 32 blocks per row), roughly 26 blocks are active per row, giving approximately 26 iterations of this
  loop per forward pass.

### Full loop body: 15 instructions

```asm
405950:  lea    0x8(%rax),%edx               ; compute blk_idx + 8 for prefetch lookahead
405953:  cmp    %ecx,%edx                    ; check if blk_idx + 8 is within bounds
405955:  jae    405966                        ; if out of bounds, skip prefetch
405957:  mov    0x534bc0(,%rax,8),%edx       ; load SparseBlock[blk_idx+8].w_offset
40595e:  prefetcht0 0x474780(,%rdx,4)        ; prefetch W_[future.w_offset] into L1
405966:  movzwl 0x534b84(,%rax,8),%edx       ; load SparseBlock[blk].x_idx (uint16, zero-extend)
40596e:  mov    0x534b80(,%rax,8),%esi       ; load SparseBlock[blk].w_offset (uint32)
405975:  vbroadcastss 0x0(%r13,%rdx,4),%ymm0 ; broadcast x[x_idx] to 8 lanes of ymm0
40597c:  movzwl 0x534b86(,%rax,8),%edx       ; load SparseBlock[blk].i_out (uint16, zero-extend)
405984:  add    $0x1,%rax                    ; blk_idx++
405988:  cmp    %ecx,%eax                   ; blk_idx < n_active_blocks ?
40598a:  vmovaps 0x40(%rsp,%rdx,4),%ymm4    ; load 8 accumulators at output position i_out
405990:  vfmadd132ps 0x474780(,%rsi,4),%ymm4,%ymm0 ; ymm0 = x[idx]*W_block + acc[i_out..i_out+7]
40599a:  vmovaps %ymm0,0x40(%rsp,%rdx,4)    ; store updated accumulators
4059a0:  jb     405950
```

| Mnemonic                     | Count  | Role                                                |
|------------------------------|--------|-----------------------------------------------------|
| `lea`                        | 1      | compute prefetch lookahead index                    |
| `cmp` (prefetch guard)       | 1      | bounds check for prefetch                           |
| `jae` (prefetch skip branch) | 1      | taken only near end of block list                   |
| `mov` (prefetch target)      | 1      | load w_offset of future block                       |
| `prefetcht0`                 | 1      | issue L1 prefetch 8 blocks ahead                    |
| `movzwl` x 2                 | 2      | load x_idx and i_out as zero-extended 16-bit fields |
| `mov` (w_offset)             | 1      | load current block weight offset                    |
| `vbroadcastss`               | 1      | broadcast scalar x[x_idx] to 8 lanes                |
| `add`                        | 1      | increment block index                               |
| `cmp` (loop bound)           | 1      | main termination check                              |
| `vmovaps` load               | 1      | load 8 accumulator floats                           |
| `vfmadd132ps`                | 1      | fused multiply-accumulate with weight block         |
| `vmovaps` store              | 1      | write 8 updated accumulators                        |
| `jb`                         | 1      | loop back-edge                                      |
| **Total**                    | **15** |                                                     |

### Overhead relative to forward_dense

The forward_dense inner loop is 7 instructions. forward_sparse is 15 instructions. The difference is 8 instructions. All
8 extra instructions are scalar: `lea`, `cmp`, `jae`, `mov` (prefetch target), `prefetcht0`, two `movzwl`, and one
`mov`. None of them execute on port 0/1 (FMA units) or port 4/7 (store units). They compete for port 2/3 (load units)
and ports 0/1/5/6 (integer ALU).

In the forward_dense inner loop, `x[j]` is already in ymm1 (broadcast once per outer iteration) and the weight column is
addressed by a simple sequential pointer. In forward_sparse, `x[x_idx]` must be gathered per block via `vbroadcastss`
with an indirect index, and the weight block must be addressed via `w_offset` loaded from the SparseBlock struct. Both
require pointer arithmetic that forward_dense avoids entirely.

The benchmark measures 3,744 ns / 15,350 cycles for forward_sparse versus 2,755 ns / 11,295 cycles for forward_dense.
The ratio is 15,350 / 11,295 = 1.36, i.e. 36% slower. This is consistent with the 8 extra scalar instructions per block
adding overhead that accumulates over the ~26 active blocks, even though they run in parallel with AVX units on
different ports.

**Prefetch strategy:** The `prefetcht0` at `40595e` targets `W_[blk+8].w_offset`, issuing a fetch 8 blocks ahead. At
~10% density (roughly 26 active blocks total), this provides approximately 2 to 3 blocks of lookahead. The prefetch
guard (`cmp` + `jae`) prevents out-of-bounds access in the final 8 iterations of the loop. The `jae` branch is predicted
not-taken for the first `n_active - 8` iterations (the steady-state case), incurring no branch misprediction penalty in
the common path.

---

## 0.6 predict() inlining map

- **Symbol:** `BM_Predict(benchmark::State&)` at `0x407330`
- **Body bounds:** `0x407330` to `0x4080e0` (next symbol: `std::basic_string::_M_create`).

**Total code size:** 2,992 bytes.

### All functions are inlined into predict()

Every neural network kernel call in predict() is inlined by the compiler. There are zero `call` instructions to tick(),
forward_dense(), or forward_sparse() in the hot path of BM_Predict. The calls present in the body are: 5 `call memcpy`
instructions (copying the 4 dendrite input buffers and the recurrent input buffer),
`call benchmark::State::StartKeepRunning`, and `call benchmark::State::FinishKeepRunning`, all outside the measured
region.

| Inlined function                                     | Back-edge address | Instructions per iteration | Notes                                                              |
|------------------------------------------------------|-------------------|----------------------------|--------------------------------------------------------------------|
| forward_dense D0 (2 inputs, 256 outputs)             | `0x407599` jne    | 7                          | Identical pattern to standalone inner loop                         |
| forward_dense D1 (2 inputs, 256 outputs)             | `0x4075d9` jne    | 7                          | Immediately follows D0, back-to-back                               |
| forward_dense D2 (2 inputs, 256 outputs)             | `0x407679` jne    | 7                          |                                                                    |
| forward_dense D3 (2 inputs, 256 outputs)             | `0x4077b9` jne    | 7                          |                                                                    |
| forward_sparse rec (256x256, 10% density)            | `0x40784f` jb     | 15                         | Identical SparseBlock pattern, prefetcht0 present                  |
| tick() hidden population (256N, 4 branches)          | `0x407bd3` jne    | 99                         | One extra vmovaps versus standalone (98)                           |
| forward_dense rec row-major (256 inputs, 64 outputs) | `0x407cf2` jne    | 21                         | Row-major accumulation: 1 broadcast + 8 vfmadd231ps per input      |
| tick_d0() slow population (64N, 1 branch)            | `0x407fb8` jne    | 70                         | One extra vmovaps versus standalone (69)                           |
| forward_dense readout (64 inputs, 8 outputs)         | `0x407ffe` jne    | 6                          | 1 vbroadcastss + 1 vfmadd231ps + 1 vmovaps store + add + cmp + jne |

### Difference between inlined tick() and standalone tick()

The standalone `BM_Tick_Hidden` inner loop runs from `0x405ff0` to `0x406254` and contains 98 instructions. The same
tick() inlined inside `BM_Predict` runs from `0x407968` to `0x407bd3` and contains 99 instructions. The extra
instruction is a `vmovaps` register-to-register copy. This appears because the surrounding predict() context has live
YMM registers more simultaneously than the isolated tick() benchmark, forcing the compiler to emit one extra register
shuffle to avoid a conflict. The functional computation is identical.

The same pattern holds for tick_d0(): standalone is 69 instructions, inlined in predict() is 70 instructions, with the
same type of extra reg-to-reg `vmovaps`.

### Row-major forward_dense for the recurrent layer

The recurrent layer forward pass inlined in predict() at `0x407c68` to `0x407cf2` uses a different code layout than the
column-major forward_dense of the dendritic layers. Instead of one `vbroadcastss` per outer iteration followed by 32
inner `vfmadd213ps`, this loop does one `vbroadcastss` per input neuron and 8 `vfmadd231ps` per iteration, accumulating
into 8 simultaneous YMM registers (ymm1 through ymm8) that each hold one 8-neuron output block. The stride on `%rax` is
`0x100` (64 floats, i.e. one full output block of 64 neurons = 8 blocks of 8). The stride on `%rdx` is 4 bytes (1 input
float per outer iteration). This produces 256 iterations total (one per input neuron) and covers all 64 output neurons
of the slow population. The 8 accumulated YMM registers are written to memory at `0x407c13` through `0x407c5b` outside
the main loop.

### frontend_bound=15.1% in predict()

The TMA metrics for predict() are: `tma_retiring=62.5%`, `tma_frontend_bound=15.1%`, `IPC=4.2`. The 2,992-byte body of
BM_Predict means the instruction stream the CPU must fetch spans a region large enough that not all of it fits in the
decoded instruction cache (DSB) simultaneously. The 9 inlined loop bodies span roughly 1,500 bytes of the hot code path.
When the outer benchmark loop transitions between the 9 inner loops in sequence, the branch predictor must redirect the
instruction fetch unit to a new loop top for each transition. The `frontend_bound=15.1%` is consistent with occasional
DSB misses at these transition points rather than sustained frontend starvation, since `tma_retiring=62.5%` confirms the
pipeline is well-utilized whenever the fetch stream is warm.

---

## Summary

| Kernel                                   | Hot loop range           | Instructions per inner iteration | Back-edge    |
|------------------------------------------|--------------------------|----------------------------------|--------------|
| tick() hidden                            | `0x405ff0` to `0x406254` | 98                               | `jne 405ff0` |
| tick_d0() slow                           | `0x406610` to `0x4067b1` | 69                               | `jne 406610` |
| h_prime triangular                       | `0x406a40` to `0x406a75` | 11                               | `jne 406a40` |
| h_prime fast-sigmoid                     | `0x406d00` to `0x406d3e` | 13                               | `jne 406d00` |
| forward_dense inner                      | `0x405c20` to `0x405c39` | 7                                | `jne 405c20` |
| forward_sparse                           | `0x405950` to `0x4059a0` | 15                               | `jb 405950`  |
| tick() inlined in predict()              | `0x407968` to `0x407bd3` | 99                               | `jne 407968` |
| tick_d0() inlined in predict()           | `0x407e0f` to `0x407fb8` | 70                               | `jne 407e0f` |
| forward_sparse inlined in predict()      | `0x4077f6` to `0x40784f` | 15                               | `jb 4077f6`  |
| forward_dense rec row-major in predict() | `0x407c68` to `0x407cf2` | 21                               | `jne 407c68` |
| forward_dense readout in predict()       | `0x407fe0` to `0x407ffe` | 6                                | `jne 407fe0` |
