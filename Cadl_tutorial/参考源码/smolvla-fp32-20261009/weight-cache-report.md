> 来源：Golem/smolvla_cadl_float/results/2026-10-08-fp32-weight-cache/report.md，2026-10-09 复制。以下为历史报告，未在本次重新运行。报告内部相对证据链接需在原实验目录中使用；本教材只附此报告及选定源码。

# FP32 constant-weight packing and shared A tiles: 2026-10-08

This software-interface experiment keeps the original tiny dimensions, FP32 multiply/add order, fixtures, model graph and accelerator simulator. It follows the [clearing/packing optimization](../2026-10-08-fp32-interface-opt/report.md). The simulator SHA256 remains `706967e03aaaeb807d919404c6adab0e11c0fad55735d45868f329547228ba72`.

## Implementation

- Recognize immutable rank-2 RHS operands from constant MLIR globals. A strict single-block use analysis binds each GEMM to its latest constant transpose producer, even when gate and up reuse one temporary allocation. Unknown aliases, mutable sources and unsupported control flow remain dynamic.
- Replace supported materialized weight transposes with strided views; the matrix values and arithmetic stay identical.
- Allocate a separate 128-byte-aligned packed RHS buffer per proven-constant GEMM. First use fills it; subsequent calls consume it directly for DMA or CPU computation. Tiny VLM MLP uses 24 KiB and Expert MLP 20 KiB of additional main memory.
- Pack one A tile per K8 group and have all active output-column tiles read it.
- Keep the seven warm timing phases and record RHS packing versus reuse counts. A first-call module ROI includes initial packing; both first-call and warm outputs are independently checked.

## Coarse module measurements

Positive reduction means lower latency. First-call cycles below are for the new implementation; the previous snapshot did not retain first-call timing. CPU/CADL ratios use the newly optimized tiled CPU baseline, not native BLAS or a direct-layout CPU implementation.

| Module     | Previous CADL warm | New CADL warm | Warm reduction | New CADL first call | New CPU/CADL warm |
| ---------- | ------------------ | ------------- | -------------- | ------------------- | ----------------- |
| vlm_mlp    | 427,600            | 215,221       | +49.67%        | 361,782             | 1.8220x           |
| expert_mlp | 341,463            | 191,538       | +43.91%        | 309,086             | 1.7031x           |
| attention  | 44,301             | 47,241        | -6.64%         | 54,995              | 1.0653x           |
| rmsnorm    | 4,149              | 4,444         | -7.11%         | 5,672               | Control           |
| rope       | 10,183             | 11,096        | -8.97%         | 17,099              | Control           |

Both engines receive the interface changes. The old/new comparison measures the combined transpose, packing and input-sharing changes, including their code-layout effects. The new runner also checks first-call outputs before the warm invocation, which can perturb cache state relative to the historical runner. RMSNorm and RoPE have no GEMM; their differences are controls, not acceleration of those operators.

Attention is 6.64% slower than the previous implementation. Its dynamic operands receive no persistent RHS cache. Its detailed packing interval is slightly shorter, while other intervals and surrounding work grow; these measurements do not isolate one cause for the overall regression. The changes to the no-GEMM controls further limit attribution to dynamic packing alone. This version demonstrates MLP gains, not a speedup across every module.

## Same-source cache-disabled control

The following programs retain the same lowered graph, transpose views, shared-A code, runner, compiler flags and hardware. Recompiling only with `SP_RHS_CACHE=0` disables RHS persistence. This comparison measures the effect of persistent packing separately from the earlier changes; instruction layout and cache behavior remain part of that effect.

| Module | Engine | Cache off warm | Cache on warm | Warm reduction | Cache off first call | Cache on first call |
| --- | --- | --- | --- | --- | --- | --- |
| vlm_mlp | baseline | 508,666 | 392,140 | 22.91% | 517,069 | 535,082 |
| vlm_mlp | cadl | 346,997 | 215,221 | 37.98% | 358,722 | 361,782 |
| expert_mlp | baseline | 408,021 | 326,209 | 20.05% | 418,104 | 435,754 |
| expert_mlp | cadl | 290,153 | 191,538 | 33.99% | 298,868 | 309,086 |

First-call minus warm cycles is not a pure packing-cost measurement: it also includes first-use instruction/data cache effects. Initial RHS packing is included in the first-call ROI; global runtime initialization stays outside both module ROIs. No full-policy amortization is assumed. These are individual simulator measurements with a fixed seed, not estimates of run-to-run variability.

## Detailed CADL GEMM stages

### vlm_mlp

| Phase        | Previous cycles | New cycles | Cycles saved |
| ------------ | --------------- | ---------- | ------------ |
| prepare_c    | 16,321          | 14,587     | +1,734       |
| pack_ab      | 149,813         | 23,283     | +126,530     |
| load_a       | 8,353           | 8,378      | -25          |
| load_b       | 8,245           | 3,983      | +4,262       |
| compute      | 60,505          | 60,266     | +239         |
| store_c      | 940             | 1,043      | -103         |
| unpack_c     | 24,481          | 22,944     | +1,537       |
| unattributed | 6,483           | 7,150      | -667         |

### expert_mlp

| Phase        | Previous cycles | New cycles | Cycles saved |
| ------------ | --------------- | ---------- | ------------ |
| prepare_c    | 14,465          | 13,590     | +875         |
| pack_ab      | 115,861         | 19,161     | +96,700      |
| load_a       | 6,984           | 6,906      | +78          |
| load_b       | 6,866           | 3,293      | +3,573       |
| compute      | 45,453          | 45,243     | +210         |
| store_c      | 948             | 1,037      | -89          |
| unpack_c     | 22,398          | 21,887     | +511         |
| unattributed | 5,868           | 6,652      | -784         |

### attention

| Phase        | Previous cycles | New cycles | Cycles saved |
| ------------ | --------------- | ---------- | ------------ |
| prepare_c    | 3,607           | 3,901      | -294         |
| pack_ab      | 12,067          | 11,722     | +345         |
| load_a       | 1,457           | 1,493      | -36          |
| load_b       | 1,350           | 1,387      | -37          |
| compute      | 2,601           | 2,588      | +13          |
| store_c      | 758             | 859        | -101         |
| unpack_c     | 5,799           | 5,898      | -99          |
| unattributed | 2,865           | 3,454      | -589         |

## Validation and scope

- 73 host tests passed in the locked server environment, including the independent SoftFloat check, immutable-source MLIR rewriting, buffer reuse, invalidation, mutable A, negative strides, tails, NaNs, zero dimensions, invalid capacity/alignment and cache-disabled operation.
- 20 coarse/detailed module simulations and four cache-disabled controls passed numerical checks, full ROI checks and normal simulator exit. Each executable also checked its first-call output.
- All previous/current output checksums agree. Original exported graphs, input/weight fixtures, original bufferized graphs and precision-reference records match byte-for-byte. Lowered kernels and wrappers intentionally change to use constant views and the new interface.
- Warm MLP calls have only RHS cache hits; dynamic Attention has only RHS pack groups.
- Tile counts, DMA byte counts and all seven detailed interval counts match the previous version. The compute commands now select the shared A tile.
- The same simulator, seed, DRAM configuration and clock are used throughout.

Weights remain in a main-memory packed cache, not permanently in accelerator SRAM. DMA byte counts are unchanged. The explicit API requires invalidation before in-place RHS mutation; automatic reuse is enabled only for compiler-proven immutable globals. The isolated Attention workload has no model KV-cache lifecycle and receives no persistent RHS cache. Gate/up do not yet share A across separate GEMM calls. Results cover these tiny modules, not full-policy inference or full-size throughput.

## Evidence

- [Module before/after and first-call timings](before-after-modules.csv)
- [Cache on/off controls](cache-ablation.csv)
- [RHS hit/packing counts and capacities](cache-statistics.csv)
- [Both engines, all GEMM phase changes](before-after-phases.csv)
- [Validated raw measurements](validation.json)
- [Cache-disabled raw records](rhs-cache-ablation.json)
- [Workload and hardware identity audit](before-after-identity.json)
- [Source changes](source-changes.diff)
- [Host tests](host-tests.log)
- [Timer perturbation](timing-overhead.csv)
