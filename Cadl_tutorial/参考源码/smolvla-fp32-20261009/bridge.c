/* FP32 interface using AQUAS hardfloat f32 multiplication and addition.
 * Each 128-byte DMA row contains four side-by-side 8x8 FP32 tiles.
 * Compile both engines with -fno-fast-math -ffp-contract=off, without LTO.
 * SP_ACCEL=1 selects RoCC; SP_ACCEL=0 is the matching software baseline.
 */
#include "bridge.h"
#include <stdio.h>
#include <string.h>
#ifndef __riscv
#include <time.h>
#endif

#ifndef SP_ACCEL
#define SP_ACCEL 0
#endif
#ifndef SP_PHASE_PROFILE
#define SP_PHASE_PROFILE 0
#endif
#if SP_ACCEL && !defined(__riscv)
#error "SP_ACCEL=1 requires the verified RV32 Rocket/RoCC simulator"
#endif
#if defined(__riscv) && __riscv_xlen != 32
#error "This bridge's DMA addresses and instruction operands require RV32"
#endif

#define SP_MAX_OPS 256
#define SP_DMA_WORDS 256
#define SP_A_TILE 8U
#define SP_B_TILE 24U
#define SP_D_TILE 40U

/* Four side-by-side 8x8 tiles: row i, tile t, column j maps to
 * data[i*32+t*8+j]. Each row is one 128-byte DMA burst. */
static float dma_a[SP_DMA_WORDS] __attribute__((aligned(128)));
static float dma_bt[SP_DMA_WORDS] __attribute__((aligned(128)));
static float dma_d[SP_DMA_WORDS] __attribute__((aligned(128)));
static float original_c[SP_DMA_WORDS];

#if SP_PHASE_PROFILE
enum sp_phase {
    SP_PREPARE_C, SP_PACK_AB, SP_LOAD_A, SP_LOAD_B,
    SP_COMPUTE, SP_STORE_C, SP_UNPACK_C, SP_PHASE_COUNT
};
static const char *const phase_names[SP_PHASE_COUNT] = {
    "prepare_c", "pack_ab", "load_a", "load_b", "compute", "store_c", "unpack_c"
};
#endif

struct profile_entry {
    uint64_t calls, cycles, tiles, dma_read_bytes, dma_write_bytes;
    uint64_t rhs_cache_hits, rhs_pack_groups;
#if SP_PHASE_PROFILE
    uint64_t phase_cycles[SP_PHASE_COUNT], phase_counts[SP_PHASE_COUNT];
#endif
};
static struct profile_entry profile[SP_MAX_OPS];
static int profiling;
static int runtime_error;

static inline void fence_memory(void) {
#ifdef __riscv
    __asm__ volatile ("fence rw, rw" ::: "memory");
#else
    __asm__ volatile ("" ::: "memory");
#endif
}

uint64_t sp_cycles(void) {
    fence_memory();
#ifdef __riscv
    uint32_t hi, low, hi_again;
    do {
        __asm__ volatile ("csrr %0, mcycleh\ncsrr %1, mcycle\ncsrr %2, mcycleh"
                          : "=&r"(hi), "=&r"(low), "=&r"(hi_again) :: "memory");
    } while (hi != hi_again);
    return ((uint64_t)hi << 32) | low;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

int sp_runtime_init(void) {
#ifdef __riscv
    uint32_t misa;
    __asm__ volatile ("csrr %0, misa" : "=r"(misa));
    if (!(misa & (1U << 5))) {
        puts("SP_ERROR Rocket F extension is unavailable");
        runtime_error = 1;
        return runtime_error;
    }
    __asm__ volatile ("csrw fcsr, zero" ::: "memory");
#endif
    return runtime_error;
}

int sp_runtime_status(void) { return runtime_error; }
void sp_profile_reset(void) { memset(profile, 0, sizeof(profile)); }
void sp_profile_enable(int enabled) { profiling = !!enabled; }
void sp_rhs_cache_invalidate(SpRhsCache *cache) {
    if (cache) cache->valid = 0;
}

void sp_format_u64(char out[21], uint64_t value) {
    char reversed[20];
    unsigned count = 0;
    do {
        reversed[count++] = (char)('0' + value % 10U);
        value /= 10U;
    } while (value);
    for (unsigned i = 0; i < count; ++i) out[i] = reversed[count - 1 - i];
    out[count] = '\0';
}

void sp_timer_calibrate(void) {
    const unsigned samples = 64;
    uint64_t minimum = UINT64_MAX, maximum = 0, total = 0;
    for (unsigned i = 0; i < samples; ++i) {
        uint64_t begin = sp_cycles();
        uint64_t delta = sp_cycles() - begin;
        if (delta < minimum) minimum = delta;
        if (delta > maximum) maximum = delta;
        total += delta;
    }
    char min_text[21], max_text[21], total_text[21];
    sp_format_u64(min_text, minimum);
    sp_format_u64(max_text, maximum);
    sp_format_u64(total_text, total);
    printf("SP_TIMER {\"unit\":\"%s\",\"samples\":%u,\"min\":%s,\"max\":%s,\"total\":%s}\n",
#ifdef __riscv
           "cycles",
#else
           "ns",
#endif
           samples, min_text, max_text, total_text);
}

#if SP_PHASE_PROFILE
static void print_phase_values(const uint64_t values[SP_PHASE_COUNT]) {
    putchar('{');
    for (unsigned phase = 0; phase < SP_PHASE_COUNT; ++phase) {
        char value[21];
        sp_format_u64(value, values[phase]);
        printf("%s\"%s\":%s", phase ? "," : "", phase_names[phase], value);
    }
    putchar('}');
}
#endif

void sp_profile_print(void) {
    for (unsigned id = 0; id < SP_MAX_OPS; ++id)
        if (profile[id].calls) {
            char calls[21], cycles[21], tiles[21], reads[21], writes[21];
            char cache_hits[21], pack_groups[21];
            sp_format_u64(calls, profile[id].calls);
            sp_format_u64(cycles, profile[id].cycles);
            sp_format_u64(tiles, profile[id].tiles);
            sp_format_u64(reads, profile[id].dma_read_bytes);
            sp_format_u64(writes, profile[id].dma_write_bytes);
            sp_format_u64(cache_hits, profile[id].rhs_cache_hits);
            sp_format_u64(pack_groups, profile[id].rhs_pack_groups);
            printf("SP_GEMM {\"index\":%u,\"calls\":%s,\"cycles\":%s,\"tile_calls\":%s,\"dma_read_bytes\":%s,\"dma_write_bytes\":%s,\"rhs_cache_hits\":%s,\"rhs_pack_groups\":%s,\"engine\":\"%s\",\"timing_mode\":\"%s\"",
                   id, calls, cycles, tiles, reads, writes,
                   cache_hits, pack_groups,
                   SP_ACCEL ? "cadl" : "baseline",
                   SP_PHASE_PROFILE ? "detailed" : "coarse");
#if SP_PHASE_PROFILE
            uint64_t phase_total = 0, timer_intervals = 0;
            for (unsigned phase = 0; phase < SP_PHASE_COUNT; ++phase) {
                phase_total += profile[id].phase_cycles[phase];
                timer_intervals += profile[id].phase_counts[phase];
            }
            char total_text[21], unattributed_text[21], intervals_text[21];
            sp_format_u64(total_text, phase_total);
            sp_format_u64(unattributed_text, profile[id].cycles - phase_total);
            sp_format_u64(intervals_text, timer_intervals);
            printf(",\"phase_cycles\":");
            print_phase_values(profile[id].phase_cycles);
            printf(",\"phase_counts\":");
            print_phase_values(profile[id].phase_counts);
            printf(",\"phase_total_cycles\":%s,\"unattributed_cycles\":%s,\"timer_intervals\":%s",
                   total_text, unattributed_text, intervals_text);
#endif
            puts("}");
        }
}

/* Phase bookkeeping occurs after each raw interval. It stays inside the
 * enclosing GEMM timer and is not subtracted from any reported measurement.
 * The coarse build removes these macros and all phase state completely. */
#if SP_PHASE_PROFILE
#define SP_PHASE_BEGIN() do { if (profiling) phase_start = sp_cycles(); } while (0)
#define SP_PHASE_END(phase) do {                                      \
    if (profiling) {                                                 \
        uint64_t elapsed = sp_cycles() - phase_start;                 \
        phase_cycles[phase] += elapsed;                              \
        phase_counts[phase] += 1;                                    \
    }                                                               \
} while (0)
#else
#define SP_PHASE_BEGIN() ((void)0)
#define SP_PHASE_END(phase) ((void)0)
#endif

#if SP_ACCEL
/* The dependent addi waits for the returned RoCC rd before the compiler can
 * proceed to a timestamp or buffer reuse. A memory clobber alone is not a
 * hardware completion acknowledgement. */
#define ROCC_COMMAND(name, opcode, funct)                                    \
    static inline uint32_t name(uint32_t rs1, uint32_t rs2) {                 \
        uint32_t rd;                                                       \
        __asm__ volatile (".insn r " #opcode ", 7, " #funct ", %0, %1, %2\n" \
                          "addi %0, %0, 0"                                 \
                          : "=r"(rd) : "r"(rs1), "r"(rs2) : "memory");      \
        return rd;                                                         \
    }
ROCC_COMMAND(rocc_compute, 0x0b, 0)
ROCC_COMMAND(rocc_load_a, 0x7b, 0)
ROCC_COMMAND(rocc_load_b, 0x7b, 1)
ROCC_COMMAND(rocc_store_c, 0x7b, 2)

static uint32_t dma_args(unsigned tile) {
    /* High half: scratchpad element address, not a tile number. Low half:
     * DRAM row pitch in bytes. Offsets fit the packed 16-bit fields. */
    return ((tile * 64U) << 16) | 128U;
}

static int check_ack(uint32_t ack) {
    if (ack) {
        printf("SP_ERROR RoCC acknowledgement=%lu\n", (unsigned long)ack);
        runtime_error = 2;
    }
    return ack != 0;
}
#endif

#if SP_RHS_CACHE
static int rhs_cache_layout(SpRhsCache *cache, intptr_t batch, intptr_t n,
                            intptr_t k, size_t *n_groups, size_t *k_groups) {
    *n_groups = (size_t)n / 32U + ((size_t)n % 32U != 0);
    *k_groups = (size_t)k / 8U + ((size_t)k % 8U != 0);
    size_t words = (size_t)batch;
    if (words > SIZE_MAX / *n_groups) goto invalid;
    words *= *n_groups;
    if (words > SIZE_MAX / *k_groups) goto invalid;
    words *= *k_groups;
    if (words > SIZE_MAX / SP_DMA_WORDS) goto invalid;
    words *= SP_DMA_WORDS;
    if (words > SIZE_MAX / sizeof(float)) goto invalid;
    if (!cache || !cache->data || ((uintptr_t)cache->data & 127U) ||
        cache->capacity_words < words ||
        (uintptr_t)cache->data > UINTPTR_MAX - words * sizeof(float)) goto invalid;
    return 1;
invalid:
    if (cache) cache->valid = 0;
    puts("SP_ERROR invalid RHS cache alignment, capacity or size");
    runtime_error = 4;
    return 0;
}
#endif

static void sp_gemm_impl(SpRhsCache *cache, int cache_requested,
             int id, intptr_t batch, intptr_t m, intptr_t n, intptr_t k,
             const float *a, intptr_t ao, intptr_t abs,
             intptr_t ars, intptr_t acs,
             const float *b, intptr_t bo, intptr_t bbs,
             intptr_t brs, intptr_t bcs,
             float *c, intptr_t co, intptr_t cbs,
             intptr_t crs, intptr_t ccs) {
    uint64_t start = profiling ? sp_cycles() : 0;
    if (id < 0 || id >= SP_MAX_OPS || batch < 0 || m < 0 || n < 0 || k < 0) {
        printf("SP_ERROR invalid GEMM id or dimensions, id=%d\n", id);
        runtime_error = 3;
        return;
    }
    const int use_rhs_cache = SP_RHS_CACHE && cache_requested;
    size_t rhs_n_groups = 0, rhs_k_groups = 0;
    int cache_warm = 0;
#if SP_RHS_CACHE
    if (use_rhs_cache && batch && m && n && k) {
        if (!rhs_cache_layout(cache, batch, n, k, &rhs_n_groups, &rhs_k_groups)) return;
        cache_warm = cache->valid && cache->packed_data == cache->data &&
            cache->source == b && cache->offset == bo && cache->batch == batch &&
            cache->n == n && cache->k == k && cache->batch_stride == bbs &&
            cache->row_stride == brs && cache->col_stride == bcs;
        /* Commit validity only after every requested slab and GEMM succeeded.
         * An early RoCC failure must not leave a valid partially used cache. */
        cache->valid = 0;
    }
#endif
    uint64_t tile_count = 0;
    uint64_t dma_read_bytes = 0, dma_write_bytes = 0;
    uint64_t rhs_cache_hits = 0, rhs_pack_groups = 0;
#if SP_PHASE_PROFILE
    uint64_t phase_start = 0;
    uint64_t phase_cycles[SP_PHASE_COUNT] = {0};
    uint64_t phase_counts[SP_PHASE_COUNT] = {0};
#endif
    for (intptr_t ib = 0; ib < batch; ++ib)
        for (intptr_t im = 0; im < m; im += 8)
            for (intptr_t jgroup = 0; jgroup < n; jgroup += 32) {
                unsigned tiles = (unsigned)((n - jgroup + 7) / 8);
                if (tiles > 4) tiles = 4;
                SP_PHASE_BEGIN();
                /* Only valid output entries are ever read from original_c;
                 * each is captured here before the final addition. */
                for (unsigned i = 0; i < 8; ++i)
                    for (unsigned t = 0; t < tiles; ++t)
                        for (unsigned j = 0; j < 8; ++j) {
                            intptr_t row = im + i, col = jgroup + t * 8 + j;
                            if (row < m && col < n)
                                original_c[i * 32 + t * 8 + j] =
                                    c[co + ib * cbs + row * crs + col * ccs];
                        }
                SP_PHASE_END(SP_PREPARE_C);
                for (intptr_t ik = 0; ik < k; ik += 8) {
                    const float *packed_b = dma_bt;
                    SP_PHASE_BEGIN();
                    /* All output tiles share the first A tile. The other
                     * three A lanes remain transferred but are never read. */
                    for (unsigned i = 0; i < 8; ++i)
                        for (unsigned x = 0; x < 8; ++x) {
                            intptr_t ar = im + i, ak = ik + x;
                            dma_a[i * 32 + x] = ar < m && ak < k
                                ? a[ao + ib * abs + ar * ars + ak * acs] : 0.0f;
                        }
                    float *rhs_destination = dma_bt;
                    if (use_rhs_cache) {
                        size_t slab = ((size_t)ib * rhs_n_groups + (size_t)jgroup / 32U)
                            * rhs_k_groups + (size_t)ik / 8U;
                        rhs_destination = cache->data + slab * SP_DMA_WORDS;
                        packed_b = rhs_destination;
                    }
                    if (!use_rhs_cache || (!cache_warm && im == 0)) {
                        /* Cached slabs initialize all 256 words once. Dynamic
                         * packing writes every active tile slot, including +0
                         * padding; inactive B tiles are never computed. */
                        unsigned packed_tiles = use_rhs_cache ? 4U : tiles;
                        for (unsigned i = 0; i < 8; ++i)
                            for (unsigned x = 0; x < 8; ++x) {
                                intptr_t ak = ik + x;
                                for (unsigned t = 0; t < packed_tiles; ++t) {
                                    intptr_t bc = jgroup + t * 8 + i;
                                    rhs_destination[i * 32 + t * 8 + x] = bc < n && ak < k
                                        ? b[bo + ib * bbs + ak * brs + bc * bcs] : 0.0f;
                                }
                            }
                        rhs_pack_groups += 1;
                    } else {
                        rhs_cache_hits += 1;
                    }
                    SP_PHASE_END(SP_PACK_AB);
#if SP_ACCEL
                    SP_PHASE_BEGIN();
                    fence_memory();
                    if (check_ack(rocc_load_a((uint32_t)(uintptr_t)dma_a, dma_args(SP_A_TILE)))) return;
                    SP_PHASE_END(SP_LOAD_A);
                    SP_PHASE_BEGIN();
                    if (check_ack(rocc_load_b((uint32_t)(uintptr_t)packed_b, dma_args(SP_B_TILE)))) return;
                    SP_PHASE_END(SP_LOAD_B);
                    dma_read_bytes += 2 * sizeof(dma_a);
                    SP_PHASE_BEGIN();
                    for (unsigned t = 0; t < tiles; ++t) {
                        uint32_t rs1 = (SP_A_TILE << 16) | (SP_B_TILE + t)
                                       | (ik ? (1U << 15) : 0);
                        uint32_t rs2 = ((SP_D_TILE + t) << 16) | (SP_D_TILE + t);
                        if (check_ack(rocc_compute(rs1, rs2))) return;
                    }
#else
                    SP_PHASE_BEGIN();
                    /* A fresh K8 dot starts from +0. The old tile sum is
                     * added once at the end, exactly as gemm.cadl does. */
                    for (unsigned t = 0; t < tiles; ++t)
                        for (unsigned i = 0; i < 8; ++i)
                            for (unsigned j = 0; j < 8; j += 2) {
                                float dot0 = 0.0f, dot1 = 0.0f;
#pragma GCC unroll 8
                                for (unsigned x = 0; x < 8; ++x) {
                                    float av = dma_a[i * 32 + x];
                                    dot0 += av * packed_b[j * 32 + t * 8 + x];
                                    dot1 += av * packed_b[(j + 1) * 32 + t * 8 + x];
                                }
                                unsigned q = i * 32 + t * 8 + j;
                                /* Preserve the first block's explicit +0
                                 * addition without reading stale tile data. */
                                float old0 = ik ? dma_d[q] : 0.0f;
                                float old1 = ik ? dma_d[q + 1] : 0.0f;
                                dma_d[q] = dot0 + old0;
                                dma_d[q + 1] = dot1 + old1;
                            }
#endif
                    SP_PHASE_END(SP_COMPUTE);
                    tile_count += tiles;
                }
#if SP_ACCEL
                if (k) {
                    SP_PHASE_BEGIN();
                    fence_memory();
                    if (check_ack(rocc_store_c((uint32_t)(uintptr_t)dma_d, dma_args(SP_D_TILE)))) return;
                    dma_write_bytes += sizeof(dma_d);
                    fence_memory();
                    SP_PHASE_END(SP_STORE_C);
                }
#endif
                SP_PHASE_BEGIN();
                for (unsigned i = 0; i < 8; ++i)
                    for (unsigned t = 0; t < tiles; ++t)
                        for (unsigned j = 0; j < 8; ++j) {
                            intptr_t row = im + i, col = jgroup + t * 8 + j;
                            if (row < m && col < n) {
                                unsigned q = i * 32 + t * 8 + j;
                                /* K=0 still performs +0 + original C, including
                                 * its signed-zero behavior, without loading D. */
                                c[co + ib * cbs + row * crs + col * ccs] =
                                    (k ? dma_d[q] : 0.0f) + original_c[q];
                            }
                        }
                SP_PHASE_END(SP_UNPACK_C);
            }
    if (use_rhs_cache && batch && m && n && k) {
        cache->source = b;
        cache->packed_data = cache->data;
        cache->offset = bo;
        cache->batch = batch;
        cache->n = n;
        cache->k = k;
        cache->batch_stride = bbs;
        cache->row_stride = brs;
        cache->col_stride = bcs;
        cache->valid = 1;
    }
    if (profiling) {
#if SP_PHASE_PROFILE
        for (unsigned phase = 0; phase < SP_PHASE_COUNT; ++phase) {
            profile[id].phase_cycles[phase] += phase_cycles[phase];
            profile[id].phase_counts[phase] += phase_counts[phase];
        }
#endif
        profile[id].cycles += sp_cycles() - start;
        profile[id].calls += 1;
        profile[id].tiles += tile_count;
        profile[id].dma_read_bytes += dma_read_bytes;
        profile[id].dma_write_bytes += dma_write_bytes;
        profile[id].rhs_cache_hits += rhs_cache_hits;
        profile[id].rhs_pack_groups += rhs_pack_groups;
    }
}

void sp_gemm(int id, intptr_t batch, intptr_t m, intptr_t n, intptr_t k,
             const float *a, intptr_t ao, intptr_t abs, intptr_t ars, intptr_t acs,
             const float *b, intptr_t bo, intptr_t bbs, intptr_t brs, intptr_t bcs,
             float *c, intptr_t co, intptr_t cbs, intptr_t crs, intptr_t ccs) {
    sp_gemm_impl(NULL, 0, id, batch, m, n, k, a, ao, abs, ars, acs,
                 b, bo, bbs, brs, bcs, c, co, cbs, crs, ccs);
}

void sp_gemm_cached_rhs(SpRhsCache *cache,
             int id, intptr_t batch, intptr_t m, intptr_t n, intptr_t k,
             const float *a, intptr_t ao, intptr_t abs, intptr_t ars, intptr_t acs,
             const float *b, intptr_t bo, intptr_t bbs, intptr_t brs, intptr_t bcs,
             float *c, intptr_t co, intptr_t cbs, intptr_t crs, intptr_t ccs) {
    sp_gemm_impl(cache, 1, id, batch, m, n, k, a, ao, abs, ars, acs,
                 b, bo, bbs, brs, bcs, c, co, cbs, crs, ccs);
}
