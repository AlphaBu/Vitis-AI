# Vitis HLS Kernel Optimization (Detailed)

Worked techniques and examples for hitting throughput/latency targets. Companion to [SKILL.md](SKILL.md).

Optimization order (do not skip steps):
1. Pass **C-simulation** (functional correctness) first.
2. **Pipeline** inner loops to II=1.
3. **Array-partition** the on-chip arrays that block II=1.
4. **Dataflow** across load/compute/store to overlap memory and compute.
5. **Tune bursts / bundles** for global-memory bandwidth.
6. Add **unroll / PERFORMANCE** for extra parallelism; re-check area/timing.
7. Confirm with **co-simulation** and the synthesis report.

---

## 1. Loop Pipeline

Overlaps iterations so a new one starts every II cycles.

```cpp
mul: for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
    y[i] = a[i] * b[i];
}
```

- Apply to the **innermost** loop to maximize throughput. Loops nested *inside* a pipelined loop are automatically unrolled.
- **Function-level pipeline**: `#pragma HLS PIPELINE` as the first statement of a function pipelines the whole function and unrolls all its loops. Use for small, fully-parallel functions.
- **Flushing/rewind**: `#pragma HLS PIPELINE II=1 rewind` allows a pipelined loop to restart with no bubble when called repeatedly.

### Diagnosing II > 1

Read the synthesis schedule report; common causes and fixes:

| Cause | Fix |
|---|---|
| Array needs >2 accesses/cycle | `ARRAY_PARTITION` the array (§4) |
| Loop-carried dependency (e.g. `acc += ...`) | Restructure (partial sums), or `#pragma HLS DEPENDENCE` if false dependency |
| Only one instance of an operator | `#pragma HLS ALLOCATION`/`BIND_OP` to add instances |
| Long single op (float div, sqrt) | Pipeline the op / raise `latency`, or use `hls_math` pipelined variants |
| `m_axi` read/write in loop body | Move memory access out via dataflow load/compute/store |

### Breaking a loop-carried dependency (reduction)

```cpp
// BAD: acc dependency forces II = latency of the adder
float acc = 0;
for (int i = 0; i < N; i++) { acc += x[i]; }

// GOOD: partial sums remove the tight recurrence → II=1
const int P = 8;
float part[P] = {0};
#pragma HLS ARRAY_PARTITION variable=part complete
for (int i = 0; i < N; i += P) {
#pragma HLS PIPELINE II=1
    for (int p = 0; p < P; p++) part[p] += x[i + p];
}
float acc = 0;
for (int p = 0; p < P; p++) acc += part[p];
```

---

## 2. Unroll

Creates parallel hardware copies of the loop body (spatial parallelism, costs area).

```cpp
for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL           // full unroll
    y[i] = f(x[i]);
}

for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor=4  // partial: 4 copies, i.e. process 4/iter
    ...
}
```

- Full unroll requires a compile-time bound.
- Partial unroll pairs with pipelining: unroll the inner loop `factor=N`, pipeline the outer → N results/cycle (needs matching array partitioning to feed it).
- Unrolling multiplies resource usage — watch LUT/DSP/BRAM.

---

## 3. Dataflow (task-level parallelism)

Runs a straight-line sequence of sub-function calls concurrently, connected by FIFOs/ping-pong buffers. The canonical **load → compute → store** structure:

```cpp
static void load(const int *g, hls::stream<int> &s, int n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        s << g[i];              // burst read from DRAM
    }
}
static void compute(hls::stream<int> &si, hls::stream<int> &so, int n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        so << (si.read() * 2);
    }
}
static void store(hls::stream<int> &s, int *g, int n) {
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
        g[i] = s.read();        // burst write to DRAM
    }
}

extern "C" {
void top(const int *in, int *out, int n) {
#pragma HLS INTERFACE m_axi     port=in  bundle=gmem0 offset=slave
#pragma HLS INTERFACE m_axi     port=out bundle=gmem1 offset=slave
#pragma HLS INTERFACE s_axilite port=n      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<int> s0, s1;
#pragma HLS STREAM variable=s0 depth=64
#pragma HLS STREAM variable=s1 depth=64
#pragma HLS DATAFLOW
    load(in, s0, n);
    compute(s0, s1, n);
    store(s1, out, n);
}
}
```

### Dataflow rules (violations serialize silently — check the report's "Dataflow" section)

- **Single producer / single consumer** per channel. A variable written by one task must be read by exactly one task.
- Channels are `hls::stream` or arrays with `#pragma HLS STREAM`; a task must not be bypassed.
- **No conditionals** around task calls; keep a linear call sequence in the region.
- **Size FIFO depth** to cover rate mismatch; too small → deadlock, too large → wasted BRAM.
- Don't read a variable after it is consumed inside the region.

### `stream_of_blocks` for block-level dataflow

Use `hls::stream_of_blocks<T>` when tasks exchange whole tiles (not element-by-element) and you want overlap without a giant FIFO.

---

## 4. Array Partition

On-chip arrays default to dual-port BRAM (2 accesses/cycle). Partitioning creates more ports so pipelined/unrolled loops can hit II=1.

```cpp
int coef[TAPS];
#pragma HLS ARRAY_PARTITION variable=coef type=complete           // all in registers
int buf[R][C];
#pragma HLS ARRAY_PARTITION variable=buf  type=cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=buf  type=block  factor=2 dim=1
```

| type | Effect | Use when |
|---|---|---|
| `complete` | Split into individual registers | Small arrays, coefficients, accumulators, any-index-any-cycle |
| `cyclic factor=N` | Round-robin into N banks (`i%N`) | Feeding an N-wide unrolled loop / strided access |
| `block factor=N` | N contiguous chunks (`i/(size/N)`) | Two halves accessed by two tasks |

- `dim=<n>` selects the dimension; `dim=0` partitions **all** dimensions.
- Match the partition factor to the loop's parallelism: unroll `factor=4` + `ARRAY_PARTITION cyclic factor=4` → 4 conflict-free accesses/cycle.
- **`ARRAY_RESHAPE`** widens the memory word instead of adding banks (fewer, wider ports) — good when you access N consecutive elements together.
- **`BIND_STORAGE`** chooses the memory kind/ports: `#pragma HLS BIND_STORAGE variable=buf type=ram_2p impl=uram`.
- **`AGGREGATE`** packs a struct into a single wide word so it moves in one transfer.

---

## 5. Memory / burst optimization

Global memory bandwidth is usually the bottleneck. See [INTERFACES.md §2](INTERFACES.md) for the full `m_axi` option table. Key techniques:

- **Sequential access → automatic bursts.** Copy a tile into local memory with a pipelined loop, compute locally, write back sequentially.
- **Widen the data path.** Use `ap_int<512>` / `hls::vector` or `AGGREGATE` so each beat moves 512 bits (matches AXI width) instead of 32.
- **Separate bundles** for inputs/outputs accessed concurrently.
- **Tune the adapter**: `max_read_burst_length=256`, `num_read_outstanding=16..32`, `latency=64` for HBM/DDR.
- **Overlap** memory with compute via dataflow (§3).

### Tiled matmul sketch (local buffers + partition + pipeline)

```cpp
void mmult_tile(const dt *A, const dt *B, acc_t *C, int M, int N, int K) {
    dt Ar[TM][TK], Br[TK][TN];
    acc_t Cr[TM][TN];
#pragma HLS ARRAY_PARTITION variable=Ar cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=Br cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=Cr complete dim=0
    // 1) burst-load tiles of A and B into Ar/Br
    // 2) compute:
    row: for (int i = 0; i < TM; i++)
      col: for (int j = 0; j < TN; j++) {
#pragma HLS PIPELINE II=1
          acc_t s = 0;
          k: for (int k = 0; k < TK; k++)
#pragma HLS UNROLL
              s += Ar[i][k] * Br[k][j];
          Cr[i][j] = s;
      }
    // 3) burst-store Cr to C
}
```

---

## 6. Performance pragma & latency control

- **`#pragma HLS PERFORMANCE target_ti=<n>`** — declare a target initiation interval for a loop/region and let HLS choose unroll/pipeline to meet it. Verify achieved II in the report.
- **`#pragma HLS LATENCY min=<a> max=<b>`** — bound loop/function latency (tool inserts/removes registers to comply).
- **`#pragma HLS LOOP_FLATTEN`** — merge perfect nested loops into one, removing inter-loop transition cycles.
- **`#pragma HLS LOOP_TRIPCOUNT min= max= avg=`** — *reporting only* estimate for variable-bound loops (does not change hardware).
- **`#pragma HLS INLINE` / `INLINE off`** — flatten a sub-function into its caller (removes call overhead, enables cross-function optimization) or keep it a separate module.

---

## 7. Reading the reports

- **C Synthesis report**: check per-loop **II** and **Latency**, **Interval**, and the **Utilization** (LUT/FF/DSP/BRAM/URAM). II=? and "achieved/target" tell you if pipelining met the goal.
- **Schedule Viewer / Dataflow Viewer**: shows why an operation is scheduled late (dependency, resource) and whether dataflow tasks overlap.
- **AXI/Burst analysis**: confirms inferred burst lengths and outstanding transactions.
- **Co-simulation**: proves the RTL matches the C model and reports measured latency/II on real vectors — always run before hardware.

---

## 8. Optimization checklist

1. C-sim passes (functional reference matches).
2. Innermost compute loop has `PIPELINE II=1`; report confirms II=1.
3. Arrays feeding pipelined loops are partitioned to remove port conflicts.
4. No loop-carried dependency limiting II (reductions use partial sums).
5. Top uses load→compute→store with `DATAFLOW` to overlap memory & compute.
6. `m_axi` access is sequential (bursts inferred); wide data path where possible.
7. Concurrently-accessed ports are on separate `bundle`s.
8. Burst/outstanding/latency options tuned for the target memory.
9. Resource usage fits the SLR/device (check Utilization).
10. Co-simulation passes and measured II/latency meet the target.
