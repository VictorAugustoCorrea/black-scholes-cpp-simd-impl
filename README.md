# Black-Scholes Option Pricing Benchmark (C++)

A C++20 implementation of the Black-Scholes option pricing model, comparing a straightforward scalar version against a block-processed, SIMD-friendly version. Both implementations are benchmarked with `std::chrono` (wall-clock time) and `RDTSC`/`RDTSCP` (CPU cycles).

This project demonstrates how to price European call and put options in C++, and how to correctly measure and compare the performance of two implementations of the same algorithm.

---

## Features

- European call and put option pricing
- Two implementations: a simple scalar version and a block-processed, vectorization-friendly version
- Automatic correctness check between both implementations on every run
- Dual timing methodology: `std::chrono` (ms, options/sec) and `RDTSC`/`RDTSCP` (cycles per option)
- Optional parallelization of the block loop with OpenMP
- No external dependencies — standard library and x86 intrinsics only

---

## Black-Scholes Formula

### Call Option

```
C = S * N(d1) - K * e^(-rT) * N(d2)
```

### Put Option

```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

Where:

```
d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

`N(x)` is the standard normal cumulative distribution function, computed via `std::erf`:

```
N(x) = 0.5 * (1 + erf(x / sqrt(2)))
```

The optimized version derives `put` from `call` via put-call parity (`put = call - S + K * e^(-rT)`), avoiding a second pair of CDF evaluations.

---

## Parameters

| Parameter | In code | Description              |
|-----------|---------|--------------------------|
| `S`       | `s0[i]` | Current stock price      |
| `K`       | `x[i]`  | Strike price             |
| `T`       | `t[i]`  | Time to maturity (years) |
| `r`       | `r`     | Risk-free interest rate  |
| `sigma`   | `sig`   | Volatility               |

---

## Build & Run

```bash
# with OpenMP (parallel block processing)
g++ -O3 -march=native -fopenmp -std=c++20 main.cpp -o Black_Scholes

# without OpenMP (pragmas are ignored, runs serially)
g++ -O3 -march=native -std=c++20 main.cpp -o Black_Scholes

# usage: ./Black_Scholes [nopt] [reps] [warmup]
./Black_Scholes 4000000 10 3
```

- `nopt` — number of options to price
- `reps` — number of measured runs
- `warmup` — number of warmup runs (discarded)

### Example output

```
$ ./Black_Scholes 2000000 8 2
Black-Scholes benchmark
  nopt   = 2000000
  reps   = 8  (warmup = 2)
  r      = 0.02
  sig    = 0.3
  rdtsc  = available
------------------------------------------------------------------------------
Max diff call: 1.907349e-05   put: 2.670288e-05
------------------------------------------------------------------------------
simpleBlackScholesFormula    | avg:   152.272 ms | best:   144.911 ms |     13.80 Mopt/s |    159 cycles/opt
optimizedBlackScholesFormula | avg:    85.571 ms | best:    84.430 ms |     23.69 Mopt/s |     89 cycles/opt
```

---

## Measurement Methodology

Each function runs a few discarded warmup iterations, followed by the measured repetitions. For every repetition:

- Wall-clock time is captured with `std::chrono::high_resolution_clock`
- CPU cycles are captured with `__rdtscp`, bounded by `_mm_lfence` to prevent out-of-order reordering across the measured region

The reported figures are the average and best time across all measured repetitions, plus average cycles per option.

---

## Limitations

The model assumes:

- Constant volatility
- Constant interest rates
- Lognormal returns
- European-style exercise, no dividends

Because of these assumptions, Black-Scholes may not accurately capture real market behavior such as volatility smiles or extreme market events.

Additionally, this implementation uses `float` (32-bit) instead of `double` to favor vectorization, which introduces differences on the order of `1e-5` between the two implementations, as shown in the correctness check above. This code is a performance benchmark, not a production-grade pricing library.

---

## Technologies

- c++20
- `<chrono>`
- `<x86intrin.h>` (RDTSCP)
- OpenMP (optional)
- `<random>`

---

## Educational Purpose

This repository was created for educational purposes in quantitative finance and low-level C++ performance measurement — comparing a scalar implementation against a block-processed one, and showing how to use `chrono` and `RDTSC` together to evaluate real CPU cost, not just wall-clock time.

Credits: Intel® oneAPI Math Kernel Library Cookbook
