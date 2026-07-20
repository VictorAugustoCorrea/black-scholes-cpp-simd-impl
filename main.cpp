#include <cmath>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <x86intrin.h>
    #define HAVE_RDTSC 1
#else
    #define HAVE_RDTSC 0
#endif

//--------------------------------------------------------------
// Constants
//--------------------------------------------------------------
#define HALF         0.5f
#define QUARTER      0.25f
#define TWO          2.0f
#define N_BUF        8
#define ALIGN_FACTOR 64
#define MY_MIN(a, b) ((a) < (b)) ? (a) : (b)

static float LOG(const float x)  { return std::log(x);  }
static float SQRT(const float x) { return std::sqrt(x); }
static float EXP(const float x)  { return std::exp(x);  }

static void V_DIV(const int n, const float* a, const float* b, float* out) {
    for (int j = 0; j < n; ++j) out[j] = a[j] / b[j];
}

static void V_LOG(const int n, const float* a, float* out) {
    for (int j = 0; j < n; ++j) out[j] = std::log(a[j]);
}

static void V_ERF(const int n, const float* a, float* out) {
    for (int j = 0; j < n; ++j) out[j] = std::erf(a[j]);
}

static void V_EXP(const int n, const float* a, float* out) {
    for (int j =0; j < n; ++j) out[j] = std::exp(a[j]);
}

static float CDF_NORM(const float x) {
    return HALF * (1.0f + std::erf(x / SQRT(2.0f)));
}

//--------------------------------------------------------------
// first version: simple implementation
//--------------------------------------------------------------
static auto simpleBlackScholesFormula(
    const int nopt,
    const float r,
    const float sig,
    const float s0[],
    const float x[],
    const float t[],
    float call[],
    float put[]) {

    for (int i = 0; i < nopt; ++i) {
        const float sig_sqrt_t = sig * SQRT(t[i]);
        const float d1 = (LOG(s0[i] / x[i]) + (r + HALF * sig * sig) * t[i]) / sig_sqrt_t;
        const float d2 = d1 - sig_sqrt_t;
        const float disc = EXP(-r * t[i]);

        call[i] = s0[i] * CDF_NORM(d1) - disc * x[i] * CDF_NORM(d2);
        put[i] = disc * x[i] * CDF_NORM(-d2) - s0[i] * CDF_NORM(-d1);
    }
}

//--------------------------------------------------------------
// second version: optimized version
// based on: Intel® oneAPI Math Kernel Library Cookbook
//--------------------------------------------------------------
static auto optimizedBlackScholesFormula(
    const int nopt,
    const float r,
    const float sig,
    const float *s0,
    const float *x,
    const float *t,
    float *call,
    float *put) {

    const float mr = -r;
    const float r_plus_half_sig2 = r + HALF * sig * sig;
    const float inv_sqrt2 = 1.0f / SQRT(2.0f);

    #pragma opm parallel for schedule(static) default(none)\
        shared(s0, x, t, call, put, nopt);\
        firstprivate(mr, sig, r_plus_half_sig2, inv_sqrt2)

    for (int i = 0; i < nopt; i += N_BUF) {
        const int n_buf = MY_MIN(N_BUF, nopt - i);

        alignas(ALIGN_FACTOR) float logRatio[N_BUF];
        alignas(ALIGN_FACTOR) float sigSqrtT[N_BUF];
        alignas(ALIGN_FACTOR) float d1[N_BUF];
        alignas(ALIGN_FACTOR) float d2[N_BUF];
        alignas(ALIGN_FACTOR) float w1[N_BUF];
        alignas(ALIGN_FACTOR) float w2[N_BUF];
        alignas(ALIGN_FACTOR) float cdf1[N_BUF];
        alignas(ALIGN_FACTOR) float cdf2[N_BUF];
        alignas(ALIGN_FACTOR) float disc[N_BUF];

        V_DIV(n_buf, s0 + i, x + i, logRatio);
        V_LOG(n_buf, logRatio, logRatio);

        #pragma simd
        for (int j = 0; j < n_buf; ++j) {
            sigSqrtT[j] = sig * SQRT(t[i + j]);
            d1[j] = (logRatio[j] + r_plus_half_sig2 * t[i + j]) / sigSqrtT[j];
            d2[j] = d1[j] - sigSqrtT[j];
            w1[j] = d1[j] * inv_sqrt2;
            w2[j] = d2[j] * inv_sqrt2;
            disc[j] = mr * t[i + j];
        }

        V_ERF(n_buf, w1, cdf1);
        V_ERF(n_buf, w2, cdf2);
        V_EXP(n_buf, disc, disc);

        #pragma simd
        for (int j = 0; j < n_buf; ++j) {
            cdf1[j] = HALF + HALF * cdf1[j];
            cdf2[j] = HALF + HALF * cdf2[j];
            call[i + j] = s0[i + j] * cdf1[j] - x[i + j] * disc[j] * cdf2[j];
            put[i + j] = call[i + j] - s0[i + j] + x[i + j] * disc[j];
        }
    }
}

static uint64_t rdtsc_start() {
#if HAVE_RDTSC
    unsigned aux;
    _mm_lfence();
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    return 0;
#endif
}

static uint64_t rdtsc_end() {
#if HAVE_RDTSC
    unsigned aux;
    _mm_lfence();
    const uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    return 0;
#endif
}

namespace {
    struct BenchResult {
        double avg_ms = 0.0;
        double best_ms = 0.0;
        double m_opts_per_sec = 0.0;
        uint64_t avg_cycles_per_opt = 0;
    };
}


template <typename Fn>
static BenchResult benchmark(const std::string& name, const int nopt, const int reps, const int warmup, Fn&& fn) {
    for (int w = 0; w < warmup; ++w) fn();

    double total_ms = 0.0;
    double best_ms = std::numeric_limits<double>::max();
    uint64_t total_cycles = 0;

    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const uint64_t c0 = rdtsc_start();

        fn();

        const uint64_t c1 = rdtsc_end();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
        total_cycles += c1 - c0;
    }

    BenchResult res;
    res.avg_ms = total_ms / reps;
    res.best_ms = best_ms;
    res.m_opts_per_sec = nopt / 1e6 / (res.best_ms / 1000.0);
    res.avg_cycles_per_opt = total_cycles / reps / static_cast<uint64_t>(nopt);

    std::cout << std::left << std::setw(28) << name
              << " | avg: " << std::right << std::setw(9) << std::fixed << std::setprecision(3) << res.avg_ms << " ms"
              << " | best: " << std::setw(9) << std::setprecision(2) << res.m_opts_per_sec << " Mopt/s";
#if HAVE_RDTSC
    std::cout << " | " << std::setw(6) << res.avg_cycles_per_opt << " cycles/opt";
#endif
    std::cout << "\n";

    return res;
}


int main(const int argc, char** argv) {
    int nopt      = 4'000'000; /* number of options */
    int reps      = 10;
    int warmup    = 3;
    char *end_ptr = nullptr;

    if (argc > 1) nopt   = static_cast<int>(std::max(1L, std::strtol(argv[1], &end_ptr, 10)));
    if (argc > 2) reps   = static_cast<int>(std::max(1L, std::strtol(argv[2], &end_ptr, 10)));
    if (argc > 3) warmup = static_cast<int>(std::max(1L, std::strtol(argv[3], &end_ptr, 10)));

    constexpr float r   = 0.02f; //risk free rate
    constexpr float sig = 0.30f; //vol

    std::vector<float> s0(nopt), x(nopt), t(nopt);
    std::vector<float> call_simple(nopt), put_simple(nopt);
    std::vector<float> call_opt(nopt), put_opt(nopt);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist_price(50.0f, 150.0f);
    std::uniform_real_distribution<float> dist_time(0.10f, 2.0f);

    for (int i = 0; i < nopt; ++i) {
        s0[i] = dist_price(rng);
        x[i] = dist_price(rng);
        t[i] = dist_time(rng);
    }

    std::cout << "Black-Scholes benchmark \n";
    std::cout << "  nopt   = " << nopt << "\n";
    std::cout << "  reps   = " << reps << "  (warmup = " << warmup << ")\n";
    std::cout << "  r      = " << r << "\n";
    std::cout << "  sig    = " << sig << "\n";

#if HAVE_RDTSC
    std::cout << "  rdtsc =  available \n";
#else
    std::cout << " rdtsc = unavailable on this architecture (cycles column will be 0)\n";
#endif
    std::cout << std::string(78, '-') << "\n";

    // --- check correctness: both results must match ---
    simpleBlackScholesFormula(
        nopt,
        r,
        sig,
        s0.data(),
        x.data(),
        t.data(),
        call_simple.data(),
        put_simple.data()
    );

    optimizedBlackScholesFormula(
        nopt,
        r,
        sig,
        s0.data(),
        x.data(),
        t.data(),
        call_opt.data(),
        put_opt.data()
    );

    double max_call_diff = 0.0, max_put_diff = 0.0;

    for (int i = 0; i < nopt; ++i) {
        max_call_diff = std::max<double>(max_call_diff, std::fabs(call_simple[i] - call_opt[i]));
        max_put_diff  = std::max<double>(max_put_diff, std::fabs(put_simple[i] - put_opt[i]));
    }

    std::cout << "Max diff call: " << std::scientific << max_call_diff
              << "   put: " << max_put_diff << std::fixed << "\n";
    std::cout << std::string(78, '-') << "\n";

    // --- benchmarks ---
    benchmark(
        "SimpleBlackScholesFormula",
        nopt,
        reps,
        warmup,
        [&] {
            simpleBlackScholesFormula(
                nopt,
                r, sig,
                s0.data(),
                x.data(),
                t.data(),
                call_simple.data(),
                put_simple.data()
                );
            }
    );

    benchmark(
        "optimizedBlackScholesFormula",
        nopt,
        reps,
        warmup,
        [&] {
            optimizedBlackScholesFormula(
                nopt,
                r, sig,
                s0.data(),
                x.data(),
                t.data(),
                call_opt.data(),
                put_opt.data()
                );
            }
    );

    return 0;
}