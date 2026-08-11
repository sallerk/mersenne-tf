// ===========================================================================
//  mersenne_tf0.9  --  GPU trial factoring of Mersenne numbers  M_p = 2^p - 1
// ===========================================================================
//
//  Finds every PRIME factor of M_p inside a user-chosen range of candidate
//  values (e.g. 1 .. 2^90), using the GPU via OpenCL.
//
//  EXACTNESS
//  ---------
//  There is no floating point anywhere in the number theory, on the host or on
//  the device.  M_p itself is never materialised -- it has p bits and p can be
//  in the hundreds of millions.  Instead every candidate q is a full 128-bit
//  integer and the test
//                          2^p mod q == 1
//  is evaluated with exact 128-bit Montgomery modular arithmetic (two 64-bit
//  limbs, products formed with mul_hi/__umulh so not one bit is discarded).
//  This is the same "arbitrary precision" guarantee Python's pow(2,p,q) gives,
//  restricted to a 128-bit modulus, which is what the hardware can do fast.
//
//  THE MATH  (p prime, q a prime factor of 2^p - 1)
//  ------------------------------------------------
//   1.  ord_q(2) divides p and is not 1, so ord_q(2) = p, and p | q-1.
//       q is odd, so 2p | q-1:            q = 2kp + 1
//   2.  p odd  =>  2 = (2^((p+1)/2))^2 mod q, so 2 is a quadratic residue,
//       so by the second supplement to quadratic reciprocity:
//                                         q = +/-1 (mod 8)
//   3.  Combining, with q = 2kp+1:
//           p = 1 (mod 4)  =>  k = 0 or 3 (mod 4)
//           p = 3 (mod 4)  =>  k = 0 or 1 (mod 4)
//       Half of all k are eliminated before anything else happens.
//
//  PRE-FACTORING  (candidates are filtered before they cost GPU time)
//  -----------------------------------------------------------------
//  A multithreaded segmented sieve removes every k whose q = 2kp+1 is divisible
//  by a small prime s:   2kp + 1 = 0 (mod s)  <=>  k = -(2p)^-1 (mod s).
//  Such a q is composite, and any prime factor of M_p hiding inside it is a
//  separate candidate that is tested on its own.  With the default sieve bound
//  only about 9% of k survive, so the GPU spends its time almost exclusively on
//  candidates that are actually prime.  The (few thousand) sieve primes that
//  themselves fall inside the requested range are tested directly on the CPU,
//  so no prime factor can be lost to the sieve.
//
//  Build:  build.bat        (needs only Visual Studio; no CUDA/OpenCL SDK)
//  Run:    mersenne_tf0.9.exe [--config config.txt] [--selftest] [--list-devices]
// ===========================================================================

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <io.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>

#include "tf_kernel.cl.h"

// ===========================================================================
//  Section 1 -- Minimal OpenCL binding, loaded from the driver at runtime.
//  Declaring the handful of entry points we use keeps the program free of any
//  SDK dependency: OpenCL.dll ships with the GPU driver.
// ===========================================================================

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef cl_uint   cl_bool;
typedef cl_ulong  cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef intptr_t  cl_context_properties;
typedef cl_uint   cl_platform_info;
typedef cl_uint   cl_device_info;
typedef cl_uint   cl_program_build_info;
typedef cl_uint   cl_kernel_work_group_info;

typedef struct _cl_platform_id*   cl_platform_id;
typedef struct _cl_device_id*     cl_device_id;
typedef struct _cl_context*       cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem*           cl_mem;
typedef struct _cl_program*       cl_program;
typedef struct _cl_kernel*        cl_kernel;
typedef struct _cl_event*         cl_event;

#define CL_SUCCESS                        0
#define CL_TRUE                           1
#define CL_FALSE                          0
#define CL_DEVICE_TYPE_GPU                (1 << 2)
#define CL_DEVICE_TYPE_ALL                0xFFFFFFFF
#define CL_PLATFORM_NAME                  0x0902
#define CL_DEVICE_MAX_COMPUTE_UNITS       0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE     0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY     0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE         0x101F
#define CL_DEVICE_NAME                    0x102B
#define CL_DEVICE_VERSION                 0x102F
#define CL_PROGRAM_BUILD_LOG              0x1183
#define CL_KERNEL_WORK_GROUP_SIZE         0x11B0
#define CL_MEM_READ_WRITE                 (1 << 0)
#define CL_MEM_WRITE_ONLY                 (1 << 1)
#define CL_MEM_READ_ONLY                  (1 << 2)
#define CL_MEM_ALLOC_HOST_PTR             (1 << 4)
#define CL_MAP_WRITE                      (1 << 1)

#define CLAPI __stdcall

typedef cl_int (CLAPI *P_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CLAPI *P_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (CLAPI *P_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CLAPI *P_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (CLAPI *P_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                              void (CLAPI*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (CLAPI *P_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_command_queue (CLAPI *P_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_ulong*, cl_int*);
typedef cl_program (CLAPI *P_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CLAPI *P_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                                         void (CLAPI*)(cl_program, void*), void*);
typedef cl_int (CLAPI *P_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (CLAPI *P_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (CLAPI *P_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*);
typedef cl_int (CLAPI *P_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_mem (CLAPI *P_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (CLAPI *P_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*,
                                               cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *P_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                                              cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *P_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                                 const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *P_clFinish)(cl_command_queue);
typedef cl_int (CLAPI *P_clFlush)(cl_command_queue);
typedef void*  (CLAPI *P_clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_ulong,
                                             size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*);
typedef cl_int (CLAPI *P_clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*,
                                                  cl_uint, const cl_event*, cl_event*);
typedef cl_int (CLAPI *P_clWaitForEvents)(cl_uint, const cl_event*);
typedef cl_int (CLAPI *P_clReleaseEvent)(cl_event);
typedef cl_int (CLAPI *P_clReleaseMemObject)(cl_mem);
typedef cl_int (CLAPI *P_clReleaseKernel)(cl_kernel);
typedef cl_int (CLAPI *P_clReleaseProgram)(cl_program);
typedef cl_int (CLAPI *P_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CLAPI *P_clReleaseContext)(cl_context);

static struct {
    HMODULE dll;
    P_clGetPlatformIDs        GetPlatformIDs;
    P_clGetPlatformInfo       GetPlatformInfo;
    P_clGetDeviceIDs          GetDeviceIDs;
    P_clGetDeviceInfo         GetDeviceInfo;
    P_clCreateContext         CreateContext;
    P_clCreateCommandQueue    CreateCommandQueue;
    P_clCreateCommandQueueWithProperties CreateCommandQueueWithProperties;
    P_clCreateProgramWithSource CreateProgramWithSource;
    P_clBuildProgram          BuildProgram;
    P_clGetProgramBuildInfo   GetProgramBuildInfo;
    P_clCreateKernel          CreateKernel;
    P_clGetKernelWorkGroupInfo GetKernelWorkGroupInfo;
    P_clSetKernelArg          SetKernelArg;
    P_clCreateBuffer          CreateBuffer;
    P_clEnqueueWriteBuffer    EnqueueWriteBuffer;
    P_clEnqueueReadBuffer     EnqueueReadBuffer;
    P_clEnqueueNDRangeKernel  EnqueueNDRangeKernel;
    P_clFinish                Finish;
    P_clFlush                 Flush;
    P_clEnqueueMapBuffer      EnqueueMapBuffer;
    P_clEnqueueUnmapMemObject EnqueueUnmapMemObject;
    P_clWaitForEvents         WaitForEvents;
    P_clReleaseEvent          ReleaseEvent;
    P_clReleaseMemObject      ReleaseMemObject;
    P_clReleaseKernel         ReleaseKernel;
    P_clReleaseProgram        ReleaseProgram;
    P_clReleaseCommandQueue   ReleaseCommandQueue;
    P_clReleaseContext        ReleaseContext;
} CL;

static bool load_opencl(std::string& err)
{
    CL.dll = LoadLibraryA("OpenCL.dll");
    if (!CL.dll) {
        err = "OpenCL.dll not found. Install/update your GPU driver (it ships the OpenCL runtime).";
        return false;
    }
#define GETPROC(name, field, required)                                        \
    CL.field = (P_cl##name)GetProcAddress(CL.dll, "cl" #name);                \
    if (!CL.field && required) { err = "missing entry point cl" #name; return false; }

    GETPROC(GetPlatformIDs, GetPlatformIDs, true)
    GETPROC(GetPlatformInfo, GetPlatformInfo, true)
    GETPROC(GetDeviceIDs, GetDeviceIDs, true)
    GETPROC(GetDeviceInfo, GetDeviceInfo, true)
    GETPROC(CreateContext, CreateContext, true)
    GETPROC(CreateCommandQueue, CreateCommandQueue, false)
    GETPROC(CreateCommandQueueWithProperties, CreateCommandQueueWithProperties, false)
    GETPROC(CreateProgramWithSource, CreateProgramWithSource, true)
    GETPROC(BuildProgram, BuildProgram, true)
    GETPROC(GetProgramBuildInfo, GetProgramBuildInfo, true)
    GETPROC(CreateKernel, CreateKernel, true)
    GETPROC(GetKernelWorkGroupInfo, GetKernelWorkGroupInfo, true)
    GETPROC(SetKernelArg, SetKernelArg, true)
    GETPROC(CreateBuffer, CreateBuffer, true)
    GETPROC(EnqueueWriteBuffer, EnqueueWriteBuffer, true)
    GETPROC(EnqueueReadBuffer, EnqueueReadBuffer, true)
    GETPROC(EnqueueNDRangeKernel, EnqueueNDRangeKernel, true)
    GETPROC(Finish, Finish, true)
    GETPROC(Flush, Flush, true)
    GETPROC(EnqueueMapBuffer, EnqueueMapBuffer, true)
    GETPROC(EnqueueUnmapMemObject, EnqueueUnmapMemObject, true)
    GETPROC(WaitForEvents, WaitForEvents, true)
    GETPROC(ReleaseEvent, ReleaseEvent, true)
    GETPROC(ReleaseMemObject, ReleaseMemObject, true)
    GETPROC(ReleaseKernel, ReleaseKernel, true)
    GETPROC(ReleaseProgram, ReleaseProgram, true)
    GETPROC(ReleaseCommandQueue, ReleaseCommandQueue, true)
    GETPROC(ReleaseContext, ReleaseContext, true)
#undef GETPROC
    if (!CL.CreateCommandQueue && !CL.CreateCommandQueueWithProperties) {
        err = "no command-queue constructor exported by OpenCL.dll";
        return false;
    }
    return true;
}

// ===========================================================================
//  Section 2 -- Exact 128-bit unsigned integer arithmetic (host side)
// ===========================================================================

struct U128 { uint64_t lo, hi; };

static inline U128 u128_make(uint64_t hi, uint64_t lo) { U128 r; r.hi = hi; r.lo = lo; return r; }
static inline U128 u128_from(uint64_t v)               { return u128_make(0, v); }
static inline bool u128_is_zero(U128 a)                { return (a.lo | a.hi) == 0; }
static inline bool u128_eq(U128 a, U128 b)             { return a.lo == b.lo && a.hi == b.hi; }
static inline bool u128_ge(U128 a, U128 b)             { return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo); }
static inline bool u128_gt(U128 a, U128 b)             { return a.hi > b.hi || (a.hi == b.hi && a.lo >  b.lo); }
static inline bool u128_lt(U128 a, U128 b)             { return u128_gt(b, a); }

static inline U128 u128_add(U128 a, U128 b)
{
    U128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (r.lo < a.lo ? 1u : 0u); return r;
}
static inline U128 u128_sub(U128 a, U128 b)
{
    U128 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (a.lo < b.lo ? 1u : 0u); return r;
}
static inline U128 u128_shl1(U128 a)
{
    U128 r; r.hi = (a.hi << 1) | (a.lo >> 63); r.lo = a.lo << 1; return r;
}
static inline U128 u128_shr1(U128 a)
{
    U128 r; r.lo = (a.lo >> 1) | (a.hi << 63); r.hi = a.hi >> 1; return r;
}
static inline U128 u128_shl(U128 a, unsigned n)
{
    if (n == 0) return a;
    if (n >= 128) return u128_make(0, 0);
    if (n >= 64)  return u128_make(a.lo << (n - 64), 0);
    return u128_make((a.hi << n) | (a.lo >> (64 - n)), a.lo << n);
}
static inline int u128_bitlen(U128 a)
{
    if (a.hi) { unsigned long i; _BitScanReverse64(&i, a.hi); return (int)i + 65; }
    if (a.lo) { unsigned long i; _BitScanReverse64(&i, a.lo); return (int)i + 1; }
    return 0;
}
static inline bool u128_bit(U128 a, int i)
{
    return i < 64 ? ((a.lo >> i) & 1u) != 0 : ((a.hi >> (i - 64)) & 1u) != 0;
}
// 128 x 64 -> low 128 bits.  Callers guarantee (via range validation) no wrap.
static inline U128 u128_mul_u64(U128 a, uint64_t b)
{
    U128 r;
    r.lo = a.lo * b;
    r.hi = __umulh(a.lo, b) + a.hi * b;
    return r;
}
// exact 128/128 division (binary long division; used only in setup/printing)
static void u128_divmod(U128 a, U128 d, U128& quot, U128& rem)
{
    quot = u128_make(0, 0);
    rem  = u128_make(0, 0);
    if (u128_is_zero(d)) return;
    for (int i = u128_bitlen(a) - 1; i >= 0; --i) {
        rem = u128_shl1(rem);
        if (u128_bit(a, i)) rem.lo |= 1u;
        if (u128_ge(rem, d)) {
            rem = u128_sub(rem, d);
            if (i < 64) quot.lo |= (1ull << i); else quot.hi |= (1ull << (i - 64));
        }
    }
}
static inline uint32_t u128_mod_u32(U128 a, uint32_t s)
{
    // 2^64 mod s, then Horner over the two limbs.  All intermediates < 2^64.
    uint64_t two64 = (uint64_t)((~0ull % s) + 1) % s;   // (2^64-1 mod s) + 1
    uint64_t t = ((a.hi % s) * two64) % s;
    return (uint32_t)((t + (a.lo % s)) % s);
}
static std::string u128_to_dec(U128 a)
{
    if (u128_is_zero(a)) return "0";
    std::string s;
    U128 ten = u128_from(10), q, r;
    while (!u128_is_zero(a)) {
        u128_divmod(a, ten, q, r);
        s.push_back((char)('0' + (char)r.lo));
        a = q;
    }
    std::reverse(s.begin(), s.end());
    return s;
}
// The search bounds are nearly always powers of two, and a 21-digit decimal is
// hard to read at a glance -- "1180591620717411303424" is 2^70.  Exact powers
// print as "2^70"; a value that is not one prints as the nearest power with two
// decimals, truncated (never rounded up) and marked "~" so it cannot be taken
// for exact.  Short values stay decimal, where they are already clearer.
static inline double u128_to_dbl(U128 v)
{
    return (double)v.hi * 18446744073709551616.0 + (double)v.lo;
}

static std::string u128_to_pow2(U128 v)
{
    if (v.hi == 0 && v.lo == 0) return "0";
    const int bits = u128_bitlen(v);                 // 1-based: bitlen(1) == 1
    U128 vm1 = u128_sub(v, u128_from(1));
    const bool exact = ((v.hi & vm1.hi) == 0 && (v.lo & vm1.lo) == 0);
    if (exact && bits > 10) return "2^" + std::to_string(bits - 1);
    if (v.hi == 0 && v.lo < (1ull << 32)) return u128_to_dec(v);   // still short enough to read
    // The double rounds 2^127-1 up to 2^127, which would print an exponent the
    // value does not reach.  Its bit length is exact and pins the answer to
    // [bits-1, bits), so truncate into that window and the "~" only ever
    // understates.
    double l = floor(log2(u128_to_dbl(v)) * 100.0) / 100.0;
    if (l < (double)(bits - 1))   l = (double)(bits - 1);
    if (l > (double)bits - 0.01)  l = (double)bits - 0.01;
    char buf[64];
    snprintf(buf, sizeof(buf), "~2^%.2f", l);
    return std::string(buf);
}

// Accepts "12345", "2^90", "2^90-1", "1_000_000", "0x..." is not supported.
static bool parse_u128(std::string t, U128& out, std::string& err)
{
    t.erase(std::remove(t.begin(), t.end(), '_'), t.end());
    t.erase(std::remove(t.begin(), t.end(), ','), t.end());
    if (t.empty()) { err = "empty value"; return false; }

    // "2^n", "2^n-m", "2^n+m", and "2^n+2^m" / "2^n-2^m"
    size_t caret = t.find('^');
    if (caret != std::string::npos) {
        std::string base = t.substr(0, caret), rest = t.substr(caret + 1);
        std::string tail;
        char op = 0;
        size_t m = rest.find_first_of("+-");
        if (m != std::string::npos) { op = rest[m]; tail = rest.substr(m + 1); rest = rest.substr(0, m); }
        if (base != "2") { err = "only base 2 is supported in the ^ form (got '" + base + "')"; return false; }
        int e = atoi(rest.c_str());
        if (e < 0 || e > 127) { err = "exponent must be 0..127 in the 2^n form"; return false; }
        U128 v = u128_shl(u128_from(1), (unsigned)e);
        if (op) {
            U128 d;
            if (!parse_u128(tail, d, err)) return false;      // tail may itself be 2^m
            if (op == '-') {
                if (u128_lt(v, d)) { err = "value underflow"; return false; }
                v = u128_sub(v, d);
            } else {
                U128 nv = u128_add(v, d);
                if (u128_lt(nv, v)) { err = "value exceeds 2^128-1"; return false; }
                v = nv;
            }
        }
        out = v;
        return true;
    }
    U128 v = u128_make(0, 0);
    const U128 TEN = u128_from(10);
    for (char c : t) {
        if (!isdigit((unsigned char)c)) { err = "not a number: '" + t + "'"; return false; }
        // v = v*10 + d, with overflow detection
        U128 hi_check, dummy;
        const U128 LIMIT = u128_make(~0ull, ~0ull);
        u128_divmod(LIMIT, TEN, hi_check, dummy);
        if (u128_gt(v, hi_check)) { err = "value exceeds 2^128-1"; return false; }
        v = u128_mul_u64(v, 10);
        U128 nv = u128_add(v, u128_from((uint64_t)(c - '0')));
        if (u128_lt(nv, v)) { err = "value exceeds 2^128-1"; return false; }
        v = nv;
    }
    out = v;
    return true;
}

// ===========================================================================
//  Section 3 -- Host Montgomery arithmetic (mirror of the kernel) + primality
//  Used to independently re-verify every factor the GPU reports.
// ===========================================================================

static inline uint64_t h_addc(uint64_t a, uint64_t b, uint64_t* c)
{
    uint64_t s = a + b; *c = (s < a) ? 1ull : 0ull; return s;
}
static inline uint64_t h_mac(uint64_t t, uint64_t a, uint64_t b, uint64_t* C)
{
    uint64_t hi = __umulh(a, b), lo = a * b, c1, c2;
    lo = h_addc(lo, t,  &c1);
    lo = h_addc(lo, *C, &c2);
    *C = hi + c1 + c2;
    return lo;
}
static U128 h_mont_mul(U128 a, U128 b, U128 m, uint64_t mp)
{
    uint64_t A[2] = { a.lo, a.hi }, B[2] = { b.lo, b.hi }, M[2] = { m.lo, m.hi };
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (int i = 0; i < 2; ++i) {
        uint64_t C = 0, cc, mu;
        t0 = h_mac(t0, A[0], B[i], &C);
        t1 = h_mac(t1, A[1], B[i], &C);
        t2 = h_addc(t2, C, &cc);
        t3 = cc;
        mu = t0 * mp;
        C  = 0;
        (void)h_mac(t0, mu, M[0], &C);
        t0 = h_mac(t1, mu, M[1], &C);
        t1 = h_addc(t2, C, &cc);
        t2 = t3 + cc;
    }
    U128 r = u128_make(t1, t0);
    if (t2 || u128_ge(r, m)) r = u128_sub(r, m);
    return r;
}
static inline U128 h_mod_dbl(U128 x, U128 m)
{
    x = u128_shl1(x);
    if (u128_ge(x, m)) x = u128_sub(x, m);
    return x;
}
static inline U128 h_mod_add(U128 a, U128 b, U128 m)
{
    U128 s = u128_add(a, b);
    if (u128_ge(s, m)) s = u128_sub(s, m);
    return s;
}
static inline uint64_t h_neg_inv64(uint64_t m0)
{
    uint64_t x = m0;
    for (int i = 0; i < 6; ++i) x = x * (2ull - m0 * x);
    return 0ull - x;
}
static U128 h_r_mod(U128 m)
{
    U128 r = u128_from(1);
    for (int i = 0; i < 128; ++i) r = h_mod_dbl(r, m);
    return r;
}
// 2^e mod m, exact.  Same algorithm the kernel runs.
static U128 h_pow2_mod(uint64_t e, U128 m)
{
    if (u128_eq(m, u128_from(1))) return u128_from(0);
    uint64_t mp  = h_neg_inv64(m.lo);
    U128 one     = h_r_mod(m);
    U128 x       = one;
    int bits = 0; { uint64_t t = e; while (t) { ++bits; t >>= 1; } }
    for (int b = bits - 1; b >= 0; --b) {
        x = h_mont_mul(x, x, m, mp);
        if ((e >> b) & 1ull) x = h_mod_dbl(x, m);
    }
    return h_mont_mul(x, u128_from(1), m, mp);   // out of Montgomery form
}
// a^e mod m in Montgomery domain (a already in Montgomery form)
static U128 h_mont_pow(U128 a_mont, U128 e, U128 m, uint64_t mp, U128 one_mont)
{
    U128 x = one_mont;
    for (int b = u128_bitlen(e) - 1; b >= 0; --b) {
        x = h_mont_mul(x, x, m, mp);
        if (u128_bit(e, b)) x = h_mont_mul(x, a_mont, m, mp);
    }
    return x;
}
// Miller-Rabin.  Deterministic for n < 3.3e24 with these bases; above that it is
// a strong probable-prime test (reported honestly by the caller).
static bool h_is_prime(U128 n)
{
    static const uint32_t small[] = { 2,3,5,7,11,13,17,19,23,29,31,37 };
    if (u128_lt(n, u128_from(2))) return false;
    for (uint32_t s : small) {
        if (u128_eq(n, u128_from(s))) return true;
        if (n.hi == 0 && n.lo % s == 0) return false;
        if (n.hi != 0 && u128_mod_u32(n, s) == 0) return false;
    }
    if ((n.lo & 1u) == 0) return false;

    U128 nm1 = u128_sub(n, u128_from(1));
    U128 d = nm1;
    int r = 0;
    while ((d.lo & 1u) == 0) { d = u128_shr1(d); ++r; }

    uint64_t mp   = h_neg_inv64(n.lo);
    U128 one      = h_r_mod(n);
    U128 neg_one  = u128_sub(n, one);

    for (uint32_t a : small) {
        U128 a_mont = one;
        for (uint32_t i = 1; i < a; ++i) a_mont = h_mod_add(a_mont, one, n);
        U128 x = h_mont_pow(a_mont, d, n, mp, one);
        if (u128_eq(x, one) || u128_eq(x, neg_one)) continue;
        bool witness = true;
        for (int i = 1; i < r; ++i) {
            x = h_mont_mul(x, x, n, mp);
            if (u128_eq(x, neg_one)) { witness = false; break; }
        }
        if (witness) return false;
    }
    return true;
}
static bool h_is_prime_u64(uint64_t n) { return h_is_prime(u128_from(n)); }

// modular inverse of a mod m (m prime here), extended Euclid, exact
static uint64_t modinv_u64(uint64_t a, uint64_t m)
{
    int64_t old_r = (int64_t)(a % m), r = (int64_t)m;
    int64_t old_s = 1, s = 0;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t t = old_r - q * r; old_r = r; r = t;
        t = old_s - q * s;         old_s = s; s = t;
    }
    if (old_r != 1) return 0;                 // not invertible
    int64_t res = old_s % (int64_t)m;
    if (res < 0) res += (int64_t)m;
    return (uint64_t)res;
}

// ===========================================================================
//  Section 4 -- Configuration file
// ===========================================================================

enum PauseMode { PAUSE_AUTO = 0, PAUSE_ALWAYS, PAUSE_NEVER };
static int g_pause_mode = PAUSE_AUTO;      // read by main() on every exit path

struct Config {
    int         pause_mode    = PAUSE_AUTO;
    // The job itself lives in worktodo.txt, not here: config.txt is machine
    // settings, worktodo.txt is what to work on, as GIMPS clients do it.
    std::string worktodo_file = "worktodo.txt";
    uint64_t    exponent      = 0;
    U128        factor_min    = u128_from(3);
    U128        factor_max    = u128_shl(u128_from(1), 40);
    int         bit_lo        = 0;      // as written in worktodo, for reporting
    int         bit_hi        = 0;
    uint32_t    sieve_primes  = 0;          // 0 = auto (resolve_sieve_limit)
    uint32_t    segment_size  = 1u << 22;
    int         threads       = 0;          // 0 = auto
    int         platform      = -1;         // -1 = auto
    int         device        = 0;
    bool        stop_on_factor = false;
    std::string results_file  = "results.txt";   // GIMPS submission lines only
    std::string log_file      = "runlog.txt";    // human record of every run
    int         arithmetic    = 0;      // 0 = auto, 96 or 128 to force a kernel
    int         workgroup     = 0;      // 0 = ask the driver
    int         gpu_slots     = 3;      // batches in flight on the device
    bool        checkpoint    = true;
    std::string checkpoint_file = "auto";
    double      checkpoint_seconds = 30.0;
};

// sieve_primes = auto.  The best bound is where the CPU sieve and the GPU finish
// together, so it tracks the GPU's cost per candidate, which is proportional to
// bitlen(p) (the number of Montgomery squarings).  The two brackets below came
// from measurement on one machine and are a starting point, not a universal
// optimum: the balance point moves with the ratio of CPU sieve speed to GPU
// test speed.  Two brackets are enough because the curve is flat -- being off
// by 4x costs only a few percent -- so a machine whose optimum differs loses
// little by leaving this on auto.
static uint32_t resolve_sieve_limit(const Config& cfg, uint64_t p)
{
    if (cfg.sieve_primes) return cfg.sieve_primes;
    int pbits = 0; { uint64_t t = p; while (t) { ++pbits; t >>= 1; } }
    return (pbits <= 32) ? 500000u : 1000000u;
}

static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
//  Value parsers.
//
//  Every number in the file goes through parse_u128, so every key accepts the
//  same forms as the exponent and the bounds -- "4194304", "4_194_304", "2^22"
//  -- and a typo is reported with its line number rather than silently reading
//  as 0 and taking a default.  strtoul("2^22") is 2 and strtoul("50O000") is 50:
//  both are legal-looking configs that quietly run something else.
//
//  The ranges below are what the rest of the program can actually use.  They are
//  deliberately wide -- they exist to catch a slip (a dropped digit, a stray
//  letter), not to second-guess anyone benchmarking.
// ---------------------------------------------------------------------------

static std::string range_msg(int64_t lo, int64_t hi)
{
    return "value out of range (" + std::to_string(lo) + " .. " + std::to_string(hi) + ")";
}

static bool parse_cfg_uint(const std::string& val, uint64_t lo, uint64_t hi,
                           uint32_t& out, std::string& err)
{
    U128 v;
    if (!parse_u128(val, v, err)) return false;
    if (v.hi != 0 || v.lo < lo || v.lo > hi) { err = range_msg((int64_t)lo, (int64_t)hi); return false; }
    out = (uint32_t)v.lo;
    return true;
}

static bool parse_cfg_int(const std::string& val, int64_t lo, int64_t hi,
                          int& out, std::string& err)
{
    std::string t = val;
    bool neg = false;
    if (!t.empty() && (t[0] == '-' || t[0] == '+')) { neg = (t[0] == '-'); t = trim(t.substr(1)); }
    U128 v;
    if (!parse_u128(t, v, err)) return false;
    if (v.hi != 0 || v.lo > (uint64_t)INT64_MAX) { err = range_msg(lo, hi); return false; }
    int64_t n = neg ? -(int64_t)v.lo : (int64_t)v.lo;
    if (n < lo || n > hi) { err = range_msg(lo, hi); return false; }
    out = (int)n;
    return true;
}

// The one value that is genuinely fractional.  strtod must consume the whole
// string; the range test also rejects NaN and the HUGE_VAL of an overflow.
static bool parse_cfg_double(const std::string& val, double lo, double hi,
                             double& out, std::string& err)
{
    const char* s = val.c_str();
    char* end = nullptr;
    double d = strtod(s, &end);
    while (*end && isspace((unsigned char)*end)) ++end;
    if (end == s || *end) { err = "not a number: '" + val + "'"; return false; }
    if (!(d >= lo && d <= hi)) {
        char buf[128];
        snprintf(buf, sizeof(buf), "value out of range (%g .. %g)", lo, hi);
        err = buf;
        return false;
    }
    out = d;
    return true;
}

static bool parse_cfg_bool(const std::string& val, bool& out, std::string& err)
{
    std::string v = val;
    for (auto& c : v) c = (char)tolower((unsigned char)c);
    if (v == "1" || v == "true"  || v == "yes" || v == "on")  { out = true;  return true; }
    if (v == "0" || v == "false" || v == "no"  || v == "off") { out = false; return true; }
    err = "expected 1/0, true/false, yes/no or on/off (got '" + val + "')";
    return false;
}

// ---------------------------------------------------------------------------
//  worktodo.txt -- what to work on.  Two accepted forms, first entry wins:
//
//     Factor=<assignment_id>,<exponent>,<bit_lo>,<bit_hi>
//         The GIMPS assignment line, as PrimeNet hands it out and as mfaktc
//         reads it.  Paste an assignment straight in.  The id may be anything
//         (PrimeNet's 32-hex-digit key, or N/A when you have no assignment).
//
//     exponent   = 9147253
//     factor_min = 1
//     factor_max = 2^70
//         The plain form, for a range that is not a whole bit level.
//
//  Bit levels are the natural unit here -- the search runs one at a time and
//  reports each as it clears -- so the Factor= form is preferred.
// ---------------------------------------------------------------------------
static bool load_worktodo(const std::string& path, Config& cfg, std::string& err)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open '" + path + "' -- it holds the job to run"; return false; }

    char line[1024];
    int  lineno = 0;
    bool have_exp = false, have_min = false, have_max = false;
    while (fgets(line, sizeof(line), f)) {
        ++lineno;
        std::string s(line);
        size_t hash = s.find_first_of("#;");
        if (hash != std::string::npos) s = s.substr(0, hash);
        s = trim(s);
        if (s.empty()) continue;

        std::string lower = s;
        for (auto& c : lower) c = (char)tolower((unsigned char)c);

        if (lower.rfind("factor=", 0) == 0) {
            // Factor=<id>,<exponent>,<bit_lo>,<bit_hi>
            std::vector<std::string> parts;
            std::string rest = s.substr(7);
            size_t pos = 0;
            for (;;) {
                size_t c = rest.find(',', pos);
                parts.push_back(trim(rest.substr(pos, c == std::string::npos ? c : c - pos)));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
            if (parts.size() < 4) {
                fclose(f);
                err = path + " line " + std::to_string(lineno) +
                      ": expected Factor=<id>,<exponent>,<bit_lo>,<bit_hi>";
                return false;
            }
            std::string perr;
            U128 e;
            if (!parse_u128(parts[1], e, perr) || e.hi != 0 || e.lo < 2) {
                fclose(f); err = path + " line " + std::to_string(lineno) + ": bad exponent"; return false;
            }
            int lo = atoi(parts[2].c_str()), hi = atoi(parts[3].c_str());
            if (lo < 0 || hi <= lo || hi > 127) {
                fclose(f);
                err = path + " line " + std::to_string(lineno) +
                      ": bit levels must satisfy 0 <= bit_lo < bit_hi <= 127";
                return false;
            }
            cfg.exponent   = e.lo;
            cfg.bit_lo     = lo;
            cfg.bit_hi     = hi;
            cfg.factor_min = (lo == 0) ? u128_from(1) : u128_shl(u128_from(1), (unsigned)lo);
            cfg.factor_max = u128_shl(u128_from(1), (unsigned)hi);
            fclose(f);
            return true;
        }

        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            fclose(f);
            err = path + " line " + std::to_string(lineno) + ": expected 'key = value' or 'Factor=...'";
            return false;
        }
        std::string key = trim(lower.substr(0, eq)), val = trim(s.substr(eq + 1));
        std::string perr;
        if (key == "exponent" || key == "p") {
            U128 v;
            if (!parse_u128(val, v, perr) || v.hi != 0 || v.lo < 2) {
                fclose(f); err = path + " line " + std::to_string(lineno) + ": bad exponent"; return false;
            }
            cfg.exponent = v.lo; have_exp = true;
        } else if (key == "factor_min" || key == "min") {
            if (!parse_u128(val, cfg.factor_min, perr)) {
                fclose(f); err = path + " line " + std::to_string(lineno) + ": " + perr; return false;
            }
            have_min = true;
        } else if (key == "factor_max" || key == "max") {
            if (!parse_u128(val, cfg.factor_max, perr)) {
                fclose(f); err = path + " line " + std::to_string(lineno) + ": " + perr; return false;
            }
            have_max = true;
        } else {
            fclose(f);
            err = path + " line " + std::to_string(lineno) + ": unknown key '" + key +
                  "' (worktodo holds the job; machine settings go in config.txt)";
            return false;
        }
    }
    fclose(f);
    if (!have_exp || !have_min || !have_max) {
        err = path + ": needs either a Factor= line or all of exponent/factor_min/factor_max";
        return false;
    }
    return true;
}

static bool load_config(const std::string& path, Config& cfg, std::string& err)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open config file '" + path + "'"; return false; }
    char line[1024];
    int lineno = 0;

    // Every rejection closes the file and points at the offending line.
    auto fail = [&](const std::string& msg) -> bool {
        fclose(f);
        err = "line " + std::to_string(lineno) + ": " + msg;
        return false;
    };

    while (fgets(line, sizeof(line), f)) {
        ++lineno;
        std::string s(line);
        size_t hash = s.find('#');
        if (hash != std::string::npos) s = s.substr(0, hash);
        s = trim(s);
        if (s.empty()) continue;
        size_t eq = s.find('=');
        if (eq == std::string::npos) return fail("expected 'key = value'");
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        for (auto& c : key) c = (char)tolower((unsigned char)c);
        if (val.empty()) return fail(key + ": no value");

        std::string perr;
        if (key == "exponent" || key == "p" || key == "mp" ||
            key == "factor_min" || key == "lower" || key == "min" ||
            key == "factor_max" || key == "upper" || key == "max") {
            return fail("'" + key + "' has moved to worktodo.txt (see that file for the\n"
                        "        format).  config.txt is machine settings only.");
        } else if (key == "worktodo_file") {
            cfg.worktodo_file = val;
        } else if (key == "sieve_primes") {
            if (val == "auto") cfg.sieve_primes = 0u;
            else if (!parse_cfg_uint(val, 2, 1000000000u, cfg.sieve_primes, perr)) return fail(key + ": " + perr);
        } else if (key == "segment_size") {
            if (!parse_cfg_uint(val, 4096, 1u << 28, cfg.segment_size, perr)) return fail(key + ": " + perr);
        } else if (key == "threads") {
            if (!parse_cfg_int(val, 0, 4096, cfg.threads, perr)) return fail(key + ": " + perr);
        } else if (key == "platform") {
            if (!parse_cfg_int(val, -1, 1024, cfg.platform, perr)) return fail(key + ": " + perr);
        } else if (key == "device") {
            if (!parse_cfg_int(val, 0, 1024, cfg.device, perr)) return fail(key + ": " + perr);
        } else if (key == "stop_on_factor") {
            if (!parse_cfg_bool(val, cfg.stop_on_factor, perr)) return fail(key + ": " + perr);
        } else if (key == "results_file") {
            cfg.results_file = val;
        } else if (key == "log_file") {
            cfg.log_file = val;
        } else if (key == "workgroup") {
            if (!parse_cfg_int(val, 0, 65536, cfg.workgroup, perr)) return fail(key + ": " + perr);
        } else if (key == "gpu_slots") {
            if (!parse_cfg_int(val, 2, 8, cfg.gpu_slots, perr)) return fail(key + ": " + perr);
        } else if (key == "arithmetic") {
            if      (val == "auto") cfg.arithmetic = 0;
            else if (val == "64")   cfg.arithmetic = 64;
            else if (val == "72")   cfg.arithmetic = 72;
            else if (val == "96")   cfg.arithmetic = 96;
            else if (val == "128")  cfg.arithmetic = 128;
            else return fail("arithmetic must be auto/64/72/96/128");
        } else if (key == "checkpoint") {
            if (!parse_cfg_bool(val, cfg.checkpoint, perr)) return fail(key + ": " + perr);
        } else if (key == "checkpoint_file") {
            cfg.checkpoint_file = val;
        } else if (key == "checkpoint_seconds") {
            if (!parse_cfg_double(val, 0.0, 86400.0, cfg.checkpoint_seconds, perr)) return fail(key + ": " + perr);
        } else if (key == "pause_on_exit") {
            if      (val == "auto")                                    cfg.pause_mode = PAUSE_AUTO;
            else if (val == "1" || val == "true"  || val == "always")   cfg.pause_mode = PAUSE_ALWAYS;
            else if (val == "0" || val == "false" || val == "never")    cfg.pause_mode = PAUSE_NEVER;
            else return fail("pause_on_exit must be auto/always/never");
        } else {
            return fail("unknown key '" + key + "'");
        }
    }
    fclose(f);
    return true;
}

// ===========================================================================
//  Section 5 -- GPU context
// ===========================================================================

struct Gpu {
    cl_platform_id   platform = nullptr;
    cl_device_id     device   = nullptr;
    cl_context       ctx      = nullptr;
    cl_command_queue queue    = nullptr;   // kernels
    cl_command_queue xfer     = nullptr;   // host->device copies, so they can
                                           // overlap with compute on the copy engine
    cl_program       program  = nullptr;
    cl_kernel        kernel   = nullptr;   // 128-bit path
    cl_kernel        kernel96 = nullptr;   // 96-bit path (32-bit limbs)
    cl_kernel        kernel96x2 = nullptr; // 96-bit, two candidates per work item
    cl_kernel        kernel64 = nullptr;   // 64-bit path (two 32-bit limbs), q < 2^64
    cl_kernel        kernel72 = nullptr;   // 72-bit path (24-bit limbs), q < 2^72
    cl_kernel        kernel72L = nullptr;  // as above, lazy reduction, q < 2^70
    std::string      name;
    size_t           wg_size  = 64;
    size_t           wg96     = 64;
    size_t           wg96x2   = 64;
    size_t           wg64     = 64;
    size_t           wg72     = 64;
    size_t           wg72L    = 64;
    cl_uint          cus      = 0;
};

static std::string dev_string(cl_device_id d, cl_device_info info)
{
    size_t n = 0;
    if (CL.GetDeviceInfo(d, info, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "";
    std::vector<char> buf(n + 1, 0);
    CL.GetDeviceInfo(d, info, n, buf.data(), nullptr);
    return std::string(buf.data());
}

static bool enumerate(std::vector<cl_platform_id>& plats,
                      std::vector<std::vector<cl_device_id>>& devs, std::string& err)
{
    cl_uint np = 0;
    if (CL.GetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        err = "no OpenCL platforms found (is a GPU driver installed?)";
        return false;
    }
    plats.resize(np);
    CL.GetPlatformIDs(np, plats.data(), nullptr);
    devs.resize(np);
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (CL.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        devs[i].resize(nd);
        CL.GetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, nd, devs[i].data(), nullptr);
    }
    return true;
}

static void list_devices()
{
    std::vector<cl_platform_id> plats;
    std::vector<std::vector<cl_device_id>> devs;
    std::string err;
    if (!enumerate(plats, devs, err)) { printf("  %s\n", err.c_str()); return; }
    for (size_t i = 0; i < plats.size(); ++i) {
        size_t n = 0; CL.GetPlatformInfo(plats[i], CL_PLATFORM_NAME, 0, nullptr, &n);
        std::vector<char> pn(n + 1, 0);
        CL.GetPlatformInfo(plats[i], CL_PLATFORM_NAME, n, pn.data(), nullptr);
        printf("  platform %d: %s\n", (int)i, pn.data());
        for (size_t j = 0; j < devs[i].size(); ++j) {
            cl_uint cu = 0, mhz = 0; cl_ulong mem = 0;
            CL.GetDeviceInfo(devs[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            CL.GetDeviceInfo(devs[i][j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
            CL.GetDeviceInfo(devs[i][j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
            printf("    device %d: %s  [%u CUs, %u MHz, %llu MB, %s]\n",
                   (int)j, dev_string(devs[i][j], CL_DEVICE_NAME).c_str(), cu, mhz,
                   (unsigned long long)(mem >> 20), dev_string(devs[i][j], CL_DEVICE_VERSION).c_str());
        }
    }
}

static bool gpu_init(Gpu& g, int want_platform, int want_device, std::string& err)
{
    std::vector<cl_platform_id> plats;
    std::vector<std::vector<cl_device_id>> devs;
    if (!enumerate(plats, devs, err)) return false;

    int pi = -1;
    if (want_platform >= 0) {
        if ((size_t)want_platform >= plats.size() || devs[want_platform].empty()) {
            err = "platform " + std::to_string(want_platform) + " has no GPU device";
            return false;
        }
        pi = want_platform;
    } else {
        for (size_t i = 0; i < plats.size(); ++i) if (!devs[i].empty()) { pi = (int)i; break; }
    }
    if (pi < 0) { err = "no OpenCL GPU device found"; return false; }
    if ((size_t)want_device >= devs[pi].size()) {
        err = "device index " + std::to_string(want_device) + " out of range on that platform";
        return false;
    }
    g.platform = plats[pi];
    g.device   = devs[pi][want_device];
    g.name     = dev_string(g.device, CL_DEVICE_NAME);
    CL.GetDeviceInfo(g.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(g.cus), &g.cus, nullptr);

    cl_int st = CL_SUCCESS;
    g.ctx = CL.CreateContext(nullptr, 1, &g.device, nullptr, nullptr, &st);
    if (!g.ctx || st != CL_SUCCESS) { err = "clCreateContext failed (" + std::to_string(st) + ")"; return false; }

    if (CL.CreateCommandQueueWithProperties) {
        g.queue = CL.CreateCommandQueueWithProperties(g.ctx, g.device, nullptr, &st);
    } else {
        g.queue = CL.CreateCommandQueue(g.ctx, g.device, 0, &st);
    }
    if (!g.queue || st != CL_SUCCESS) { err = "clCreateCommandQueue failed (" + std::to_string(st) + ")"; return false; }

    if (CL.CreateCommandQueueWithProperties) {
        g.xfer = CL.CreateCommandQueueWithProperties(g.ctx, g.device, nullptr, &st);
    } else {
        g.xfer = CL.CreateCommandQueue(g.ctx, g.device, 0, &st);
    }
    if (!g.xfer || st != CL_SUCCESS) { err = "clCreateCommandQueue(transfer) failed"; return false; }

    const char* srcs[3] = { TF_KERNEL_SOURCE_A, TF_KERNEL_SOURCE_B, TF_KERNEL_SOURCE_C };
    size_t      lens[3] = { strlen(srcs[0]), strlen(srcs[1]), strlen(srcs[2]) };
    g.program = CL.CreateProgramWithSource(g.ctx, 3, srcs, lens, &st);
    if (!g.program || st != CL_SUCCESS) { err = "clCreateProgramWithSource failed"; return false; }

    st = CL.BuildProgram(g.program, 1, &g.device, "-cl-std=CL1.2", nullptr, nullptr);
    if (st != CL_SUCCESS) {
        size_t n = 0;
        CL.GetProgramBuildInfo(g.program, g.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::vector<char> log(n + 1, 0);
        CL.GetProgramBuildInfo(g.program, g.device, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        err = std::string("OpenCL kernel build failed:\n") + log.data();
        return false;
    }
    g.kernel = CL.CreateKernel(g.program, "mersenne_tf", &st);
    if (!g.kernel || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf) failed"; return false; }
    g.kernel96 = CL.CreateKernel(g.program, "mersenne_tf96", &st);
    if (!g.kernel96 || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf96) failed"; return false; }
    g.kernel96x2 = CL.CreateKernel(g.program, "mersenne_tf96x2", &st);
    if (!g.kernel96x2 || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf96x2) failed"; return false; }
    g.kernel64 = CL.CreateKernel(g.program, "mersenne_tf64", &st);
    if (!g.kernel64 || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf64) failed"; return false; }
    g.kernel72 = CL.CreateKernel(g.program, "mersenne_tf72", &st);
    if (!g.kernel72 || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf72) failed"; return false; }
    g.kernel72L = CL.CreateKernel(g.program, "mersenne_tf72L", &st);
    if (!g.kernel72L || st != CL_SUCCESS) { err = "clCreateKernel(mersenne_tf72L) failed"; return false; }

    size_t maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg_size = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel96, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg96 = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel96x2, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg96x2 = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel64, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg64 = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel72, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg72 = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    maxwg = 64;
    CL.GetKernelWorkGroupInfo(g.kernel72L, g.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxwg), &maxwg, nullptr);
    g.wg72L = (maxwg >= 256) ? 256 : (maxwg ? maxwg : 64);
    return true;
}

static void gpu_free(Gpu& g)
{
    if (g.kernel64) CL.ReleaseKernel(g.kernel64);
    if (g.kernel72L) CL.ReleaseKernel(g.kernel72L);
    if (g.kernel72)  CL.ReleaseKernel(g.kernel72);
    if (g.kernel96x2) CL.ReleaseKernel(g.kernel96x2);
    if (g.kernel96) CL.ReleaseKernel(g.kernel96);
    if (g.kernel)  CL.ReleaseKernel(g.kernel);
    if (g.program) CL.ReleaseProgram(g.program);
    if (g.xfer)    CL.ReleaseCommandQueue(g.xfer);
    if (g.queue)   CL.ReleaseCommandQueue(g.queue);
    if (g.ctx)     CL.ReleaseContext(g.ctx);
    g = Gpu();
}

// ===========================================================================
//  Section 6 -- Pre-factoring sieve + work pipeline
// ===========================================================================

struct SievePrime {
    uint32_t s;      // the prime
    uint32_t k0;     // k = k0 (mod s)  <=>  s | 2kp+1
    uint32_t invW;   // W^-1 mod s, where W is the wheel modulus
};

// ---------------------------------------------------------------------------
//  The wheel.
//
//  Rather than enumerate every k and then strike out the ones whose candidate is
//  divisible by 3, 5, 7 or 11, we simply never enumerate those k at all.  Work
//  modulo W = 4 * 3 * 5 * 7 * 11 = 4620:
//
//    * the factor 4 carries the  q = +/-1 (mod 8)  rule (2 classes of 4 survive)
//    * each prime s in {3,5,7,11} kills the single class k = -(2p)^-1 (mod s)
//
//  960 of the 4620 residues survive -- 20.8% of all k, against 50% for the mod-4
//  rule alone.  The CPU therefore touches 2.4x fewer k, and the four densest
//  strike-out patterns (1/3 + 1/5 + 1/7 + 1/11 = 0.77 marks per k) disappear from
//  the sieve entirely.  Together that is ~3.5x less host work per unit of search
//  space.  This is what mfaktc's 4620 "classes" are doing.
//
//  A prime s that divides 2p cannot divide q = 2kp+1 at all, so it constrains
//  nothing and is left out of W (this only arises for p in {3,5,7,11}).
// ---------------------------------------------------------------------------
struct Wheel {
    uint64_t              W = 4;
    std::vector<uint32_t> classes;      // surviving residues, ascending
};

static Wheel build_wheel(uint64_t p)
{
    static const uint32_t WP[] = { 3, 5, 7, 11 };
    Wheel wh;
    std::vector<uint32_t> used, k0;
    for (uint32_t s : WP) {
        if (p % s == 0) continue;                   // s == p: never divides q
        uint64_t inv = modinv_u64((2 * p) % s, s);
        if (!inv) continue;
        wh.W *= s;
        used.push_back(s);
        k0.push_back((uint32_t)((s - inv) % s));    // k = -(2p)^-1 (mod s)
    }
    const uint32_t c4 = (p % 4 == 1) ? 3u : 1u;     // allowed k mod 4 (besides 0)

    for (uint64_t r = 0; r < wh.W; ++r) {
        uint32_t m4 = (uint32_t)(r & 3u);
        if (m4 != 0 && m4 != c4) continue;
        bool ok = true;
        for (size_t j = 0; j < used.size(); ++j)
            if ((uint32_t)(r % used[j]) == k0[j]) { ok = false; break; }
        if (ok) wh.classes.push_back((uint32_t)r);
    }
    return wh;
}

// ---------------------------------------------------------------------------
//  Checkpointing.
//
//  A class is the natural unit: the run is a loop over wheel classes, so after
//  each phase of classes finishes and the GPU queue has drained, everything
//  below that class index is provably complete.  Resuming skips those classes.
//  Factors are already appended to the results file the moment they are found,
//  so a checkpoint only has to carry them for the end-of-run recap.
// ---------------------------------------------------------------------------
struct CheckpointData {
    uint64_t p = 0, wheel = 0, classes_total = 0, classes_done = 0;
    uint64_t level = 0;                 // index into the bit-level list

    U128     fmin = u128_make(0, 0), fmax = u128_make(0, 0);
    uint64_t k_scanned = 0, gpu_tested = 0;
    double   seconds = 0;
    std::vector<U128> factors;
};

static std::string checkpoint_path(const Config& cfg, uint64_t p)
{
    if (cfg.checkpoint_file != "auto") return cfg.checkpoint_file;
    return "checkpoint_" + std::to_string(p) + ".txt";
}

static void checkpoint_save(const std::string& path, const CheckpointData& cp)
{
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    // version 2: the scan runs one bit level at a time, so classes_done counts
    // classes inside "level" rather than across the whole range.  A version 1
    // file means something different and is rejected rather than misread.
    fprintf(f, "version=2\n");
    fprintf(f, "p=%llu\n",             (unsigned long long)cp.p);
    fprintf(f, "fmin=%s\n",            u128_to_dec(cp.fmin).c_str());
    fprintf(f, "fmax=%s\n",            u128_to_dec(cp.fmax).c_str());
    fprintf(f, "wheel=%llu\n",         (unsigned long long)cp.wheel);
    fprintf(f, "level=%llu\n",         (unsigned long long)cp.level);
    fprintf(f, "classes_total=%llu\n", (unsigned long long)cp.classes_total);
    fprintf(f, "classes_done=%llu\n",  (unsigned long long)cp.classes_done);
    fprintf(f, "k_scanned=%llu\n",     (unsigned long long)cp.k_scanned);
    fprintf(f, "gpu_tested=%llu\n",    (unsigned long long)cp.gpu_tested);
    fprintf(f, "seconds=%.3f\n",       cp.seconds);
    for (const U128& q : cp.factors) fprintf(f, "factor=%s\n", u128_to_dec(q).c_str());
    fclose(f);
    // replace atomically-ish: never leave a half-written checkpoint in place
    remove(path.c_str());
    rename(tmp.c_str(), path.c_str());
}

static bool checkpoint_load(const std::string& path, CheckpointData& cp)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    bool have_version = false;
    while (fgets(line, sizeof(line), f)) {
        std::string s = trim(line);
        size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string k = s.substr(0, eq), v = s.substr(eq + 1);
        std::string perr;
        if      (k == "version")       have_version = (v == "2");
        else if (k == "level")         cp.level = strtoull(v.c_str(), nullptr, 10);
        else if (k == "p")             cp.p = strtoull(v.c_str(), nullptr, 10);
        else if (k == "fmin")          parse_u128(v, cp.fmin, perr);
        else if (k == "fmax")          parse_u128(v, cp.fmax, perr);
        else if (k == "wheel")         cp.wheel = strtoull(v.c_str(), nullptr, 10);
        else if (k == "classes_total") cp.classes_total = strtoull(v.c_str(), nullptr, 10);
        else if (k == "classes_done")  cp.classes_done = strtoull(v.c_str(), nullptr, 10);
        else if (k == "k_scanned")     cp.k_scanned = strtoull(v.c_str(), nullptr, 10);
        else if (k == "gpu_tested")    cp.gpu_tested = strtoull(v.c_str(), nullptr, 10);
        else if (k == "seconds")       cp.seconds = atof(v.c_str());
        else if (k == "factor")      { U128 q; if (parse_u128(v, q, perr)) cp.factors.push_back(q); }
    }
    fclose(f);
    return have_version;
}

// Diagnostic (--nogpu): run the whole sieve/pack pipeline but never submit to
// the device.  The difference between this and a normal run is exactly the cost
// of the PCIe transfer plus the kernel.
static bool g_nogpu = false;

// Diagnostic (--noxfer): launch the kernels but skip the host->device upload.
// Results are meaningless; the point is to price the transfer.
static bool g_noxfer = false;

// Ctrl-C / console close: stop at the next phase boundary and save a checkpoint
// rather than losing the run.
static std::atomic<bool> g_interrupt(false);

static BOOL WINAPI console_ctrl_handler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT ||
        type == CTRL_CLOSE_EVENT || type == CTRL_SHUTDOWN_EVENT) {
        g_interrupt.store(true);
        printf("\n  interrupted - winding down and saving a checkpoint at the last\n"
               "  completed class (that part of the range is re-done on resume)...\n");
        fflush(stdout);
        return TRUE;
    }
    return FALSE;
}

static std::vector<uint32_t> primes_below(uint32_t limit)
{
    std::vector<uint32_t> out;
    if (limit < 3) return out;
    std::vector<bool> comp(limit, false);
    for (uint32_t i = 2; (uint64_t)i * i < limit; ++i)
        if (!comp[i])
            for (uint64_t j = (uint64_t)i * i; j < limit; j += i) comp[(size_t)j] = true;
    for (uint32_t i = 2; i < limit; ++i) if (!comp[i]) out.push_back(i);
    return out;
}

// ---------------------------------------------------------------------------
//  Pinned staging buffers.
//
//  clEnqueueWriteBuffer from ordinary pageable memory has to stage through a
//  driver-owned pinned buffer, and that copy runs synchronously on the calling
//  thread -- measured at 0.15 s of a 0.48 s run, none of it overlapped with
//  compute.  Allocating the staging memory ourselves with CL_MEM_ALLOC_HOST_PTR
//  and having the sieve threads write survivors straight into it removes the
//  copy: the upload becomes a pure DMA the transfer queue can overlap with the
//  previous batch's kernel.
//
//  A worker holds a buffer from the moment it starts packing until the GPU has
//  consumed that batch, so a handful of buffers covers many more threads.
// ---------------------------------------------------------------------------
class PinnedPool {
public:
    bool init(Gpu& g, int nbuf, size_t bytes, std::string& err)
    {
        for (int i = 0; i < nbuf; ++i) {
            cl_int st = CL_SUCCESS;
            cl_mem m = CL.CreateBuffer(g.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       bytes, nullptr, &st);
            if (!m || st != CL_SUCCESS) break;
            void* p = CL.EnqueueMapBuffer(g.xfer, m, CL_TRUE, CL_MAP_WRITE, 0, bytes,
                                          0, nullptr, nullptr, &st);
            if (!p || st != CL_SUCCESS) { CL.ReleaseMemObject(m); break; }
            mems_.push_back(m);
            ptrs_.push_back((uint32_t*)p);
            used_.push_back(false);
        }
        if (mems_.size() < 3) { err = "could not allocate pinned staging buffers"; return false; }
        return true;
    }
    void watch(std::atomic<bool>* abort) { abrt_ = abort; }
    void poke() { cv_.notify_all(); }      // wake blocked workers after an abort
    int acquire()
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return stop_ || (abrt_ && abrt_->load()) || any_free(); });
        // Leaving because of an abort, not because a buffer came free: pass the
        // wake-up on.  There are many more workers than buffers, so most of them
        // are parked here at any moment, and a release() only ever notifies one.
        if (stop_ || (abrt_ && abrt_->load())) { lk.unlock(); cv_.notify_all(); return -1; }
        for (size_t i = 0; i < used_.size(); ++i)
            if (!used_[i]) { used_[i] = true; return (int)i; }
        return -1;
    }
    void release(int id)
    {
        if (id < 0) return;
        { std::lock_guard<std::mutex> lk(m_); used_[id] = false; }
        cv_.notify_one();
    }
    void stop() { { std::lock_guard<std::mutex> lk(m_); stop_ = true; } cv_.notify_all(); }
    uint32_t* ptr(int id) { return ptrs_[id]; }
    int count() const { return (int)mems_.size(); }
    void shutdown(Gpu& g)
    {
        for (size_t i = 0; i < mems_.size(); ++i) {
            CL.EnqueueUnmapMemObject(g.xfer, mems_[i], ptrs_[i], 0, nullptr, nullptr);
            CL.ReleaseMemObject(mems_[i]);
        }
        if (!mems_.empty()) CL.Finish(g.xfer);
        mems_.clear(); ptrs_.clear(); used_.clear();
    }
private:
    bool any_free() const { for (bool u : used_) if (!u) return true; return false; }
    std::vector<cl_mem>    mems_;
    std::vector<uint32_t*> ptrs_;
    std::vector<bool>      used_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool>* abrt_ = nullptr;
    bool stop_ = false;
};

struct Batch {
    uint32_t* data  = nullptr;      // into a pinned buffer, held until consumed
    uint32_t  count = 0;
    int       buf   = -1;           // pool id, released once the GPU is done
    U128     base_k = u128_make(0, 0);
    uint64_t scanned = 0;           // k values covered by this segment
};

// bounded producer/consumer queue
class BatchQueue {
public:
    explicit BatchQueue(size_t cap) : cap_(cap) {}
    // false = refused because the queue is shutting down; the caller still owns
    // the batch and must give its pinned buffer back.
    bool push(Batch&& b) {
        std::unique_lock<std::mutex> lk(m_);
        cv_full_.wait(lk, [&] { return q_.size() < cap_ || stop_; });
        if (stop_) return false;
        q_.push_back(std::move(b));
        cv_empty_.notify_one();
        return true;
    }
    bool pop(Batch& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_empty_.wait(lk, [&] { return !q_.empty() || done_ || stop_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        cv_full_.notify_one();
        return true;
    }
    void finish() { { std::lock_guard<std::mutex> lk(m_); done_ = true; } cv_empty_.notify_all(); }
    void abort()  { { std::lock_guard<std::mutex> lk(m_); stop_ = true; } cv_empty_.notify_all(); cv_full_.notify_all(); }
private:
    std::deque<Batch> q_;
    std::mutex m_;
    std::condition_variable cv_empty_, cv_full_;
    size_t cap_;
    bool done_ = false, stop_ = false;
};

// "45s", "12m34s", "1h23m", "3d04h"
static std::string fmt_duration(double s)
{
    if (!(s >= 0.0) || s > 3.1e9) return "--";
    unsigned long long t = (unsigned long long)(s + 0.5);
    unsigned long long h = t / 3600, m = (t % 3600) / 60, sec = t % 60;
    char buf[64];
    if (h >= 24)      snprintf(buf, sizeof(buf), "%llud%02lluh", h / 24, h % 24);
    else if (h)       snprintf(buf, sizeof(buf), "%lluh%02llum", h, m);
    else if (m)       snprintf(buf, sizeof(buf), "%llum%02llus", m, sec);
    else              snprintf(buf, sizeof(buf), "%llus", sec);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
//  The progress line is rewritten in place with a leading '\r'.  That only works
//  while it fits the window: one character too many and the console wraps it, so
//  the '\r' returns to the start of the *wrapped* row and every update leaves
//  the previous one stranded above it.  Ask the console how wide it is and stay
//  inside that.  Redirected output has no console and no wrapping either, so it
//  gets the full line.
// ---------------------------------------------------------------------------
static int console_width()
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) return 1000;
    int w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    return (w < 40) ? 80 : w;
}

// Wipe the progress line before printing anything that scrolls.  Redirected
// output reports no console and gets the full-line width, so cap the wipe at a
// sane column count rather than writing a thousand spaces into a log file.
static void clear_line()
{
    int w = console_width() - 1;
    if (w > 160) w = 160;
    printf("\r%*s\r", w, "");
}

struct RunStats {
    uint64_t k_scanned   = 0;    // candidates in the surviving wheel classes
    uint64_t gpu_tested  = 0;    // candidates that survived the sieve
    double   seconds     = 0;
    bool     interrupted = false;
};

struct Found { U128 q; };

// ---------------------------------------------------------------------------
//  Test the sieve primes that lie inside the requested range directly on the
//  CPU.  The sieve intentionally removes any q divisible by a small prime; that
//  would also remove q == s itself, so those few thousand values are checked
//  here instead.  This is what makes the pre-factoring lossless.
// ---------------------------------------------------------------------------
static void check_small_primes_in_range(uint64_t p, U128 fmin, U128 fmax,
                                        const std::vector<uint32_t>& primes,
                                        std::vector<U128>& found)
{
    uint64_t twop = 2 * p;
    for (uint32_t s : primes) {
        U128 q = u128_from(s);
        if (u128_lt(q, fmin) || u128_gt(q, fmax)) continue;
        if (s <= 2) continue;
        if ((uint64_t)(s - 1) % twop != 0) continue;          // must be 2kp+1
        if (u128_eq(h_pow2_mod(p, q), u128_from(1)))
            found.push_back(q);
    }
}

// ---------------------------------------------------------------------------
//  Announce one factor: full detail to the console the moment the GPU flags it,
//  plus one line appended to the results file.  Nothing here is taken from the
//  GPU on trust -- every field is recomputed on the CPU.
// ---------------------------------------------------------------------------
// Identifies the program in GIMPS result lines, as mfaktc's version string does.
#define TF_PROGRAM_ID "mersenne_tf 1.0"

static void report_factor(uint64_t p, U128 q, const Config& cfg)
{
    bool verified = u128_eq(h_pow2_mod(p, q), u128_from(1));
    U128 kq, r;
    u128_divmod(u128_sub(q, u128_from(1)), u128_from(2 * p), kq, r);
    bool form_ok = u128_is_zero(r);
    bool prime   = h_is_prime(q);
    const char* pl = prime ? (u128_bitlen(q) <= 81 ? "prime" : "probable prime") : "COMPOSITE";

    clear_line();                                // wipe the progress line
    printf("\n  *** FACTOR FOUND ***   %s\n", u128_to_dec(q).c_str());
    printf("      k         = %s   (q = 2kp+1: %s)\n", u128_to_dec(kq).c_str(), form_ok ? "yes" : "NO");
    printf("      q mod 8   = %u\n", (uint32_t)(q.lo & 7u));
    printf("      size      = %d bits\n", u128_bitlen(q));
    printf("      2^p mod q = 1 : %s  (recomputed on the CPU)\n", verified ? "VERIFIED" : "*** FAILED ***");
    printf("      q is      : %s\n", pl);

    // GIMPS manual-submission format.  Paste results.txt straight into
    // https://www.mersenne.org/manual_result/ -- these are the exact lines that
    // page parses, so nothing has to be reformatted by hand.
    FILE* rf = fopen(cfg.results_file.c_str(), "a");
    if (rf) {
        fprintf(rf, "M%llu has a factor: %s [TF:%d:%d:%s]\n",
                (unsigned long long)p, u128_to_dec(q).c_str(),
                cfg.bit_lo, cfg.bit_hi, TF_PROGRAM_ID);
        fclose(rf);
        printf("      logged to : %s\n", cfg.results_file.c_str());
    }
    if (!verified) printf("      *** NOT logged as verified -- CPU check failed ***\n");
    printf("\n");
    fflush(stdout);
}

// ---------------------------------------------------------------------------
//  How many k in [a, b] sit on the wheel -- i.e. how many candidates a stretch
//  of the range actually holds.  Whole wheel periods contribute NCLASS each and
//  only the leftover tail needs looking at, so this is exact and costs one pass
//  over the class list.
// ---------------------------------------------------------------------------
static U128 wheel_k_count(const Wheel& wh, U128 a, U128 b)
{
    if (u128_lt(b, a)) return u128_from(0);
    const uint64_t W = wh.W;
    U128 span = u128_add(u128_sub(b, a), u128_from(1));
    U128 full, rem;
    u128_divmod(span, u128_from(W), full, rem);
    const uint32_t amod = u128_mod_u32(a, (uint32_t)W);
    uint64_t tail = 0;
    for (uint32_t r : wh.classes)
        if ((uint64_t)((r + W - amod) % W) < rem.lo) ++tail;
    return u128_add(u128_mul_u64(full, (uint64_t)wh.classes.size()), u128_from(tail));
}

// ---------------------------------------------------------------------------
//  Bit levels.
//
//  Trial factoring is tracked by bit level -- "M_p is cleared to 2^70" -- so the
//  search runs one level at a time and finishes each before starting the next.
//  A level is everything below 2^40, then 2^40..2^50, 2^50..2^60, and every
//  power of two above that.
//
//  The split is done in k, not in q: the pieces then partition [kmin, kmax]
//  exactly, with no double counting at the seams, so the level candidate counts
//  add up to the run's own scanned total.  A level the range stops inside keeps
//  the bound actually reached as its upper end -- claiming the power of two
//  above it belongs to whoever finishes the level.
// ---------------------------------------------------------------------------
struct Level {
    U128 klo, khi;          // k range, inside [kmin, kmax]
    U128 qlo;               // candidate bounds: [qlo, qhi_excl)
    U128 qhi_excl;
    U128 qhi_disp;          // what to print as the upper end
    U128 candidates;        // k on the wheel inside this level
    int  bit_lo, bit_hi;    // the level as GIMPS names it: 2^bit_lo .. 2^bit_hi
    bool full;              // false when the range stops inside this level, in
                            // which case it is NOT a cleared bit level and must
                            // not be reported to GIMPS as one
};

static std::vector<Level> build_levels(uint64_t p, const Wheel& wh, U128 fmin, U128 fmax,
                                       U128 kmin, U128 kmax)
{
    const uint64_t twop = 2 * p;
    const U128     ONE  = u128_from(1);
    std::vector<Level> out;
    if (u128_lt(kmax, kmin)) return out;

    U128 klo = kmin, qlo = fmin;
    int prev_e = u128_bitlen(fmin) - 1;               // where this range starts
    if (prev_e < 0) prev_e = 0;
    for (int e = 40; e <= 127; ++e) {
        if (e > 40 && e < 50) continue;               // below 2^60 the levels are
        if (e > 50 && e < 60) continue;               // decades, not single bits
        U128 V = u128_shl(ONE, (unsigned)e);
        if (u128_ge(qlo, V)) continue;                // level already behind us
        // last k whose q = 2kp+1 is still below V
        U128 ksplit, dummy;
        u128_divmod(u128_sub(V, u128_from(2)), u128_from(twop), ksplit, dummy);
        const bool truncated = u128_lt(kmax, ksplit);
        U128 khi = truncated ? kmax : ksplit;
        if (u128_lt(khi, klo)) continue;

        Level L;
        L.klo = klo; L.khi = khi;
        L.qlo = qlo;
        L.qhi_excl = truncated ? u128_add(fmax, ONE) : V;
        L.qhi_disp = truncated ? fmax : V;
        L.candidates = wheel_k_count(wh, klo, khi);
        L.bit_lo = prev_e;
        L.bit_hi = e;
        L.full   = !truncated;
        if (!u128_is_zero(L.candidates)) out.push_back(L);   // empty level: nothing to scan

        klo = u128_add(khi, ONE);
        qlo = V;
        prev_e = e;
        if (u128_gt(klo, kmax)) break;
    }
    return out;
}

// Compact form for the screen: the bottom level is "under 2^40", the rest name
// the power-of-two band they cover.
static std::string level_label(const Level& L)
{
    if (u128_lt(L.qlo, u128_shl(u128_from(1), 40)))
        return "<" + u128_to_pow2(L.qhi_disp);
    return u128_to_pow2(L.qlo) + ".." + u128_to_pow2(L.qhi_disp);
}

// One line per level, appended the moment that level is finished -- so a run
// stopped later keeps every level it did clear.
static void log_level(const Config& cfg, uint64_t p, const Level& L,
                      const std::vector<U128>& factors)
{
    int nf = 0;
    for (const U128& q : factors)
        if (u128_ge(q, L.qlo) && u128_lt(q, L.qhi_excl)) ++nf;

    clear_line();
    printf("  level %s cleared: %s candidates, %d factor(s)\n",
           level_label(L).c_str(), u128_to_dec(L.candidates).c_str(), nf);
    fflush(stdout);

    // One GIMPS "no factor" line per bit level, written as the level clears.
    // A level that found something is covered by its "has a factor" line, so it
    // is not also reported clean.
    if (nf == 0 && L.full) {
        FILE* rf = fopen(cfg.results_file.c_str(), "a");
        if (!rf) { printf("  ! could not write to %s\n", cfg.results_file.c_str()); return; }
        fprintf(rf, "no factor for M%llu from 2^%d to 2^%d [%s]\n",
                (unsigned long long)p, L.bit_lo, L.bit_hi, TF_PROGRAM_ID);
        fclose(rf);
    }
}

// ---------------------------------------------------------------------------
//  Append the end-of-run record.  Written on every completed run, factor or
//  not: a file that only appears when something is found cannot tell "this
//  range is clean" apart from "this range was never searched", and the clean
//  result is the one a trial-factoring run usually produces.  One line, same
//  key=value shape as the factor lines, so a run's whole history greps.
// ---------------------------------------------------------------------------
static void report_run(uint64_t p, const Config& cfg,
                       const std::vector<U128>& factors, const RunStats& stats)
{
    FILE* rf = fopen(cfg.log_file.c_str(), "a");
    if (!rf) {
        printf("  ! could not write the run record to %s\n", cfg.log_file.c_str());
        fflush(stdout);
        return;
    }
    // fmt_duration is whole seconds, which would log a short run as "0s".
    char tbuf[32];
    if (stats.seconds < 60.0) snprintf(tbuf, sizeof(tbuf), "%.2fs", stats.seconds);
    else                      snprintf(tbuf, sizeof(tbuf), "%s", fmt_duration(stats.seconds).c_str());

    SYSTEMTIME lt; GetLocalTime(&lt);
    fprintf(rf, "%04d-%02d-%02d %02d:%02d:%02d  p=%llu  range=%s..%s  status=%s  factors=%d"
                "  scanned=%llu  tested=%llu  time=%s\n",
            lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute, lt.wSecond,
            (unsigned long long)p,
            u128_to_pow2(cfg.factor_min).c_str(), u128_to_pow2(cfg.factor_max).c_str(),
            stats.interrupted ? "interrupted" : "complete",
            (int)factors.size(),
            (unsigned long long)stats.k_scanned, (unsigned long long)stats.gpu_tested, tbuf);
    fclose(rf);
    printf("          run recorded in %s\n", cfg.log_file.c_str());
    fflush(stdout);
}

// ---------------------------------------------------------------------------
//  Main range runner.  Returns false only on hard errors.
// ---------------------------------------------------------------------------
static bool run_range(Gpu& g, const Config& cfg, uint64_t p, U128 fmin, U128 fmax,
                      bool verbose, std::vector<U128>& factors, RunStats& stats,
                      std::string& err)
{
    const uint64_t twop = 2 * p;
    const U128     ONE  = u128_from(1);

    // ---- k range from the factor range: q = 2kp+1 in [fmin, fmax] ----------
    if (u128_lt(fmin, u128_from(3))) fmin = u128_from(3);
    U128 kmin, kmax, rem, tmp;
    {   // kmin = ceil((fmin-1)/2p), kmax = floor((fmax-1)/2p)
        U128 a = u128_sub(fmin, ONE);
        u128_divmod(a, u128_from(twop), kmin, rem);
        if (!u128_is_zero(rem)) kmin = u128_add(kmin, ONE);
        if (u128_is_zero(kmin)) kmin = ONE;
        U128 b = u128_sub(fmax, ONE);
        u128_divmod(b, u128_from(twop), kmax, tmp);
    }
    if (u128_lt(kmax, kmin)) return true;   // nothing to do

    // ---- the wheel: which k are worth enumerating at all -------------------
    const Wheel  wh     = build_wheel(p);
    const uint64_t W    = wh.W;
    const size_t NCLASS = wh.classes.size();

    // ---- sieve primes: precompute k0(s) and W^-1 (mod s) once -------------
    std::vector<uint32_t> primes = primes_below(resolve_sieve_limit(cfg, p));
    std::vector<SievePrime> sp;
    sp.reserve(primes.size());
    for (uint32_t s : primes) {
        if (W % s == 0) continue;                   // 2,3,5,7,11: the wheel has them
        uint64_t tp = twop % s;
        if (tp == 0) continue;                      // s == p: never divides q
        uint64_t inv = modinv_u64(tp, s);
        if (inv == 0) continue;
        SievePrime e;
        e.s    = s;
        e.k0   = (uint32_t)((s - inv) % s);         // k = -(2p)^-1 (mod s)
        e.invW = (uint32_t)modinv_u64(W % s, s);
        if (e.invW == 0) continue;
        sp.push_back(e);
    }

    // Record a factor exactly once, whichever path found it.  A prime below the
    // sieve bound can be reported both by the direct CPU check and by the GPU
    // (the adaptive prime cap means small segments no longer strike out q == s),
    // and a resumed run already carries its earlier factors.
    auto add_factor = [&](const U128& q) {
        for (const U128& f : factors) if (u128_eq(f, q)) return;

        // Only prime factors are reported.  A composite divisor is a product of
        // prime factors that each divide M_p in their own right, so it carries
        // no new information: either they are inside the range and reported
        // separately, or they are below factor_min.  Miller-Rabin only calls a
        // number composite when it has an actual witness, so this can never
        // discard a genuine prime.
        if (!h_is_prime(q)) {
            if (verbose) {
                clear_line();
                printf("  note: composite divisor %s seen; its prime factors divide M_%llu\n"
                       "        too and are either reported below or under factor_min.\n",
                       u128_to_dec(q).c_str(), (unsigned long long)p);
                fflush(stdout);
            }
            return;
        }
        factors.push_back(q);
        if (verbose) report_factor(p, q, cfg);
    };

    // ---- small primes that are themselves inside the range ----------------
    {
        std::vector<U128> small_hits;
        check_small_primes_in_range(p, fmin, fmax, primes, small_hits);
        for (const U128& q : small_hits) add_factor(q);
    }

    // ---- GPU buffers ------------------------------------------------------
    const uint32_t SEG = cfg.segment_size;
    const int NSLOT = std::max(2, std::min(8, cfg.gpu_slots));
    cl_int st = CL_SUCCESS;
    std::vector<cl_mem>   idx_buf(NSLOT, nullptr);
    std::vector<cl_event> evt(NSLOT, nullptr);      // kernel completion
    std::vector<cl_event> wr_evt(NSLOT, nullptr);   // upload completion
    std::vector<Batch>    live(NSLOT);
    for (int i = 0; i < NSLOT; ++i) {
        idx_buf[i] = CL.CreateBuffer(g.ctx, CL_MEM_READ_ONLY, (size_t)SEG * sizeof(uint32_t), nullptr, &st);
        if (!idx_buf[i] || st != CL_SUCCESS) { err = "clCreateBuffer(index) failed"; return false; }
    }
    const uint32_t FOUND_CAP = 256;
    uint32_t zero = 0;
    cl_mem cnt_buf = CL.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &st);
    cl_mem res_buf = CL.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, FOUND_CAP * 16, nullptr, &st);
    if (!cnt_buf || !res_buf || st != CL_SUCCESS) { err = "clCreateBuffer(results) failed"; return false; }
    CL.EnqueueWriteBuffer(g.queue, cnt_buf, CL_TRUE, 0, sizeof(uint32_t), &zero, 0, nullptr, nullptr);

    int pbits = 0; { uint64_t t = p; while (t) { ++pbits; t >>= 1; } }

    // ---- pick the arithmetic width for a bit level -------------------------
    //  Narrower is faster, because the cost is the accumulation around the
    //  multiplies rather than the multiplies themselves: sizing the limbs to the
    //  candidate instead of to the container measures 1.46x, and skipping the
    //  per-squaring reduction on top of that 1.78x.  Each kernel is exact only
    //  below its own bound, so the choice is made per level from that level's
    //  upper bound -- levels are powers of two, so a level never straddles one.
    //      < 2^70   lazy 24-bit limbs   (needs 4m <= R = 2^72 for the bound)
    //      < 2^72   24-bit limbs
    //      < 2^96   32-bit limbs
    //      else     64-bit limbs
    //  arithmetic = 72/96/128 forces a width; a forced width that cannot hold
    //  the candidates is ignored, never silently used.
    auto pick_kernel = [&](U128 level_hi_excl, cl_kernel& K, size_t& WG, const char*& nm) {
        const U128 B64 = u128_shl(u128_from(1), 64);
        const U128 B70 = u128_shl(u128_from(1), 70);
        const U128 B72 = u128_shl(u128_from(1), 72);
        const U128 B96 = u128_shl(u128_from(1), 96);
        const int  a   = cfg.arithmetic;
        if      (!u128_gt(level_hi_excl, B64) && (a == 0 || a == 64)) { K = g.kernel64;  WG = g.wg64;  nm = "64-bit, two 32-bit limbs"; }
        else if (!u128_gt(level_hi_excl, B70) && (a == 0 || a == 72 || a == 64)) { K = g.kernel72L; WG = g.wg72L; nm = "72-bit, 24-bit limbs, lazy"; }
        else if (!u128_gt(level_hi_excl, B72) && (a == 0 || a == 72)) { K = g.kernel72;  WG = g.wg72;  nm = "72-bit, three 24-bit limbs"; }
        else if (!u128_gt(level_hi_excl, B96) && (a == 0 || a == 96 || a == 72)) { K = g.kernel96; WG = g.wg96; nm = "96-bit, three 32-bit limbs"; }
        else                                                          { K = g.kernel;   WG = g.wg_size; nm = "128-bit, two 64-bit limbs"; }
        if (cfg.workgroup > 0) WG = (size_t)cfg.workgroup;
    };

    // Pull back any hits the GPU has flagged and announce them straight away,
    // so a long run reports a factor within milliseconds instead of at the end.
    uint32_t reported = 0;
    auto drain_hits = [&]() -> uint32_t {
        uint32_t cnt = 0;
        CL.EnqueueReadBuffer(g.queue, cnt_buf, CL_TRUE, 0, sizeof(cnt), &cnt, 0, nullptr, nullptr);
        uint32_t n = std::min(cnt, FOUND_CAP);
        if (n > reported) {
            std::vector<U128> raw(n - reported);
            CL.EnqueueReadBuffer(g.queue, res_buf, CL_TRUE, (size_t)reported * 16,
                                 (size_t)(n - reported) * 16, raw.data(), 0, nullptr, nullptr);
            for (const U128& q : raw) add_factor(q);
            reported = n;
        }
        return cnt;
    };

    auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0;
    std::atomic<uint64_t> scanned_total(0), survivors_total(0);
    std::atomic<bool> abort_flag(false);

    // Pinned staging pool.  Fall back to fewer buffers if the driver will not
    // pin that much; three is the minimum the pipeline needs.
    PinnedPool pool;
    {
        bool okp = false;
        // enough to keep every slot busy plus a few being packed
        for (int nbuf : { NSLOT + 5, NSLOT + 3, NSLOT + 1, 3 }) {
            std::string perr;
            if (pool.init(g, nbuf, (size_t)SEG * sizeof(uint32_t), perr)) { okp = true; break; }
            pool.shutdown(g);
        }
        if (!okp) { err = "could not allocate pinned staging buffers - lower segment_size"; return false; }
        pool.watch(&abort_flag);
    }

    uint64_t total_k = 0;
    {   // candidates in the surviving wheel classes (for the progress bar)
        U128 span = u128_add(u128_sub(kmax, kmin), ONE);
        U128 qq, rr;
        u128_divmod(span, u128_from(W), qq, rr);
        U128 tot = u128_mul_u64(qq, (uint64_t)NCLASS);
        total_k = tot.hi ? ~0ull : tot.lo;
    }

    // ---- the bit levels this run clears, lowest first ----------------------
    const std::vector<Level> levels = build_levels(p, wh, fmin, fmax, kmin, kmax);
    if (levels.empty()) return true;                 // no candidate in the range

    // ---- checkpoint: resume a previous run of the identical job ------------
    const bool use_cp = cfg.checkpoint && verbose;
    const std::string cp_path = checkpoint_path(cfg, p);
    size_t   level_start   = 0;
    size_t   class_start   = 0;
    uint64_t resumed_scan  = 0, resumed_gpu = 0;
    double   resumed_secs  = 0;
    if (use_cp) {
        CheckpointData cp;
        if (checkpoint_load(cp_path, cp) && cp.p == p &&
            u128_eq(cp.fmin, fmin) && u128_eq(cp.fmax, fmax) &&
            cp.wheel == W && cp.classes_total == NCLASS && cp.classes_done <= NCLASS &&
            cp.level < levels.size()) {
            level_start  = (size_t)cp.level;
            class_start  = (size_t)cp.classes_done;
            resumed_scan = cp.k_scanned;
            resumed_gpu  = cp.gpu_tested;
            resumed_secs = cp.seconds;
            for (const U128& q : cp.factors) factors.push_back(q);
            printf("  resuming from %s: level %s, %llu of %llu classes done\n"
                   "               (%llu of %llu levels already cleared)\n\n",
                   cp_path.c_str(), level_label(levels[level_start]).c_str(),
                   (unsigned long long)class_start, (unsigned long long)NCLASS,
                   (unsigned long long)level_start, (unsigned long long)levels.size());
        } else if (FILE* probe = fopen(cp_path.c_str(), "rb")) {
            // Silently starting over would look like the checkpoint had failed.
            fclose(probe);
            printf("  note: %s does not match this job (different range, or written by a\n"
                   "        version before the level-by-level scan) - starting fresh.\n\n",
                   cp_path.c_str());
        }
    }
    scanned_total.store(resumed_scan);
    survivors_total.store(resumed_gpu);

    auto last_cp = std::chrono::steady_clock::now();
    const char* last_knm = nullptr;         // so the width is announced on change

    struct ClassInfo { U128 kc0; uint64_t N; uint64_t seg0; };

    // ---- one bit level at a time, lowest first -----------------------------
    //  The level is the outer loop so that each one is finished, logged and
    //  checkpointed before the next begins.  Scanning classes across the whole
    //  range instead would leave every level equally unfinished until the very
    //  end, and a run stopped halfway could claim none of them.
    for (size_t li = level_start; li < levels.size() && !abort_flag.load(); ++li) {
        const Level&      L   = levels[li];
        const U128        lk0 = L.klo, lk1 = L.khi;
        const std::string lvl = level_label(L);

        // Widest candidate in this level decides the kernel.
        cl_kernel   K  = nullptr;
        size_t      WG = 64;
        const char* knm = "";
        pick_kernel(L.qhi_excl, K, WG, knm);
        if (verbose && knm != last_knm) {
            clear_line();
            printf("  arithmetic : %s   (level %s)\n", knm, lvl.c_str());
            fflush(stdout);
            last_knm = knm;
        }

        // ---- phase sizing, per level --------------------------------------
        //  A phase is a group of whole classes.  It ends with a full GPU drain,
        //  so it is a safe checkpoint boundary; make it big enough that the
        //  drain is a rounding error, small enough that little work is lost on
        //  a crash.  A low level holds few k per class and so wants many more
        //  classes per phase than a high one -- hence sizing it here.
        uint64_t segs_per_class;
        {
            U128 span = u128_add(u128_sub(lk1, lk0), ONE);
            U128 qq, rr;
            u128_divmod(span, u128_from(W), qq, rr);
            uint64_t n_avg  = qq.hi ? ~0ull : qq.lo;
            segs_per_class  = (n_avg + SEG - 1) / SEG;
            if (segs_per_class == 0) segs_per_class = 1;
        }
        //  Each phase boundary costs a full GPU drain plus a worker respawn, so
        //  in principle phases want to be long.  Measured: widening them 8x
        //  (256 -> 2048 segments) changed throughput by less than the run-to-run
        //  noise, so keep them short and buy finer checkpoint granularity.
        const uint64_t PHASE_SEGS = 256;
        size_t phase_classes = (size_t)((PHASE_SEGS + segs_per_class - 1) / segs_per_class);
        if (phase_classes < 1)      phase_classes = 1;
        if (phase_classes > NCLASS) phase_classes = NCLASS;

        // ---- progress is per level, so it needs this level's own baseline --
        //  Resuming lands inside a level with some of its classes already done,
        //  and the running counter cannot tell that work from earlier levels'.
        //  The classes below class_start hold a known number of k, so count them
        //  once here rather than starting the level's progress bar at zero.
        const double   level_total = u128_to_dbl(L.candidates);
        const uint64_t level_base  = scanned_total.load();
        uint64_t level_done0 = 0;
        if (li == level_start && class_start > 0) {
            const uint32_t kmod = u128_mod_u32(lk0, (uint32_t)W);
            for (size_t c = 0; c < class_start && c < NCLASS; ++c) {
                uint32_t r   = wh.classes[c];
                U128     kc0 = u128_add(lk0, u128_from((uint64_t)((r + W - kmod) % W)));
                if (u128_gt(kc0, lk1)) continue;
                U128 qq, rr;
                u128_divmod(u128_sub(lk1, kc0), u128_from(W), qq, rr);
                level_done0 += qq.lo + 1;
            }
        }

        // ---- run the wheel classes of this level, a phase at a time -------
        for (size_t cb = (li == level_start ? class_start : 0);
             cb < NCLASS && !abort_flag.load(); cb += phase_classes) {
            const size_t ce = std::min(NCLASS, cb + phase_classes);

            std::vector<ClassInfo> cls;
            std::vector<uint64_t>  seg0s;
            uint64_t nseg = 0;
            cls.reserve(ce - cb);
            seg0s.reserve(ce - cb);
            for (size_t c = cb; c < ce; ++c) {
                ClassInfo info;
                uint32_t r    = wh.classes[c];
                uint32_t kmod = u128_mod_u32(lk0, (uint32_t)W);
                info.kc0 = u128_add(lk0, u128_from((uint64_t)((r + W - kmod) % W)));
                if (u128_gt(info.kc0, lk1)) {
                    info.N = 0;
                } else {
                    U128 qq, rr;
                    u128_divmod(u128_sub(lk1, info.kc0), u128_from(W), qq, rr);
                    U128 n2 = u128_add(qq, ONE);
                    if (n2.hi) { err = "range too large for one run - lower factor_max"; return false; }
                    info.N = n2.lo;
                }
                info.seg0 = nseg;
                nseg += (info.N + SEG - 1) / SEG;
                cls.push_back(info);
                seg0s.push_back(info.seg0);
            }
            if (nseg == 0) continue;

            std::atomic<uint64_t> next_seg(0);
            BatchQueue queue(NSLOT + 1);   // in-flight batches must not exceed the pool

            // Stopping early has to go through here, never through abort_flag alone.
            // Most sieve threads are parked in pool.acquire() waiting for a buffer at
            // any given moment, and a condition variable only re-tests its predicate
            // when it is signalled: raising the flag silently leaves them asleep.
            // They then never exit, so the finisher below never joins them, so it
            // never calls queue.finish(), so the consumer loop never ends and the
            // pool.poke() after that loop -- the wake-up they are waiting for -- is
            // never reached.  That circular wait is what hung Ctrl-C.
            auto request_abort = [&] { abort_flag.store(true); pool.poke(); };

            // Where this phase started.  An aborted phase is only partly scanned, so
            // an interrupt checkpoint has to name this boundary rather than the one
            // ahead of it; see the interrupt block at the end of the loop.
            const uint64_t phase_scan0 = scanned_total.load();
            const uint64_t phase_gpu0  = survivors_total.load();

            int nthreads = cfg.threads > 0 ? cfg.threads : (int)std::thread::hardware_concurrency() - 1;
            if (nthreads < 1) nthreads = 1;
            if (nthreads > 32) nthreads = 32;

            std::vector<std::thread> workers;
            for (int w = 0; w < nthreads; ++w) {
                workers.emplace_back([&] {
                    std::vector<uint64_t> mark;   // bitmap: 1 = struck out by the sieve
                    std::vector<uint32_t> offs(sp.size());   // running strike position
                    for (;;) {
                        uint64_t s = next_seg.fetch_add(1);
                        if (s >= nseg || abort_flag.load()) break;

                        // decode the global segment index into (class, offset).
                        // upper_bound-1 lands on the last class sharing this seg0,
                        // which is the non-empty one when empty classes precede it.
                        size_t ci = (size_t)(std::upper_bound(seg0s.begin(), seg0s.end(), s)
                                             - seg0s.begin() - 1);
                        uint64_t off = (s - cls[ci].seg0) * (uint64_t)SEG;
                        uint32_t len = (uint32_t)std::min<uint64_t>(SEG, cls[ci].N - off);

                        Batch b;
                        b.base_k  = u128_add(cls[ci].kc0, u128_mul_u64(u128_from(off), W));
                        b.scanned = len;

                        // A bitmap, not a byte array: a 4M-candidate segment is
                        // 512 KB this way and stays resident in L2, which is worth
                        // several times the cost of the bit twiddling.
                        const size_t nw = ((size_t)len + 63) / 64;
                        mark.assign(nw, 0);

                        // ---- pre-factoring: strike out every k whose q has a
                        //      small prime factor -------------------------------
                        //  A prime only earns its keep if the candidates it removes
                        //  cost the GPU more than the setup costs here; past about
                        //  len/8 it strikes too few positions to be worth it, which
                        //  also stops tiny ranges paying for a huge prime table.
                        const uint32_t pcap = (len > 8192) ? (len / 8) : 1024u;

                        // First strike position for each prime, once per segment.
                        size_t np = 0;
                        for (const SievePrime& e : sp) {
                            if (e.s > pcap) break;                  // sp is ascending
                            uint32_t b0 = u128_mod_u32(b.base_k, e.s);
                            offs[np++] = (uint32_t)(((uint64_t)((e.k0 + e.s - b0) % e.s) * e.invW) % e.s);
                        }

                        // Strike in L1-sized blocks rather than sweeping the whole
                        // segment per prime.  The bitmap is 512 KB, so a prime whose
                        // stride exceeds a cache line misses on nearly every hit when
                        // swept end to end -- measured 5.5 cycles per strike against
                        // the 1-2 a bitmap sieve should cost.  Running all primes
                        // through one 32 KB block keeps that block resident; the cost
                        // is re-entering the prime loop per block, with the running
                        // offsets in a sequential array the prefetcher handles.
                        const uint32_t BLK = 1u << 18;              // bits = 32 KB
                        for (uint32_t bs = 0; bs < len; bs += BLK) {
                            const uint32_t be = std::min(len, bs + BLK);
                            for (size_t i = 0; i < np; ++i) {
                                const uint32_t s = sp[i].s;
                                uint64_t j = offs[i];
                                for (; j < be; j += s) mark[j >> 6] |= 1ull << (j & 63);
                                offs[i] = (uint32_t)j;
                            }
                        }
                        if (len & 63) mark[nw - 1] |= ~0ull << (len & 63);   // pad tail

                        // Pack survivors directly into pinned memory.  The buffer is
                        // only acquired now, after the marking, so it is held for a
                        // small fraction of the segment's processing time.
                        int bid = pool.acquire();
                        if (bid < 0) break;                       // pool stopped
                        uint32_t* out = pool.ptr(bid);
                        uint32_t   cnt = 0;
                        for (size_t w = 0; w < nw; ++w) {
                            uint64_t v = ~mark[w];
                            uint32_t base = (uint32_t)(w << 6);
                            while (v) {
                                unsigned long t;
                                _BitScanForward64(&t, v);
                                out[cnt++] = base + (uint32_t)t;
                                v &= v - 1;
                            }
                        }
                        b.data = out; b.count = cnt; b.buf = bid;

                        scanned_total.fetch_add(len);
                        survivors_total.fetch_add(cnt);
                        if (cnt && queue.push(std::move(b))) continue;
                        pool.release(bid);
                    }
                });
            }

            // ---- consumer: feed the GPU ---------------------------------------
            uint64_t launches = 0;
            std::thread finisher([&] { for (auto& t : workers) t.join(); queue.finish(); });

            Batch b;
            while (queue.pop(b)) {
                if (g_nogpu) { pool.release(b.buf); ++launches; continue; }   // host-only timing
                int slot = (int)(launches % NSLOT);
                if (evt[slot]) {
                    CL.WaitForEvents(1, &evt[slot]);     // slot's kernel done -> reusable
                    CL.ReleaseEvent(evt[slot]);
                    evt[slot] = nullptr;
                }
                if (wr_evt[slot]) { CL.ReleaseEvent(wr_evt[slot]); wr_evt[slot] = nullptr; }
                pool.release(live[slot].buf);            // its kernel has finished
                live[slot] = std::move(b);
                const uint32_t n = live[slot].count;

                // Upload on the transfer queue so the copy engine can work while the
                // previous batch is still computing; the kernel below waits on it.
                if (!g_noxfer) {
                    st = CL.EnqueueWriteBuffer(g.xfer, idx_buf[slot], CL_FALSE, 0,
                                               n * sizeof(uint32_t), live[slot].data, 0, nullptr, &wr_evt[slot]);
                    if (st != CL_SUCCESS) { err = "clEnqueueWriteBuffer failed"; request_abort(); break; }
                    CL.Flush(g.xfer);
                }

                uint64_t base_lo = live[slot].base_k.lo, base_hi = live[slot].base_k.hi;
                uint64_t step = W, pe = p;
                int argi = 0;
                CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &idx_buf[slot]);
                CL.SetKernelArg(K, argi++,sizeof(uint32_t), &n);
                CL.SetKernelArg(K, argi++,sizeof(uint64_t), &base_lo);
                CL.SetKernelArg(K, argi++,sizeof(uint64_t), &base_hi);
                CL.SetKernelArg(K, argi++,sizeof(uint64_t), &step);
                CL.SetKernelArg(K, argi++,sizeof(uint64_t), &twop);
                CL.SetKernelArg(K, argi++,sizeof(uint64_t), &pe);
                CL.SetKernelArg(K, argi++,sizeof(int32_t),  &pbits);
                CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &cnt_buf);
                CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &res_buf);
                CL.SetKernelArg(K, argi++,sizeof(uint32_t), &FOUND_CAP);

                size_t local  = WG;
                size_t global = ((n + local - 1) / local) * local;
                st = CL.EnqueueNDRangeKernel(g.queue, K, 1, nullptr, &global, &local,
                                             wr_evt[slot] ? 1u : 0u,
                                             wr_evt[slot] ? &wr_evt[slot] : nullptr, &evt[slot]);
                if (st != CL_SUCCESS) { err = "clEnqueueNDRangeKernel failed (" + std::to_string(st) + ")"; request_abort(); break; }
                ++launches;

                // periodic: progress + check whether the GPU has flagged anything
                auto now = std::chrono::steady_clock::now();
                if (verbose && std::chrono::duration<double>(now - last_print).count() > 0.5) {
                    last_print = now;
                    double el  = std::chrono::duration<double>(now - t0).count();
                    uint64_t sc = scanned_total.load(), sv = survivors_total.load();

                    // Rates must count only work done in THIS run: on a resumed run
                    // the totals start at the checkpointed values, which would
                    // otherwise be credited to this run's clock.
                    uint64_t sc_now = sc - resumed_scan;
                    uint64_t sv_now = sv - resumed_gpu;
                    double rate  = el > 0 ? sv_now / el : 0.0;
                    double kps   = (el > 0) ? (double)sc_now / el : 0.0;   // k scanned per second

                    // The percentage tracks the level being scanned, not the whole
                    // job -- a level is the unit of work that finishes and gets
                    // logged.  Both ETAs are shown so the two never disagree: one
                    // for this level, one for everything still to do.
                    double lvl_done = (double)level_done0 + (double)(sc - level_base);
                    double frac     = level_total > 0 ? lvl_done / level_total : 0.0;
                    if (frac > 1.0) frac = 1.0;
                    double eta_lvl  = (kps > 0 && level_total > lvl_done)
                                    ? (level_total - lvl_done) / kps : -1.0;
                    double eta_job  = (kps > 0 && total_k > sc)
                                    ? (double)(total_k - sc) / kps : -1.0;

                    // Which wheel classes this phase is on.  A phase is usually one
                    // class; on small ranges it groups several, so name the span.
                    char cls[48];
                    if (ce - cb == 1)
                        snprintf(cls, sizeof(cls), "cls %llu/%llu",
                                 (unsigned long long)(cb + 1), (unsigned long long)NCLASS);
                    else
                        snprintf(cls, sizeof(cls), "cls %llu-%llu/%llu",
                                 (unsigned long long)(cb + 1), (unsigned long long)ce,
                                 (unsigned long long)NCLASS);

                    // Widest layout that fits the window, so the line stays a
                    // single line on an 80-column console as readily as on a wide
                    // one.  The choice only ever narrows: durations change length
                    // as a run goes on ("59m03s" -> "1h23m"), and re-widening on
                    // the short ones would make the line flicker between layouts.
                    const int    wmax   = console_width() - 1;
                    const double sieved = sc ? 100.0 * (1.0 - (double)sv / (double)sc) : 0.0;
                    static int   tier   = 0;
                    char line[512];
                    for (;; ++tier) {
                        switch (tier) {
                        case 0:
                            snprintf(line, sizeof(line),
                                     "  %s %6.2f%%  %llu tested (%.0f M/s)  %s  sieved %.1f%%  elapsed %s  ETA %s  job %s",
                                     lvl.c_str(), frac * 100.0, (unsigned long long)sv, rate / 1e6,
                                     cls, sieved, fmt_duration(el).c_str(),
                                     fmt_duration(eta_lvl).c_str(), fmt_duration(eta_job).c_str());
                            break;
                        case 1:
                            snprintf(line, sizeof(line),
                                     "  %s %6.2f%%  %llu tested (%.0f M/s)  %s  ETA %s  job %s",
                                     lvl.c_str(), frac * 100.0, (unsigned long long)sv, rate / 1e6,
                                     cls, fmt_duration(eta_lvl).c_str(), fmt_duration(eta_job).c_str());
                            break;
                        case 2:
                            snprintf(line, sizeof(line), "  %s %6.2f%%  %.0f M/s  %s  ETA %s",
                                     lvl.c_str(), frac * 100.0, rate / 1e6, cls,
                                     fmt_duration(eta_lvl).c_str());
                            break;
                        default:
                            snprintf(line, sizeof(line), "  %s %6.2f%%  %.0f M/s  ETA %s",
                                     lvl.c_str(), frac * 100.0, rate / 1e6,
                                     fmt_duration(eta_lvl).c_str());
                            if ((int)strlen(line) > wmax) line[wmax] = '\0';
                            break;
                        }
                        if ((int)strlen(line) <= wmax || tier >= 3) break;
                    }

                    // Pad back to the previous length so a line that shrinks does
                    // not leave the tail of the longer one behind it.
                    static int prev_len = 0;
                    int len = (int)strlen(line);
                    printf("\r%s%*s", line, (prev_len > len ? prev_len - len : 0), "");
                    prev_len = len;
                    fflush(stdout);
                }
                // Polling costs a pipeline drain (the read is blocking on an
                // in-order queue), so do it rarely.  64 batches is still well under
                // a tenth of a second, i.e. still "immediately" from a user's view.
                if ((launches % 64) == 0) {
                    uint32_t cnt = drain_hits();          // announces new hits at once
                    if ((cnt && cfg.stop_on_factor) || g_interrupt.load()) request_abort();
                }
            }
            queue.abort();
            pool.poke();                       // release anyone blocked on a buffer
            finisher.join();
            CL.Finish(g.xfer);
            CL.Finish(g.queue);
            for (int i = 0; i < NSLOT; ++i) {
                if (evt[i])    { CL.ReleaseEvent(evt[i]);    evt[i]    = nullptr; }
                if (wr_evt[i]) { CL.ReleaseEvent(wr_evt[i]); wr_evt[i] = nullptr; }
                pool.release(live[i].buf);
                live[i] = Batch();
            }
            if (!err.empty()) break;

            // ---- phase finished: the GPU is drained, so every class below ce is
            //      provably complete and this is a safe checkpoint boundary ------
            drain_hits();
            if (use_cp && !abort_flag.load()) {
                auto now = std::chrono::steady_clock::now();
                bool due  = std::chrono::duration<double>(now - last_cp).count() >= cfg.checkpoint_seconds;
                bool last = (ce >= NCLASS);
                if (due || last) {
                    CheckpointData cp;
                    cp.p = p; cp.fmin = fmin; cp.fmax = fmax; cp.wheel = W;
                    cp.level         = li;
                    cp.classes_total = NCLASS;
                    cp.classes_done  = ce;
                    cp.k_scanned  = scanned_total.load();
                    cp.gpu_tested = survivors_total.load();
                    cp.seconds    = resumed_secs +
                                    std::chrono::duration<double>(now - t0).count();
                    cp.factors    = factors;
                    checkpoint_save(cp_path, cp);
                    last_cp = now;
                }
            }
            if (g_interrupt.load()) {
                // Save where the run is provably complete, then unwind.  The
                // interrupt stops the sieve threads mid-phase, so the classes in
                // [cb, ce) were only partly scanned -- naming ce here would resume
                // past candidates that were never tested and could hide a factor.
                // The phase start is the last drained boundary, so it is the honest
                // one; at most one phase of work is repeated.  Any factor already
                // found inside the partial phase is still carried over.
                if (use_cp) {
                    CheckpointData cp;
                    cp.p = p; cp.fmin = fmin; cp.fmax = fmax; cp.wheel = W;
                    cp.level         = li;
                    cp.classes_total = NCLASS;
                    cp.classes_done  = cb;
                    cp.k_scanned  = phase_scan0;
                    cp.gpu_tested = phase_gpu0;
                    cp.seconds    = resumed_secs +
                                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    cp.factors    = factors;
                    checkpoint_save(cp_path, cp);
                    printf("  checkpoint written to %s (level %s, %llu of %llu classes done)\n"
                           "  re-run with the same config.txt to continue.\n",
                           cp_path.c_str(), lvl.c_str(),
                           (unsigned long long)cb, (unsigned long long)NCLASS);
                }
                request_abort();
                break;
            }
        }   // end of the class-phase loop

        // ---- level finished: every class of it is drained and complete -----
        if (abort_flag.load() || !err.empty()) break;
        if (verbose) log_level(cfg, p, L, factors);
        if (use_cp) {
            CheckpointData cp;
            cp.p = p; cp.fmin = fmin; cp.fmax = fmax; cp.wheel = W;
            cp.level         = li + 1;         // start of the next level
            cp.classes_total = NCLASS;
            cp.classes_done  = 0;
            cp.k_scanned  = scanned_total.load();
            cp.gpu_tested = survivors_total.load();
            cp.seconds    = resumed_secs +
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            cp.factors    = factors;
            if (li + 1 < levels.size()) checkpoint_save(cp_path, cp);
            last_cp = std::chrono::steady_clock::now();
        }
    }

    CL.Finish(g.queue);
    if (verbose) clear_line();

    // ---- collect any hits from the final, unpolled batches ----------------
    uint32_t cnt = drain_hits();
    if (cnt > FOUND_CAP)
        printf("  ! %u factors found but only %u could be stored - narrow the range\n", cnt, FOUND_CAP);

    pool.stop();
    pool.shutdown(g);
    for (int i = 0; i < NSLOT; ++i) if (idx_buf[i]) CL.ReleaseMemObject(idx_buf[i]);
    CL.ReleaseMemObject(cnt_buf);
    CL.ReleaseMemObject(res_buf);

    std::sort(factors.begin(), factors.end(), [](const U128& a, const U128& b) { return u128_lt(a, b); });
    factors.erase(std::unique(factors.begin(), factors.end(),
                              [](const U128& a, const U128& b) { return u128_eq(a, b); }), factors.end());

    stats.k_scanned  = scanned_total.load();
    stats.gpu_tested = survivors_total.load();
    stats.seconds    = resumed_secs +
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats.interrupted = g_interrupt.load();

    // The whole range is done -- drop the checkpoint so the next run starts clean.
    if (use_cp && err.empty() && !abort_flag.load())
        remove(cp_path.c_str());

    return err.empty();
}

// ===========================================================================
//  Section 6b -- Pure device benchmark
//
//  Everything else in this program measures the whole pipeline (sieve + PCIe +
//  kernel), which cannot tell you which of those is the limit.  This uploads a
//  single batch once and then launches the kernel on it repeatedly with no host
//  work in between, so the number it prints is the GPU's ceiling and nothing
//  else.  Results are ignored -- the candidates are arbitrary.
// ===========================================================================
static void bench_device(Gpu& g, const Config& cfg)
{
    const uint32_t N = 1u << 22;
    const int      REPS = 20;

    std::vector<uint32_t> idx(N);
    for (uint32_t i = 0; i < N; ++i) idx[i] = i;

    cl_int st = CL_SUCCESS;
    cl_mem ibuf = CL.CreateBuffer(g.ctx, CL_MEM_READ_ONLY,  (size_t)N * 4, nullptr, &st);
    cl_mem cbuf = CL.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, 4, nullptr, &st);
    cl_mem rbuf = CL.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, 256 * 16, nullptr, &st);
    if (st != CL_SUCCESS) { printf("  benchmark: buffer allocation failed\n"); return; }
    CL.EnqueueWriteBuffer(g.queue, ibuf, CL_TRUE, 0, (size_t)N * 4, idx.data(), 0, nullptr, nullptr);
    uint32_t zero = 0;
    CL.EnqueueWriteBuffer(g.queue, cbuf, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr);

    printf("\n  pure GPU kernel throughput (one batch, no sieve, no transfers)\n");
    printf("  %-12s %-6s %-10s %-12s %s\n", "exponent", "pbits", "candidate", "kernel", "rate");

    for (uint64_t p : { (uint64_t)1277, (uint64_t)82589933, (uint64_t)1152921504606847601ull }) {
        int pbits = 0; { uint64_t t = p; while (t) { ++pbits; t >>= 1; } }
        uint64_t twop = 2 * p;

      for (int which = 0; which < 3; ++which) {
        cl_kernel K   = (which == 0) ? g.kernel : (which == 1) ? g.kernel96 : g.kernel96x2;
        size_t    WG  = (which == 0) ? g.wg_size : (which == 1) ? g.wg96 : g.wg96x2;
        const char* nm = (which == 0) ? "128-bit/64" : (which == 1) ? "96-bit/32" : "96-bit x2";
        const uint32_t per_item = (which == 2) ? 2u : 1u;
        if (cfg.workgroup > 0) WG = (size_t)cfg.workgroup;

        for (int level : { 70, 90 }) {
            // k such that q = 2kp+1 lands at about 2^level
            U128 kbase, rem;
            u128_divmod(u128_shl(u128_from(1), (unsigned)level), u128_from(twop), kbase, rem);
            if (u128_is_zero(kbase)) kbase = u128_from(1);
            uint64_t base_lo = kbase.lo, base_hi = kbase.hi, step = 4;
            uint32_t cap = 256, n = N;
            int argi = 0;
            CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &ibuf);
            CL.SetKernelArg(K, argi++,sizeof(uint32_t), &n);
            CL.SetKernelArg(K, argi++,sizeof(uint64_t), &base_lo);
            CL.SetKernelArg(K, argi++,sizeof(uint64_t), &base_hi);
            CL.SetKernelArg(K, argi++,sizeof(uint64_t), &step);
            CL.SetKernelArg(K, argi++,sizeof(uint64_t), &twop);
            CL.SetKernelArg(K, argi++,sizeof(uint64_t), &p);
            CL.SetKernelArg(K, argi++,sizeof(int32_t),  &pbits);
            CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &cbuf);
            CL.SetKernelArg(K, argi++,sizeof(cl_mem),   &rbuf);
            CL.SetKernelArg(K, argi++,sizeof(uint32_t), &cap);

            size_t items  = (N + per_item - 1) / per_item;
            size_t local  = WG;
            size_t global = ((items + local - 1) / local) * local;
            for (int w = 0; w < 3; ++w)              // warm up / clock ramp
                CL.EnqueueNDRangeKernel(g.queue, K, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            CL.Finish(g.queue);

            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r)
                CL.EnqueueNDRangeKernel(g.queue, K, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            CL.Finish(g.queue);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            printf("  %-12llu %-6d 2^%-8d %-12s %.0f M/s\n",
                   (unsigned long long)p, pbits, level, nm, (double)N * REPS / el / 1e6);
        }
      }
    }
    CL.ReleaseMemObject(ibuf);
    CL.ReleaseMemObject(cbuf);
    CL.ReleaseMemObject(rbuf);
    (void)cfg;
}

// ===========================================================================
//  Section 7 -- Self test
// ===========================================================================

struct TestCase {
    uint64_t p;
    const char* fmin;
    const char* fmax;
    const char* expected;    // space separated
};

static bool selftest(Gpu& g, Config cfg)
{
    // Known factorisations; the last one exercises the full 128-bit path with a
    // 76-bit factor of M_193.
    static const TestCase cases[] = {
        { 11,  "3", "1000",              "23 89" },
        { 23,  "3", "1000",              "47" },
        { 29,  "3", "10000",             "233 1103 2089" },
        { 37,  "3", "1000",              "223" },
        { 43,  "3", "100000",            "431 9719" },
        { 101, "3", "2^44",              "7432339208719" },
        { 67,  "3", "2^40",              "193707721 761838257287" },
        { 193, "61654440233248340616000", "61654440233248340617000", "61654440233248340616559" },
    };

    cfg.stop_on_factor = false;
    cfg.checkpoint     = false;
    int fails = 0;

    // Run every case through EVERY kernel that can represent it.  Left to itself
    // the auto rule would send each case down one path and leave the others
    // untested -- and the narrow kernels are exact only below their own bound,
    // so "it worked on the 96-bit path" says nothing about the 72-bit one.
    // A case whose factors exceed a width is skipped for that width, not failed.
    for (int width : { 64, 72, 96, 128 }) {
      cfg.arithmetic = width;
      printf("  -- %s --\n",
             width == 64  ? "64-bit arithmetic (two 32-bit limbs)"
             : width == 72 ? "72-bit arithmetic (24-bit limbs; lazy below 2^70)"
             : width == 96 ? "96-bit arithmetic (32-bit limbs)"
                           : "128-bit arithmetic (64-bit limbs)");
      for (const TestCase& tc : cases) {
        if (width == 64 || width == 72) {        // only cases that fit the width
            U128 hi; std::string e2;
            parse_u128(tc.fmax, hi, e2);
            if (u128_gt(hi, u128_shl(u128_from(1), (unsigned)width))) {
                printf("  [skip] p=%-5llu range %-26s -> above 2^%d\n",
                       (unsigned long long)tc.p, tc.fmin, width);
                continue;
            }
        }
        U128 fmin, fmax;
        std::string perr;
        parse_u128(tc.fmin, fmin, perr);
        parse_u128(tc.fmax, fmax, perr);

        std::vector<U128> got;
        RunStats stats;
        std::string err;
        if (!run_range(g, cfg, tc.p, fmin, fmax, false, got, stats, err)) {
            printf("  [FAIL] p=%llu : %s\n", (unsigned long long)tc.p, err.c_str());
            ++fails;
            continue;
        }
        std::string gs;
        for (size_t i = 0; i < got.size(); ++i) { if (i) gs += " "; gs += u128_to_dec(got[i]); }
        bool ok = (gs == tc.expected);
        // every reported factor must also pass the independent CPU check
        for (const U128& q : got)
            if (!u128_eq(h_pow2_mod(tc.p, q), u128_from(1))) ok = false;
        printf("  [%s] p=%-5llu range %-26s -> %s\n", ok ? " ok " : "FAIL",
               (unsigned long long)tc.p, tc.fmin, gs.empty() ? "(none)" : gs.c_str());
        if (!ok) { printf("         expected: %s\n", tc.expected); ++fails; }
      }
    }
    printf("\n  %s\n", fails == 0 ? "all self tests passed" : "SELF TESTS FAILED");
    return fails == 0;
}

// ===========================================================================
//  Section 8 -- main
// ===========================================================================

static void banner()
{
    printf("=======================================================================\n");
    printf(" mersenne_tf0.9  --  GPU trial factoring of M_p = 2^p - 1  (exact 128-bit)\n");
    printf("=======================================================================\n");
}

static int run_main(int argc, char** argv)
{
    std::string cfg_path = "config.txt";
    bool do_selftest = false, do_list = false, do_bench = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)      cfg_path = argv[++i];
        else if (a == "--selftest")               do_selftest = true;
        else if (a == "--bench")                  do_bench = true;
        else if (a == "--nogpu")                  g_nogpu = true;
        else if (a == "--noxfer")                 g_noxfer = true;
        else if (a == "--list-devices")           do_list = true;
        else if (a == "--help" || a == "-h") {
            banner();
            printf("\nusage: mersenne_tf0.9 [--config FILE] [--selftest] [--list-devices]\n\n"
                   "  Reads the exponent and the factor bounds from a plain text config\n"
                   "  file (default config.txt).  See README.md.\n");
            return 0;
        } else {
            printf("unknown argument '%s' (try --help)\n", a.c_str());
            return 2;
        }
    }

    banner();

    std::string err;
    if (!load_opencl(err)) { printf("\nERROR: %s\n", err.c_str()); return 1; }

    if (do_list) { printf("\nOpenCL GPU devices:\n"); list_devices(); return 0; }

    // The self test also honours the device / tuning settings from the file,
    // but unlike a real run it carries on with defaults if the file is bad.
    Config cfg;
    if (!load_config(cfg_path, cfg, err)) {
        printf("\nERROR: %s\n", err.c_str());
        if (!do_selftest) return 1;
        cfg = Config();
    }
    // The job lives in worktodo.txt.  The self test brings its own cases, so it
    // runs without one; every other mode needs it.
    if (!do_selftest && !do_bench) {
        if (!load_worktodo(cfg.worktodo_file, cfg, err)) {
            printf("\nERROR: %s\n", err.c_str());
            return 1;
        }
    } else if (cfg.exponent == 0) {
        cfg.exponent = 1277;                  // harmless default for --bench
    }
    g_pause_mode = cfg.pause_mode;
    if (cfg.segment_size < 4096) cfg.segment_size = 4096;

    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

    Gpu gpu;
    if (!gpu_init(gpu, cfg.platform, cfg.device, err)) {
        printf("\nERROR: %s\n\nAvailable devices:\n", err.c_str());
        list_devices();
        return 1;
    }
    printf("\n  GPU        : %s (%u compute units, work group %u)\n",
           gpu.name.c_str(), gpu.cus, (unsigned)gpu.wg_size);

    if (do_bench) {
        bench_device(gpu, cfg);
        gpu_free(gpu);
        return 0;
    }

    if (do_selftest) {
        printf("\n  running self tests (each result is re-verified on the CPU)\n\n");
        bool ok = selftest(gpu, cfg);
        gpu_free(gpu);
        return ok ? 0 : 1;
    }

    // ---- validate the job -------------------------------------------------
    const uint64_t p = cfg.exponent;
    if (p < 3) {
        printf("\nERROR: exponent must be an odd prime >= 3 (M_2 = 3 is prime and has no factors to find).\n");
        gpu_free(gpu); return 1;
    }
    if (p >= (1ull << 62)) {
        printf("\nERROR: exponent too large (needs 2p to fit in 64 bits).\n");
        gpu_free(gpu); return 1;
    }
    if (!h_is_prime_u64(p)) {
        printf("\nERROR: exponent %llu is not prime.\n"
               "       For composite n, 2^n-1 is composite and its factors do NOT all have\n"
               "       the form 2kn+1, so this search would be unsound.  Factor 2^a-1 for a\n"
               "       prime divisor a of n instead.\n", (unsigned long long)p);
        gpu_free(gpu); return 1;
    }
    U128 LIMIT = u128_shl(u128_from(1), 127);
    if (u128_ge(cfg.factor_max, LIMIT)) {
        printf("\n  note: factor_max clamped to 2^127-1 (the 128-bit arithmetic limit).\n");
        cfg.factor_max = u128_sub(LIMIT, u128_from(1));
    }
    if (u128_gt(cfg.factor_min, cfg.factor_max)) {
        printf("\nERROR: factor_min > factor_max.\n");
        gpu_free(gpu); return 1;
    }

    printf("  exponent p : %llu   (M_p = 2^%llu - 1, %llu bits, prime exponent)\n",
           (unsigned long long)p, (unsigned long long)p, (unsigned long long)p);
    printf("  worktodo   : %s\n", cfg.worktodo_file.c_str());
    printf("  factor range: %s .. %s\n",
           u128_to_pow2(cfg.factor_min).c_str(), u128_to_pow2(cfg.factor_max).c_str());
    printf("               (%d-bit .. %d-bit candidates)\n",
           u128_bitlen(cfg.factor_min), u128_bitlen(cfg.factor_max));
    {
        Wheel wh = build_wheel(p);
        printf("  candidates : q = 2kp+1, q = +/-1 (mod 8), k on a wheel mod %llu\n",
               (unsigned long long)wh.W);
        printf("               %llu of %llu residues survive (%.1f%% of all k)\n",
               (unsigned long long)wh.classes.size(), (unsigned long long)wh.W,
               100.0 * wh.classes.size() / (double)wh.W);
    }
    printf("  pre-sieve  : primes below %u%s\n", resolve_sieve_limit(cfg, p),
           cfg.sieve_primes ? "" : "  (auto)");
    printf("  arithmetic : narrowest exact kernel, chosen per bit level\n");
    printf("\n");

    std::vector<U128> factors;
    RunStats stats;
    bool ok = run_range(gpu, cfg, p, cfg.factor_min, cfg.factor_max, true, factors, stats, err);
    if (!ok) { printf("\nERROR: %s\n", err.c_str()); gpu_free(gpu); return 1; }

    // ---- report -----------------------------------------------------------
    printf("  scanned    : %llu candidates in the allowed classes\n", (unsigned long long)stats.k_scanned);
    printf("  pre-sieved : %llu removed before the GPU (%.1f%%)\n",
           (unsigned long long)(stats.k_scanned - stats.gpu_tested),
           stats.k_scanned ? 100.0 * (1.0 - (double)stats.gpu_tested / (double)stats.k_scanned) : 0.0);
    printf("  GPU tested : %llu candidates in %.2f s (%.2f M/s)\n",
           (unsigned long long)stats.gpu_tested, stats.seconds,
           stats.seconds > 0 ? stats.gpu_tested / stats.seconds / 1e6 : 0.0);
    printf("\n");

    // Each factor was already announced in full (and logged) the moment it was
    // found; this is just the recap.
    if (factors.empty()) {
        // An interrupted run cleared only part of the range, so it cannot claim
        // the range is clean -- and neither does the record written below.
        printf(stats.interrupted
                   ? "  RESULT: no factor of M_%llu in the part of the range searched.\n"
                   : "  RESULT: no factor of M_%llu in the range.\n",
               (unsigned long long)p);
    } else {
        printf("  RESULT: %d factor(s) of M_%llu found, logged to %s:\n",
               (int)factors.size(), (unsigned long long)p, cfg.results_file.c_str());
        for (const U128& q : factors)
            printf("          %s  (%d bits)\n", u128_to_dec(q).c_str(), u128_bitlen(q));
    }
    report_run(p, cfg, factors, stats);

    gpu_free(gpu);
    return 0;
}

// ---------------------------------------------------------------------------
//  Keep the window open when the program was started by double-clicking it,
//  so the results do not vanish.  When it was launched from an existing shell
//  (or with redirected output) there is nothing to keep open, so it exits.
// ---------------------------------------------------------------------------
static bool launched_by_doubleclick()
{
    DWORD pids[4];
    DWORD n = GetConsoleProcessList(pids, 4);
    return n <= 1 && _isatty(_fileno(stdin)) != 0;
}

int main(int argc, char** argv)
{
    int rc = run_main(argc, argv);

    bool hold = (g_pause_mode == PAUSE_ALWAYS) ||
                (g_pause_mode == PAUSE_AUTO && launched_by_doubleclick());
    if (hold) {
        printf("\n  Press Enter to close this window . . . ");
        fflush(stdout);
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {}
    }
    return rc;
}
