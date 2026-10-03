// tf_kernel.cl.h -- OpenCL C source for the Mersenne trial-factoring kernel.
//
// Embedded as a raw string literal so the executable is fully self-contained.
//
// ALL ARITHMETIC IS EXACT INTEGER ARITHMETIC.  There is not a single float or
// double in this file.  Candidate factors are full 128-bit integers held as two
// 64-bit limbs; products are formed with mul_hi() so no bits are ever lost.
//
// Test performed per candidate q:      2^p mod q == 1  ?
// Method: Montgomery modular arithmetic, radix 2^64, n = 2 limbs (CIOS).
//   * mont(x) denotes x*R mod q with R = 2^128.
//   * Because the base is exactly 2, the "multiply by base" step of binary
//     exponentiation is a modular DOUBLING, not a multiplication:
//         2 * mont(y) = 2*y*R = mont(2y)
//     so the whole powmod is  bitlen(p)  squarings plus popcount(p) doublings.
//   * mont(1) = R mod q, obtained by 128 modular doublings of 1.
//   * 2^p mod q == 1  <=>  mont(2^p) == mont(1), so no final conversion is
//     needed -- we compare in the Montgomery domain.
//
// That describes the 128-bit kernel, the reference.  Below 2^88 -- every level
// a real search visits -- the work is done by Section H instead: three 28- or
// 30-bit limbs in a program compiled for the one exponent being tested.

#pragma once

// ===========================================================================
//  Shared helpers -- the first piece of both programs.
//
//  The main program (Sections A, B, D: the 128- and 96-bit kernels and the
//  sieve) and each per-exponent program (Sections E, H) are separate source
//  lists handed to clCreateProgramWithSource; both need these.
// ===========================================================================
static const char* TF_KERNEL_SOURCE_SHARED = R"CLC(
// add with carry-out
inline ulong addc(ulong a, ulong b, ulong *carry)
{
    ulong s = a + b;
    *carry = (s < a) ? 1UL : 0UL;
    return s;
}

// Build the candidate q = 2kp+1 for one index.  k and q are formed with 64-bit
// arithmetic: a couple of multiplies once per candidate, against pbits squarings
// in the loop, so it is not worth splitting into limbs.
inline void build_q(uint iidx, ulong base_lo, ulong base_hi, ulong step,
                    ulong twop, ulong *q_lo_out, ulong *q_hi_out)
{
    ulong i = (ulong)iidx, c;
    ulong klo = i * step, khi = mul_hi(i, step);
    ulong k_lo = addc(base_lo, klo, &c);
    ulong k_hi = base_hi + khi + c;

    ulong qlo = k_lo * twop;
    ulong qhi = mul_hi(k_lo, twop) + k_hi * twop;
    ulong q_lo = addc(qlo, 1UL, &c);
    ulong q_hi = qhi + c;

    *q_lo_out = q_lo;
    *q_hi_out = q_hi;
}

// A 64-bit survivor count carried in two 32-bit words.  OpenCL 1.2 has
// atomic_add on global uint as core, while 64-bit atomics need
// cl_khr_int64_base_atomics, which is not universally present -- the same
// reason the bitmap is uint32.  One word is not enough: a phase at the GIMPS
// wavefront holds far more than 2^32 survivors (2^73..2^74 at p = 9147253 has
// 2.1e10 in a single CLASS), and the wrap showed up as the progress line's
// "sieved" sawtoothing between 80% and 100% -- and, less visibly, as a wrong
// candidate count in the run's own report.
inline void phase_add(__global uint *phase_total, uint n)
{
    uint old = atomic_add(&phase_total[0], n);
    if (old + n < old) atomic_inc(&phase_total[1]);   // unsigned wrap = carry
}

// One huge sieve prime's strikes on a segment's bitmap, from its first strike
// offs[i] to the end; then offs[i] becomes its first strike in the segment
// after, should that one follow on (see sieve_offsets).  A global atomic per
// strike, since two primes can land in the same word.  Here rather than in the
// sieve section because two kernels walk: sieve_mark_huge, and the fused TF
// kernel, which carries the next segment's walk -- see mersenne_tfL_gs.
inline void huge_walk(__global uint *bits, __global const uint *s_tab,
                      __global uint *offs, uint i, uint len)
{
    uint s = s_tab[i], j = offs[i];
    for (; j < len; j += s)
        atomic_or(&bits[j >> 5], 1u << (j & 31u));
    offs[i] = j - len;
}
)CLC";

static const char* TF_KERNEL_SOURCE_A = R"CLC(
// ---------------------------------------------------------------------------
// 128-bit unsigned integer, little-endian limbs
// ---------------------------------------------------------------------------
typedef struct { ulong lo, hi; } u128;

inline u128 u128_sub(u128 a, u128 b)
{
    u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - ((a.lo < b.lo) ? 1UL : 0UL);
    return r;
}

inline int u128_ge(u128 a, u128 b)
{
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
}

inline int u128_eq(u128 a, u128 b)
{
    return (a.hi == b.hi) && (a.lo == b.lo);
}

// t + a*b + C  ->  returns low 64 bits, leaves high 64 bits in *C.
// Exact: t + a*b + C <= (2^64-1) + (2^64-1)^2 + (2^64-1) = 2^128 - 1, so the
// 128-bit result always fits and the carry chain below cannot overflow.
inline ulong mac(ulong t, ulong a, ulong b, ulong *C)
{
    ulong hi = mul_hi(a, b);
    ulong lo = a * b;
    ulong c1, c2;
    lo = addc(lo, t,  &c1);
    lo = addc(lo, *C, &c2);
    *C = hi + c1 + c2;
    return lo;
}

// ---------------------------------------------------------------------------
// Montgomery multiplication, CIOS, 2 limbs.   returns a*b*R^-1 mod m
// mp = -m^-1 mod 2^64.   Requires m odd.  m < 2^127 guarantees t2 == 0.
// ---------------------------------------------------------------------------
inline u128 mont_mul(u128 a, u128 b, u128 m, ulong mp)
{
    ulong A[2]; A[0] = a.lo; A[1] = a.hi;
    ulong B[2]; B[0] = b.lo; B[1] = b.hi;
    ulong M[2]; M[0] = m.lo; M[1] = m.hi;

    ulong t0 = 0, t1 = 0, t2 = 0, t3 = 0;

    for (int i = 0; i < 2; ++i) {
        ulong C = 0, cc, mu, dummy;

        // t += a * b[i]
        t0 = mac(t0, A[0], B[i], &C);
        t1 = mac(t1, A[1], B[i], &C);
        t2 = addc(t2, C, &cc);
        t3 = cc;

        // t = (t + mu*m) / 2^64,  mu chosen so the low limb cancels
        mu = t0 * mp;
        C  = 0;
        dummy = mac(t0, mu, M[0], &C);   // low limb is 0 by construction
        t0    = mac(t1, mu, M[1], &C);
        t1    = addc(t2, C, &cc);
        t2    = t3 + cc;
    }

    u128 r; r.lo = t0; r.hi = t1;
    if (t2 != 0 || u128_ge(r, m))
        r = u128_sub(r, m);
    return r;
}

// x = 2x mod m.   Requires x < m < 2^127 so the shift cannot lose the top bit.
inline u128 mod_dbl(u128 x, u128 m)
{
    x.hi = (x.hi << 1) | (x.lo >> 63);
    x.lo <<= 1;
    if (u128_ge(x, m))
        x = u128_sub(x, m);
    return x;
}

// -m^-1 mod 2^64 by Newton iteration (x <- x*(2 - m*x) doubles correct bits).
// Seed x = m0 is correct mod 2^3 for odd m0; 6 iterations reach >= 64 bits.
inline ulong neg_inv64(ulong m0)
{
    ulong x = m0;
    for (int i = 0; i < 6; ++i)
        x = x * (2UL - m0 * x);
    return 0UL - x;
}

// R mod m  with R = 2^128.  Exact, no division.
//
// Naively this is 128 modular doublings of 1.  Instead start at 2^b where
// b = bitlen(m): since 2^(b-1) <= m < 2^b we have 2^b mod m = 2^b - m, a single
// subtraction, leaving only 128-b doublings.  For the 70-90 bit candidates that
// real searches use this removes more than half of the work.
inline u128 r_mod(u128 m)
{
    int b = m.hi ? (128 - clz(m.hi)) : (64 - clz(m.lo));   // bit length, <= 127

    u128 p2;                                              // p2 = 2^b
    if (b >= 64) { p2.hi = 1UL << (b - 64); p2.lo = 0; }
    else         { p2.hi = 0; p2.lo = 1UL << b; }

    u128 r = u128_sub(p2, m);                             // = 2^b mod m
    for (int i = b; i < 128; ++i)
        r = mod_dbl(r, m);
    return r;
}

// ---------------------------------------------------------------------------
// Kernel.
//
// Each work item takes one surviving candidate index, rebuilds
//      k = base_k + step * idx        (128-bit)
//      q = k * twop + 1               (128-bit, twop = 2p)
// and tests whether q divides 2^p - 1.
//
// The host guarantees q < 2^127 and that every q here already survived the
// small-prime pre-sieve, so no candidate with a tiny factor wastes a thread.
// ---------------------------------------------------------------------------
__kernel void mersenne_tf(
    __global const uint  *idx,          // surviving candidate indices
    __global const uint  *n_buf,        // how many (device-side: the sieve writes it)
    const ulong           base_lo,      // k of index 0 (low limb)
    const ulong           base_hi,      // k of index 0 (high limb)
    const ulong           step,         // k increment per index unit
    const ulong           twop,         // 2*p
    const ulong           pexp,         // p
    const int             pbits,        // bit length of p
    __global uint        *found_count,  // atomic counter
    __global ulong2      *found,        // reported factors
    const uint            found_cap)
{
    uint gid = get_global_id(0);
    uint n = n_buf[0];
    if (gid >= n) return;

    ulong i = (ulong)idx[gid];
    ulong c;

    // k = base_k + step*i
    u128 k;
    ulong mlo = i * step;
    ulong mhi = mul_hi(i, step);
    k.lo = addc(base_lo, mlo, &c);
    k.hi = base_hi + mhi + c;

    // q = k*twop + 1
    u128 q;
    ulong qlo = k.lo * twop;
    ulong qhi = mul_hi(k.lo, twop) + k.hi * twop;
    q.lo = addc(qlo, 1UL, &c);
    q.hi = qhi + c;

    ulong mp   = neg_inv64(q.lo);       // q is odd by construction
    u128  one  = r_mod(q);              // mont(1)
    u128  x    = one;                   // invariant: x == mont(2^e)

    // left-to-right binary exponentiation over the bits of p
    for (int b = pbits - 1; b >= 0; --b) {
        x = mont_mul(x, x, q, mp);              // e -> 2e
        if ((pexp >> b) & 1UL)
            x = mod_dbl(x, q);                  // e -> e+1   (base is 2)
    }

    // x == mont(2^p);  2^p == 1 (mod q)  <=>  x == mont(1)
    if (u128_eq(x, one)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) {
            found[slot] = (ulong2)(q.lo, q.hi);
        }
    }
}

)CLC";

// MSVC limits one string literal to 16380 bytes, so the source is carried in
// parts and handed to clCreateProgramWithSource as an array.
static const char* TF_KERNEL_SOURCE_B = R"CLC(
// ===========================================================================
//  96-bit path -- three 32-bit limbs.
//
//  Identical mathematics to the kernel above, but every multiply is a 32-bit
//  mul_hi/mul_lo pair, which consumer GeForce hardware executes natively.  A
//  64x64->128 multiply is emulated in software there and costs several
//  instructions, so the 128-bit kernel above pays roughly 2x for arithmetic it
//  does not need whenever the candidate fits in 96 bits -- which covers every
//  bit level a real search visits.  This is the same reason every mfaktc kernel
//  is named _mul32 or _mul24.
//
//  The host selects this kernel automatically when factor_max < 2^96.
// ===========================================================================
// uint3 rather than a struct: the (u96)(a,b,c) literal form and .x/.y/.z access
// are built in, and it costs nothing at runtime.
typedef uint3 u96;

inline uint addc32(uint a, uint b, uint *carry)
{
    uint s = a + b;
    *carry = (s < a) ? 1u : 0u;
    return s;
}

inline u96 u96_sub(u96 a, u96 b)
{
    uint br0 = (a.x < b.x) ? 1u : 0u;
    uint x = a.x - b.x;
    uint y = a.y - b.y;
    uint br1 = (a.y < b.y || (br0 && y == 0u)) ? 1u : 0u;
    y -= br0;
    uint z = a.z - b.z - br1;
    return (u96)(x, y, z);
}

inline int u96_ge(u96 a, u96 b)
{
    if (a.z != b.z) return a.z > b.z;
    if (a.y != b.y) return a.y > b.y;
    return a.x >= b.x;
}

inline int u96_eq(u96 a, u96 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

// t + a*b + C  (32-bit): low 32 returned, high 32 left in *C.  Same bound
// argument as the 64-bit version -- the sum never exceeds 2^64 - 1.
inline uint macc(uint t, uint a, uint b, uint *C)
{
    uint hi = mul_hi(a, b);
    uint lo = a * b;
    uint c1, c2;
    lo = addc32(lo, t,  &c1);
    lo = addc32(lo, *C, &c2);
    *C = hi + c1 + c2;
    return lo;
}

// Montgomery multiply mod m, CIOS, 3 limbs.  mp = -m^-1 mod 2^32, m odd.
inline u96 mont_mul96(u96 a, u96 b, u96 m, uint mp)
{
    uint A[3]; A[0] = a.x; A[1] = a.y; A[2] = a.z;
    uint B[3]; B[0] = b.x; B[1] = b.y; B[2] = b.z;
    uint M[3]; M[0] = m.x; M[1] = m.y; M[2] = m.z;

    uint t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    for (int i = 0; i < 3; ++i) {
        uint C = 0, cc, mu, dummy;

        t0 = macc(t0, A[0], B[i], &C);
        t1 = macc(t1, A[1], B[i], &C);
        t2 = macc(t2, A[2], B[i], &C);
        t3 = addc32(t3, C, &cc);
        t4 = cc;

        mu = t0 * mp;
        C  = 0;
        dummy = macc(t0, mu, M[0], &C);      // low limb cancels
        t0    = macc(t1, mu, M[1], &C);
        t1    = macc(t2, mu, M[2], &C);
        t2    = addc32(t3, C, &cc);
        t3    = t4 + cc;
    }

    u96 r = (u96)(t0, t1, t2);
    if (t3 != 0 || u96_ge(r, m)) r = u96_sub(r, m);
    return r;
}

// ---------------------------------------------------------------------------
//  Dedicated Montgomery SQUARING.
//
//  The exponentiation is nothing but x*x, and a square needs fewer products
//  than a general multiply: the off-diagonal terms a_i*a_j (i<j) each occur
//  twice, so compute them once and double the sum.  For three limbs that is
//  3 cross + 3 diagonal = 6 products instead of 9, and with the 9 of the
//  Montgomery reduction the multiply count drops from 18 to 15.
//
//  Separated Operand Scanning: form the full 192-bit square first, then reduce.
// ---------------------------------------------------------------------------

// t[0..5] = a*a exactly; t[6] = 0
//
//  Accumulated in 64-bit columns rather than with hand-rolled 32-bit carries: a
//  32x32 product lands in a ulong whole (one instruction), the carry into the
//  next limb is simply its high half, and ">>= 32" is a register selection.
//  Measured 1.38x against the compare-and-select version it replaces.
//
//  It never wraps: at each step carry (< 2^32) + limb (< 2^32) + product
//  (<= 2^64 - 2^33 + 1) <= 2^64 - 1.
inline void sqr96(u96 a, uint *t)
{
    uint A0 = a.x, A1 = a.y, A2 = a.z;

    // cross terms, one copy each, at limbs 1..4
    ulong p01 = (ulong)A0 * A1;
    ulong p02 = (ulong)A0 * A2;
    ulong p12 = (ulong)A1 * A2;

    ulong acc = p01;
    uint s1 = (uint)acc; acc >>= 32;
    acc += p02;
    uint s2 = (uint)acc; acc >>= 32;
    acc += p12;
    uint s3 = (uint)acc; acc >>= 32;
    uint s4 = (uint)acc;

    // 2S -- fits, since S < 2*b^5
    uint d1 = s1 << 1;
    uint d2 = (s2 << 1) | (s1 >> 31);
    uint d3 = (s3 << 1) | (s2 >> 31);
    uint d4 = (s4 << 1) | (s3 >> 31);
    uint d5 = (s4 >> 31);

    // diagonals: A0^2 spans limbs 0-1, A1^2 limbs 2-3, A2^2 limbs 4-5
    ulong q0 = (ulong)A0 * A0;
    ulong q1 = (ulong)A1 * A1;
    ulong q2 = (ulong)A2 * A2;

    acc  = q0;
    t[0] = (uint)acc; acc >>= 32;
    acc += (ulong)d1;
    t[1] = (uint)acc; acc >>= 32;
    acc += q1 + (ulong)d2;
    t[2] = (uint)acc; acc >>= 32;
    acc += (ulong)d3;
    t[3] = (uint)acc; acc >>= 32;
    acc += q2 + (ulong)d4;
    t[4] = (uint)acc; acc >>= 32;
    acc += (ulong)d5;
    t[5] = (uint)acc; acc >>= 32;
    t[6] = (uint)acc;
}

inline u96 mont_sqr96(u96 a, u96 m, uint mp)
{
    uint t[7];
    sqr96(a, t);

    uint M0 = m.x, M1 = m.y, M2 = m.z;
    for (int i = 0; i < 3; ++i) {
        uint  mu  = t[i] * mp;
        ulong acc = (ulong)t[i] + (ulong)mu * M0;   // low limb cancels
        acc >>= 32;
        acc += (ulong)t[i+1] + (ulong)mu * M1;
        t[i+1] = (uint)acc; acc >>= 32;
        acc += (ulong)t[i+2] + (ulong)mu * M2;
        t[i+2] = (uint)acc; acc >>= 32;
        for (int k = i + 3; k < 7; ++k) {           // fixed length: no divergence
            acc += (ulong)t[k];
            t[k] = (uint)acc; acc >>= 32;
        }
    }
    u96 r = (u96)(t[3], t[4], t[5]);                // < 2m, so one subtraction
    if (t[6] != 0 || u96_ge(r, m)) r = u96_sub(r, m);
    return r;
}

// x = 2x mod m.  The bit shifted out of the top limb is tracked explicitly, so
// this stays correct for every m < 2^96 (2x can reach 2^97).
inline u96 mod_dbl96(u96 x, u96 m)
{
    uint carry = x.z >> 31;
    x.z = (x.z << 1) | (x.y >> 31);
    x.y = (x.y << 1) | (x.x >> 31);
    x.x = x.x << 1;
    if (carry || u96_ge(x, m)) x = u96_sub(x, m);
    return x;
}

inline uint neg_inv32(uint m0)
{
    uint x = m0;                     // correct mod 2^3 for odd m0
    for (int i = 0; i < 5; ++i)      // 3 -> 6 -> 12 -> 24 -> 48 bits
        x = x * (2u - m0 * x);
    return 0u - x;
}

// R mod m with R = 2^96.  b = bitlen(m); 2^b - m is one subtraction (and for
// b == 96 the subtraction from 0 wraps to exactly 2^96 - m), then 96-b doublings.
inline u96 r_mod96(u96 m)
{
    int b;
    if      (m.z) b = 96 - clz(m.z);
    else if (m.y) b = 64 - clz(m.y);
    else          b = 32 - clz(m.x);

    u96 p2 = (u96)(0u, 0u, 0u);
    if (b < 32)       p2.x = 1u << b;
    else if (b < 64)  p2.y = 1u << (b - 32);
    else if (b < 96)  p2.z = 1u << (b - 64);
    // b == 96: p2 stays 0, and 0 - m wraps to 2^96 - m, which is what we want

    u96 r = u96_sub(p2, m);
    for (int i = b; i < 96; ++i)
        r = mod_dbl96(r, m);
    return r;
}

// ---------------------------------------------------------------------------
//  Two candidates per work item.
//
//  A single candidate is one long dependency chain: every squaring needs the
//  previous result, and inside a squaring the carry chain is serial too.  That
//  leaves the multiply pipes waiting on latency rather than short of work.
//  Running two independent candidates in the same thread interleaves two such
//  chains, which the compiler can schedule against each other, at the cost of
//  roughly double the registers.
// ---------------------------------------------------------------------------
__kernel void mersenne_tf96x2(
    __global const uint  *idx,
    __global const uint  *n_buf,
    const ulong           base_lo,
    const ulong           base_hi,
    const ulong           step,
    const ulong           twop,
    const ulong           pexp,
    const int             pbits,
    __global uint        *found_count,
    __global ulong2      *found,
    const uint            found_cap)
{
    uint gid = get_global_id(0);
    uint n = n_buf[0];
    uint nhalf = (n + 1) >> 1;           // "half" is a reserved type name here
    if (gid >= nhalf) return;

    // gid and gid+nhalf rather than 2*gid and 2*gid+1, so neighbouring threads
    // still read neighbouring indices and the loads stay coalesced.
    uint i0 = gid;
    uint i1 = gid + nhalf;
    int  have1 = (i1 < n);
    if (!have1) i1 = i0;                 // harmless duplicate; report is guarded

    ulong q0_lo, q0_hi, q1_lo, q1_hi;
    build_q(idx[i0], base_lo, base_hi, step, twop, &q0_lo, &q0_hi);
    build_q(idx[i1], base_lo, base_hi, step, twop, &q1_lo, &q1_hi);
    u96 q0 = (u96)((uint)q0_lo, (uint)(q0_lo >> 32), (uint)q0_hi);
    u96 q1 = (u96)((uint)q1_lo, (uint)(q1_lo >> 32), (uint)q1_hi);

    uint mp0 = neg_inv32(q0.x), mp1 = neg_inv32(q1.x);
    u96 one0 = r_mod96(q0),     one1 = r_mod96(q1);
    u96 x0 = one0,              x1 = one1;

    for (int b = pbits - 1; b >= 0; --b) {
        x0 = mont_sqr96(x0, q0, mp0);
        x1 = mont_sqr96(x1, q1, mp1);
        if ((pexp >> b) & 1UL) {
            x0 = mod_dbl96(x0, q0);
            x1 = mod_dbl96(x1, q1);
        }
    }

    if (u96_eq(x0, one0)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q0_lo, q0_hi);
    }
    if (have1 && u96_eq(x1, one1)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q1_lo, q1_hi);
    }
}

__kernel void mersenne_tf96(
    __global const uint  *idx,
    __global const uint  *n_buf,
    const ulong           base_lo,
    const ulong           base_hi,
    const ulong           step,
    const ulong           twop,
    const ulong           pexp,
    const int             pbits,
    __global uint        *found_count,
    __global ulong2      *found,
    const uint            found_cap)
{
    uint gid = get_global_id(0);
    uint n = n_buf[0];
    if (gid >= n) return;

    ulong i = (ulong)idx[gid];
    ulong c;

    // k and q are still built with 64-bit arithmetic: that is a couple of
    // multiplies once per candidate, against pbits Montgomery multiplies in the
    // loop below, so it is not worth splitting into limbs.
    ulong klo = i * step, khi = mul_hi(i, step);
    ulong k_lo = addc(base_lo, klo, &c);
    ulong k_hi = base_hi + khi + c;

    ulong qlo = k_lo * twop;
    ulong qhi = mul_hi(k_lo, twop) + k_hi * twop;
    ulong q_lo = addc(qlo, 1UL, &c);
    ulong q_hi = qhi + c;

    u96 q = (u96)((uint)q_lo, (uint)(q_lo >> 32), (uint)q_hi);

    uint mp  = neg_inv32(q.x);          // q odd by construction
    u96  one = r_mod96(q);
    u96  x   = one;

    for (int b = pbits - 1; b >= 0; --b) {
        x = mont_sqr96(x, q, mp);
        if ((pexp >> b) & 1UL)
            x = mod_dbl96(x, q);
    }

    if (u96_eq(x, one)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) {
            found[slot] = (ulong2)(q_lo, q_hi);
        }
    }
}

)CLC";


// ===========================================================================
//  Section D -- the sieve, on the device.
//
//  Up to 1.1 this ran on the CPU: mark a bitmap, bit-scan it, upload four bytes
//  per survivor.  That cost the host ~9 s of the 10 s run and pushed ~79 GB
//  across PCIe per bit level, and the GPU sat 26% below its own kernel ceiling
//  waiting for it.  Here the whole pipeline stays on the device and the TF
//  kernel reads a count the sieve wrote.
//
//  The bitmap is uint32 words, not uint64: OpenCL 1.2 has atomic_or on global
//  uint as core, while 64-bit atomics need cl_khr_int64_base_atomics, which is
//  not universally present.
//
//  A set bit means STRUCK OUT, matching the CPU sieve's `mark`.
// ===========================================================================
static const char* TF_KERNEL_SOURCE_D = R"CLC(

// Exact n mod s without an integer division.
//
// recip = floor(2^32 / s), computed once per prime on the host.  mul_hi gives
// floor(n * recip / 2^32), which under-estimates floor(n/s) by at most one for
// every n < 2^32, so a single conditional subtract makes it exact.  This is
// integer arithmetic throughout -- no float reciprocal, in keeping with the
// rest of the program.
//
// Worth doing because the sieve's inner loop is one modulo per (thread, prime),
// ~6.4e10 of them per bit level, and that was 73% of the sieve's cost.
inline uint mod_recip(uint n, uint s, uint recip)
{
    uint q = mul_hi(n, recip);
    uint r = n - q * s;
    return (r >= s) ? (r - s) : r;
}

// First index in this segment struck by each sieve prime.
//
// From scratch (step == 0) it mirrors the host exactly: b0 = base_k mod s
// (Horner over the two limbs, as u128_mod_u32 does), then
// offs = (k0 - b0) * W^-1 (mod s).  s < 2^28, so every intermediate here is far
// inside 64 bits.  A 64-bit remainder by a runtime divisor is a long software
// sequence on a GPU, so it takes only the two it needs: base_hi is zero unless
// k >= 2^64, and k0 and b0 are both below s, so their difference needs none.
//
// Most segments do not need that.  A class's segments run back to back, so a
// segment usually starts exactly where the previous one ended, and then each
// offset is the previous one moved back by that segment's length: step is that
// length, and the update is one mod_recip.  The host launches this for the
// primes below the huge tier only -- huge_walk leaves each huge prime's next
// offset behind as it finishes, for nothing.  Measured on an RTX 3070 at
// p = 27886007, 2^66..2^67, sieve_primes 5.5M (380000 primes): 0.34 s per bit
// level from scratch every segment, 0.09 s this way.
__kernel void sieve_offsets(
    __global const uint  *s_tab,
    __global const uint  *k0_tab,
    __global const uint  *invW_tab,
    __global const uint  *rc_tab,
    const uint            nprimes,
    const ulong           base_lo,
    const ulong           base_hi,
    const uint            step,       // 0 = from scratch, else the previous segment's length
    __global uint        *offs)
{
    uint i = get_global_id(0);
    if (i >= nprimes) return;

    if (step) {
        uint s = s_tab[i], o = offs[i];
        uint d = mod_recip(step, s, rc_tab[i]);
        offs[i] = (o >= d) ? (o - d) : (o + s - d);
        return;
    }

    ulong s  = (ulong)s_tab[i];
    ulong b0 = base_lo % s;
    if (base_hi) {
        ulong two64 = ((0xFFFFFFFFFFFFFFFFUL % s) + 1UL) % s;   // 2^64 mod s
        b0 = (((base_hi % s) * two64) % s + b0) % s;
    }
    ulong k0 = (ulong)k0_tab[i];
    ulong d  = (k0 >= b0) ? (k0 - b0) : (k0 + s - b0);
    offs[i] = (uint)((d * (ulong)invW_tab[i]) % s);
}

// ---------------------------------------------------------------------------
//  Every prime up to the huge tier, into one LDS tile per group.
//
//  This is the first kernel to touch the bitmap in a segment, so it *builds* each
//  tile rather than loading it, and it resets the compaction counter.
//
//  Primes below 64 (13 of them: the wheel has the rest) hit every 32-bit word,
//  so they are not struck at all: a thread computes whole words of them in
//  registers.  Each prime's strikes in a word are its 64-bit comb of multiples
//  shifted by where its first one falls, and that position moves on by a fixed
//  amount from one of the thread's words to the next, so a word costs a shift,
//  an OR and a compare-subtract per prime -- the divisions are once per
//  (thread, prime).
//  Until 1.4 these primes had a kernel of their own (sieve_mark_small), one
//  thread per word paying a division per (word, prime) and writing the bitmap
//  out for this kernel to read back in: 0.57 s per bit level, against 0.12 s
//  added here (RTX 3070, p = 27886007, 2^66..2^67).
//
//  The rest are struck into the tile, which is then written back, so the bitmap
//  makes one trip out for all of these primes.  Its
//  threads split the *primes* rather than the window, so two threads can hit
//  the same word and the strikes need atomics -- but __local atomics into
//  shared memory, not global ones, which is the entire difference.  The first
//  version of this kernel used one global atomic_or per strike and cost 2.1 s
//  of a 3.3 s sieve for the 4.2e10 strikes above 2048.
//
//  Each item costs one division, to find its first strike:
//
//  * a prime below 2048 is split into (prime, sub-tile) items of
//    TIER_SUB_WORDS, so 67 does not leave one thread striking the whole tile
//    ~1950 times while its neighbours idle.  Consecutive threads take the
//    sub-tiles of one prime, so a warp's loops run the same length.  Until 1.4
//    these primes had a kernel of their own (sieve_mark_tier) in which a thread
//    owned a window of words and paid a division per (window, prime); folding
//    them in here measured 2.35 s -> 2.04 s for the two kernels together
//    (p = 27886007, 2^66..2^67, sieve_primes 1M).  The strikes are what cost,
//    not the divisions -- this saves a pass over the bitmap and a launch.
//  * a larger prime is one item per (prime, tile): near 2048 it strikes a
//    131072-bit tile ~64 times, near the huge threshold once or not at all.
// ---------------------------------------------------------------------------
#define TIER_SUB_WORDS 256u     // 8192 bits; 4096 and 16384 measured ~4% slower
#ifndef LARGE_WG                // from the host, so launch and declaration agree
#define LARGE_WG 256
#endif

inline uint first_strike(uint o, uint s, uint rc, uint start)
{
    if (o >= start) return o;
    uint rem = mod_recip(start - o, s, rc);
    return start + (rem ? (s - rem) : 0u);
}

__kernel __attribute__((reqd_work_group_size(LARGE_WG, 1, 1)))
void sieve_mark_large(
    __global uint        *bits,
    __global const uint  *s_tab,
    __global const uint  *rc_tab,
    __global const uint  *offs,
    const uint            tier_first,       // primes [0, tier_first) are below 64: in registers
    const uint            tier_last,        // [tier_first, tier_last) below 2048: sub-tile items
    const uint            first,            // the rest: one item per prime
    const uint            last,
    const uint            len,
    const uint            tile_words,       // a power of two, >= TIER_SUB_WORDS
    __global uint        *count,            // zeroed here for the compaction
    __local  uint        *tile)
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const uint w0  = get_group_id(0) * tile_words;
    const uint nw  = (len + 31u) >> 5;

    if (get_global_id(0) == 0) count[0] = 0u;

    // A thread's words are lsz apart, so a warp stores consecutive words.  The
    // eight are a compile-time count so that acc stays in registers; a tile of
    // other than 8 * lsz words just takes more passes, or fewer words in one.
    for (uint base = 0; base < tile_words; base += 8u * lsz) {
        uint acc[8];
        for (uint k = 0; k < 8u; ++k) acc[k] = 0u;
        const uint b0 = (w0 + base + lid) * 32u;        // first bit of the first word
        for (uint q = 0; q < tier_first; ++q) {
            uint  s = s_tab[q], rc = rc_tab[q];
            ulong comb = 0;                              // bits 0, s, 2s, ... below 64
            for (uint b = 0; b < 64u; b += s) comb |= 1UL << b;
            uint r = first_strike(offs[q], s, rc, b0) - b0;   // first strike in the word, < s
            uint d = mod_recip(lsz * 32u, s, rc);             // ... and in the next one
            for (uint k = 0; k < 8u; ++k) {
                acc[k] |= (uint)(comb << r);             // strikes r, r + s, ... below 32
                r = (r >= d) ? (r - d) : (r + s - d);
            }
        }
        for (uint k = 0; k < 8u; ++k) {
            uint i = base + lid + k * lsz, w = w0 + i;
            if (i < tile_words) tile[i] = (w < nw) ? acc[k] : 0xFFFFFFFFu;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tstart = w0 * 32u;
    uint tend   = (w0 + tile_words) * 32u;
    if (tend > len) tend = len;

    const uint nsub    = tile_words / TIER_SUB_WORDS;
    const uint sub_log = 31u - clz(nsub);
    const uint nitems  = (tier_last - tier_first) << sub_log;
    for (uint it = lid; it < nitems; it += lsz) {
        uint i  = tier_first + (it >> sub_log);
        uint ss = tstart + (it & (nsub - 1u)) * (TIER_SUB_WORDS * 32u);
        uint se = ss + TIER_SUB_WORDS * 32u;
        if (se > tend) se = tend;
        if (ss >= se) continue;
        uint s = s_tab[i];
        for (uint j = first_strike(offs[i], s, rc_tab[i], ss); j < se; j += s)
            atomic_or(&tile[(j >> 5) - w0], 1u << (j & 31u));
    }

    for (uint i = first + lid; i < last; i += lsz) {
        uint s = s_tab[i];
        for (uint j = first_strike(offs[i], s, rc_tab[i], tstart); j < tend; j += s)
            atomic_or(&tile[(j >> 5) - w0], 1u << (j & 31u));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = lid; i < tile_words; i += lsz) {
        uint w = w0 + i;
        if (w < nw) bits[w] = tile[i];
    }
}

// ---------------------------------------------------------------------------
//  Primes larger than the tile: one thread owns one prime, for the whole segment.
//
//  sieve_mark_large pays one division per (prime, tile) to find where a prime
//  first strikes that tile.  That is a good trade only while a prime strikes the
//  tile several times.  Above the tile size it strikes 0 or 1 times, so the
//  division is nearly all waste -- and at a 32768-bit tile with a 1M bound that
//  described 77694 of the 80897 primes in the tier, 41.4M divisions per segment
//  against a small fraction of that in strikes.
//
//  Here there is no tiling and no division at all: offs[] already holds the
//  prime's first strike in this segment, so the walk is a chain of additions.
//  The strikes go straight to the bitmap, which needs a global atomic_or because
//  two primes can land in the same word.  Those atomics are nearly all this
//  costs: 1.33 s of 1.54 s per bit level at sieve_primes 5.5M (RTX 3070,
//  p = 27886007, 2^66..2^67, measured by dropping them) -- the walk itself is
//  cheap.  They still pay: without these strikes 23% more candidates survive.
//  And they are a cost of the L2, not of the SMs, which the TF kernel keeps busy
//  while leaving the L2 nearly idle -- so where the TF kernel is the fused one,
//  it carries this walk for the next segment and the atomics hide under its
//  arithmetic (see mersenne_tfL_gs).  This kernel then runs only for a phase's
//  first segment, and on the unfused path.
//
//  Must run AFTER sieve_mark_large: that kernel stages tiles through LDS and
//  writes them back with plain stores, which would clobber these bits; and it
//  reads offs[], though never these entries.  The queue is in-order, so enqueue
//  order is enough.
// ---------------------------------------------------------------------------
__kernel void sieve_mark_huge(
    __global uint        *bits,
    __global const uint  *s_tab,
    __global uint        *offs,
    const uint            first,
    const uint            last,
    const uint            len)
{
    uint i = first + get_global_id(0);
    if (i < last) huge_walk(bits, s_tab, offs, i, len);
}

// ---------------------------------------------------------------------------
//  Bitmap -> index list, plus the count the TF kernel will read.
//
//  Output order is deliberately not defined: idx is only ever read as idx[i] by
//  a thread that then works on that value alone, so nothing depends on the
//  order and a prefix sum would be pure cost.  One atomic per non-empty word.
// ---------------------------------------------------------------------------
//  The group claims its output range ONCE.  The obvious version -- every thread
//  doing atomic_add(count, popcount) for its own word -- puts two global atomics
//  on two addresses for every non-empty word, and at 23% survival almost every
//  word is non-empty: about a million atomics per segment onto two cache lines.
//  Here the group prefix-sums the per-word counts in LDS, one thread claims the
//  total, and each thread writes at its own offset inside that claim, which is
//  256x fewer global atomics for one scan over 256 values.
//
//  COMPACT_WG is a compile-time size because the scan array must be, so the host
//  launches this kernel with exactly that local size.
#define COMPACT_WG 256

__kernel __attribute__((reqd_work_group_size(COMPACT_WG, 1, 1)))
void sieve_compact(
    __global const uint  *bits,
    const uint            len,
    __global uint        *idx,
    __global uint        *count,
    __global uint        *phase_total)
{
    __local uint   scan[COMPACT_WG];
    __local uint   gbase;
    __local ushort sidx[COMPACT_WG * 32];   // 32 survivors per thread, worst case

    const uint lid = get_local_id(0);
    const uint w   = get_global_id(0);
    const uint nw  = (len + 31u) >> 5;

    // Threads past the end still take part: every one of them has to reach the
    // barriers below, so this cannot early-return the way the old version did.
    uint base = w << 5, v = 0u;
    if (w < nw) {
        v = ~bits[w];                        // clear bit = survivor
        uint valid = len - base;             // last word: ignore bits past len
        if (valid < 32u) v &= (1u << valid) - 1u;
    }
    const uint pc = popcount(v);

    // inclusive scan (Hillis-Steele): read the whole array, then write it
    scan[lid] = pc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = 1; off < COMPACT_WG; off <<= 1) {
        uint t = (lid >= off) ? scan[lid - off] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] += t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint total = scan[COMPACT_WG - 1];
    const uint excl  = scan[lid] - pc;        // this thread's offset in the claim

    // phase_total lets the host learn the survivor count once per phase instead
    // of reading it back per segment.  That read was blocking on an in-order
    // queue, so it drained the whole pipeline every segment -- measured at about
    // 180 us each, ~1 s per bit level.
    if (lid == 0) {
        gbase = 0u;
        if (total) {
            gbase = atomic_add(count, total);
            phase_add(phase_total, total);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Staged through LDS rather than written straight out.  A thread owns one
    // word, so it holds 0..32 survivors and would write them as a short run at
    // an offset unrelated to its neighbours' -- 32 threads of a warp writing 32
    // scattered short runs, which is the worst case for the coalescer and was
    // measured to be what this kernel spends its time on (its cost tracks the
    // survivor count, not the word count).  So: park them in LDS, then have the
    // threads change roles and copy out with consecutive threads on consecutive
    // addresses.  An index is chunk-relative here and fits in 16 bits, which
    // halves the LDS this needs.
    uint at = excl;
    while (v) {
        uint lsb = v & (0u - v);
        sidx[at++] = (ushort)(lid * 32u + (31u - clz(lsb)));   // ctz is CL 2.0
        v &= v - 1u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint chunk_base = get_group_id(0) * (COMPACT_WG * 32u);
    for (uint t = lid; t < total; t += COMPACT_WG)
        idx[gbase + t] = chunk_base + (uint)sidx[t];
}
)CLC";


// ===========================================================================
//  Section E -- the sieve fused into trial factoring.
//
//  The split pipeline runs sieve_compact to turn the bitmap into a global index
//  list, which the TF kernel then reads back.  That is one extra launch, one
//  write and one read of four bytes per survivor -- about 165 GB per bit level
//  at the shipped depth, on top of a kernel that is otherwise compute-bound.
//
//  Here a work group takes one chunk of the bitmap, compacts it into LDS with
//  the same prefix-sum sieve_compact uses, and immediately tests what it found.
//  The index never reaches global memory at all.  It also sizes itself: the
//  split path cannot know the survivor count without reading it back, so it
//  launches enough threads to cover every candidate and lets four in five exit.
//
//  This section is only the compaction.  It goes into each per-exponent
//  program, and the fused kernel that calls it, mersenne_tfL_gs, is in
//  Section H next to its split twin -- both call the same tfL_one().
// ===========================================================================
static const char* TF_KERNEL_SOURCE_E = R"CLC(

#ifndef TFGS_WG                 // -DTFGS_WG comes from the host, so the
#define TFGS_WG    256          // launch and reqd_work_group_size agree
#endif
#define TFGS_CHUNK (TFGS_WG * 32)

// One group's chunk of the bitmap -> a list of survivors in LDS.  Same prefix
// sum sieve_compact uses; the difference is only that nothing reaches global
// memory.  Every thread must call this -- it has barriers.
//
// Returns the number of survivors, and leaves them in sidx as chunk-relative
// indices (< TFGS_CHUNK, so 16 bits is enough and the array is half the size).
inline uint tfgs_compact(
    __global const uint *bits, const uint len, const uint lid, const uint grp,
    __local uint *scan, __local ushort *sidx, __global uint *phase_total)
{
    const uint w  = grp * TFGS_WG + lid;
    const uint nw = (len + 31u) >> 5;

    // Threads past the end still take part: they have barriers to reach.
    uint v = 0u;
    if (w < nw) {
        v = ~bits[w];                        // clear bit = survivor
        uint valid = len - (w << 5);         // last word: ignore bits past len
        if (valid < 32u) v &= (1u << valid) - 1u;
    }
    const uint pc = popcount(v);

    scan[lid] = pc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = 1; off < TFGS_WG; off <<= 1) {
        uint t = (lid >= off) ? scan[lid - off] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] += t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint nloc = scan[TFGS_WG - 1];

    uint at = scan[lid] - pc;
    while (v) {
        uint lsb = v & (0u - v);
        sidx[at++] = (ushort)(lid * 32u + (31u - clz(lsb)));   // ctz is CL 2.0
        v &= v - 1u;
    }

    // The host still wants the survivor count for the progress line; with no
    // sieve_compact to write it, one atomic per group carries it instead.
    if (lid == 0 && nloc) phase_add(phase_total, nloc);
    barrier(CLK_LOCAL_MEM_FENCE);
    return nloc;
}
)CLC";

// ===========================================================================
//  Section H -- three L-bit limbs, compiled once per exponent.
//
//  This is the trial-factoring kernel for every q below 2^88.  It is built per
//  exponent, with
//
//    -DLB=28|30            limb width; Montgomery radix 2^L, R = 2^(3L)
//    -DMTF_P, -DMTF_PBITS  the exponent, so the bit loop unrolls with every
//                          doubling known at compile time
//    -DMTF_TOP             how many leading bits of p the start value absorbs
//    -DMTF_REDUCE          (optional) reduce after each doubling -- see below
//
//  Values are kept lazily in [0, 2q) with normalised limbs.
//
//  FREE DOUBLING.  "x = 2x mod q" is not done at all: the next squaring is
//  handed 2x instead, as limbs shifted left by one.  That is exact while
//  16q <= R:  x < 2q  ->  2x < 4q  ->  ((2x)^2 + mu q) / R < 16q^2/R + q <= 2q,
//  so the squaring's output is back in [0, 2q) with no compare and no subtract.
//  It holds for q < 2^(3L-4): 2^80 at L = 28, 2^86 at L = 30.  The worst-case
//  column with doubled limbs is 2^59.32 / 2^63.32, inside the 64-bit
//  accumulators; the reduction carry is below 2^31.32 at L = 28, so there it is
//  shifted as one 32-bit word.
//
//  MTF_REDUCE keeps a reducing doubling (shift, subtract 2q when needed) for
//  2^(3L-4) <= q < 2^(3L-2), where free doubling's bound fails but the lazy
//  squaring's (4q <= R) still holds.  Only 2^86..2^88 uses it by default.
//
//  START VALUE.  The leading MTF_TOP bits of p, value e, are not squared
//  through: the loop starts from mont(2^e) = 2^(3L+e) mod q, which pow2_lazy
//  computes directly by long division.  That replaces both those squarings and
//  the old mont(1) = R mod q chain of 3L modular doublings.
//
//  FINAL TEST.  Without mont(1) to compare against, the last value is taken out
//  of Montgomery form (one REDC) and compared with the plain residue a factor
//  must leave.
//
//  Measured against the 1.3 kernels (RTX 3070, M candidates/s): 2^66 3010 ->
//  5097, 2^74 2478 -> 4294, 2^81 2500 -> 3815, 2^87 2519 -> 3093.
// ===========================================================================
static const char* TF_KERNEL_SOURCE_H = R"CLC(

#define LM ((1u << LB) - 1u)

typedef struct { uint x, y, z; } u3;        // three L-bit limbs, little-endian

inline int u3_eq(u3 a, u3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
inline int u3_ge(u3 a, u3 b)
{
    if (a.z != b.z) return a.z > b.z;
    if (a.y != b.y) return a.y > b.y;
    return a.x >= b.x;
}
inline u3 u3_sub(u3 a, u3 b)                // assumes a >= b
{
    uint bx = (a.x < b.x) ? 1u : 0u;
    uint x  = (a.x - b.x) & LM;
    uint by = (a.y < b.y + bx) ? 1u : 0u;
    uint y  = (a.y - b.y - bx) & LM;
    uint z  = (a.z - b.z - by) & LM;
    u3 r; r.x = x; r.y = y; r.z = z;
    return r;
}
inline u3 u3_dbl(u3 a)                      // 2a; a < 2^(3L-1)
{
    u3 r;
    r.z = (a.z << 1) | (a.y >> (LB - 1));
    r.y = ((a.y << 1) | (a.x >> (LB - 1))) & LM;
    r.x = (a.x << 1) & LM;
    return r;
}
inline u3 q_to_u3(ulong q_lo, ulong q_hi)
{
    u3 q;
    q.x = (uint)(q_lo & LM);
    q.y = (uint)((q_lo >> LB) & LM);
    q.z = (uint)(((q_lo >> (2*LB)) | (q_hi << (64 - 2*LB))) & LM);
    return q;
}

// -q^-1 mod 2^L.  (3q) xor 2 is right to 5 bits for odd q; each Newton step
// doubles that, so three reach 40 >= L.
inline uint neg_invL(uint m0)
{
    uint x = (3u * m0) ^ 2u;
    x = x * (2u - m0 * x);
    x = x * (2u - m0 * x);
    x = x * (2u - m0 * x);
    return (0u - x) & LM;
}

#if LB <= 28
#define CARRY(t) ((ulong)(uint)((t) >> LB))
#else
#define CARRY(t) ((t) >> LB)
#endif

// Montgomery square of (a << s), s in {0, 1}: the doubling rides in on the
// shift.  a < 2q, normalised; the result is < 2q, normalised.  The cross terms
// are doubled before the multiply (2a_i < 2^(L+2) <= 2^32) rather than after.
inline u3 msqr(u3 a, u3 m, uint mp, uint s)
{
    uint x0 = a.x << s, x1 = a.y << s, x2 = a.z << s;
    uint d0 = x0 << 1, d1 = x1 << 1;
    ulong T0 = (ulong)x0 * x0;
    ulong T1 = (ulong)d0 * x1;
    ulong T2 = (ulong)d0 * x2 + (ulong)x1 * x1;
    ulong T3 = (ulong)d1 * x2;
    ulong T4 = (ulong)x2 * x2;
    ulong T5 = 0;
    ulong M0 = m.x, M1 = m.y, M2 = m.z;
    for (int i = 0; i < 3; ++i) {
        uint mu = ((uint)T0 * mp) & LM;
        T0 += (ulong)mu * M0;                // low L bits cancel
        T1 += (ulong)mu * M1;
        T2 += (ulong)mu * M2;
        T1 += CARRY(T0);
        T0 = T1; T1 = T2; T2 = T3; T3 = T4; T4 = T5; T5 = 0;
    }
    uint r0 = (uint)T0 & LM;
    T1 += CARRY(T0);
    uint r1 = (uint)T1 & LM;
    uint r2 = (uint)T2 + (uint)(T1 >> LB);  // < 2q < 2^(3L-1): no bits lost
    u3 r; r.x = r0; r.y = r1; r.z = r2;
    return r;
}

// a * R^-1 mod q, for a < 2q normalised: the result is in [0, q].
inline u3 redc(u3 a, u3 m, uint mp)
{
    ulong T0 = a.x, T1 = a.y, T2 = a.z;
    ulong M0 = m.x, M1 = m.y, M2 = m.z;
    for (int i = 0; i < 3; ++i) {
        uint mu = ((uint)T0 * mp) & LM;
        T0 += (ulong)mu * M0;
        T1 += (ulong)mu * M1;
        T2 += (ulong)mu * M2;
        T1 += (T0 >> LB);
        T0 = T1; T1 = T2; T2 = 0;
    }
    uint r0 = (uint)T0 & LM;
    T1 += (T0 >> LB);
    uint r1 = (uint)T1 & LM;
    uint r2 = (uint)T2 + (uint)(T1 >> LB);
    u3 r; r.x = r0; r.y = r1; r.z = r2;
    return r;
}

// 128-bit shift right by n, -128 < n < 128; a negative n shifts left.
inline void sh128(ulong lo, ulong hi, int n, ulong *olo, ulong *ohi)
{
    if (n >= 64)      { *olo = hi >> (n - 64); *ohi = 0; }
    else if (n > 0)   { *olo = (lo >> n) | (hi << (64 - n)); *ohi = hi >> n; }
    else if (n == 0)  { *olo = lo; *ohi = hi; }
    else if (n > -64) { int k = -n; *ohi = (hi << k) | (lo >> (64 - k)); *olo = lo << k; }
    else              { *ohi = lo << (-n - 64); *olo = 0; }
}

// 2^E mod q, lazily: in [0, 2q), as three limbs.  Needs E >= bitlen(q) = b.
//
// Long division of a power of two, POW2_CHUNK bits at a time.  r starts at 2^b
// (< 2q, as q is odd and 2^(b-1) < q).  Each step forms y = r * 2^s and
// subtracts qhat * q, with qhat from the top bits of y and a reciprocal of the
// top 32 bits of q:
//
//   d' = floor(q / 2^(b-32)) + 1                    in (2^31, 2^32]
//   w0 = floor((2^32-1) / (floor(d'/2^16) + 1))     one 32-bit division, < 2^48/d'
//   w1 = w0 2^15 + floor(w0 e0 / 2^33),  e0 = 2^48 - d' w0     (one Newton step)
//        so  2^63/d' (1 - 2^-27.8) - 1 < w1 <= 2^63/d'
//   qhat = floor(y_t w1 / 2^64),  y_t = floor(y / 2^(b-33)) < 2^(34+s)
//
// qhat never exceeds floor(y/q) and falls short of it by less than
// 1 + 2^(s-30) + 2^(s+1-27.8) + 2^-28 < 2 for s <= 26, so r stays in [0, 2q).
#define POW2_CHUNK 26
inline u3 pow2_lazy(ulong q_lo, ulong q_hi, int E)
{
    int b = q_hi ? 128 - (int)clz(q_hi) : 64 - (int)clz(q_lo);
    ulong t_lo, t_hi;
    sh128(q_lo, q_hi, b - 32, &t_lo, &t_hi);
    ulong dp  = t_lo + 1UL;
    uint  d16 = (uint)(dp >> 16) + 1u;
    uint  w0  = 0xFFFFFFFFu / d16;
    ulong e0  = (1UL << 48) - dp * (ulong)w0;
    ulong w1  = ((ulong)w0 << 15) + (((ulong)w0 * e0) >> 33);
    ulong r_lo, r_hi;
    if (b >= 64) { r_lo = 0; r_hi = 1UL << (b - 64); } else { r_lo = 1UL << b; r_hi = 0; }
    for (int D = E - b; D > 0; ) {
        int s = D < POW2_CHUNK ? D : POW2_CHUNK;
        D -= s;
        ulong y_lo = r_lo << s, y_hi = (r_hi << s) | (r_lo >> (64 - s));
        ulong yt, yt_hi;
        sh128(y_lo, y_hi, b - 33, &yt, &yt_hi);     // y_t < 2^60: yt_hi == 0
        ulong lo = (ulong)(uint)yt * w1;
        ulong qh = (((yt >> 32) * w1) + (lo >> 32)) >> 32;
        ulong pl = qh * q_lo, ph = mul_hi(qh, q_lo) + qh * q_hi;
        r_lo = y_lo - pl;
        r_hi = y_hi - ph - (y_lo < pl ? 1UL : 0UL);
    }
    return q_to_u3(r_lo, r_hi);
}

// The whole test for one candidate index: 1 if q = 2kp+1 divides 2^p - 1.
inline int tfL_one(uint c, const ulong base_lo, const ulong base_hi,
                   const ulong step, const ulong twop, ulong *q_lo, ulong *q_hi)
{
    build_q(c, base_lo, base_hi, step, twop, q_lo, q_hi);
    u3   q  = q_to_u3(*q_lo, *q_hi);
    uint mp = neg_invL(q.x);

    const int top = MTF_PBITS - MTF_TOP;    // bits left for the loop, >= 1
    u3 x = pow2_lazy(*q_lo, *q_hi, 3*LB + (int)(MTF_P >> top));
    u3 want;

#ifdef MTF_REDUCE
    u3 q2 = u3_dbl(q);                      // 2q < 2^(3L-1)
    #pragma unroll
    for (int b = top - 1; b >= 0; --b) {
        x = msqr(x, q, mp, 0u);
        if ((MTF_P >> b) & 1) {
            x = u3_dbl(x);                  // < 4q <= R: still three limbs
            if (u3_ge(x, q2)) x = u3_sub(x, q2);
        }
    }
    // x = mont(2^p): a factor iff its plain value is 1
    want.x = 1u; want.y = 0u; want.z = 0u;
#else
    uint s = 0u;
    #pragma unroll
    for (int b = top - 1; b >= 0; --b) {
        x = msqr(x, q, mp, s);
        s = (uint)((MTF_P >> b) & 1);
    }
    // p is odd, so its last doubling is still pending: x = mont(2^(p-1)), and
    // a factor iff 2^(p-1) = 2^-1 = (q+1)/2 (mod q)
    want.x = ((q.x >> 1) | ((q.y & 1u) << (LB - 1))) + 1u;  // q odd: no carry
    want.y =  (q.y >> 1) | ((q.z & 1u) << (LB - 1));
    want.z =   q.z >> 1;
#endif
    return u3_eq(redc(x, q, mp), want);
}

// pexp and pbits are unused -- p is compiled in -- but kept so every TF kernel
// takes the same arguments.
__kernel void mersenne_tfL(
    __global const uint  *idx,
    __global const uint  *n_buf,
    const ulong           base_lo,
    const ulong           base_hi,
    const ulong           step,
    const ulong           twop,
    const ulong           pexp,
    const int             pbits,
    __global uint        *found_count,
    __global ulong2      *found,
    const uint            found_cap)
{
    uint gid = get_global_id(0);
    if (gid >= n_buf[0]) return;
    ulong q_lo, q_hi;
    if (tfL_one(idx[gid], base_lo, base_hi, step, twop, &q_lo, &q_hi)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q_lo, q_hi);
    }
}

// The sieve fused in (Section E): one candidate per thread per pass over what
// the group compacted.  Two or four per thread, as the 1.3 kernels paired
// them, measured within 1% of one in the split form: this kernel is
// issue-bound, not latency-bound.
//
// It also does the NEXT segment's huge-prime walk (huge_walk), a share per
// group, after its own candidates.  That walk is global atomics, which load the
// L2 and leave the SMs idle, and this kernel is the reverse, so here they
// overlap: +0.10 to +0.23 s per bit level on this kernel (four interleaved
// pairs) against 1.63 s for the same walk as sieve_mark_huge (RTX 3070,
// p = 27886007, 2^66..2^67, sieve_primes 5.5M).  At the start of the kernel it
// cost +0.64 s, and between the compaction and the tests +1.4 s.  The next segment's bitmap is another
// slot's, so nothing here reads what the walk writes.  hfirst == hlast: none.
__kernel __attribute__((reqd_work_group_size(TFGS_WG, 1, 1)))
void mersenne_tfL_gs(
    __global const uint  *bits,
    const uint            len,
    const ulong           base_lo,
    const ulong           base_hi,
    const ulong           step,
    const ulong           twop,
    const ulong           pexp,
    const int             pbits,
    __global uint        *found_count,
    __global ulong2      *found,
    const uint            found_cap,
    __global uint        *phase_total,
    __global uint        *hbits,            // the next segment's bitmap
    __global const uint  *hs_tab,           // sieve primes
    __global uint        *hoffs,            // and their offsets in that segment
    const uint            hfirst,           // the huge ones: [hfirst, hlast)
    const uint            hlast,
    const uint            hlen)             // that segment's length
{
    __local ushort sidx[TFGS_CHUNK];
    __local uint   scan[TFGS_WG];

    const uint lid = get_local_id(0), grp = get_group_id(0);
    const uint nloc = tfgs_compact(bits, len, lid, grp, scan, sidx, phase_total);

    const uint chunk_base = grp * TFGS_CHUNK;
    for (uint t = lid; t < nloc; t += TFGS_WG) {
        ulong q_lo, q_hi;
        if (tfL_one(chunk_base + (uint)sidx[t], base_lo, base_hi, step, twop, &q_lo, &q_hi)) {
            uint slot = atomic_inc(found_count);
            if (slot < found_cap) found[slot] = (ulong2)(q_lo, q_hi);
        }
    }

    const uint ng  = get_num_groups(0);
    const uint per = (hlast - hfirst + ng - 1) / ng;
    const uint h0  = hfirst + grp * per;
    const uint h1  = min(h0 + per, hlast);
    for (uint i = h0 + lid; i < h1; i += TFGS_WG)
        huge_walk(hbits, hs_tab, hoffs, i, hlen);
}
)CLC";
