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

#pragma once

static const char* TF_KERNEL_SOURCE_A = R"CLC(
// ---------------------------------------------------------------------------
// 128-bit unsigned integer, little-endian limbs
// ---------------------------------------------------------------------------
typedef struct { ulong lo, hi; } u128;

// add with carry-out
inline ulong addc(ulong a, ulong b, ulong *carry)
{
    ulong s = a + b;
    *carry = (s < a) ? 1UL : 0UL;
    return s;
}

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
    const uint            n,            // how many
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

// Build the candidate q = 2kp+1 for one index.  k and q are formed with 64-bit
// arithmetic: a couple of multiplies once per candidate, against pbits squarings
// in the loop, so it is not worth splitting into limbs.
inline u96 build_q(uint iidx, ulong base_lo, ulong base_hi, ulong step,
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
    return (u96)((uint)q_lo, (uint)(q_lo >> 32), (uint)q_hi);
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
    const uint            n,
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
    uint nhalf = (n + 1) >> 1;           // "half" is a reserved type name here
    if (gid >= nhalf) return;

    // gid and gid+nhalf rather than 2*gid and 2*gid+1, so neighbouring threads
    // still read neighbouring indices and the loads stay coalesced.
    uint i0 = gid;
    uint i1 = gid + nhalf;
    int  have1 = (i1 < n);
    if (!have1) i1 = i0;                 // harmless duplicate; report is guarded

    ulong q0_lo, q0_hi, q1_lo, q1_hi;
    u96 q0 = build_q(idx[i0], base_lo, base_hi, step, twop, &q0_lo, &q0_hi);
    u96 q1 = build_q(idx[i1], base_lo, base_hi, step, twop, &q1_lo, &q1_hi);

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
    const uint            n,
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

// MSVC caps a string literal at 16380 bytes, so the source is handed to the
// driver in pieces; they are concatenated by clCreateProgramWithSource.
static const char* TF_KERNEL_SOURCE_C = R"CLC(
// ===========================================================================
//  THE 64-BIT PATH -- two 32-bit limbs.
//
//  Below 2^64 a candidate needs only two limbs, and the product count halves:
//  3 for the square (a0^2, a0a1, a1^2) plus 4 for the reduction, against 15 for
//  the three-limb paths.  Product count is what sets the ceiling -- an ablation
//  with the accumulation stripped out measured 2.08x for 15 products -- so this
//  is the largest single win available, and it applies to every level below
//  2^64, which is half of a "from 2^40" job.
//
//  Montgomery, radix 2^32, R = 2^64, mp = -q^-1 mod 2^32.
//  VALID ONLY FOR q < 2^64.
// ===========================================================================
typedef struct { uint x, y; } u64v;        // two 32-bit limbs, little-endian

inline int u64v_ge(u64v a, u64v b)
{
    if (a.y != b.y) return a.y > b.y;
    return a.x >= b.x;
}

inline int u64v_eq(u64v a, u64v b) { return a.x == b.x && a.y == b.y; }

inline u64v u64v_sub(u64v a, u64v b)       // assumes a >= b
{
    uint bx = (a.x < b.x) ? 1u : 0u;
    u64v r; r.x = a.x - b.x; r.y = a.y - b.y - bx;
    return r;
}

// x = 2x mod m.  2x can reach 2^65, so the bit above limb 1 is explicit.
inline u64v mod_dbl64(u64v a, u64v m)
{
    uint over = a.y >> 31;
    u64v r; r.y = (a.y << 1) | (a.x >> 31); r.x = a.x << 1;
    if (over || u64v_ge(r, m)) r = u64v_sub(r, m);
    return r;
}

inline u64v r_mod64(u64v m)                // 2^64 mod m
{
    int b = m.y ? (32 + (32 - clz(m.y))) : (32 - clz(m.x));
    u64v p2; p2.x = 0; p2.y = 0;
    if (b < 32)      p2.x = 1u << b;
    else if (b < 64) p2.y = 1u << (b - 32);
    // b == 64: p2 stays 0 and 0 - m wraps to exactly 2^64 - m
    u64v r;
    if (b == 64) { uint bx = (0u < m.x) ? 1u : 0u; r.x = 0u - m.x; r.y = 0u - m.y - bx; }
    else         r = u64v_sub(p2, m);
    for (int i = b; i < 64; ++i) r = mod_dbl64(r, m);
    return r;
}

// Montgomery squaring, radix 2^32, two limbs.  64-bit accumulators as in the
// 96-bit path: carry (< 2^32) + limb (< 2^32) + product (<= 2^64 - 2^33 + 1)
// never exceeds 2^64 - 1.
inline u64v mont_sqr64(u64v a, u64v m, uint mp)
{
    uint A0 = a.x, A1 = a.y;
    uint M0 = m.x, M1 = m.y;

    ulong p01 = (ulong)A0 * A1;            // one copy; doubled below
    ulong q0  = (ulong)A0 * A0;
    ulong q1  = (ulong)A1 * A1;

    uint s0 = (uint)p01, s1 = (uint)(p01 >> 32);
    uint d0 = s0 << 1;                     // 2*p01, kept exact across the shift
    uint d1 = (s1 << 1) | (s0 >> 31);
    uint d2 = s1 >> 31;

    uint t[5];
    ulong acc = q0;
    t[0] = (uint)acc; acc >>= 32;
    acc += (ulong)d0;
    t[1] = (uint)acc; acc >>= 32;
    acc += q1 + (ulong)d1;
    t[2] = (uint)acc; acc >>= 32;
    acc += (ulong)d2;
    t[3] = (uint)acc; acc >>= 32;
    t[4] = (uint)acc;

    for (int i = 0; i < 2; ++i) {
        uint  mu = t[i] * mp;
        ulong ac = (ulong)t[i] + (ulong)mu * M0;    // low limb cancels
        ac >>= 32;
        ac += (ulong)t[i+1] + (ulong)mu * M1;
        t[i+1] = (uint)ac; ac >>= 32;
        for (int k = i + 2; k < 5; ++k) {           // fixed length: no divergence
            ac += (ulong)t[k];
            t[k] = (uint)ac; ac >>= 32;
        }
    }

    u64v r; r.x = t[2]; r.y = t[3];                 // < 2m, so one subtraction
    if (t[4] != 0 || u64v_ge(r, m)) r = u64v_sub(r, m);
    return r;
}

__kernel void mersenne_tf64(
    __global const uint  *idx,
    const uint            n,
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
    if (gid >= n) return;

    ulong q_lo, q_hi;
    build_q(idx[gid], base_lo, base_hi, step, twop, &q_lo, &q_hi);

    u64v q; q.x = (uint)q_lo; q.y = (uint)(q_lo >> 32);

    uint mp  = neg_inv32(q.x);              // q odd by construction
    u64v one = r_mod64(q);
    u64v x   = one;

    for (int b = pbits - 1; b >= 0; --b) {
        x = mont_sqr64(x, q, mp);
        if ((pexp >> b) & 1UL)
            x = mod_dbl64(x, q);
    }

    if (u64v_eq(x, one)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q_lo, q_hi);
    }
}

// ===========================================================================
//  THE 72-BIT PATH -- three 24-bit limbs.
//
//  The 96-bit kernel sizes its arithmetic to the container, not to the
//  candidate.  Below 2^72 three 24-bit limbs are enough, and that changes the
//  economics: a 24x24 product is 48 bits, so a 64-bit column holds several of
//  them plus carries with no masking and no normalisation.  Only the bottom
//  column is touched per reduction round.  Same fifteen products as the 96-bit
//  path, measured 1.46x faster -- the accumulation around them is the cost, not
//  the multiplies.
//
//  Montgomery, radix 2^24, R = 2^72, mp = -q^-1 mod 2^24.
//  VALID ONLY FOR q < 2^72.  The host selects per bit level and never straddles.
// ===========================================================================
typedef struct { uint x, y, z; } u72;      // three 24-bit limbs, little-endian

#define M24 0x00FFFFFFu

inline int u72_ge(u72 a, u72 b)
{
    if (a.z != b.z) return a.z > b.z;
    if (a.y != b.y) return a.y > b.y;
    return a.x >= b.x;
}

inline int u72_eq(u72 a, u72 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

inline u72 u72_sub(u72 a, u72 b)            // assumes a >= b
{
    uint bx = (a.x < b.x) ? 1u : 0u;
    uint x  = (a.x - b.x) & M24;
    uint by = (a.y < b.y + bx) ? 1u : 0u;
    uint y  = (a.y - b.y - bx) & M24;
    uint z  = (a.z - b.z - by) & M24;
    u72 r; r.x = x; r.y = y; r.z = z;
    return r;
}

inline uint neg_inv24(uint m0)              // -m^-1 mod 2^24, m odd
{
    uint x = m0;
    for (int i = 0; i < 5; ++i) x = x * (2u - m0 * x);   // inverse mod 2^32
    return (0u - x) & M24;
}

// x = 2x mod m, for x < m.
inline u72 mod_dbl72(u72 a, u72 m)
{
    uint z = (a.z << 1) | (a.y >> 23);
    uint y = ((a.y << 1) | (a.x >> 23)) & M24;
    uint x = (a.x << 1) & M24;
    uint over = z >> 24;
    z &= M24;
    u72 r; r.x = x; r.y = y; r.z = z;
    if (over || u72_ge(r, m)) r = u72_sub(r, m);
    return r;
}

inline u72 r_mod72(u72 m)                   // 2^72 mod m
{
    int b;
    if      (m.z) b = 48 + (32 - clz(m.z));
    else if (m.y) b = 24 + (32 - clz(m.y));
    else          b =      (32 - clz(m.x));

    u72 r;
    if (b == 72) {                          // 0 - m wraps to 2^72 - m
        uint bx = (0u < m.x) ? 1u : 0u;
        uint x  = (0u - m.x) & M24;
        uint by = (0u < m.y + bx) ? 1u : 0u;
        uint y  = (0u - m.y - bx) & M24;
        uint z  = (0u - m.z - by) & M24;
        r.x = x; r.y = y; r.z = z;
    } else {
        u72 p2; p2.x = 0; p2.y = 0; p2.z = 0;
        if      (b < 24) p2.x = 1u << b;
        else if (b < 48) p2.y = 1u << (b - 24);
        else             p2.z = 1u << (b - 48);
        r = u72_sub(p2, m);
    }
    for (int i = b; i < 72; ++i) r = mod_dbl72(r, m);
    return r;
}

// Montgomery squaring, radix 2^24.  Columns stay unnormalised in 64-bit
// accumulators: each starts below 3*2^48, every round adds mu*M_j < 2^48 plus a
// carry below 2^26, so after three rounds a column is still below 2^51.
//
// eager = 1 reduces the result to [0, m) as the 96-bit path does.
// eager = 0 leaves it in [0, 2m), which is legitimate only when 4m <= R, i.e.
// m < 2^70 -- the caller guarantees that by bit level.
inline u72 mont_sqr72(u72 a, u72 m, uint mp, int eager)
{
    ulong X0 = a.x, X1 = a.y, X2 = a.z;
    ulong M0 = m.x, M1 = m.y, M2 = m.z;

    ulong T0 = X0 * X0;
    ulong T1 = (X0 * X1) << 1;
    ulong T2 = ((X0 * X2) << 1) + X1 * X1;
    ulong T3 = (X1 * X2) << 1;
    ulong T4 = X2 * X2;
    ulong T5 = 0;

    for (int i = 0; i < 3; ++i) {
        uint mu = ((uint)T0 * mp) & M24;
        T0 += (ulong)mu * M0;                // low 24 bits cancel
        T1 += (ulong)mu * M1;
        T2 += (ulong)mu * M2;
        T1 += (T0 >> 24);                    // the only carry that moves
        T0 = T1; T1 = T2; T2 = T3; T3 = T4; T4 = T5; T5 = 0;
    }

    ulong acc = T0;
    uint r0 = (uint)acc & M24; acc >>= 24;
    acc += T1;
    uint r1 = (uint)acc & M24; acc >>= 24;
    acc += T2;
    uint r2 = (uint)acc & M24; acc >>= 24;

    u72 r; r.x = r0; r.y = r1; r.z = r2;
    if (eager) {
        if ((uint)acc != 0u || u72_ge(r, m)) r = u72_sub(r, m);
    }
    // lazy: the result is < 2m < 2^71, so limb z is < 2^23 and acc is 0 --
    // nothing is dropped by leaving it unreduced here.
    return r;
}

inline u72 q_to_u72(ulong q_lo, ulong q_hi)
{
    u72 q;
    q.x = (uint)(q_lo & M24);
    q.y = (uint)((q_lo >> 24) & M24);
    q.z = (uint)(((q_lo >> 48) | (q_hi << 16)) & M24);
    return q;
}

__kernel void mersenne_tf72(
    __global const uint  *idx,
    const uint            n,
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
    if (gid >= n) return;

    ulong q_lo, q_hi;
    build_q(idx[gid], base_lo, base_hi, step, twop, &q_lo, &q_hi);
    u72 q = q_to_u72(q_lo, q_hi);

    uint mp  = neg_inv24(q.x);
    u72  one = r_mod72(q);
    u72  x   = one;

    for (int b = pbits - 1; b >= 0; --b) {
        x = mont_sqr72(x, q, mp, 1);
        if ((pexp >> b) & 1UL)
            x = mod_dbl72(x, q);
    }

    if (u72_eq(x, one)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q_lo, q_hi);
    }
}

// ---------------------------------------------------------------------------
//  Lazy variant.  VALID ONLY FOR q < 2^70.
//
//  Every value is kept in [0, 2m) instead of [0, m), which removes the
//  conditional subtract from the squaring -- the hot path, run pbits times.
//  The bound holds because 4m <= R = 2^72 when m < 2^70:
//    * squaring: inputs < 2m and 4m <= R  =>  output < 2m;
//    * doubling: input < 2m  =>  2x < 4m, and one subtraction of 2m brings it
//      back under 2m;
//    * representation: 2m < 2^71, so limb z stays below 2^23 and never
//      overflows its 24 bits;
//    * the final compare normalises both sides to [0, m) first.
// ---------------------------------------------------------------------------
__kernel void mersenne_tf72L(
    __global const uint  *idx,
    const uint            n,
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
    if (gid >= n) return;

    ulong q_lo, q_hi;
    build_q(idx[gid], base_lo, base_hi, step, twop, &q_lo, &q_hi);
    u72 q = q_to_u72(q_lo, q_hi);

    // 2q, for the doubling step's single correction
    u72 q2;
    {
        uint z = (q.z << 1) | (q.y >> 23);
        uint y = ((q.y << 1) | (q.x >> 23)) & M24;
        uint x = (q.x << 1) & M24;
        q2.x = x; q2.y = y; q2.z = z;        // q < 2^70 so 2q < 2^71: z fits
    }

    uint mp  = neg_inv24(q.x);
    u72  one = r_mod72(q);                   // in [0, q)
    u72  x   = one;

    for (int b = pbits - 1; b >= 0; --b) {
        x = mont_sqr72(x, q, mp, 0);         // stays in [0, 2q)
        if ((pexp >> b) & 1UL) {
            uint z = (x.z << 1) | (x.y >> 23);
            uint y = ((x.y << 1) | (x.x >> 23)) & M24;
            uint w = (x.x << 1) & M24;
            x.x = w; x.y = y; x.z = z;       // 2x < 4q < 2^72: z fits in 24 bits
            if (u72_ge(x, q2)) x = u72_sub(x, q2);
        }
    }

    if (u72_ge(x, q)) x = u72_sub(x, q);     // normalise both sides once
    if (u72_ge(one, q)) one = u72_sub(one, q);

    if (u72_eq(x, one)) {
        uint slot = atomic_inc(found_count);
        if (slot < found_cap) found[slot] = (ulong2)(q_lo, q_hi);
    }
}
)CLC";
