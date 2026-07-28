# mersenne_tf0.9 — GPU trial factoring of Mersenne numbers

Finds every **prime factor of `M_p = 2^p - 1`** inside a range of candidate values you
choose (e.g. everything below `2^90`), using your GPU, with **exact integer arithmetic** —
no floating point, no rounding, anywhere in the number theory.

`M_p` itself is never constructed. It has `p` bits, and `p` may be in the hundreds of
millions; the program only ever works modulo the candidate, which is at most 128 bits.

---

## Requirements

| | |
|---|---|
| GPU | Any OpenCL 1.2 GPU — NVIDIA, AMD or Intel. Verified on an RTX 3070. |
| Runtime | `OpenCL.dll`, which **ships with your GPU driver**. Nothing to install. |
| Build | Visual Studio 2019/2022 with "Desktop development with C++". |

No CUDA Toolkit, no OpenCL SDK, no headers and no import libraries are needed: the
program resolves the OpenCL entry points from the driver at run time.

## Build

```bash
build.bat
```

Produces `mersenne_tf0.9.exe`. The build script finds Visual Studio on its own.

## Run

```bash
mersenne_tf0.9.exe
```

| command | what it does |
|---|---|
| `mersenne_tf0.9.exe` | runs the job described in `config.txt` |
| `mersenne_tf0.9.exe --config myjob.txt` | uses a different config file |
| `mersenne_tf0.9.exe --selftest` | checks every kernel against known factorisations |
| `mersenne_tf0.9.exe --list-devices` | lists OpenCL GPUs with their platform/device indices |

Performance diagnostics, used to produce the numbers further down:

| command | what it measures |
|---|---|
| `mersenne_tf0.9.exe --bench` | the kernel alone — no sieve, no transfers |
| `mersenne_tf0.9.exe --nogpu` | the host pipeline alone — never submits to the GPU |
| `mersenne_tf0.9.exe --noxfer` | kernels without the upload, pricing the transfer |

While it runs, a single line is rewritten in place:

```
  2^61..2^62  15.48%  6953609162 tested (1069 M/s)  cls 129-160/960  sieved 79.3%  elapsed 7s  ETA 5s  job 47m59s
```

The line leads with the bit level being scanned — the bottom one reads `<2^40` — and
**the percentage is that level's**, not the whole job's, because the level is the unit of
work that finishes and gets logged. `ETA` follows it: time left on this level, with `job`
giving the whole configured range. Resuming picks the percentage up where it left off
(a resume at class 176 of 960 starts at 18.7%), rather than restarting at zero.

`cls` is the wheel class within the level, and level plus class is the resume point a
checkpoint records. Where a phase groups several classes it reads `cls 129-160/960` as
above; on a big level it is a single `cls 454/960`.

The line is rewritten in place, so it must fit the window — one character too many and the
console wraps it, the `\r` returns to the start of the *wrapped* row, and every update
strands the previous one above it. It is built to the console's actual width, dropping the
least important fields when there is no room, and never re-widens mid-run (durations change
length as a run goes on, and re-widening on the short ones would make it flicker):

| console | line |
|---|---|
| 120 | full, as above (113 chars) |
| 100 | `2^60..2^61  92.75%  5806976062 tested (1054 M/s)  cls 833-896/960  ETA 0s  job 48m41s` |
| 80 | `2^60..2^61  92.60%  1054 M/s  cls 833-896/960  ETA 0s` |
| 45 | `2^60..2^61  90.56%  1043 M/s  ETA 0s` |

`ETA` extrapolates from the search space covered so far in **this** run — on a resumed
run the checkpointed progress counts towards the percentage but not towards the rate,
so neither figure is distorted by work done in an earlier session. Durations read as
`45s`, `12m34s`, `1h23m`, `3d04h`.

---

## The input file

Everything is in `config.txt`. The three lines that matter:

```ini
exponent   = 1277        # the p in M_p = 2^p - 1.  Must be prime.
factor_min = 3           # lower bound of the factor search
factor_max = 2^90        # upper bound
```

Numbers can be written as plain decimal (`1180591620717411303424`), with separators
(`1_180_591_620_717_411_303_424`), or as powers of two with an optional offset:
`2^90`, `2^90-1`, `2^70+2^60`. That applies to every numeric key, not just the bounds —
`segment_size = 2^22` and `sieve_primes = 1_000_000` are both fine.

Values are validated as the file is read. A number that does not parse, or one outside
the range its key accepts, stops the program with the line number:

```
ERROR: line 55: sieve_primes: not a number: '50O000'
ERROR: line 100: segment_size: value out of range (4096 .. 268435456)
ERROR: line 68: checkpoint: expected 1/0, true/false, yes/no or on/off (got 'maybe')
```

The alternative is worse than it sounds: `strtoul("50O000")` is 50 and `strtoul("2^22")`
is 2, so without the check a typo runs happily with settings nobody chose. Each key's
range is documented next to it in `config.txt`.

The remaining keys are tuning and output:

| key | meaning |
|---|---|
| `sieve_primes` | pre-factoring bound. `auto` picks it from the exponent size; 200000 measured best here — see Tuning |
| `segment_size` | candidates per sieve segment / GPU batch (default `4194304`) |
| `threads` | CPU sieve threads, `0` = auto |
| `platform`, `device` | which GPU (see `--list-devices`), `-1` = auto |
| `stop_on_factor` | `1` to stop at the first factor instead of scanning the whole range |
| `results_file` | run log: one line per factor, one end-of-run line per run (written even when nothing is found), and one line per bit level cleared |
| `pause_on_exit` | `auto` waits for Enter only when the exe was double-clicked; `always` / `never` to force it |
| `checkpoint` | `1` to save progress and resume automatically (default) |
| `checkpoint_file` | `auto` = `checkpoint_<exponent>.txt` |
| `checkpoint_seconds` | minimum interval between checkpoint writes (default 30) |
| `arithmetic` | `auto` picks the narrowest exact kernel per bit level; force `64` / `72` / `96` / `128` to compare |
| `workgroup` | GPU work-group size, `0` = ask the driver (flat 64–1024 here) |
| `gpu_slots` | batches in flight on the device, 2–8 (default 3; avoid 2) |

`config.txt` documents every key inline, with the measurements behind each default.

---

## What it guarantees

**Every prime factor of `M_p` in `[factor_min, factor_max]` is found.**

Composite divisors are deliberately *not* reported. If `q = s·t` divides `M_p`, then `s`
and `t` each divide `M_p` on their own and each is reported in its own right, so nothing
is lost — you get the prime factorisation of `M_p` restricted to your range.

Each hit is printed **the moment it is found** — the host polls the GPU every 64
batches, well under a tenth of a second, rather than reporting at the end of the run —
with everything needed to check it independently:

```
  *** FACTOR FOUND ***   7432339208719
      k         = 36793758459   (q = 2kp+1: yes)
      q mod 8   = 7
      size      = 43 bits
      2^p mod q = 1 : VERIFIED  (recomputed on the CPU)
      q is      : prime
      logged to : results.txt
```

Scanning then continues, and the end of the run recaps everything found. Set
`stop_on_factor = 1` to stop at the first one instead.

Every run also appends one end-of-run line, whether or not it found anything — a clean
range is the usual result and is worth recording, since "searched, nothing there" and
"never searched" are otherwise indistinguishable.

On top of that, **the search runs one bit level at a time** and logs each level the moment
it is cleared, which is how trial factoring is normally quoted ("M_p is cleared to 2^70").
A run that is interrupted, or even killed outright, keeps every level it finished:

```
2026-07-28 10:12:40  p=8249309  level=3..2^40  status=complete  factors=0  candidates=13847
2026-07-28 10:12:41  p=8249309  level=2^40..2^50  status=complete  factors=0  candidates=14166323
2026-07-28 10:12:43  p=8249309  q=313603386094415369  k=19007857876  bits=59  verified  prime
2026-07-28 10:12:43  p=8249309  level=2^50..2^60  status=complete  factors=1  candidates=14506315114
2026-07-28 10:12:46  p=8249309  level=2^60..2^61  status=complete  factors=0  candidates=14520495283
2026-07-28 10:12:52  p=8249309  level=2^61..2^62  status=complete  factors=0  candidates=29040990567
2026-07-28 10:13:04  p=8249309  level=2^62..2^63  status=complete  factors=0  candidates=58081981135
2026-07-28 10:13:05  p=8249309  range=1..2^70  status=interrupted  factors=1  scanned=121609147984  tested=25118888980  time=24.90s
```

The levels are everything below `2^40`, then `2^40..2^50`, `2^50..2^60`, and every power of
two above that. Ascending order means the cheap levels clear in seconds — the six above
took 25 s of a 51-minute job, because each level holds as many candidates as everything
below it put together.

They are split in `k` rather than in `q`, so the pieces partition the scan exactly and
their `candidates` add up to the run line's `scanned` — no level is double counted at a
seam. A level the range stops inside is reported at the bound actually reached
(`level=2^69..~2^69.58`), never at the power of two above it, and levels holding no
candidate at all are not printed.

`status=interrupted` on the run line marks a run stopped with Ctrl-C. It says nothing
about the levels: those are complete or absent, never partial.

The `VERIFIED` line means the CPU independently recomputed `2^p mod q` with a separate
implementation and got 1. In Python you can confirm any hit with:

```python
pow(2, 101, 7432339208719) == 1
```

---

## Checkpoints

A run is a loop over the bit levels, and inside each level a loop over the wheel classes —
960 of them for any exponent other than 3, 5, 7 or 11. Each phase of classes ends with a
full GPU drain, so at that moment every class below the current index is provably
finished. That is the checkpoint boundary. `checkpoint_<exponent>.txt` records the level,
how many of its classes are done, the statistics so far, and any factors found; it is
rewritten at most every `checkpoint_seconds`, at every level boundary, and deleted when
the range completes.

Re-run with the same config and it resumes automatically:

```
  resuming from checkpoint_8249309.txt: level 2^63..2^64, 36 of 960 classes done
               (6 of 13 levels already cleared)
```

A checkpoint is only accepted if the exponent, both bounds and the wheel all match, so
editing the range in `config.txt` starts a clean run rather than silently skipping work —
and it says so rather than restarting in silence. Checkpoints are versioned: files written
before the level-by-level scan order count classes across the whole range instead of within
a level, so they are rejected rather than misread.
Ctrl-C (or closing the window) drains the GPU, saves and exits — in well under a second.
The interrupt lands in the middle of a class, so the checkpoint names the last class that
actually *finished*: the partial one is re-done on resume rather than recorded as
complete, which is what keeps an interrupted run from stepping over untested candidates.
Factors are appended to the results file the instant they are found, so they survive even
an unclean kill.

Verified by killing a run mid-flight with `Stop-Process -Force` and resuming: the
resumed run reported the same factor and scanned exactly 36,193,259,186 candidates —
identical to an uninterrupted run.

---

## The mathematics

For **prime** `p`, every prime factor `q` of `2^p − 1` obeys two constraints:

**1. `q = 2kp + 1`.** From `2^p ≡ 1 (mod q)`, the order of 2 mod `q` divides the prime `p`
and is not 1, so it is exactly `p`. By Fermat, `p | q−1`; and `q` is odd, so `2p | q−1`.

**2. `q ≡ ±1 (mod 8)`.** `p` is odd, so `2 ≡ (2^((p+1)/2))² (mod q)`, making 2 a quadratic
residue mod `q`. By the second supplement to quadratic reciprocity this happens exactly
when `q ≡ ±1 (mod 8)`.

**Combined**, writing `q = 2kp+1`:

| | allowed `k mod 4` |
|---|---|
| `p ≡ 1 (mod 4)` | `k ≡ 0` or `3` |
| `p ≡ 3 (mod 4)` | `k ≡ 0` or `1` |

So half of all `k` are discarded before any work is done. In practice the program goes
further and folds this rule into a larger wheel — see *Pre-factoring, part 1* below —
and prints the wheel it is using at startup.

Two related facts the program relies on or enforces:

* `gcd(2^a − 1, 2^b − 1) = 2^gcd(a,b) − 1`, so for prime `p` the factors of `M_p` are not
  shared with any other `M_q`.
* If `p` is composite, `2^p − 1` is composite **and rule 1 fails**, so the search would be
  unsound. The program refuses to run on a composite exponent and says why.

---

## Pre-factoring, part 1: the wheel

Testing a candidate on the GPU costs a full modular exponentiation, so the cheapest
candidate is one that is never enumerated. Work modulo **W = 4 · 3 · 5 · 7 · 11 = 4620**:

* the factor 4 carries the `q ≡ ±1 (mod 8)` rule — 2 of every 4 residues survive;
* each prime `s ∈ {3,5,7,11}` kills exactly one residue, the `k ≡ −(2p)⁻¹ (mod s)`
  that makes `s | q`.

**960 of the 4620 residues survive** — 20.8% of all `k`, against 50% for the mod-4 rule
on its own. The CPU therefore touches 2.4x fewer `k`, and the four densest strike-out
patterns (`1/3 + 1/5 + 1/7 + 1/11 = 0.77` marks per `k`) never enter the sieve at all.
These are mfaktc's 4620 "classes", and they are also the unit of checkpointing.

A prime that divides `2p` cannot divide `q = 2kp+1`, so it constrains nothing and is
left out of `W`; that only arises for `p ∈ {3,5,7,11}`, which the self test covers.

## Pre-factoring, part 2: the sieve

The wheel handles 3, 5, 7 and 11. Every larger prime is handled by a multithreaded
segmented sieve, which strikes out each surviving `k` whose candidate is divisible by a
small prime:

```
s | 2kp + 1   <=>   k ≡ −(2p)⁻¹  (mod s)
```

For each small prime `s`, that is one modular inverse (precomputed once) and then a plain
arithmetic progression to strike out — the same structure as a sieve of Eratosthenes,
except the progression is over `k` rather than over the candidates themselves. The segment
is a **bitmap** so a 4M-candidate segment occupies 512 KB.

That bitmap is swept in **32 KB blocks** rather than end to end. A prime whose stride
exceeds a cache line otherwise misses on nearly every hit — measured 5.5 cycles per
strike against the 1–2 a bitmap sieve should cost. Running every prime through one
L1-sized block before moving on, with the strike positions carried in a sequential
array, cut the host floor of a 419 G-candidate job from **52.1 s to 44.5 s**.

With `sieve_primes = 200000` the sieve removes about **78% of the `k` the wheel let
through** — that is the "sieved" figure on the progress line, and its denominator is the
wheel classes, not all `k`. Multiplying the two stages together, **4.3% of all `k` reach
the GPU: one candidate in 23.**

Primes only earn their place: a prime that would strike fewer than about one position in
eight of a segment costs more in setup than it saves in GPU time, so the sieve skips it.
That cap also stops a small range paying for a large prime table.

The sieve would also strike out a candidate that *equals* one of the sieve primes, so
every sieve prime falling inside the requested range is tested directly on the CPU
instead. That is why raising `sieve_primes` can never hide a factor.

---

## Exact arithmetic

The property you asked for — no rounding error, like Python's big integers — is
structural here, not something checked after the fact:

* Candidates are full integers — three 32-bit limbs below `2^96`, two 64-bit limbs above.
* Products are formed with `mul_hi` (device) and `__umulh` (host), so the high half of
  every product is kept. No product is ever truncated.
* Reduction is **Montgomery CIOS**, which is exact integer arithmetic — no division, no
  reciprocal, no float. The upper bound `t + a·b + C ≤ 2^128 − 1` is respected at every
  step of the carry chain.
* `R mod q` is obtained by shifting and conditional subtraction, again exactly.
* The exponentiation needs only **squarings and modular doublings**: because the base is
  exactly 2, `2·mont(y) = mont(2y)`. That is `bitlen(p)` squarings — 27 of them for
  `p ≈ 8×10^7`, not `p` of them.

Some GPU factoring programs use floating point deliberately (48-bit mantissa tricks) and
must then bound the error. This one has no error to bound.

**Limits:** candidates up to `2^127 − 1`; exponent `p` prime and below `2^62`.

---

## Verification performed

| check | result |
|---|---|
| `--selftest`: 8 known factorisations incl. the 76-bit factor of `M_193`, run through **both** kernels | pass |
| Exhaustive diff against Python `pow(2,p,q)` for 15 exponents, all `k` up to 1.2M, both kernel widths, neither the wheel nor the sieve assumed | identical factor sets — no misses, no false hits |
| Structural check: every odd `q < 300000` scanned for `p = 11,23,29,89` | every prime divisor satisfied `q = 2kp+1` and `q ≡ ±1 (mod 8)` |
| Wheel edge cases `p ∈ {3,5,7,11,13}` against Python | match — including `p = 3`, where the wheel legitimately excludes the genuine factor 7 and the direct small-prime check catches it |
| 61-bit exponents with 65-, 72- and 72-bit factors (constructed and confirmed in Python) | all three found |
| Checkpoint: run hard-killed with `Stop-Process -Force`, then resumed | same factor, and exactly 36,193,259,186 candidates scanned — identical to an uninterrupted run |
| `M_1277` to `2^42` | no factor — consistent with it being the smallest exponent with none known |

The 65-bit case (`q = 18446744073709574033`, just above `2^64`) is included on purpose: it
lands on the limb boundary.

Every factor is re-derived on the CPU before it is reported, by a separate implementation
from the one the GPU runs — so a bug in the kernel cannot produce a false positive.

**In real use.** A search of `M_82589959` found three factors, each confirmed afterwards
with Python's arbitrary-precision `pow(2, p, q)`:

| factor | k | bits |
|---|---|---|
| 23785908193 | 144 | 35 |
| 3306241238689 | 20016 | 42 |
| 227414995493408542159 | 1376771451681 | 68 |

---

## Performance and tuning

All figures below: RTX 3070 + i7-12700KF, `p = 82589933`, candidates at the `2^70` level,
runs of ~10 s unless stated.

### How to read a throughput number

Two things distort measurements of this program, and both bit me while tuning it:

**Short runs understate.** The GPU needs seconds to reach its boost clocks and the batch
pipeline has to fill. The same range, measured over different window lengths:

| run length | 0.4 s | 1.4 s | 5.2 s | 20.5 s |
|---|---|---|---|---|
| measured | 787 M/s | 901 M/s | 915 M/s | **931 M/s** |

**Sessions differ more than runs do.** Six back-to-back runs of an identical job varied
by only ±1% (878–899 M/s), but the *same* job measured hours apart, with the same idle
desktop, has ranged from 845 to 1042 M/s. Treat any single figure as ±15% and compare
only numbers taken back to back.

**Rate scales with the exponent**, roughly inversely with `bitlen(p)`, since that is the
number of squarings per candidate:

| exponent | bits | kernel rate |
|---|---|---|
| 1277 | 11 | ~2400 M/s |
| 82589933 | 27 | ~1160 M/s |
| ~2^61 | 61 | ~510 M/s |

### Where the time goes

Three diagnostics isolate the stages, all measured back to back:

* `--bench` — the kernel alone: one batch uploaded once, launched repeatedly.
* `--nogpu` — the whole sieve/pack pipeline, never submitting.
* `--noxfer` — kernels launched but the upload skipped, pricing the transfer.

| stage | rate |
|---|---|
| host sieve + pack (`--nogpu`) | 1759 M/s |
| kernel alone, 128-bit / 64-bit limbs (`--bench`) | 833 M/s |
| kernel alone, 96-bit / 32-bit limbs (`--bench`) | **1160 M/s** |
| pipeline without the upload (`--noxfer`) | 937 M/s |
| **full pipeline** | **~890 M/s** (session range 845–1042) |

So the host has ~1.5x the capacity the GPU can consume, the upload costs ~5%, and the
pipeline runs at roughly **80–90% of the kernel-alone ceiling**. The remainder is sieve
threads competing for memory bandwidth plus per-batch launch cost.

This program started at 415 M/s here. Five changes got it to ~1940–2000:

1. **The wheel** (above): the host touches 2.4x fewer `k` and the four densest sieve
   patterns disappear. Host capacity went from ~415 M/s to ~1760 M/s of survivors.
2. **A separate transfer queue.** Everything used to go through one in-order command
   queue, so uploads and kernels *strictly alternated* — 0.23 s of host time plus
   0.41 s of kernel time came to 0.66 s instead of overlapping to 0.41 s. Uploads now
   go on their own queue and the kernel waits on the upload's event.
3. **The 96-bit kernel** and **4. squaring symmetry** (below).
5. **Pinned staging buffers.** Even on its own queue the upload still cost 0.15 s of a
   0.48 s run, because `clEnqueueWriteBuffer` from pageable memory stages through a
   driver-owned pinned buffer and that copy runs synchronously on the calling thread.
   The sieve threads now pack survivors directly into `CL_MEM_ALLOC_HOST_PTR` memory,
   so the upload is a pure DMA: **0.15 s → 0.03 s**.

An early version of this file claimed the sieve was "free" because it overlapped with
GPU work. That was wrong: it divided candidates by an assumed GPU rate that was itself
the pipeline rate — circular, and it hid the fact that the GPU was idle half the time.
`--bench`, `--nogpu` and `--noxfer` measure the pieces directly, which is what turned
each of the five wins above from a guess into a number.

### Arithmetic width

mfaktc's kernels are all `_mul32` / `_mul24` — 32-bit limbs with PTX carry chains
(`__umul32`, `__add_cc`, `__addc_cc`). Consumer GeForce cards run 32-bit multiplies
natively and emulate 64-bit ones, so there is a real gain in narrower limbs, and this
program now carries both kernels:

| | limbs | products per squaring | rate (p = 82589933) |
|---|---|---|---|
| `mersenne_tf` | 2 x 64-bit | 8 | 833 M/s |
| `mersenne_tf96`, general multiply | 3 x 32-bit | 18 | 1025 M/s |
| `mersenne_tf96`, **squaring symmetry** | 3 x 32-bit | **15** | **1193 M/s** |

The move to 32-bit limbs gained only ~25%, not the 2x a naive reading suggests: CIOS is
O(n²) in the limb count, so three narrow limbs need 18 multiply-accumulates where two
wide ones need 8. The narrower multiplies are cheaper by roughly 2.8x each, and
8 x 2.8 / 18 ≈ 1.25.

**Squaring symmetry** then added 16%. The exponentiation is nothing but `x·x`, and in a
square the off-diagonal products `a_i·a_j` (i<j) each occur twice — compute them once
and double the sum. For three limbs that is 3 cross + 3 diagonal = 6 products instead
of 9, so with the 9 of the Montgomery reduction the count falls from 18 to 15, and
15/18 ≈ the 16% observed.

`arithmetic = auto` uses the 96-bit kernel whenever `factor_max < 2^96` and the
128-bit one above that. Both are covered by `--selftest`, which runs every case
through each.

### Two optimisations that were tried and rejected

Recorded so nobody spends the effort twice.

**Barrett reduction.** mfaktc's fastest kernels use it, and their
`kernel_benchmarks.txt` shows `barrett76_mul32` at 2.5x their `95bit_mul32`. But that
comparison is against *their* float-division kernel, not against Montgomery. Counting
products for a generic 3-limb modulus:

| | products per squaring |
|---|---|
| Montgomery (interleaved reduction) | 6 square + 9 reduce = **15** |
| Barrett | 6 square + ~20 reduce = **~26** |

Barrett needs two large multiplies — the high half of `q1·m` and the low half of
`q3·q` — where Montgomery's interleaved reduction needs one. It would be a regression
here. mfaktc's Barrett variants win by hard-coding a bit level (76, 79, 87 …) so that
whole limbs of the product are known to be zero; that specialisation, not Barrett
itself, is the advantage.

**Two candidates per work item.** A single candidate is one long dependency chain, so
interleaving two independent ones should hide latency. Measured: 1069 M/s against
1060 M/s at 2^70, and slightly *worse* at 2^90. The compiler was already extracting
that parallelism, and doubling the live registers cancels the rest. The kernel was
removed again.

### Host-side tuning that turned out not to matter

Swept, measured, left at their defaults — recorded so the next person does not repeat it
(p = 82589933, 2^70, ~10 s runs on an RTX 3070):

| knob | result |
|---|---|
| work-group size | flat 64→1024 (1160 M/s ± 5); only 32 was slower at 1077 |
| `gpu_slots` (batches in flight) | 2 costs ~5%; 3 / 4 / 6 / 8 all within noise |
| `segment_size` | flat 2^21→2^23; 2^24 collapses to 398 M/s (pinned pool starves) |
| phase width (GPU drain frequency) | widening 8x changed nothing measurable — reverted, so checkpoints stay fine-grained |

There is no obvious lever left on the host side.

### `sieve_primes`

It trades CPU time against GPU time, and the optimum is where the two finish together.
Because the GPU's cost per candidate scales with `bitlen(p)`, larger exponents justify
heavier sieving. Wall clock for a fixed search space, measured back to back:

| `sieve_primes` | 100k | 200k | 500k | 1M | 2M |
|---|---|---|---|---|---|
| p = 82589933, q ≈ 2^70 | 12.35 s | 11.52 s | **10.69 s** | 10.73 s | 10.77 s |
| survivors reaching the GPU | 23.5% | 22.1% | 20.6% | 20.5% | 20.5% |

`auto` encodes that: 500000 for exponents up to 32 bits, 1000000 above.

**Those numbers are stale, and the optimum moves whenever either side gets faster.**
Re-measured on `p = 9147253, 2^40 … 2^65` after the per-level kernels and the
cache-blocked sieve:

| `sieve_primes` | 200k | 300k | 500k |
|---|---|---|---|
| wall clock | **49.8 s** | 50.5 s | 55.8 s |

(Measured back to back in one window. Timings on this machine are only comparable
within a window — see the note on measurement below.)

200000 is now optimal, not 500000 — a faster kernel makes each surviving candidate
cheaper, so paying CPU time to remove more of them stops being worth it. The shipped
`config.txt` sets 200000 explicitly for this reason; `auto` has not been re-tuned
(it is M4 in `tasks/todo.md`).

The curve is still flat near the optimum — 100k to 300k spans 1.4% — so this is worth
setting once and forgetting. Being badly wrong costs about 12%.

### A warning about measuring on this machine

Some runs collapse to roughly a tenth speed and stay there: the same job has measured
49.8 s and 341 s with no code change. It is **not** in this program — a `--nogpu` run,
with the GPU untouched, degrades by the same factor, and so does OpenCL kernel
compilation, which is plain CPU work. Instrumenting the pipeline puts it in the sieve's
marking loop, which takes 10x longer for byte-identical work; the GPU idling at 400 MHz
that it looks like at first is downstream of that.

The trigger found so far is a background video: closing it restored full speed. The
likely mechanism is memory-bandwidth contention, which a bitmap sieve is unusually
sensitive to.

Practical consequence: **compare timings only within one window, and re-measure the
baseline whenever you compare against another program.** Several numbers in earlier
versions of this file were taken in a degraded window and were wrong by 2–6x.

### What is still on the table

Head-to-head reference job, `p = 9147253, 2^40 … 2^65` — 419 G candidates in the allowed
classes, the same work mfaktc was timed on:

| state | wall |
|---|---|
| where this work started | 79 s |
| per-level kernels (64/72/96/128-bit) | 54.6 s |
| cache-blocked sieve marking | **49.8 s** |
| mfaktc | 38 s (timed before a background video was found to be corrupting measurements — worth re-timing) |

The split is now host 44.5 s against GPU 49.6 s, so it is **GPU-bound, but only just**;
the balance has flipped three times during this work, and each fix moved it back. That is
why the items only paid off together — with the sieve at 52 s, the 64-bit kernel was worth
1.5 s on the whole job despite being 1.21x on the levels it covers.

* **Fewer products per squaring** — the remaining GPU lever, and the ceiling is known.
  Two ablation kernels locate the cost precisely.

  It is not limb count, not occupancy, and not the multiply count. The 96-bit kernel
  compiles to 36 registers with no spills, so it runs at full occupancy; and if multiplies
  bound it, its 33 IMADs per squaring against this card's ~5.3×10^12 IMAD/s would give
  160 G squarings/s against the 31.6 G measured.

  Deleting the carry logic — same multiplies, same
  adds, no compare-and-select — buys **17%** (1134 → 1331 M/s). Stripping the reduction
  to its bare 15 products, with no accumulation structure at all, gives **2.09x**
  (2367 M/s). So carries are minor; what costs is the sheer volume of *accumulation*
  around the multiplies, and 2.09x is the ceiling for any design that still issues 15
  products per squaring.

  So the only way past ~2550 M/s is to issue fewer than 15 products. That is exactly what
  the 64-bit kernel does — two limbs need 3 for the square and 4 for the reduction, seven
  in all — and it measures 1.21x on the levels below 2^64 that it covers. Above 2^64 three
  limbs are unavoidable, and **Barrett does not help**: its reduction needs ~12 products
  against Montgomery's 9, so it is worse on the metric that matters. A 3-limb Karatsuba
  square saves one product; beyond that, nothing cheap is left. Scoped in
  `tasks/todo.md`. (Earlier versions of this section
  blamed shorter carry chains, then Montgomery-vs-Barrett as an instruction-count
  argument, then carry propagation; the ablations above are what actually settled it.)
* **Sieving on the GPU**, as mfaktc does with `SieveOnGPU=1`. Sending the sieve bitmap
  (512 KB per segment) instead of the survivor indices (~3.7 MB) would cut the upload
  by 7x and remove the host packing loop, at the cost of a stream compaction in the
  kernel. Less urgent now that the upload costs ~5%.

**What a bit level costs.** For `p = 82589933`, sweeping the whole range `2^70 … 2^71`
means about `3.6×10^12` candidates in the allowed classes ≈ **5 minutes** on this machine.
Each further bit level doubles that, so `2^89 … 2^90` for the same exponent is on the order
of **a decade of single-GPU time**. That is inherent to trial factoring, not to this
implementation — it is why GIMPS trial-factors to roughly `2^70`–`2^80` and then switches
to P−1 and Lucas–Lehmer. Set `factor_max` accordingly, and note that a *narrow* high band
(e.g. `factor_min = 2^75`, `factor_max = 2^75+2^60`) is perfectly practical.

**Windows TDR.** Batches are sized so each kernel runs for milliseconds, well under the
2-second display-driver watchdog. If you raise `segment_size` a great deal on a GPU that is
also driving your monitor, you may hit it.

---

## Files

| file | |
|---|---|
| `config.txt` | **your job** — exponent, factor bounds, tuning |
| `mersenne_tf0.9.exe` | the program; self-contained, no runtime dependencies but the driver |
| `mersenne_tf0.9.cpp` | host source — config, wheel, sieve, OpenCL driver, verification |
| `tf_kernel.cl.h` | OpenCL source for all five kernels (64/72/96/128-bit Montgomery) |
| `build.bat` | build script; finds Visual Studio on its own |
| `README.md` | this file |
| `results.txt` | run log: factors found, plus one line per run even when none were |
| `checkpoint_<p>.txt` | progress for an unfinished run; deleted when it completes |

The `.exe` is statically linked and resolves OpenCL from the driver at run time, so it
and a `config.txt` are all you need to copy to another machine — including one with a
different GPU vendor.
