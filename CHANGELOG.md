# Changelog

Each release is its own directory; earlier ones are kept as they shipped.

---

## 1.2 — 2026-08-10

**The sieve moved to the GPU.** Up to 1.1 the CPU marked a bitmap, bit-scanned it and
uploaded four bytes per survivor — about 79 GB per bit level. The whole pipeline is now
device-side.

Reference job throughout: `p = 86000009`, `2^66..2^67`, RTX 3070 at its **stock 270 W**
limit, configurations interleaved with an adjacent control — every earlier figure taken
under a 100 W cap was misleading, in both magnitude and ordering.

| | inner time |
|---|---|
| 1.1, CPU sieve | 9.94 s |
| **1.2, GPU sieve** | **9.19 s** |
| mfakto 0.15pre8 | 6.89 s |

### Added

- **Device-side sieve** (`sieve = gpu`, the default). `sieve_offsets` finds each prime's
  first strike in a segment; three marking kernels split the primes by size; `sieve_compact`
  turns the bitmap into an index list. The split is measured, not assumed:
  - **under 64** — one thread per 32-bit word, accumulating in a register. Such a prime
    strikes every word, so no thread divides for nothing. This kernel also writes (rather
    than ORs) the bitmap and resets the compaction counter, folding two clear passes away.
  - **64 to 2047** — a thread owns a window sized to the prime's octave, staged in shared
    memory and written back once. Rocke Verser's two-phase scheme from mfakto's
    `GPU-based-bit-sieving.txt`; worth 11.58 s → 9.53 s on its own against the first
    version, which did one global read-modify-write per strike.
  - **2048 and up** — a group stages a 32768-bit tile and its threads split the *primes*,
    striking with `__local` atomics. Worth 10.11 s → 9.53 s over one global atomic per strike.
- **`mod_recip`**, exact `n mod s` by multiply-high against a precomputed `floor(2^32/s)`,
  replacing the integer division in both marking kernels' inner loops.
- **`--sieve-only`**, which runs the sieve and never launches trial factoring. Subtracting
  two whole-run times could not separate the sieve from a compaction cost that moves with
  it; this can.
- **`sieve = cpu`** keeps the 1.1 pipeline as the reference implementation, and is what the
  device sieve is verified against.
- **`mersenne_tf72Lx2`**, two independent Montgomery chains per work item on the 72-bit
  lazy path, selected by `vector = auto`. Worth 2.8% when the GPU is the constraint.
- **`vector = auto|1|2`** to compare the two.

### Changed

- The survivor count lives in a device buffer instead of a kernel argument, so the sieve
  can write a count the trial-factoring kernel reads without a host round trip per segment.
- **`sieve_primes` default 200000 → 120000, and `auto` retuned per sieve mode**
  (120000 / 250000 on the device, unchanged at 500000 / 1000000 on the CPU). 200000 was
  the CPU-sieve optimum, where sieving overlaps with the GPU and is nearly free. On the
  device both sides compete for the same GPU, nothing is hidden, and the optimum roughly
  halves. Round-robin over a whole bit level, 4 rounds:

  | `sieve_primes` | 60000 | 90000 | **120000** | 150000 | 200000 | 300000 |
  |---|---|---|---|---|---|---|
  | median | 9.35 s | 9.16 s | **9.05 s** | 9.07 s | 9.26 s | 9.73 s |

  A finer pass puts the plateau at 110000–130000, so `auto` picking 500000 was costing
  about 8% on the device path.
- `segment_size` default 2^22 → **2^24**. The device sieve launches ~9 kernels per segment,
  and WDDM charges roughly 15 us a launch, so small segments pay it over and over. Sieve
  alone: 2^22 6.20 s, **2^24 3.28 s**, 2^26 3.64 s, 2^28 68.61 s — the last because three
  slots of a 2^28 index buffer is 3 GB and spills off an 8 GB card.
- **The progress line's rate now counts every candidate disposed of**, sieved out or
  tested, where it used to count only those reaching the GPU. The old figure moved the
  wrong way — sieving deeper removes candidates instead of testing them, so it fell exactly
  when the job got faster — and was not comparable between runs at different
  `sieve_primes`. The count beside it changed from `tested` to `done` to match, `sieved`
  still gives the split, and the end-of-run summary reports both:

  ```
  GPU tested : 19719827817 candidates in 10.93 s (1803.61 M/s on the GPU)
  overall    : 8153.03 M candidates/s disposed of (sieved or tested)
  ```
- `--bench` now covers the 72-bit kernels. It never did, which is why a 26% gap between
  the kernel's ceiling and its delivered rate went unnoticed: pure kernel 2850 M/s,
  delivered 2111 M/s. It is also useless under a power cap and trustworthy without one.

### Fixed

- **`sieved` read 100% until the first phase ended** on the device path, because the
  survivor count is accumulated on the GPU and only collected at phase boundaries. The
  progress line now picks up the running phase counter for display — twice a second, not
  the per-segment stall that design replaced.
- A **UTF-8 BOM** in `config.txt` or `worktodo.txt` produced `ERROR: line 1: expected
  'key = value'` with nothing to suggest an invisible character. Both readers now skip it.
- `build.bat` cleaned `mersenne_tf0*.obj`, a 0.9-era name that stopped matching, leaving an
  object file behind every build.

### A side effect worth knowing about

Because the device path uses essentially no CPU per candidate, it is far less sensitive to
whatever else the machine is doing. Observed accidentally when a busy desktop landed in the
middle of a benchmark, then confirmed against an adjacent control:

| | quiet machine | same job, loaded machine |
|---|---|---|
| 1.1, CPU sieve | 9.94 s | 20.9 s (+110%) |
| 1.2, GPU sieve | 9.19 s | 11.0 s (+20%) |

So on a workstation that is also being used, 1.2 is worth considerably more than the 8% the
quiet-machine figures suggest.

### Where the remaining gap is

Trial factoring is no longer the problem: removing the transfers took the delivered rate
from **2111 to 2892 M/s**, against mfakto's fitted 3101 M/s — 7% apart, and essentially the
kernel's own ceiling. The gap is **entirely the sieve**, now ~2.4 s against mfakto's ~1.3 s
(fitted from three sieve depths; residuals under 0.01 s).

Three plausible-looking optimisations were tried and **measured no better**, and are
recorded because the reasoning behind each was sound and wrong:

| change | expected | measured |
|---|---|---|
| replace the inner-loop `%` with multiply-high | ~4x fewer ops on the dominant instruction | neutral (12.86 s vs 12.88 s) |
| drop the per-segment blocking count readback | it drained an in-order queue every segment | neutral (3.65 s vs 3.28 s sieve-only) |
| collapse 12 launches per segment down to 4 | ~15 us of WDDM submit overhead each | **worse** (4.26 s vs 3.28 s) |

The last one is the informative failure: the launches were real overhead, but paying them
is cheaper than moving primes 64–2047 out of a window-owning kernel into the tile kernel.
`mod_recip` and the device-side counter were kept anyway — both strictly do less work, and
neither costs anything. The sieve is evidently bound by something none of these touched.

### Verification

- **Bit-for-bit identical survivors to the CPU sieve**: 22,255,341,964 of 89,141,611,080 at
  `sieve_primes = 50000`, chosen so the `segment_size/8` ceiling cannot bind and both paths
  use the same primes.
- Known factor still found and CPU-verified: `M8249309 has a factor: 313603386094415369`,
  from the same 905,703,107 candidates on both paths.
- `--selftest` passes all six width/vector combinations with the device sieve active.
- At `sieve_primes = 200000` the device path reports 19,719,827,817 survivors against the
  CPU path's 19,731,052,409 — 0.057% fewer, because the CPU drops primes above `len/8` on
  short trailing segments while the device applies all of them. It removes more composites,
  never fewer.

---

## 1.1 — 2026-08-10

**Reporting and documentation only. No change to what gets factored.** Every survivor
count and every result line is identical to 1.0 — verified by running both binaries over
`p = 86000009`, `2^66..2^67` at three sieve settings and comparing the counters:

| `sieve_primes` / `segment_size` | candidates reaching the GPU | 1.0 | 1.1 |
|---|---|---|---|
| 200000 / 4194304 | 19,731,052,409 | ✔ | ✔ |
| 1000000 / 4194304 | 18,286,683,492 | ✔ | ✔ |
| 1000000 / 8388608 | 17,444,225,265 | ✔ | ✔ |

`--selftest` passes all cases on the 72-, 96- and 128-bit paths.

### Fixed

- **The run header claimed a sieve depth the sieve never reached.** The segment loop
  skips any prime above `segment_size / 8` — a prime that strikes fewer than about one
  position in eight does not remove enough candidates to pay for its own setup. At the
  default `segment_size = 4194304` that ceiling is 524288, so `sieve_primes = 1000000`
  was really 524288 and the header still printed `primes below 1000000`. It now prints
  both:

  ```
  pre-sieve  : primes below 1000000, capped to 524288 by segment_size/8
  ```

  and stays as it was when the cap does not bite. Found because the survivor count at
  `sieve_primes = 1000000` disagreed with the Mertens prediction by 4.9% while 17000,
  200000 and 500000 all matched to within 0.01%; solving for the implied bound gave
  524340, i.e. 2^19.

- **The cap had no single definition.** It was an unnamed expression inside the segment
  loop, which is how the header came to contradict it. Both callers now go through
  `sieve_prime_cap_for()`, so the reported bound cannot drift from the applied one.

- **`build.bat` left an object file behind every build** — it cleaned `mersenne_tf0*.obj`,
  a leftover from the 0.9 file name, which stopped matching when the binary was renamed.

- **Banner and usage text still said `mersenne_tf0.9` in the 1.0 release** while result
  lines correctly said `mersenne_tf 1.0`. The release number now lives in one place,
  `MTF_VERSION`, with `MTF_NAME` and `TF_PROGRAM_ID` derived from it, so the banner,
  usage text, GIMPS result lines and run log cannot disagree again.

### Documented

- `config.txt`: a `CEILING` paragraph under `sieve_primes` giving the rule, the value it
  takes at the default segment size, and the header line it produces. Cross-referenced
  from `segment_size`, whose documentation did not mention that it also sets the sieve's
  depth limit. The stated range is now `2 .. 1000000000, effective value limited to
  segment_size/8`.

- **Why the ceiling is not worth tuning around**, measured rather than asserted: raising
  `segment_size` to 8388608 lets a real 1000000 sieve run, which removes 4.6% more
  candidates and costs 19% more wall time (`p = 86000009`, `2^66..2^67`).

- **`M/s` is not a speed.** It counts candidates *reaching* the GPU, so sieving deeper is
  supposed to lower it — that is the work being removed — and it is only comparable
  between runs at the same `sieve_primes`. Dropping `sieve_primes` from 200000 to 17000
  raises the rate from ~1750 to ~1830 M/s and makes the level take **19% longer**, because
  25% more candidates survive. Noted on the progress-line description in `README.md` and
  next to the tuning measurements in `config.txt`.

- Second data point for the `sieve_primes` optimum, at a GIMPS-scale exponent
  (`p = 86000009`, `2^66..2^67`): 200000 → 11.4 s, 17000 → 13.6 s, 500000 → 12.0 s.
  Confirms the 200000 chosen from `p = 9147253`.

### Known and unchanged

Below `2^60` the scan levels are decades (`<2^40`, `2^40..2^50`, `2^50..2^60`), not single
bits, so a job such as `Factor=N/A,9147253,58,59` is scanned in full but counts as part of
the `2^50..2^60` level and writes no `no factor` line — the same rule that withholds any
partially covered level. Behaviour is unchanged from 1.0 and only ever withholds a true
claim rather than making a false one; asking for `50,60` reports normally. Now stated in
`README.md` §5 instead of being a surprise. Found while verifying 1.1 against 1.0.

`auto` returns 1000000 for exponents wider than 32 bits, which meets the 524288 ceiling at
the default segment size. Left as it is on purpose: `resolve_sieve_limit()` states what the
tuning wants and the cap states what the segment allows, and the header now shows both.
Clamping would collapse them and hide the fact that the tuning asked for more. It cannot
affect results either way — for `p > 2^32` every candidate `q = 2kp+1` exceeds 2^33, so no
prime under a million can be in range for the direct check.

---

## 1.0 — 2026-08-09

Job moved out of `config.txt` into `worktodo.txt`, in PrimeNet assignment format.
Results written as GIMPS manual-submission lines. Checkpoints, bit-level scan order, and
per-level `no factor` reporting.

## 0.9 — 2026-07-28

First release. Job and settings both in `config.txt`; results in its own format.
