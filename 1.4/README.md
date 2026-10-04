# mersenne_tf 1.4 — GPU trial factoring of Mersenne numbers

Finds every **prime factor of `M_p = 2^p - 1`** inside a range you choose, using your GPU,
with **exact integer arithmetic** — no floating point anywhere in the number theory.

`M_p` itself is never constructed. It has `p` bits, and `p` may be in the hundreds of
millions; the program only ever works modulo the candidate, which is at most 128 bits.

Results are written as PrimeNet JSON result lines, the format the GIMPS manual results
page and AutoPrimeNet read, so finished work can be submitted as it is.

> Measurements, and the reasoning behind every default, are in
> [`CHANGELOG.md`](../CHANGELOG.md). This file is the manual.

**New in 1.4: about 1.7x faster** across the levels GIMPS assigns, from a new
trial-factoring kernel and a reworked sieve, and now a few percent ahead of mfaktc 0.24.1
on the same card.

*Below `2^88` the kernel is compiled for the exponent being tested.* Knowing `p` lets it
unroll the bit loop, start from `2^e` for the leading bits of `p` instead of squaring up to
them, and — below `2^86` — skip the modular doubling altogether: the next squaring absorbs
it as a shift of the limbs, which is exact while `16q` fits the Montgomery radix. Kernel
throughput at `2^66` went from 3010 to 5097 M candidates/s on an RTX 3070.

*The device sieve takes less than half the time it did in 1.3.* The largest primes are
struck from inside the trial-factoring kernel, where their memory traffic overlaps its
arithmetic. The smallest are built in registers, and each segment's starting offsets
follow from the previous segment's instead of being recomputed.

Whole runs on an RTX 3070 (stock 270 W, interleaved with 1.3 and mfaktc, medians of three
rounds):

| job | 1.3 | 1.4 | | mfaktc 0.24.1 |
|---|---|---|---|---|
| `p = 9147253`, `2^65..2^66` | 32.53 s | 18.60 s | 1.75x | 19.32 s |
| `p = 27886007`, `2^66..2^67` | 22.00 s | 13.09 s | 1.68x | 14.03 s |
| `p = 110000017`, `2^69..2^70` | 46.68 s | 27.17 s | 1.72x | 28.57 s |
| `p = 999000011`, `2^72..2^73` | 44.50 s | 25.97 s | 1.71x | 27.59 s |
| `p = 999000011`, a `2^81` slice | 44.82 s | 26.69 s | 1.68x | — |
| `p = 999000011`, a `2^87` slice | 44.58 s | 31.62 s | 1.41x | — |

1.4 was faster than mfaktc in every pairing measured, but mfaktc's times move more than
ours from one session to the next, so take the margin as a few percent. Details in the
changelog.

*`results.txt` is now in PrimeNet's JSON result format*, one line per bit level, with the
keys Prime95 writes and the checksum mfaktc adds — PrimeNet no longer accepts the text lines
earlier versions wrote. See [§5](#5-output).

`arithmetic = 64` and `72` are gone — the new 28-bit kernel beats both everywhere they
applied. Full list, with the measurements, in [`CHANGELOG.md`](../CHANGELOG.md).

---

## 1. Requirements

| | |
|---|---|
| GPU | Any OpenCL 1.2 GPU — NVIDIA, AMD or Intel. Developed on an RTX 3070. |
| Runtime | `OpenCL.dll`, which **ships with your GPU driver**. Nothing to install. |
| Build | Visual Studio 2019/2022 with "Desktop development with C++". |

No CUDA Toolkit, no OpenCL SDK, no headers, no import libraries: the program resolves the
OpenCL entry points from the driver at run time.

## 2. Build

```bash
build.bat
```

Produces `mersenne_tf.exe`. The script finds Visual Studio on its own.

## 3. The two input files

**`worktodo.txt` — what to work on.** One entry, in either form:

```ini
Factor=N/A,9147253,64,65
```

That is the PrimeNet assignment line: trial factor `M_9147253` from `2^64` to `2^65`. Paste
an assignment in as-is; the id may be PrimeNet's 32-hex-digit key or `N/A`. A key is
copied into every result line as `"aid"`; `N/A` means no assignment and adds nothing.

```ini
exponent   = 9147253      # for a range that is not a whole bit level
factor_min = 1
factor_max = 2^70
```

**`config.txt` — settings for this machine.** Which GPU, how many sieve threads, where
output goes. Nothing about the job. Keep them apart and a worktodo can move between
machines without disturbing anything tuned for one of them.

Numbers accept plain decimal (`1180591620717411303424`), separators
(`1_180_591_620_717_411_303_424`), or powers of two with an offset (`2^90`, `2^90-1`,
`2^70+2^60`) — in both files, for every numeric key.

Bad values stop the program with the line number rather than silently defaulting:

```
ERROR: line 55: sieve_primes: not a number: '50O000'
ERROR: line 100: segment_size: value out of range (4096 .. 268435456)
```

## 4. Run

```bash
mersenne_tf.exe
```

| command | what it does |
|---|---|
| `mersenne_tf.exe` | runs the job in `worktodo.txt` |
| `mersenne_tf.exe --config myjob.txt` | uses a different settings file |
| `mersenne_tf.exe --selftest` | checks every kernel against known factorisations |
| `mersenne_tf.exe --list-devices` | lists OpenCL GPUs with their platform/device indices |
| `mersenne_tf.exe --bench` | kernel throughput alone — no sieve, no transfers |
| `mersenne_tf.exe --sieve-only` | the sieve alone — never launches trial factoring |
| `mersenne_tf.exe --profile` | per-kernel device time (the run's *wall* time is meaningless) |
| `mersenne_tf.exe --no-fuse` | keeps the sieve and the test in separate kernels, to compare |
| `mersenne_tf.exe --nogpu` | the host pipeline alone — never submits to the GPU |

`--sieve-only` and `--nogpu` test nothing, so they claim nothing. They write no
`results.txt` line, log `status=diagnostic`, and neither resume from nor touch a
checkpoint, so they are safe to time on a job that is part way through.

While it runs, one line is rewritten in place:

```
  2^64..2^65  15.48%  6953609162 done (4832 M/s)  cls 129-160/960  sieved 79.3%  elapsed 7s  ETA 5s  job 47m59s
```

- **`2^64..2^65`** — the bit level being scanned. The bottom one reads `<2^40`.
- **the percentage is that level's**, not the whole job's, because the level is the unit
  of work that finishes and gets reported. Resuming picks it up where it left off.
- **`ETA`** is for this level; **`job`** is the whole configured range.
- **`cls`** is the wheel class within the level. Level plus class is the resume point.
- **`done` and `M/s` count every candidate disposed of** — sieved out *or* tested on the
  GPU — not just the ones that reached the GPU. That makes the rate a measure of progress
  and comparable between runs at different `sieve_primes`. Up to 1.1 it counted only GPU
  arrivals, which moved the wrong way: sieving deeper removes candidates instead of testing
  them, so the rate fell exactly when the job got faster.
- **`sieved`** is the share the pre-factoring removed, so `done x (1 - sieved)` is what the
  GPU actually tested. The end-of-run summary reports both rates separately.

The line is built to your console's width and never wraps; on a narrow window it drops the
least important fields rather than spilling onto a second line.

## 5. Output

### `results.txt` — for GIMPS

PrimeNet result lines and nothing else: one JSON line per bit level, in the format the
[manual results page](https://www.mersenne.org/manual_result/) and AutoPrimeNet read.

```
{"status":"NF", "exponent":9147253, "worktype":"TF", "bitlo":62, "bithi":63, "rangecomplete":true, "program":{"name":"mersenne_tf", "version":"1.4"}, "timestamp":"2026-10-03 22:49:17", "os":{"os":"Windows", "architecture":"x86_64"}, "checksum":{"version":1, "checksum":"D3C07086"}}
{"status":"F", "exponent":48205429, "worktype":"TF", "factors":["2176310738837111"], "bitlo":50, "bithi":60, "rangecomplete":true, "program":{"name":"mersenne_tf", "version":"1.4"}, "timestamp":"2026-10-03 22:49:11", "os":{"os":"Windows", "architecture":"x86_64"}, "checksum":{"version":1, "checksum":"024C35F3"}}
```

- **One line per bit level, written as the level finishes:** `"NF"` if it found nothing,
  `"F"` with every factor it found otherwise. A run stopped part way still submits
  everything it finished.
- **`rangecomplete`** says whether every candidate in the level was tested. A level your
  range only partly covers is not claimed: it gets a line only if it found a factor, and
  then with `"rangecomplete":false`. The same goes for the level `stop_on_factor` stops in.
- The keys are Prime95's, in Prime95's order. `"aid"` is there when the worktodo line has
  a PrimeNet assignment key, and `"user"`/`"computer"` when `config.txt` sets them.
- `"os"` and `"checksum"` are what mfaktc 0.24 adds to a TF result: a CRC32 over the
  result's fields, which mfaktc says PrimeNet uses to validate TF results. The self test
  checks the layout against real mfaktc lines.
- The file is written under `results.txt.lck`, the lock AutoPrimeNet and mfaktc use, so
  AutoPrimeNet can read it while a run is going.

Above `2^60` a level is one bit. Below it the levels are decades — `<2^40`, `2^40..2^50`,
`2^50..2^60` — so a job like `Factor=N/A,9147253,58,59` is scanned in full but counts as
part of the `2^50..2^60` level and produces no `NF` line. Ask for `50,60` if you want that
range claimed. This only ever withholds a true claim, never makes a false one, and GIMPS
assignments do not reach that far down.

**AutoPrimeNet** submits only result lines from programs it knows, by name, and
`mersenne_tf` is not on its list yet (AutoPrimeNet 2.0.1). Until it is, submit
`results.txt` on the [manual results page](https://www.mersenne.org/manual_result/).

### `runlog.txt` — for you

One line per run, including interrupted ones:

```
2026-08-09 18:33:44  p=9147253  range=2^64..2^65  status=complete  factors=0  scanned=209521354534  tested=46343035241  time=26.89s
```

This is the file to read when you want to know what happened. It is kept separate so
`results.txt` stays machine-parseable.

### On screen, the moment a factor is found

```
  *** FACTOR FOUND ***   348318885503
      k         = 497063   (q = 2kp+1: yes)
      q mod 8   = 7
      size      = 39 bits
      2^p mod q = 1 : VERIFIED  (recomputed on the CPU)
      q is      : prime
```

`VERIFIED` means the CPU independently recomputed `2^p mod q` with a separate
implementation and got 1. Check any hit yourself in Python:

```python
pow(2, 350377, 348318885503) == 1
```

## 6. Stopping and resuming

Ctrl-C (or closing the window) drains the GPU, saves and exits in well under a second.
Re-run with the same files and it resumes:

```
  resuming from checkpoint_9147253.txt: level 2^64..2^65, 36 of 960 classes done
```

A checkpoint is only accepted if the exponent and both bounds still match, so editing the
job starts a clean run rather than silently skipping work — and it says so rather than
restarting in silence.

A factor goes into `results.txt` with its level's line, when the level finishes. Until
then the checkpoint carries it, so a run stopped or killed part way through a level
reports it when resumed — once, with the whole level. With `checkpoint = 0` there is
nothing to resume, so Ctrl-C writes the level's factors at once, marked
`"rangecomplete":false`.

If the GPU fails part way through (a kernel fault, or a driver reset), the run stops with
`ERROR: GPU error ...` and claims nothing since its last checkpoint: no `results.txt`
line, no `complete` in the run log. Re-run to resume from that checkpoint.

## 7. Settings reference

Every key is documented inline in `config.txt` with the measurement behind its default.
The ones worth knowing:

| key | meaning |
|---|---|
| `worktodo_file` | where to read the job from (default `worktodo.txt`) |
| `sieve` | `gpu` runs pre-factoring on the device (default); `cpu` is the 1.1 pipeline, kept as the reference — the two produce identical survivor sets |
| `vector` | candidates per work item on the 96-bit path (`2^88..2^96`): `auto` (1), `1`, `2` |
| `sieve_primes` | pre-factoring bound. `auto` scales it from the exponent *and* from where the sieve runs — the device path wants a few million, the CPU path much less. On `sieve = cpu` it is capped at `segment_size/8`; the run header says so when the cap bites. The device sieve applies every prime below the bound: 5.5 M to 16 M measure the same there, and 32 M is slower — see `CHANGELOG.md` |
| `arithmetic` | `auto` picks the narrowest exact kernel per bit level — three 28-bit limbs below `2^80`, 30-bit to `2^88`, 32-bit to `2^96`, else 64-bit; force `84`/`90`/`96`/`128` to compare |
| `threads` | CPU sieve threads, `0` = auto (cores − 1) |
| `platform`, `device` | which GPU (see `--list-devices`), `-1` = auto |
| `stop_on_factor` | `1` to stop at the first factor instead of scanning the whole range; the job ends there and keeps no checkpoint |
| `results_file`, `log_file` | the two output files above |
| `user`, `computer` | optional; copied into every result line. Neither the manual results page nor AutoPrimeNet needs them |
| `checkpoint`, `checkpoint_seconds` | progress saving, on by default |
| `segment_size`, `workgroup`, `gpu_slots` | tuning; the defaults were measured, leave them alone unless benchmarking |

## 8. How long will it take?

Candidate count scales as **1/p** and **doubles with every bit level**. On an RTX 3070 with
`p ≈ 9.1M`:

| range | time | 1.3, same window |
|---|---|---|
| `2^64 .. 2^65` | ~10 s | ~17 s |
| `2^40 .. 2^65` (everything below, plus that level) | ~20 s | ~31 s |
| each further bit level | double the one before | |

Higher levels gain about as much — 1.7x up to `2^81`, 1.4x at `2^87`; see the table at
the top.

So `2^69..2^70` is roughly 32x the `2^64..2^65` level. That is inherent to trial factoring,
not to this implementation — it is why GIMPS trial-factors to about `2^70`–`2^80` and then
switches to P−1 and Lucas–Lehmer.

A small exponent is *slower*, not faster: at `p = 127` the same `2^70` bound has 65,000x
more candidates than at `p = 9.1M`.

## 9. Files

| file | |
|---|---|
| `mersenne_tf.exe` | the program; self-contained apart from the driver |
| `worktodo.txt` | the job |
| `config.txt` | machine settings |
| `results.txt` | PrimeNet result lines (JSON) |
| `runlog.txt` | run history |
| `checkpoint_<p>.txt` | progress for an unfinished run; deleted when it completes |
| `mersenne_tf.cpp`, `tf_kernel.cl.h` | host source and OpenCL kernels |
| `build.bat` | build script |

The `.exe` is statically linked and resolves OpenCL from the driver at run time, so it plus
`config.txt` and `worktodo.txt` are all you need to copy to another machine — including one
with a different GPU vendor.
