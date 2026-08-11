# mersenne_tf 1.3 — GPU trial factoring of Mersenne numbers

Finds every **prime factor of `M_p = 2^p - 1`** inside a range you choose, using your GPU,
with **exact integer arithmetic** — no floating point anywhere in the number theory.

`M_p` itself is never constructed. It has `p` bits, and `p` may be in the hundreds of
millions; the program only ever works modulo the candidate, which is at most 128 bits.

Results are written in the format the GIMPS manual submission page expects, so finished
work can be pasted straight back.

> Measurements, and the reasoning behind every default, are in
> [`CHANGELOG.md`](../CHANGELOG.md). This file is the manual.

**New in 1.3** — the device sieve no longer gets more expensive the deeper it goes. In
1.2 its cost scaled with the *number of primes* rather than with the work they did, so
sieving deeper made runs slower and `sieve_primes` had to default to a shallow 120000.
That is fixed and the default is now twenty times deeper, which is most of a **1.19x**
whole-run gain. Full list, with the measurements, in [`CHANGELOG.md`](../CHANGELOG.md).

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
an assignment in as-is; the id may be PrimeNet's 32-hex-digit key or `N/A`.

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

Only lines the [manual submission page](https://www.mersenne.org/manual_result/) parses:

```
M350377 has a factor: 348318885503 [TF:38:39:mersenne_tf 1.3]
no factor for M9147253 from 2^64 to 2^65 [mersenne_tf 1.3]
```

A `no factor` line is written **as each bit level clears**, so a run stopped part way still
submits everything it finished. A level your range only partly covers is deliberately *not*
reported — that claim belongs to whoever finishes it.

Above `2^60` a level is one bit. Below it the levels are decades — `<2^40`, `2^40..2^50`,
`2^50..2^60` — so a job like `Factor=N/A,9147253,58,59` is scanned in full but counts as
part of the `2^50..2^60` level and produces no `no factor` line. Ask for `50,60` if you
want that range claimed. This only ever withholds a true claim, never makes a false one,
and GIMPS assignments do not reach that far down.

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
restarting in silence. Factors are appended to `results.txt` the instant they are found,
so they survive even an unclean kill.

## 7. Settings reference

Every key is documented inline in `config.txt` with the measurement behind its default.
The ones worth knowing:

| key | meaning |
|---|---|
| `worktodo_file` | where to read the job from (default `worktodo.txt`) |
| `sieve` | `gpu` runs pre-factoring on the device (default); `cpu` is the 1.1 pipeline, kept as the reference — the two produce identical survivor sets |
| `vector` | candidates per work item: `auto` (2 where available), `1`, `2` |
| `sieve_primes` | pre-factoring bound. `auto` scales it from the exponent *and* from where the sieve runs — the device path wants a few million, the CPU path much less. On `sieve = cpu` it is capped at `segment_size/8`; the run header says so when the cap bites. Deeper is better on the device up to about 8 M and flat from 2 M to 16 M — that was the other way round before the sieve stopped scaling with the prime count, see `CHANGELOG.md` |
| `arithmetic` | `auto` picks the narrowest exact kernel per bit level — 24-bit limbs below `2^70`, 28-bit to `2^82`, 30-bit to `2^88`, 32-bit to `2^96`, else 64-bit; force `64`/`72`/`84`/`90`/`96`/`128` to compare |
| `threads` | CPU sieve threads, `0` = auto (cores − 1) |
| `platform`, `device` | which GPU (see `--list-devices`), `-1` = auto |
| `stop_on_factor` | `1` to stop at the first factor instead of scanning the whole range |
| `results_file`, `log_file` | the two output files above |
| `checkpoint`, `checkpoint_seconds` | progress saving, on by default |
| `segment_size`, `workgroup`, `gpu_slots` | tuning; the defaults were measured, leave them alone unless benchmarking |

## 8. How long will it take?

Candidate count scales as **1/p** and **doubles with every bit level**. On an RTX 3070 with
`p ≈ 9.1M`:

| range | time |
|---|---|
| `2^64 .. 2^65` | ~17 s |
| `2^40 .. 2^65` (everything below, plus that level) | ~32 s |
| each further bit level | double the one before |

(1.2 was ~27 s and ~50 s for the same two.)

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
| `results.txt` | GIMPS submission lines |
| `runlog.txt` | run history |
| `checkpoint_<p>.txt` | progress for an unfinished run; deleted when it completes |
| `mersenne_tf.cpp`, `tf_kernel.cl.h` | host source and OpenCL kernels |
| `build.bat` | build script |

The `.exe` is statically linked and resolves OpenCL from the driver at run time, so it plus
`config.txt` and `worktodo.txt` are all you need to copy to another machine — including one
with a different GPU vendor.
