# mersenne_tf

GPU trial factoring of Mersenne numbers, `M_p = 2^p - 1`, with exact integer arithmetic.

`M_p` is never constructed. It has `p` bits, and `p` may be in the hundreds of millions;
the program only ever works modulo the candidate, which is at most 128 bits. Results are
written as PrimeNet JSON result lines, ready for the
[GIMPS manual results page](https://www.mersenne.org/manual_result/).

## Performance

mersenne_tf 1.4 against mfaktc 0.24.1 on the same RTX 3070, at its stock 270 W. Whole runs,
start-up included; each figure is the median of three rounds, with the programs
alternating job by job.

| job | **mersenne_tf 1.4** | mfaktc 0.24.1 | |
|---|---|---|---|
| `p = 9147253`, `2^65..2^66` | **18.60 s** | 19.32 s | 4% faster |
| `p = 27886007`, `2^66..2^67` | **13.09 s** | 14.03 s | 7% faster |
| `p = 110000017`, `2^69..2^70` | **27.17 s** | 28.57 s | 5% faster |
| `p = 999000011`, `2^72..2^73` | **25.97 s** | 27.59 s | 6% faster |

mersenne_tf won all 48 pairings measured. mfaktc's own times move more than ours from one
session to the next, though, so read the margin as a few percent. Where the time goes, and
what was tried and dropped, is in [CHANGELOG.md](CHANGELOG.md).

## Download

[**mersenne_tf-1.4-win64.zip**](https://github.com/sallerk/mersenne-tf/releases/download/v1.4/mersenne_tf-1.4-win64.zip)
— a **prebuilt 64-bit Windows binary** plus its source. You do not need Visual Studio, a
CUDA Toolkit or an OpenCL SDK to run it; the only requirement is `OpenCL.dll`, which ships
with your GPU driver. The manual is [1.4/README.md](1.4/README.md), and what changed in
each version is in [CHANGELOG.md](CHANGELOG.md).

GitHub's green *Code → Download ZIP* button gives you the whole repository instead, with
every earlier version in its own directory — the link above is the way to get 1.4 on its
own.

## Quick start

Unzip it and run:

```bash
mersenne_tf.exe --selftest
mersenne_tf.exe
```

Edit `worktodo.txt` to say what to factor, `config.txt` to say which GPU to use.

Or build from source — Visual Studio 2019/2022 with "Desktop development with C++",
nothing else:

```bash
cd 1.4
build.bat
```

## Requirements

| | |
|---|---|
| GPU | any OpenCL 1.2 GPU — NVIDIA, AMD or Intel. Developed on an RTX 3070. |
| Runtime | `OpenCL.dll`, which **ships with your GPU driver**. Nothing to install. |
| OS | the prebuilt binaries are 64-bit Windows. The source has no Windows-specific number theory in it, but the build script and the OpenCL loader are Windows-only as written. |

The `.exe` is statically linked and resolves OpenCL from the driver at run time, so it plus
`config.txt` and `worktodo.txt` are all you need to copy to another machine — including one
with a different GPU vendor.

## Reading the output

One line, rewritten in place:

```
  2^64..2^65  15.48%  6953609162 done (4832 M/s)  cls 129-160/960  sieved 79.3%  elapsed 7s  ETA 5s  job 47m59s
```

- **`2^64..2^65`** is the bit level being scanned, and **the percentage is that level's**,
  not the whole job's — the level is the unit of work that finishes and gets reported.
  `ETA` is for the level; `job` is the whole configured range.
- **`cls`** is the wheel class within the level. Level plus class is the resume point.
- **`done` and `M/s` count every candidate disposed of** — sieved out *or* tested on the
  GPU. That makes the rate a measure of progress rather than of GPU traffic, and
  comparable between runs at different sieve depths.
- **`sieved`** is the share pre-factoring removed, so `done x (1 - sieved)` is what the GPU
  actually tested.

The line is built to your console's width and never wraps; on a narrow window it drops the
least important fields rather than spilling onto a second one.

## Results

`results.txt` gets PrimeNet result lines and nothing else: one JSON line per bit level, in
the format Prime95 writes, for the [manual results page](https://www.mersenne.org/manual_result/):

```
{"status":"NF", "exponent":9147253, "worktype":"TF", "bitlo":62, "bithi":63, "rangecomplete":true, "program":{"name":"mersenne_tf", "version":"1.4"}, "timestamp":"2026-10-03 22:49:17", "os":{"os":"Windows", "architecture":"x86_64"}, "checksum":{"version":1, "checksum":"D3C07086"}}
{"status":"F", "exponent":48205429, "worktype":"TF", "factors":["2176310738837111"], "bitlo":50, "bithi":60, "rangecomplete":true, "program":{"name":"mersenne_tf", "version":"1.4"}, "timestamp":"2026-10-03 22:49:11", "os":{"os":"Windows", "architecture":"x86_64"}, "checksum":{"version":1, "checksum":"024C35F3"}}
```

A level's line is written **as the level finishes**: `NF`, or `F` with every factor it
found. A run stopped part way still submits everything it finished. A level your range
only partly covers is deliberately *not* claimed — it gets a line only if it found a
factor, marked `"rangecomplete":false`.

The lines carry the checksum mfaktc 0.24 adds to TF results, the PrimeNet assignment key
from `worktodo.txt` when there is one, and optionally your user and computer names. The
file is written under `results.txt.lck`, the lock AutoPrimeNet uses, but AutoPrimeNet does
not list `mersenne_tf` among the programs whose results it submits yet, so for now submit
`results.txt` by hand.

Every factor is re-verified on the CPU, by a separate implementation, before it is written.
`runlog.txt` gets one human-readable line per run including interrupted ones; it is kept
separate so `results.txt` stays machine-parseable.

## Resuming

Ctrl-C (or closing the window) drains the GPU, saves and exits in well under a second.
Re-run with the same files and it resumes at the level and class it reached:

```
  resuming from checkpoint_9147253.txt: level 2^64..2^65, 36 of 960 classes done
```

A checkpoint is accepted only if the exponent and both bounds still match, so editing the
job starts a clean run rather than silently skipping work — and it says so. A factor found
in a level that has not finished is kept in the checkpoint, so a run that is stopped or
killed reports it, with its level, when resumed.

## Scope and limitations

- **Trial factoring only.** No P-1, no ECM, no PRP or Lucas–Lehmer. For P-1 on the GPU see
  [Mp_p-1_gpu](https://github.com/sallerk/Mp_p-1_gpu).
- **Windows / MSVC only.** No Makefile, no Linux build, no CI. The number theory is portable
  C++ and the kernels are plain OpenCL 1.2, but the build script and the OpenCL loader are
  Windows-only as written.
- **One job per run.** The first entry in `worktodo.txt` wins; there is no queue and no
  PrimeNet automation. Results are written for you to upload manually, and a finished job
  stays in `worktodo.txt`.
- **The exponent must be prime**, and below `2^62`. For composite `n` the factors of
  `2^n - 1` do not all have the form `2kn+1`, so the search would be unsound — the program
  refuses rather than silently returning a wrong "no factor".
- **Candidates are capped at `2^127-1`**, the arithmetic limit. Well above anything GIMPS
  trial-factors.
- **Above `2^96` throughput drops.** Exact-width kernels cover the band that matters — in
  1.4, 28-bit limbs below `2^80`, 30-bit to `2^88`, 32-bit to `2^96` — and everything past
  that falls back to the general 128-bit path.
- **Deep levels are inherently expensive.** Candidate count doubles with every bit level;
  that is trial factoring, not this implementation.
- **Tuned on one GPU.** The defaults were measured on an RTX 3070. They should be sane
  elsewhere, but no AMD or Intel device has been benchmarked.

## Name

"Mersenne, trial factoring". It divides — it does **not** implement P-1, ECM or any
primality test.
