# mersenne_tf

GPU trial factoring of Mersenne numbers, `M_p = 2^p - 1`, with exact integer arithmetic.

`M_p` is never constructed. It has `p` bits, and `p` may be in the hundreds of millions;
the program only ever works modulo the candidate, which is at most 128 bits. Results are
written in the format the [GIMPS manual submission page](https://www.mersenne.org/manual_result/)
expects.

## Download

Each version has its own release, with a **prebuilt 64-bit Windows binary** inside — you do
not need Visual Studio, a CUDA Toolkit or an OpenCL SDK to run it. The only requirement is
`OpenCL.dll`, which ships with your GPU driver.

| version | download | |
|---|---|---|
| **1.3** | [**mersenne_tf-1.3-win64.zip**](https://github.com/sallerk/mersenne-tf/releases/download/v1.3/mersenne_tf-1.3-win64.zip) | current. A device sieve that pays to run deep, and 28-/30-bit limbs that removed a 21% cliff above `2^72`. **1.18x** over 1.2 at `2^66`, **1.39x** at `2^76`. Start here — see [1.3/README.md](1.3/README.md). |
| 1.2 | [mersenne_tf-1.2-win64.zip](https://github.com/sallerk/mersenne-tf/releases/download/v1.2/mersenne_tf-1.2-win64.zip) | moved the sieve to the GPU — the whole pipeline is device-side. Kept as-is. |
| 1.1 | [mersenne_tf-1.1-win64.zip](https://github.com/sallerk/mersenne-tf/releases/download/v1.1/mersenne_tf-1.1-win64.zip) | reports the sieve bound it actually applies, not the one you asked for. Kept as-is. |
| 1.0 | [mersenne_tf-1.0-win64.zip](https://github.com/sallerk/mersenne-tf/releases/download/v1.0/mersenne_tf-1.0-win64.zip) | job moved to `worktodo.txt`; results in GIMPS manual-submission format. Kept as-is. |
| 0.9 | [mersenne_tf-0.9-win64.zip](https://github.com/sallerk/mersenne-tf/releases/download/v0.9/mersenne_tf-0.9-win64.zip) | job and settings both in `config.txt`; results in its own format. Kept as-is. |

All of them are on the [releases page](https://github.com/sallerk/mersenne-tf/releases).
GitHub's green *Code → Download ZIP* button gives you the **whole repository**, every
version at once — the links above are the way to get one version on its own.

Each release is also its own directory in this repository, kept exactly as it shipped.
What changed between them is in [CHANGELOG.md](CHANGELOG.md).

## Quick start

Unzip a release and run it:

```bash
mersenne_tf.exe --selftest
mersenne_tf.exe
```

Edit `worktodo.txt` to say what to factor, `config.txt` to say which GPU to use.
(In 0.9 both live in `config.txt`.)

Or build from source — Visual Studio 2019/2022 with "Desktop development with C++",
nothing else:

```bash
cd 1.3
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

`results.txt` gets only lines the [manual submission page](https://www.mersenne.org/manual_result/)
parses:

```
M350377 has a factor: 348318885503 [TF:38:39:mersenne_tf 1.3]
no factor for M9147253 from 2^64 to 2^65 [mersenne_tf 1.3]
```

A `no factor` line is written **as each bit level clears**, so a run stopped part way still
submits everything it finished. A level your range only partly covers is deliberately *not*
reported — that claim belongs to whoever finishes it.

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
job starts a clean run rather than silently skipping work — and it says so. Factors are
appended to `results.txt` the instant they are found, so they survive even an unclean kill.

## Scope and limitations

- **Trial factoring only.** No P-1, no ECM, no PRP or Lucas–Lehmer. For P-1 on the GPU see
  [Mp_p-1_gpu](https://github.com/sallerk/Mp_p-1_gpu).
- **Windows / MSVC only.** No Makefile, no Linux build, no CI. The number theory is portable
  C++ and the kernels are plain OpenCL 1.2, but the build script and the OpenCL loader are
  Windows-only as written.
- **One job per run.** The first entry in `worktodo.txt` wins; there is no queue and no
  PrimeNet automation. Results are written for you to upload manually.
- **The exponent must be prime**, and below `2^62`. For composite `n` the factors of
  `2^n - 1` do not all have the form `2kn+1`, so the search would be unsound — the program
  refuses rather than silently returning a wrong "no factor".
- **Candidates are capped at `2^127-1`**, the arithmetic limit. Well above anything GIMPS
  trial-factors.
- **Above `2^96` throughput drops.** Exact-width kernels cover the band that matters —
  24-bit limbs below `2^70`, 28-bit to `2^82`, 30-bit to `2^88`, 32-bit to `2^96` — and
  everything past that falls back to the general 128-bit path.
- **Deep levels are inherently expensive.** Candidate count doubles with every bit level;
  that is trial factoring, not this implementation.
- **Tuned on one GPU.** The defaults were measured on an RTX 3070. They should be sane
  elsewhere, but no AMD or Intel device has been benchmarked.

## Name

"Mersenne, trial factoring". It divides — it does **not** implement P-1, ECM or any
primality test.
