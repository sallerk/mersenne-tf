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
