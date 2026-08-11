# mersenne_tf

GPU trial factoring of Mersenne numbers, `M_p = 2^p - 1`, with exact integer arithmetic.

| version | |
|---|---|
| [**1.3**](1.3/) | current. The device sieve stopped scaling with the prime count, so it now pays to sieve deep — 1.14x over 1.2. Start here — see [1.3/README.md](1.3/README.md). |
| [1.2](1.2/) | previous. Moved the sieve to the GPU — the whole pipeline is device-side. Kept as-is. |
| [1.1](1.1/) | Reports the sieve bound it actually applies, not the one you asked for. Kept as-is. |
| [1.0](1.0/) | job moved to `worktodo.txt`; results in GIMPS manual-submission format. Kept as-is. |
| [0.9](0.9/) | job and settings both in `config.txt`; results in its own format. Kept as-is. |

What changed between them is in [CHANGELOG.md](CHANGELOG.md).

## Quick start

```bash
cd 1.3
build.bat
mersenne_tf.exe --selftest
mersenne_tf.exe
```

Edit `worktodo.txt` to say what to factor, `config.txt` to say which GPU to use.
