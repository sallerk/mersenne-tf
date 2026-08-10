# mersenne_tf

GPU trial factoring of Mersenne numbers, `M_p = 2^p - 1`, with exact integer arithmetic.

| version | |
|---|---|
| [**1.0**](1.0/) | current. Job lives in `worktodo.txt`; results are written in GIMPS manual-submission format. Start here — see [1.0/README.md](1.0/README.md). |
| [0.9](0.9/) | previous. Job and settings both in `config.txt`; results in its own format. Kept as-is. |

## Quick start

```bash
cd 1.0
build.bat
mersenne_tf.exe --selftest
mersenne_tf.exe
```

Edit `worktodo.txt` to say what to factor, `config.txt` to say which GPU to use.
