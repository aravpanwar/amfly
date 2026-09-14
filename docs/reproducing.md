# Reproducing this

Everything published here is computed from pinned files and asserted in the
test suite, so you should get the same numbers. If you do not, that is a real
finding and worth reporting.

## 1. Install

```
pip install -e .[dev]
```

CPU works and is the reference. For clips longer than a few hundred
milliseconds you want a CUDA build of torch, because the CPU path runs about
200ms per 0.1ms step at 166,700 neurons, which is roughly 25 minutes per 700ms
of simulated time.

```
python -c "from amfly.sim.backend import describe; print(describe())"
```

`cuda_available: false` means the slow path. `torch` from PyPI is CPU-only by
default.

## 2. Fetch the connectome

```
python scripts/fetch_data.py --out data
```

About 1.1GB across three files, verified by SHA-256 against
`data/manifest.json`. It refuses to proceed on a mismatch, which is deliberate:
a silently updated release would change the numbers underneath you.

`syn-points` and `syn-partners` are not fetched. They are per-synapse
coordinates, another 19.5GB, and LIF dynamics do not use them.

## 3. Verify the counts

```
python -m pytest tests/ -q
```

Expect these exactly:

| | |
|---|---|
| Neurons | 166,700 |
| Directed connections | 25,582,938 |
| Synapses | 124,177,617 |
| Descending neurons | 1,314 |
| Thermoreceptors (`TRN_VP*`) | 25 |

Without the dataset present the slow gates skip and the rest still run:

```
python -m pytest tests/ -q -m "not slow"
```

## 4. Run it

```
python scripts/run_sim.py --data data --ms 700 --out runs/001
```

700ms is the minimum worth running, because the dial commits its decision 500ms
ahead and a shorter run shows no movement at all.

## 5. Figures and clip

```
python scripts/make_figures.py --run runs/001 --out out/
```

Stills always. The mp4 and gif need ffmpeg on PATH, and their absence is
reported rather than silently skipped.

## What you should see

With chamber 0 heated, measured on the full network:

- Divergence begins around step 18, about 1.8ms, which is one synaptic delay
  after the stimulus arrives.
- The heated chamber differs by a few hundred neurons per step.
- The other four chambers and the operator stay bit-identical to each other for
  the whole run, distance exactly 0.

That last line is the one to check. If unheated chambers drift apart, something
is wrong: either determinism settings are off, or a non-deterministic reduction
has crept in. It is not the piece working.

## Determinism caveat

Bit-identity is guaranteed within one machine and one configuration. Identical
results across different GPUs, CUDA versions or torch builds are not promised,
which is why every run records its backend in `provenance.json`.
