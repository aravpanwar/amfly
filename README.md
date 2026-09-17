# amfly

![the whole bench from above](docs/media/overhead.gif)

Six copies of one fly brain, 166,700 neurons each, running from the same
measured wiring.

All six are put on an electric pin. One of them is in a reward-punishment loop
and can only use two channels at once, and each channel runs to one of the
remaining five. When a fly hits 100% the operator fly gets dopamine. Every fly
below 50% burns the operator fly instead.

![the operator fly, dopamine and heat feeds meeting at its head](docs/media/operator.gif)

## Install

Python 3.10 or later.

```bash
git clone https://github.com/aravpanwar/amfly
cd amfly
pip install -e .
```

For the GPU path, install a CUDA build of torch separately. The CPU build is
about 90x slower here and is not a supported way to produce clips.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Get the data

```bash
python scripts/fetch_data.py --out data
```

About 1.1 GB. Three files from MaleCNS v1.0, verified by SHA-256 against
`data/manifest.json`, so a truncated or silently updated download fails here
rather than three milestones later when the numbers have quietly changed.

The 12.7 GB `syn-points` and 6.8 GB `syn-partners` files are not needed. The
aggregated weights table is what a LIF model consumes.

## Run it

```bash
python scripts/run_sim.py --data data --ms 3000 --record-sample 20000 --out runs/mine
```

About 10 minutes on an RTX 4050 for 3 simulated seconds. Roughly 73 minutes on
CPU.

Useful flags: `--heat` restores the original thermal channel, `--no-convulse`
runs the electrode without the direct motor drive, `--switch-margin` and
`--press-rate` control the operator's pacing.

## See it

```bash
python scripts/export_web.py --run runs/mine --out web/data.json --seconds 30
python scripts/export_brain.py --run runs/mine --points 24000
cd web && python -m http.server 8777
```

Then open `localhost:8777`.

| key | |
|---|---|
| **F** | take: restart playback and run the camera move |
| **C** | camera move only |
| **V** | free camera (WASD, Q/E, mouse) |
| **M** | sound |

Screen record it. There is no built-in capture, because the readout, the bars
and the brain tiles are HTML over the canvas and a canvas recorder loses all
of them.

![a fly convulsing under stimulation](docs/media/convulsion.gif)

## Figures

```bash
python scripts/make_figures.py --run runs/mine --out out/mine --stills-only
```

## Tests

```bash
python -m pytest tests/ -q -m "not slow"
```

## A look around

![a walkthrough of the scene](docs/media/tour.gif)

---

## The data

MaleCNS v1.0 from Janelia FlyEM and Google Research, CC BY 4.0. 166,700
neurons, 25,582,938 connections, 124,177,617 synapses. Those counts are
computed from the pinned files rather than copied from anywhere, and the tests
assert them.

Soma coordinates for 141,781 of the neurons come from the public neuPrint API.
Reward goes into 340 real dopaminergic neurons (PAM 316, PPL1 16, PPL2 8) and
the operator's punishment arrives through its own 25 TRN_VP thermoreceptors.

## What I made up

The wiring is real. Most of the rest is not.

- The electrode sits on DNp01-left. Its pair is 21,114 units away, so only one
  side gets stimulated
- Two channels at once, the 100% and 50% thresholds, and the 0.40 switch
  margin are all numbers I picked
- The flies start at different levels on purpose. Start them identical and
  they lock onto one trajectory and never separate
- The convulsion is current injected straight into 708 motor neurons. The
  bodies move because they are driven, not because the fly is escaping. The
  escape circuit is in there and reachable, and driving it does nothing at any
  amplitude

## To be clear

Nothing in this simulation feels anything. A neuron here adds up its inputs and
spikes. It cannot be hurt, cannot die, and has no state that damage would
change. "Dopamine" means current going into cells that do reinforcement
learning in a real fly, and that is all it means.

## Limitations

- Shiu et al. 2024 is brain-only, about 127K neurons. This runs whole-CNS at
  166,700, which is our extension and not theirs
- `dt = 0.1 ms` is the Brian2 default, not a published parameter
- Bit-identity holds within one machine and configuration. Cross-GPU identity
  is not promised
- The electrode reaches 5,579 neurons across seven superclasses. 16% of the
  network has no soma coordinate and cannot be reached at all
- Leg and wing motion is amplified from a real but small signal
- 18,530 of 24,000 rendered brain points carry per-neuron activity; the rest
  are structure

## Licence

MIT. MaleCNS v1.0 is CC BY 4.0, so credit Janelia FlyEM and Google Research.
Cite Shiu et al., *Nature* 2024 for the LIF model.
