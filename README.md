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

## What is measured, and what is authored

**Measured.** 166,700 neurons, 25,582,938 connections, 124,177,617 synapses.
MaleCNS v1.0, Janelia FlyEM and Google Research, CC BY 4.0. Counts computed
from the pinned files and asserted in the test suite. Soma coordinates for
141,781 neurons from the public neuPrint API. The 340 dopaminergic neurons
reward goes into: PAM 316, PPL1 16, PPL2 8. The 25 TRN_VP thermoreceptors the
operator's punishment arrives through.

**Authored.**

- The electrode site, on DNp01-left. The pair sits 21,114 units apart, so the
  stimulation is unilateral
- The two-channel limit. A rule of the piece, not a property of the fly
- The 100% reward and 50% neglect thresholds
- The 0.40 switch margin
- The starting levels, seeded apart deliberately: flies that begin identical
  lock onto one trajectory and never separate
- The convulsion, below

## The convulsion is not an escape reflex

The bodies move because current is injected directly into 708 VNC motor
neurons. That is authored, and it is not the fly escaping.

The escape circuit **is** in the dataset, DNp01 plus DNp02, 03, 04, 09 and 11,
and it is reachable from the stimulation site. Driving it produces no motor
change at any amplitude: 3,713 motor spikes at 20 mV against 3,686 at 400 mV,
and against a realistic baseline it comes out at **0.92x**, marginally fewer
than with escape off.

The alarm bell is wired, reachable, and ringing it does nothing. Driving the
motor stage directly reaches 2.00x at 60 mV, which is what the bodies are
doing on screen.

## Nothing here experiences anything

Stimulating PAM is current injected into neurons that participate in
reinforcement learning in a real fly. **Nothing in this simulation experiences
reward, and nothing experiences pain.** A neuron here integrates input and
spikes. It cannot be damaged, cannot die, and has no state that harm would
change.

The piece is about determinism: six identical programs, differing only in what
is done to them.

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
