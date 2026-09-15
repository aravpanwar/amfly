# Getting to 3D

What it would take to put the six flies in bodies, what is genuinely hard about
it, and what can be had cheaply along the way.

Written after reading the closest precedents through the GitHub API rather than
guessing. The short version: the rendering is easy, the bodies are not, and the
reason is a real open problem rather than missing effort.

## The state of the art, verified

`artem-x-meta/fly-arena` is the nearest existing thing: MaleCNS connectome plus
FlyGym/MuJoCo physics, actively developed. Its own README says:

> Motor neurons are not individually mapped to muscles.

> FlyGym supplies an engineered walking controller.

> Running the full graph does not demonstrate that natural behaviour emerges.

Its `fly_arena/motor.py` opens with "engineered motor primitives and scratch-data
IK" and carries hand-written states: `"hold"`, `"IDLE"`, `escape_turn`,
`feed_mouth_targets`. The fly walks because someone wrote a gait and an inverse
kinematics solver, not because the connectome drives the legs.

`TuragaLab/flybody` (848 stars) is the body model, and it is trained with
reinforcement learning against a reward, not driven by a connectome.
`NeLy-EPFL/flygym` (311 stars) is the same picture.

**So there is no working connectome-drives-body pipeline in public.** Every
project either runs the brain or runs the body and couples them loosely. That is
the thing to know before promising a 3D fly that moves because it was heated.

## What we already have for it

| piece | status |
|---|---|
| 708 VNC motor neurons, with real muscle types (`Acc. ti flexor MN`, `DLMn a, b`) | present in the loader |
| 1,314 descending neurons | already driving the dials |
| 1,846 ascending neurons, for body-to-brain feedback | present, unused |
| Six brains running deterministically at 17 ms/step on the GPU | done |
| Heat producing measurable divergence | done |

The motor neurons are there and typed. That is the raw material; it is the
mapping from their spikes to muscle forces that does not exist.

## Three levels, in order

### Level 1: chambers, no bodies (half a day, no research risk)

A Three.js scene. Six boxes. Each chamber's colour and glow driven by its heat
level, straight from the recorded `heat` array. Six spike-trace panels beside
them, the same data the 2D figures already use.

No fly models, no physics, no motor neurons. Pure playback of arrays we already
dump. This is the version that looks like a piece and carries no risk at all.

**Do this one first.** It is a rendering job, not a research job, and it gets the
project a shareable clip without claiming anything untrue.

### Level 2: bodies that sit there and twitch (2-3 days, modest risk)

Load the flybody mesh (Apache-2.0, `TuragaLab/flybody`) into the Three.js scene.
Six of them, one per chamber, posed and still.

Then drive *something* visible from real motor-neuron spikes: leg twitch
amplitude, wing buzz rate, a posture that tenses as a chamber heats. Map the 708
motor neurons to a handful of aggregate signals rather than to individual
muscles, and say in the README that it is an aggregate readout.

This is honest and achievable. It is also where the piece would actually land
emotionally: six flies, one visibly more agitated than the others, because of
what the sixth one is doing to a dial.

**Measured, 2026-09-14: heat does reach the motor neurons, and the response is
monotonic in heat.** Five chambers held at 8, 6, 4, 2 and 0 mV from step 0, over
3,000 steps, total spikes across the 708 `vnc_motor` neurons:

| chamber | heat | motor spikes | vs operator |
|---|---|---|---|
| 0 | 8 mV | 30,891 | **+3.1%** |
| 1 | 6 mV | 30,711 | +2.5% |
| 2 | 4 mV | 30,022 | +0.2% |
| 3 | 2 mV | 29,701 | -0.9% |
| 4 | 0 mV | 29,963 | +0.0% |
| operator | never heated | 29,963 | reference |

The unheated chamber matches the operator exactly, which is the control working.
The ordering is right and the signal is real.

**But 3% is small.** Averaged over 708 neurons that is roughly 0.3 extra spikes
per step at full heat, against a baseline of about 10. Level 2 is therefore
viable but needs care:

- Drive a *cumulative* or *integrated* quantity, not an instantaneous one. A 3%
  rate difference integrated over seconds is visible; frame to frame it is
  nothing.
- Amplify honestly and say so. Mapping 0 to 3% onto 0 to 100% of a posture range
  is an authored visual gain, and the README must state the real number beside
  it. `project.md`'s register does not allow an unlabelled 30x exaggeration.
- Expect subtlety rather than thrashing. Five flies with slightly different
  tension is the truthful picture, not one convulsing while four sit still.

Do not start Level 2 expecting dramatic motion. Start it expecting a small,
real, monotonic signal that has to be rendered carefully to be seen at all.

### Level 3: flies that walk because they were heated (weeks, may not work)

MuJoCo physics per fly, motor neurons mapped to actuators, closed loop with
ascending feedback. Six physics sims plus six brains.

This is the part nobody has done. The obstacles, in order of how likely they are
to stop it:

1. **Spikes to muscle force has no established mapping.** Rate-code each motor
   neuron into an actuator target and the result is noise, not walking. This is
   the whole problem, and `fly-arena` sidestepped it with engineered gaits.
2. **A fly that cannot stand shows nothing.** Without a working gait the six
   flies collapse identically and the divergence is invisible regardless of
   whether it is real.
3. **Cost.** Six MuJoCo flybody instances at 0.2 ms steps with adhesive claws,
   alongside six 166,700-neuron brains, on a 6 GB laptop.

`project.md` already flagged 3 and recommended Rapier over MuJoCo. Obstacles 1
and 2 are the ones that decide whether this is possible at all.

**If Level 3 is attempted, the honest framing is the one `fly-arena` uses: an
engineered gait whose parameters are modulated by connectome activity.** Not
"the connectome walks the fly". Claiming the latter is the thing that gets a
project taken apart in the comments, and `project.md`'s register rules it out.

## Recommended order

1. Finish the 2D clip. It is close and it is what proves the piece is real.
2. Measure whether heat changes motor-neuron activity. One hour, decides Level 2.
3. Build Level 1, the glowing chambers. Half a day, no risk, shareable.
4. Only then decide on Level 2, with the measurement in hand.
5. Treat Level 3 as a separate project with its own README and its own negative
   results, not as a promise attached to this one.

## The rule that should not bend

`project.md`: *"If the divergence doesn't read as meaningful on a plot, no amount
of rendering will save it."*

Rendering makes a true thing legible. It cannot make an unproven thing true. Any
3D work that arrives before the plot is convincing is decoration, and the shelf
is already full of decorated flies.

## Reference implementation, pulled 2026-09-15

`Lulzx/fly-brain` is **MIT licensed**, so its code and assets can be reused with
attribution. The repo itself is 249 MB, nearly all bundled connectome data we
already have, so pull individual files rather than cloning:

```
gh api repos/Lulzx/fly-brain/contents/src/arena.js --jq '.content' | base64 -d > ref/arena.js
```

**What is worth taking:**

| file | why |
|---|---|
| `src/arena.js` (452 lines) | the whole browser scene: Three.js, OrbitControls, RoomEnvironment, EffectComposer with GTAO ambient occlusion |
| `src/arena-batches.js` | instanced rendering for many flies at once, which is exactly the six-chamber case |
| `src/render-resolution.js` | resolution scaling, for keeping frame rate on a laptop |
| `art/fly/fly-full-body.blend` | **an actual fly model**, MIT. Needs exporting to .glb for the web. |

`fly-brain` is the only project found that gets a connectome simulation running
in a browser at all, which is why it is the right reference for Level 1 and
Level 2. Note it simulates a single fly; the six-instance batching is ours.

**What is not worth taking:** their simulation approach. `fly-arena` and
`flybody` both hand-author motion, and `fly-brain` renders one fly rather than a
comparison of several, so there is no existing technique for the thing this
piece actually shows.
