# Negative results

Kept because the README register promises them, and because the failures are
more informative than the successes.

## 2026-09-14: first divergence run does not read

**Run:** 30ms simulated, 300 steps, chamber 0 heated, full 166,700-neuron network.

**What was expected:** five chambers, one heated, visibly separating on a rate plot.

**What happened:** the divergence is real and causal but one neuron wide. At step
216 chamber 0 had 118 active neurons in the recorded subset against chamber 1's
117, a symmetric difference of exactly 1. By step 250 the two were identical
again. Total spike counts over the run were bit-identical across all six
instances (475,695 each), and the maximum per-step rate difference was 1 spike.

The plot shows six traces that look the same, because they very nearly are.

**Three causes, all mine, none of them the connectome's:**

1. **The heat barely arrived.** `Heat.ramp_steps` is 2000 (200ms at dt=0.1ms)
   but the run was 300 steps, so the stimulus reached 15% of its 8.0mV target
   and spent the whole run ramping. The chamber was warmed, not heated.

2. **The network was oscillating in lockstep.** Feeding 6.0mV into 25
   thermoreceptors every single step drove a strong synchronised population
   rhythm at roughly 2.3ms. That is the drive ringing through the network, not
   fly-like activity, and a globally synchronised network resists divergence:
   a perturbation gets swamped by the next drive cycle.

3. **Population rate is the wrong observable.** Chamber 0 genuinely diverged in
   *which* neurons fired while the *count* stayed the same, so a rate trace
   hides precisely the thing the piece is about. Identical totals were being
   read as identical behaviour.

**Kept anyway:** the causal chain is sound. Gate 2 passes on the real network,
so heating chamber 0 provably leaves the other four and the operator
bit-identical to a control. The mechanism works; the parameters and the
observable were wrong.

**Changes:** shorten the ramp, drive with a sparser non-tonic input, and measure
divergence on spike identity rather than on population rate.

### Resolution

**Run:** 60ms, 600 steps, chamber 0 heated, phasic drive into 698 strided
`cb_sensory` neurons, 0.5ms pulse every 10ms, heat ramp shortened to 20ms.

Heat reached its full 8.0mV target. Divergence began at step 18, about 1.8ms,
which is one synaptic delay after the stimulus arrived. The heated chamber now
differs from the others by roughly 700 neurons per step, with a cumulative
spike-identity distance of 140,439 over the run, against the 1 neuron at 1 step
that the tonic version produced.

Total spike counts were 999,772 for the heated chamber against 998,825 for each
of the other five, a difference of 947 spikes. Note the five unheated instances
still matched each other exactly, which is the isolation property holding while
the heated one moves.

So all three causes were real and all three were mine. The connectome was never
the problem.

**Retained lesson:** a synchronised network resists divergence. Tonic drive is
the natural thing to reach for and it quietly destroys the phenomenon, because
every perturbation is overwritten by the next drive cycle before it can
propagate. Phasic drive leaves the network free to carry a difference forward.

## 2026-09-14: the dial reaches only two of five chambers

**Test:** replayed 300 steps of real operator descending-neuron activity (6,500
DN spike events across 1,035 distinct DNs) through the dial readout.

**What works:** the dial moves. 17 switches over the window, so the readout is
responsive to real spiking rather than sitting stuck.

**What does not:** it visited only chambers 0 and 1. Block spike totals at the
end of the window were `[228, 231, 150, 152, 184]`. Blocks 0 and 1 are
persistently more active than the rest, so an argmax over raw counts almost
never selects blocks 2, 3 or 4.

Three of the five chambers would never be heated. The piece is five chambers and
one operator; a dial that can only reach two of them is not the piece.

**Why it happens:** the blocks are contiguous slices of sorted bodyId, and DN
firing rates are not uniform across that ordering. Splitting 1,314 DNs into five
equal-sized blocks equalises *neuron count*, not *activity*. Nothing forces the
five blocks to be comparably active, and measurably they are not.

**Note the partition itself is fine.** Measured, the blocks hold 263/263/263/
263/262 DNs spanning 181/183/142/81/180 distinct DN types, so no block is a
single functional group and the mapping carries no anatomical claim. The problem
is purely that raw argmax over unequal baselines has a fixed winner.

**Options, none of them chosen yet:**

1. Normalise each block by its own running baseline, so the dial responds to
   which block is *unusually* active rather than which is loudest. Keeps the
   rule legible and keeps it driven by actual spikes.
2. Partition by activity instead of by index, so the five blocks are matched at
   rest. Requires a calibration pass, and the split then depends on a prior run.
3. Leave it, and accept that the operator has favourites. Defensible as a
   statement, but it makes three chambers decorative, and a viewer counting
   chambers will notice.

Option 1 is the likely fix: it preserves "driven by actual spike activity, not
an RNG with a skin on it" while removing a fixed winner that is an artefact of
bodyId ordering rather than anything the fly is doing.

### Resolution

Each block is now divided by its own slow baseline before the argmax, so the
dial responds to which block is *unusually* active rather than which is loudest.

Replayed against the same 300 steps of real operator DN activity:

| | before | after |
|---|---|---|
| chambers reached | 0, 1 | 0, 1, 2, 3, 4 |
| switches | 17 | 59 |
| time per chamber | not measured | 86, 61, 31, 90, 32 |

All five reachable, and no chamber is decorative. Still driven entirely by
spikes, still no RNG, and the rule is still one paragraph to read.

The 700ms run in flight at the time was killed at step 2900 of 7000, about 11
minutes in, because it was using the old readout and would have produced a clip
in which three of the five chambers were never heated.

## 2026-09-14: the operator appeared to diverge, and had not

**Run:** r002, 300ms, 3000 steps, dial latency shortened to 40ms so switches
were visible. 23 dial switches, all five chambers heated.

**What the run reported:**

```
instance 0 diverged at step 18
instance 2 diverged at step 426
instance 3 diverged at step 458
instance 4 diverged at step 495
instance 5 diverged at step 1502    <- the operator
```

Instance 5 is the operator. It is never heated, and `Heat.injection()` provably
never writes its column: verified directly, its maximum heat over the whole run
is exactly 0.0.

**Cause: the divergence reference was itself being heated.** The reference was
`CHAMBERS[1]`, chosen earlier precisely because it was unheated. That was true
while the dial sat still. Once the dial moved, chamber 1 was selected at step
435 and started being heated, so every subsequent "distance from the reference"
was distance from a moving target. The operator looked like it diverged because
the thing it was being compared against had changed.

Note instance 2 diverged at 426, slightly before 435, so that one is its own
genuine heating. Everything after 435 is contaminated.

**Fix:** the reference is now the operator, which is the only instance
structurally guaranteed never to be heated. Chambers are all heatable by
definition, so no chamber can serve as a stable reference in a run where the
dial moves.

**Retained lesson:** "currently unheated" is not the same property as "cannot be
heated". The first was an observation about one run and the second is a
structural guarantee, and only the second is safe to build a measurement on.

Worth stating plainly: the piece was not broken. The measurement was. But the
run reported "the operator diverged", which is exactly the claim the piece
must not make falsely, and it would have been rendered into a clip.

## 2026-09-14: the dial chatters, and the latency does not pace it

**Run:** r003, 3000ms, 30,000 steps, authored 500ms dial latency, GPU.

**What it reported:** 725 dial switches. Every chamber heated to the full 8.0mV.
All five diverged. The operator at exactly 0.

**What actually happened:** the median dial hold is **4 steps**, and 688 of the
725 holds are shorter than 100 steps (10ms). Sampling the log every 100 steps
showed a handful of slow, deliberate-looking switches; that was an aliasing
artefact. Underneath, the dial re-decides roughly ten thousand times a second.

**The latency is not a pacing mechanism.** `DIAL_LATENCY_MS` delays when a
decision takes effect, but a fresh decision is computed every step and queued.
The result is a 500ms-delayed copy of a stream that changes every 0.1ms, not a
dial that holds a position for 500ms. I had assumed the latency implied
hysteresis. It does not.

**Consequences, all visible in the numbers:**

- Cumulative distance: chamber 0 reached 43,548,286 while chambers 1, 2, 3 and 4
  reached 1,313, 1,185, 1,560 and 1,465. Chamber 0 is the only one heated
  continuously, from step 18 before the dial began moving. Every other chamber
  is heated in 4-step slivers that leave almost nothing behind.
- Chamber 3 was reported as re-converging at step 29,727. That is not a finding
  about the connectome, it is the chatter.
- Peak heat reaching 8.0mV in every chamber is misleading: the ramp climbs while
  a chamber is selected and decays when it is not, so rapid switching lets all
  five touch the ceiling without any of them being meaningfully heated.

**What this does not affect:** the operator still sits at exactly 0, so
isolation holds. Divergence onsets are still real. The simulation is correct;
the control signal on top of it is not doing what the piece needs.

**Fix:** the dial needs hysteresis, a minimum dwell time once it commits to a
chamber, so a decision persists long enough for the heat to matter and for a
viewer to read the causality. Latency and dwell are different properties and the
piece needs both.
