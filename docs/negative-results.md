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

## 2026-09-14: the dwell fix worked, and exposed a deeper limit

**Run:** r004, 3000ms, 30,000 steps, 300ms dwell added, GPU.

**The dwell fix is confirmed.** Switches dropped from 725 to 8, exactly the
maximum the arithmetic allows (25,000 usable steps / 3,000 dwell). Median hold
is 3,000 steps, precisely the dwell, with zero holds under 100 steps against 688
before. The sequence `0, 3, 0, 3, 0, 2, 4, 1, 3` visits all five chambers, and
every chamber spent at least 2,805 steps at full 8.0mV heat.

**And it did not make the piece readable.** Per chamber:

| chamber | steps at full heat | total distance | peak |
|---|---|---|---|
| 0 | 13,637 | 43,548,269 | 2,102 |
| 1 | 2,805 | 264 | **1** |
| 2 | 2,805 | 612 | **1** |
| 3 | 6,191 | 1,290 | **1** |
| 4 | 2,805 | 436 | **1** |
| operator | 0 | 0 | 0 |

Chambers 1 to 4 reach a peak spike-identity distance of exactly **one neuron**
and stay there, across thousands of steps of full heat. Chamber 3 was heated for
6,191 steps and never exceeded 1. Chamber 0 passed 1,138 within its first 1,000
steps.

**It is not duration and it is not intensity.** Both were ample. The difference
is *when* the heat arrives. Chamber 0 is heated from step 18, while the network
is still settling out of its initial conditions. Every other chamber is heated
into an already-established driven rhythm, and that rhythm absorbs the
perturbation: one neuron flips, the next drive pulse overwrites it, nothing
compounds.

This is the same phenomenon as the tonic-drive failure recorded above, in a
subtler form. A network locked to a periodic drive is resistant to perturbation
regardless of how large the perturbation is, because the drive re-synchronises
it every cycle. Making the drive phasic was enough to let a *settling* network
diverge. It is not enough to let a *settled* one diverge.

**What is not in doubt:** the operator is exactly 0 across the whole run, so
isolation holds, and chamber 0 demonstrates the mechanism works when the network
is susceptible. The simulation is correct. The regime is wrong.

**Options, none chosen yet:**

1. Weaker or sparser background drive, so the network is not locked to a rhythm
   that overwrites perturbations. Risks the network going silent.
2. Heat a larger population than the 25 TRN_VP thermoreceptors, so the
   perturbation is large relative to the drive.
3. Start with the dial already moving, so no chamber gets the privileged
   settling-phase heating that makes chamber 0 incomparable to the rest.

Option 1 is the likely one, and it is the same lesson twice: a synchronised
network does not diverge.

### Correction: it is fan-out, not drive synchronisation

The explanation above, that a settled network locked to a periodic drive absorbs
the perturbation, is **wrong**. A sweep of six regimes, all heating only after
the network had settled, says so plainly:

| regime | peak distance | mean rate |
|---|---|---|
| baseline (r004) | 1 | 1,653 |
| weaker drive, 2mV | 1 | 1,646 |
| sparser drive, every 21st neuron | 1 | 1,688 |
| slower drive, every 50ms | 1 | 1,662 |
| **heat all 4,882 cb_sensory** | **1,446** | 1,661 |
| heat 200mV into TRN_VP (25x) | 1 | 1,653 |

Changing the drive does nothing. Raising heat intensity 25x does nothing. The
only variable that mattered was **how many neurons the heat is injected into**.

The cause is fan-out:

| | TRN_VP | cb_sensory |
|---|---|---|
| neurons | 25 | 4,882 |
| directly downstream | 551 | 10,566 |
| outgoing synapses | 50,329 | 2,267,868 |

45x the synaptic mass. Twenty-five thermoreceptors cannot inject enough signal
into a 166,700-neuron network to move its trajectory, and turning the voltage up
does not help because those 25 cells saturate: a neuron can only spike so often,
so past threshold the extra millivolts buy nothing.

Chamber 0 diverged in earlier runs not because it was heated during the settling
phase, but because it was heated **continuously for 13,637 steps**, accumulating
a great many one-neuron perturbations, while the others were heated in blocks of
2,805. The settling-phase story fit the data and was not the mechanism.

**Consequence for the piece.** `TRN_VP` is the anatomically correct
thermoreceptor population and it is too small to carry the premise. The choice
is between:

1. Heating a broader sensory population, which diverges convincingly but is no
   longer specifically "heat" and must not be described as such.
2. Keeping TRN_VP and accepting that a chamber must be heated for thousands of
   steps before it separates, which is honest but makes the dial nearly
   irrelevant, since chamber 0's dominance comes from duration alone.

Not yet decided. Option 1 must not be labelled heat if it is taken.

## 2026-09-14: the dials started mid-range, and that alone broke the piece

Five continuous dials replaced the single switch. The dials worked: driven by
real operator activity they reached genuinely different levels. The chambers
still came out identical to each other, and it took a long chain of eliminations
to find why.

**Ruled out, each by measurement:**

- *Too few thermoreceptors.* No. Five instances held at constant 0, 2, 4, 6 and
  8 mV separate by about 2,400 neurons. The 25 TRN_VP cells discriminate fine.
- *The background drive masking the heat.* No. All 25 thermoreceptors are inside
  `cb_sensory` and the drive hits 698 of those, but excluding them, including
  them and removing the drive entirely all separated correctly.
- *Injecting millivolts instead of firing rates.* No. `tel-0s/flyverse-core`
  drives sensory neurons in Hz (`base_hz=1.0, max_hz=150.0`) rather than as
  injected voltage, which is the better convention and worth adopting, but
  rate-coding the same ramp gave 1682 for all five: identical, unchanged.
- *Cumulative measurement swamped by a shared opening.* Partly true and not the
  cause. A trailing 500-step window still gave 847,939 against 847,948 for
  chambers at 8.0 mV and 0.0 mV.

**The actual cause.** The dials were initialised to 0.5, half scale, so every
chamber began *identically heated* at 4 mV. The network locks onto a shared
trajectory from the first step and never escapes it, however far the levels
diverge afterwards. Starting mid-range was my own change, made earlier the same
day so that levels could move both directions from the outset.

The comparison that isolates it, same brain, same drive, same final levels:

| onset | distance from operator |
|---|---|
| constant 8.0/8.0/0.6/0.0/0.0 from step 0 | 1663, 1663, 1342, 0, 0 |
| linear ramp from **zero**, then hold | 979, 979, 349, 0, 0 |
| **abrupt switch-on at step 1000** | **0, 0, 0, 0, 0** |
| the dials' own history, starting at 0.5 | 1678, 1678, 1678, 1678, 1678 |

Two things fall out of that table. Ramping is fine, so long as it starts from
zero. And heat that arrives late does nothing at all: a chamber switched to full
heat at step 1000 and held there for another 1000 steps ended at distance zero.

**The rule.** The chambers must differ from the very first step. From zero they
do, because each rises at its own rate. From a shared nonzero start they never
recover, and no amount of later separation in the dial levels repairs it.

**After the fix**, levels 58%, 89%, 0%, 0%, 0% gave distances 68463, 68464, 106,
0, 0 with the operator at 0: heated chambers diverge, untouched chambers sit at
exactly zero.

## 2026-09-14: turning the heat up does not help, and past 8 mV it actively hurts

The obvious response to a weak signal is more of it. Measured, across four
amplitudes and two onsets, 3,000 steps, five chambers at amp, 0.75, 0.5, 0.25
and 0 of full:

| amplitude | onset | distance from operator |
|---|---|---|
| 8 mV | step 0 | 2622, 2411, 2360, 2361, 0 |
| 40 mV | step 0 | 2622, 2622, 2622, 2622, 0 |
| 200 mV | step 0 | 2622, 2622, 2622, 2622, 0 |
| 1000 mV | step 0 | 2622, 2622, 2622, 2622, 0 |
| 8 mV | step 5000 | **0, 0, 0, 0, 0** |
| 200 mV | step 5000 | **0, 0, 0, 0, 0** |
| 1000 mV | step 5000 | **0, 0, 0, 0, 0** |

**Two findings, both against intuition.**

**More heat destroys the gradient.** At 8 mV the five chambers are
distinguishable from one another. At 40 mV and above they are identical: every
chamber reads 2622 and 2170 spikes regardless of whether it was set to 100% or
25%. The thermoreceptors saturate. A neuron has a refractory period of 2.2 ms,
so it cannot exceed roughly 450 Hz however hard it is driven, and once all 25
are at that ceiling the dial position stops carrying information. **8 mV is
near the top of the useful range, not the bottom.**

**Late heat does nothing at any amplitude.** Starting at step 5000 gives
distance zero at 8 mV, at 200 mV, and at 1000 mV. A 125-fold increase changes
nothing at all. Whatever the mechanism, it is not something amplitude reaches.

**Consequence.** The only lever that works is *when* the heat starts, not how
hard. Chambers must be heated from step 0 or they never separate. That collides
directly with the authored 500 ms dial latency, which guarantees nothing is
heated for the first 5,000 steps. The two cannot both be kept as they are.
