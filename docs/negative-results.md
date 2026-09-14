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
