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
