# From heat to electrical stimulation

Planned 2026-09-16. Not yet built.

## Why change at all

"Heat" implies cooking, and the model cannot support that. There is no tissue
damage, no nociception, and nothing that degrades: a neuron here integrates
input and spikes, and it cannot be injured. `docs/what-heat-is.md` sets out
that gap in full, and it is the weakest claim in the piece.

Electrical stimulation is a better fit for what the simulation literally does.
It injects millivolts of depolarising current into neurons. That was always
the mechanism; heat was an interpretive layer on top of it, and one that
promised more than the model delivers.

**Millivolts is already the real unit.** Volts is therefore more honest than
degrees, not less.

## The finding that makes this possible

`data/soma.npz` holds real soma coordinates for **141,781 of 166,700 neurons**
(85.1%), fetched from the public neuPrint API.

That means an electrode can be placed at an actual 3D position and stimulate
by physical distance, which is exactly what a real electrode does. No
hand-picking of cell types, and no "why those neurons?" to answer.

**The thermoreceptors have no coordinates at all: 0 of 25.** They are sensory
afferents whose cell bodies sit in the periphery. So the current heat channel
could never have been placed physically even in principle. The electrode model
is the first version of this that has a location.

## The stimulation model

A point electrode at a chosen site. Every neuron gets current scaled by
distance, on a Gaussian falloff:

    injection(i) = amplitude * exp(-(d_i / radius)^2)

Measured at the candidate site, with radius 6000: 1,418 neurons within one
radius, 952 in falloff-weighted terms. That is a local population, not a
labelled line, which is the point. An electrode excites what is near it.

Neurons with no coordinates receive nothing, and that is declared: 14.9% of
the network is unreachable by the electrode because neuPrint has no soma
position for it.

## Where the electrode goes

On `DNp01`, the giant fibre escape command neuron, which IS in the dataset.

Measured: two neurons, a bilateral pair 21,114 units apart, both with
coordinates. An electrode on one captures that neuron plus ~1,400 local
others at radius 6000. Placing one electrode near both is not possible at any
plausible radius, so the stimulation is unilateral and must be described that
way.

This site is chosen, and the README says so. What is NOT chosen is which
neurons it then excites: that follows from measured soma positions.

## What this fixes

The fly currently sits still while being hurt, because warmth does not drive
escape. Measured reachability from the thermoreceptors: 0 escape neurons at
one hop, 5 at two, all 36 at three. The alarm bell is wired but never rung.

An electrode on `DNp01` drives the escape pathway directly, so the fly can
actually convulse. That is an authored intervention standing in for a
nociceptive input the connectome does not contain, and it gets labelled as
exactly that. It is not an emergent response and must never be presented as
one.

## What does NOT change

The compulsion loop reads a chamber level in 0..1 and does not know what that
level represents. Confirmed by reading it: reward fires at 100%, punishment
from chambers below 50%, and both are downstream of a single number.

So reward, punishment, habituation, escalating neglect, the grip of two, the
switching margin and every recorded result survive the rename untouched.

The operator's punishment still arrives through its own thermoreceptors,
which is now a deliberate asymmetry worth stating: the five are stimulated
electrically, and the one at the console is burned. Different mechanisms,
because they are different acts.

## Units

Chamber level becomes **millivolts at the electrode**, which is the actual
simulation parameter rather than a metaphor. The UI reads mV. The 8 mV ceiling
stays: measured, above 40 mV the response saturates and a chamber at 25% is
indistinguishable from one at 100%.

## Open question

Continuous level, or discrete pulses? Pulses would look better in a clip and
match how stimulation is really delivered, but they change the mechanism
rather than the label, so the parameter sweep would need rerunning.

## Measured, after building it

**The electrode works and is isolated.** At the `DNp01`-left site with radius
6000 it reaches 5,579 neurons above 1% weight. Stimulating chamber 0 alone
gives 45 `DNp01` spikes in chamber 0 and **0 in chamber 4**, so every spike is
caused by the electrode and isolation is intact.

**Unilateral, as predicted.** `DNp01`-left sits at weight 1.000 and
`DNp01`-right at 0.0000. One electrode reaches one side, and the piece says so.

**Pulse rate scales with level.** Over 400ms at levels 1.0, 0.5 and 0.1, the
chambers spent 80, 12 and 8 steps mid-pulse. Amplitude per pulse is fixed and
the level sets frequency, which is how a stimulator is actually used.

### A correction to this plan

This document claimed the electrode would drive the escape pathway where
warmth could not. **Measured, that is not true.** Against the old thermo
channel under identical conditions:

| channel | DNp01 spikes | all spikes | DNp01 share |
|---|---|---|---|
| electrode | 45 | 1,026,707 | 0.0044% |
| thermo | 42 | 947,073 | 0.0044% |

Identical share. Warmth already rang the alarm bell at a low rate, and the
electrode rings it at the same low rate, so `docs/what-heat-is.md` is wrong to
say the thermoreceptors never drive DNp01.

**So the electrode is not justified by making the fly convulse.** It is
justified by having a physical position, by stimulating a local population
rather than a labelled line, and by matching what the simulation literally
does. Those are real and they stand on their own.

Visible convulsion, if it is wanted, still needs `DNp01` driven directly and
hard, declared as an authored intervention. That is a separate change and must
not be presented as something the electrode produces on its own.

## Making the fly visibly convulse: measured, and harder than expected

Two routes tested. The obvious one does not work.

### Driving the escape command neurons does nothing

The 12 escape neurons (DNp01, 02, 03, 04, 09, 11, two each) were driven
directly at a range of amplitudes, with the 708 VNC motor neurons as the
readout since those are what a convulsion would actually show.

**Motor output saturates immediately.** At 20 mV it is 3,713 spikes; at 400 mV
it is 3,686. Twenty times the current for no change, because the escape
neurons hit their refractory ceiling and everything downstream flattens. This
is the same saturation already recorded for heat in
`docs/negative-results.md`: amplitude is not a lever in this model.

**Worse, against a realistic baseline it is slightly negative.** With the
phasic sensory drive the real runs use, motor spikes before the escape burst
were 3,126 over 300 steps and 2,875 during it: **0.92x**. Driving the escape
pathway *reduces* motor output marginally, because the network is already
saturated by the baseline drive and the added input lands on neurons that are
mostly refractory.

So the escape circuit cannot deliver a convulsion here. It is present and
wired, and driving it changes nothing visible.

### Driving the motor neurons directly does work

The 708 VNC motor neurons, driven directly against the same realistic
baseline:

| amplitude | motor spikes | vs off |
|---|---|---|
| 0 mV | 2,875 | 1.00x |
| 10 mV | 3,447 | 1.20x |
| 25 mV | 4,098 | 1.43x |
| 60 mV | 5,019 | 1.75x |

This scales properly rather than saturating, and 1.75x is a large enough
change to animate against. Compare the existing figure of about 3% that heat
moves motor firing by.

### What this means, and the honesty cost

A convulsion would have to come from **stimulating motor neurons directly**,
and that is a bigger authored intervention than driving the escape circuit
would have been. It is not the escape reflex: it is current injected into the
output stage, bypassing the command neurons that would normally decide to
fire it.

If it ships it must be labelled exactly that way, in the README and on screen.
Something like: "motor neurons stimulated directly. This is not an escape
response; the connectome contains the escape circuit but driving it produces
no motor change in this model."

The alternative is accepting that the fly does not visibly convulse, and
saying why, which is itself the more interesting finding: **the alarm bell is
wired, reachable, and ringing it does nothing.**

### Not yet decided

Whether to ship the direct motor drive at all. It is a real result either way
and both options are defensible; what is not defensible is showing a
convulsion and implying the fly produced it.

## Shipped: the convulsion, and where the honesty lives

The flies convulse. The 708 VNC motor neurons are driven directly alongside
the electrode, scaled by chamber level.

**No label in the scene.** This is an art piece, not a paper or a production
system, and an on-screen disclaimer would clutter the one thing the scene is
for. The declaration lives in the README, in full, where it can be read
properly rather than skimmed over a shot.

What the README must say, plainly:

> The bodies convulse because current is injected directly into 708 motor
> neurons. That is an authored intervention. It is not an escape reflex: the
> escape circuit is present in the connectome and reachable from the
> stimulation site, and driving it produces no motor change in this model at
> any amplitude.

### The fix that made it work

Driving the motor stage only during a pulse gave 1.16x and saturated there.
At 50 Hz a pulse is 4 steps in 200, so the drive was off 98% of the time, and
a body that moves only inside a 0.4 ms window flickers rather than convulses.

Held continuously between pulses, with a 1.6x kick on the pulse itself so the
discharge still lands:

| motor drive | motor spikes | vs off |
|---|---|---|
| off | 6,532 | 1.00x |
| 25 mV | 11,008 | 1.69x |
| 60 mV | 13,045 | 2.00x |
| 100 mV | 14,302 | 2.19x |

60 mV doubles motor firing, against the roughly 3% that heat moved it by.

**Isolation still holds.** With chamber 0 stimulated and the motor drive
active, spikes per instance are [1029552, 0, 0, 0, 0, 0]: every other chamber
and the operator sit at exactly zero.
