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
