# What "heat" actually is in this piece

A question worth answering precisely, because the README depends on it and
because it is the kind of thing that gets a project taken apart in comments.

## Three different things

**1. Sensing warmth.** A fly detects ambient temperature through
thermoreceptors. This is what the model does: 25 `TRN_VP*` neurons are driven,
and that signal propagates inward through real measured wiring.

**2. Pain.** A real fly senses noxious heat through multidendritic nociceptors
(class IV md neurons) in the body wall, which are a different population with a
different pathway.

**These are not in the MaleCNS connectome.** Checked directly: zero neurons
match `md`, `CIV`, `Noci` or `ppk` prefixes. The dataset is the central nervous
system, and those sensors live in the peripheral body wall. So the piece cannot
model nociception even in principle, because the wiring for it was never
imaged.

**3. Tissue damage.** Actually burning a fly destroys cells: membranes fail,
neurons die, the network degrades. **Nothing in this model can represent that.**
A neuron here integrates input and spikes. It cannot be damaged, cannot die, and
has no state that burning would change. The connectome is fixed wiring; there is
no mechanism by which any of it degrades.

## Why there is no thrashing, and why that is a real gap

The obvious objection: burn something and it flails. Nociception drives a
whole-body escape reflex, and a fly should roll and thrash, not sit there with a
3% change in motor firing.

That objection is correct about flies and it identifies a genuine limit here.

**The escape machinery IS in the dataset.** Checked: `DNp01` (the giant fiber
escape command neuron), `DNp02`, `DNp03`, `DNp04`, `DNp09`, `DNp11`, and 34
giant-fiber-related neurons. The wiring that produces an escape response is
present and measured.

**What is missing is the sensor that triggers it.** Nociceptors live in the
peripheral body wall and were never imaged, so nothing in this model can deliver
the signal that would normally drive DNp01. The thermoreceptors that ARE present
report ambient warmth, which in a real fly produces preference behaviour, walking
away from a warm patch, not an escape reflex.

**And the two are connected.** Measured reachability from the 25
thermoreceptors: 0 escape neurons at one synaptic hop, 5 at two hops, and all 36
at three hops. The path from warmth to the escape command exists in the measured
wiring and is short.

So the model has the fly's alarm bell, the wire that rings it, and none of its
smoke detectors. Warmth-level input simply does not drive DNp01 hard enough to
fire. In a real fly that trigger comes from nociceptors, and those are not in
this dataset.

That is a more interesting fact than a missing connection would have been: the
escape reflex is reachable in principle, and what is absent is the sensor that
would justify firing it.

### What this means for the piece

The honest description is **a fly detecting rising warmth**, not a fly being
burned. A viewer who expects thrashing is expecting nociception, and the dataset
cannot provide it.

Two things follow:

1. The README must not imply burning or agony. What is shown is a thermal
   stimulus and its measured consequences.
2. If a future version wants visible distress, the only truthful route is to
   drive the escape pathway directly and **say so**: "DNp01 stimulated directly,
   standing in for a nociceptive input the connectome does not contain." That is
   an authored intervention, not an emergent response, and labelling it as
   emergent would be the exact overclaim `project.md` warns about.

## So what is the operator doing

Turning up a **warmth signal** in each chamber. The fly's thermoreceptors fire
harder, that propagates through its brain, and its trajectory diverges from the
flies whose dials are lower.

That is real, measured, and it is the whole of what happens. Everything else is
the viewer's inference.

## Why this is fine, and arguably better

The piece was never about pain. `project.md` is explicit:

> **The piece is about determinism, not cruelty.** Nothing in the simulation
> experiences anything, and the README must say so plainly. The dread is
> structural: a thing with no mind is deciding this.

The horror is that a mindless thing is deterministically altering five other
mindless things, and could not have done otherwise. It does not require, and is
not improved by, a claim that anything is suffering. A model that claimed to
simulate pain would be both false and a worse piece.

## What the README must not say

- Not "torture", not "pain", not "burning", not "damage".
- Not "the fly is hurt" or any phrasing implying experience or injury.

## What it can say

- The operator raises and lowers a thermal stimulus in five sealed chambers.
- The stimulus enters through the fly's real thermoreceptor neurons.
- The five diverge measurably and only because of what is done to them.
- Nothing in the simulation experiences anything.

## The honest caveat, for the caveats section

> Heat here is a warmth signal injected into 25 thermosensory neurons. It is not
> nociception: the MaleCNS connectome contains no nociceptors, since those sit in
> the peripheral body wall and this dataset is the central nervous system. No
> tissue damage is modelled, and the model has no mechanism that could represent
> it.
