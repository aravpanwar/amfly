# Decisions

Why things are the way they are, including the places where this project
deliberately does the opposite of what `project.md` specified.

## Deterministic pull over event-driven push

`project.md` says "event-driven propagation is not optional". Reversed here.

That instruction was written for a realtime target. This project is offline: it
dumps to disk and renders later, so speed buys nothing it needs. A push kernel
with atomic accumulation has a non-deterministic reduction order, and float
addition is not associative. Summing the same 10,000 values in two orders
differs by about 20% of a single 0.275mV synaptic weight.

Injected every step, that would decorrelate the six instances from arithmetic
noise alone, producing a convincing fake of the exact phenomenon the piece is
about, while the validation gate appeared to pass. The cheapest possible way to
be wrong.

So: six instances batched as six columns, one shared CSR with sorted indices,
one matmul per step, identical reduction order for every column. Bit-identity
becomes structural rather than something to hope for.

Cost is roughly 200ms per step on CPU at 166,700 neurons, so about 25 minutes
per 700ms of simulated time. Acceptable for an offline piece.

## Divergence measured by spike identity, not rate

The first run diverged in *which* neurons fired while the totals stayed
identical, so the rate plot showed six lines on top of each other and hid the
subject entirely. See `docs/negative-results.md`.

Divergence is now the Hamming distance between spike vectors. Two instances are
the same only while the same neurons fire at the same step.

## The reference is an unheated chamber

Measuring against the heated chamber makes every other instance show the same
large distance, so the plot reads as five chambers diverging when one did.
Against an unheated reference, only the heated chamber moves and the rest sit
flat at zero.

## Phasic drive, not tonic

Driving the thermoreceptors every step made the whole network ring at the drive
rate. A globally synchronised network resists divergence, because each
perturbation is overwritten by the next drive cycle before it can propagate.

Brief pulses into a broader sensory population leave the network free to carry
a difference forward. This was the single largest factor in the divergence
becoming visible.

## No RNG anywhere in the divergence path

The README promises chamber selection is "driven by actual spike activity, not
an RNG with a skin on it", so symmetry is broken only by the operator's own
spikes. Chambers start identical and stay deterministic.

There is no random seed in the pipeline at all. Even the recorded neuron sample
is strided rather than drawn, and the dial breaks ties by lowest index.

## No 3D, no physics engine, for now

`project.md` Phase 3 specifies flybody, Three.js and Rapier. Deferred entirely.

The deliverable is clips of traces decorrelating, and the trace panels are the
piece. If the divergence does not read on a plot, no amount of rendering saves
it. Bodies are a later concern if the project gets traction.

## MIT, and our own loader

`neurofly-kit` is AGPL-3.0, which is viral and would force the whole repo AGPL.
The loader is about 200 lines. Writing it keeps the licence free and keeps the
dataset isolated behind one module, which `project.md` asks for anyway.

## Counts are computed, never copied

The widely repeated 165,122 neurons is a stale v0.9 figure that propagated
through other projects' READMEs. v1.0 is 166,700.

Everything published is computed from the pinned files and asserted in
`tests/test_loader.py`, so a changed download fails loudly rather than quietly
producing a different piece.
