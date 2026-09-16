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

Measured cost at 166,700 neurons: 9.5ms per step on an RTX 4050, 146.9ms on
CPU. The GPU path uses the same batched formulation, one `torch.sparse.mm` over
all six columns with a fixed reduction order, so determinism is preserved rather
than traded away for the 15.4x.

Cross-backend bit-identity is not *promised*, but on this machine it is
observed. The same 300-step configuration run on GPU and on CPU produced
byte-identical output: same SHA-256 on the rate array, all 88,085 spike events
matching, identical divergence onsets and dial history.

That is a stronger result than the design requires and it should be treated as a
happy accident of this hardware rather than a guarantee. Determinism within one
backend is structural, because the reduction order is fixed by the CSR indices.
Agreement *between* backends depends on both reducing in the same order, which
is not something either library promises. The gates test the property that is
guaranteed; the cross-backend check is recorded because it is informative, not
because it is relied upon.

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

## Buttons switch on a margin, never on a clock

A held button keeps its slot until another chamber beats it by
`switch_margin`. There is no timer anywhere in the selection path.

This was not the first design. Re-deciding on a fixed cadence was tried at
four values and every one of them produced a regular pattern, because a clock
decides *when* the operator acts and leaves it only the choice of *which*. At
100ms four of the five chambers sat within one percent of each other, which is
a round robin rather than a preference. Removing the cadence was worse: the
grip is then re-auctioned every step, so the three unheld chambers receive
press and release in near-equal alternation and cancel out into a flat braid at
mid-range.

A margin fixes both because it makes switching depend on the *size* of the
difference rather than on elapsed time. Small fluctuations are ignored, a
decisive block takes the slot, and the rhythm therefore comes from the
connectome rather than from a constant.

The value, 0.40, is authored and must be declared in the README. What it
controls is how decisive the operator must be before it changes its mind, not
which chamber it picks. Below about 0.2 the braid returns, because the margin
is then smaller than the fluctuation it exists to ignore.
