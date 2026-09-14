# README draft

Held here rather than in README.md until there is a clip to lead with.
project.md says to hold the post until then, and the same applies to the repo
front page: the video does what no copy can.

Register notes, from project.md: no adjectives, no overclaiming. The projects
on the shelf that read as serious are the ones that do not oversell. Publish
negative results.

---

## amfly

*nomouth*

Six instances of one connectome. Five in chambers, one outside at a dial.

The MaleCNS connectome is run six times in parallel. Five instances are
embodied in sealed chambers. The sixth sits outside at a dial, and its
descending-neuron activity drives heat into the other five.

The operator is not special. Same graph, same weights, same parameters. It
differs only in where it sits in the wiring. It does not know the chambers
exist, it receives no feedback from them, and it could not have done otherwise.

At t=0 all six spike trains are bit-identical. They decorrelate only because of
what is done to them.

**Nothing in the simulation experiences anything.** The piece is about
determinism, not cruelty. The dread is structural: a thing with no mind is
deciding this.

### What is real and what is authored

The wiring is real and measured. Everything else is authored.

- The connectome is MaleCNS v1.0 from Janelia FlyEM and Google Research, CC BY 4.0.
- The neuron model is leaky integrate-and-fire per Shiu et al., Nature 2024.
- The chambers, the dial, the heat and the six-way framing are authored by me.

Chamber selection is driven by actual spike activity. There is no RNG anywhere
in the divergence path, and no random seed in the pipeline at all: even the
recorded neuron sample is strided rather than drawn.

### Numbers

Computed from the v1.0 files directly and asserted in `tests/test_loader.py`.
They are not copied from anyone else's README, including the widely repeated
165,122, which is a stale v0.9 figure.

| | |
|---|---|
| Neurons | 166,700 |
| Directed connections | 25,582,938 |
| Synapses | 124,177,617 |
| Descending neurons | 1,314 |
| Thermoreceptors used | 25 (`TRN_VP*`) |

These are three different quantities and they get conflated constantly. 25.6M
is the connection count, not the synapse count.

### Honest caveats

- **Shiu et al. is brain-only**, about 127,000 neurons. This runs whole-CNS at
  166,700 including the nerve cord. That extension is mine, not theirs.
- **dt=0.1ms is the Brian2 default clock**, not a published parameter of the paper.
- **Connection weights are counts of detected synaptic contacts**, not measured
  physiological strengths.
- **Glutamate is treated as inhibitory** (GluCl), following Shiu. This surprises
  people coming from vertebrate work.
- **1.9% of neurons (3,177) have unclear or missing neurotransmitter predictions**
  and default to excitatory. Run `--silence-unclear-nt` to silence them instead
  and see how much the result depends on that choice.
- **Bit-identity is guaranteed within one machine and configuration.** Identical
  results across different GPUs are not promised. On this machine the CUDA and
  numpy backends happen to agree byte for byte, which is observed rather than
  relied upon.

### Negative results

In `docs/negative-results.md`, including the first divergence run, which did not
work and why.

### Credits

MaleCNS v1.0, Janelia FlyEM and Google Research, CC BY 4.0.
Shiu et al., *A leaky integrate-and-fire computational model based on the
connectome of the entire adult Drosophila brain*, Nature, 2024.
