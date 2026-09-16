# Where this stands

Everything is committed and pushed. Tests green, README still blank by choice.

## The piece changed, and for the better

It is no longer an indifferent operator. It is a trapped one. All three of the
ideas that got it there came from you:

- **Buttons, not dials.** Press and a chamber climbs, release and it falls, and
  only two can be pressed at once. Nothing achieved stays achieved. This alone
  took mean reward from 0.03 to 0.94.
- **Reward only for new maxes.** Holding a chamber builds tolerance and stops
  paying, exactly as a feed does. Took reward from 0.99 to 0.30 and broke the
  operator's habit of camping on two chambers forever.
- **Escalating neglect.** Time at the floor accumulates as debt, so abandoning a
  chamber is not free.

The loop runs through real measured circuits: reward into the 340 dopaminergic
neurons present in the data (PAM 316, PPL1 16, PPL2 8), punishment into the
operator's own TRN_VP thermoreceptors, the same 25 cells heated in every
chamber.

## Closed: the buttons now switch on a margin, not a clock

This was the last thing standing between the piece and a clip, and it is done.

A held button keeps its slot until a challenger beats it by `switch_margin`,
currently 0.40. Every timer-based version read as a metronome, because a clock
decides when the operator acts and leaves it only the choice of which chamber.
Dropping the timer without a margin was worse: three chambers braided into a
flat line at mid-range. Full measurements in `docs/negative-results.md`.

Chamber 2's "rescue count", which cost an evening, was never real. The metric
counted time above 80% heat for a chamber that sat around 40%. That is written
up as the standing lesson: **suspect the instrument before the system.**

## Then

1. Export the winning run: `export_web.py`, `export_brain.py`, `make_figures.py`
2. Reload the scene, record a clip with **R**
3. The overnight sweep, `python scripts/sweep.py --hours 7`, which now explores
   the margin and press rate rather than parameters that no longer decide much
4. The README, which is now genuinely worth writing since the design has settled

`scripts/replay_dials.py` tests button parameters in seconds instead of the
thirteen minutes a real run costs. Build its cache once with `--build-dn`.

## To bring the scene up

    cd web && python -m http.server 8777

Keys: **R** record, **G** glass, **S** sound.

## Things that must go in the README

- The two-button limit is a rule of the piece, not a property of the fly
- The 100% and 50% thresholds are chosen, and so is the 0.40 switch margin
- Stimulating PAM is current injected into neurons that participate in
  reinforcement learning. Nothing here experiences reward and the README must
  not imply otherwise
- Chamber starting spread is authored
- Leg and wing motion is amplified from a real but small signal, about 3%
- Only some brain points carry per-neuron activity, 3,152 of 24,000
- project.md chose indifference over cruelty and this reverses that;
  `--no-compulsion` still runs the original
