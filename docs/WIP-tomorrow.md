# Where this stands, end of 2026-09-14

Nothing pushed after commit 56. Two commits are local-only and the contrast fix
is uncommitted in the working tree, deliberately, so the dates land tomorrow.

## The design changed today

The operator no longer picks which chamber is heated. It holds five dials, one
per chamber, and sets how much each gets, continuously. Nothing is ever off.

This was the right call for the piece and it also fixed the divergence problem:
only continuously heated chambers accumulate, and under one switch four chambers
sat cold at any moment.

## Working

- Five dials driven by five descending-neuron blocks, levels spanning 0 to 100%.
- Divergence from the operator: 2.98 million, up from 239 when the dials were
  first wired and 68,464 after the lockstep fix.
- The operator is exactly 0 in every run. Isolation has never broken.
- GPU at 15x CPU, both backends byte-identical on this machine.

## The open bug

Chambers held at clearly different heat (8.0, 8.0, 0.6, 0.0, 0.0 mV) come out
nearly identical to each other: full-network Hamming distance from the operator
reads 1678 for all five, differing by at most 1.

**This is a bug, not a limit.** A direct test holding five instances at constant
0/2/4/6/8 mV produced roughly 2,400 differing neurons between them. Same brain,
same levels, opposite result. So the 25 TRN_VP thermoreceptors are sufficient to
discriminate heat levels, and something about how the dials deliver that heat is
not working.

### Diagnostic result: the values are right, the ramp is the problem

| case | distance from operator |
|---|---|
| constant 0..8 mV, 3000 steps | 0, 2361, 2360, 2411, 2622 |
| constant 0..8 mV, 2000 steps | 0, 1692, 1678, 1703, 1663 |
| **dials3 final levels held constant** | **1663, 1663, 1342, 0, 0** |

The third row uses the dials' own final levels (8.0, 8.0, 0.6, 0.0, 0.0 mV) and
separates exactly as the piece needs. **So the dial values are correct.** The
bug is that they arrive gradually.

Refinement, measured: the dials do reach a full 8 mV spread by step 1250 and
hold it for the last 64% of the run, so "bunched the whole time" is not right
either. They start together at 4 mV and take about 1250 steps to separate. All
five accumulate near-identical divergence during that shared opening, and the
cumulative total is dominated by it afterwards.

### Fix to try first

Start the levels apart rather than all at 4 mV, or raise `slew_rate` so
separation happens in the first hundred steps instead of the first thousand.
Starting apart is the cleaner option but needs care: the starting spread must
still be set by operator activity, not authored, or the piece stops being
"driven by actual spike activity".

A second option is to measure divergence over a trailing window rather than
cumulatively from step 0, so an early common phase does not swamp the later
separation. That is a change to the measurement, not the piece, and is probably
worth doing regardless.

## Two things I had wrong today, corrected

- **"The 25 thermoreceptors are too few."** Wrong. The direct test proves they
  discriminate fine. The fan-out note in negative-results.md is corrected in a
  later section but the earlier text is still there for the record.
- **"A settled network absorbs perturbation."** Wrong. A six-regime sweep showed
  drive changes do nothing and only injection population mattered, and even that
  conclusion is now superseded by the constant-level test above.

Both are recorded in docs/negative-results.md rather than quietly edited away.

## Next

1. Read the diagnostic result.
2. Fix the dial delivery bug it points at.
3. Rerun 3 seconds, check all five chambers separate from each other.
4. Then render the clip.
