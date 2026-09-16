# Scene polish, in order of what a clip gains

Written 2026-09-16, after the shadow pass. The scene is structurally done:
flies mounted on pins, wired console, everything grounded, one shadow caster.
What follows is about the footage, not the simulation.

Ranked by how much each changes a recorded clip.

## 1. Camera movement  (IN PROGRESS)

The largest gap. `R` records a locked-off shot while the camera sits where it
was left, so every clip is a screen recording of a static scene.

A scripted path on a keypress: start wide, push past the operator, track down
the cable run, settle on the row. Repeatable, identical every take, and it is
the one thing that cannot be fixed afterwards.

Free orbit stays. The path is a separate key so a take is one keypress rather
than a manual orbit performed while recording.

## 2. React at the moment of reward

Reward is currently a bar and a green underlight, which is information rather
than an event. It is the hook of the piece, the doomscrolling mechanic, and
it should be felt.

A brief exposure lift or bloom on the frames where a chamber reaches 100%.
Measured on the confirmed run, 120 frames of 900 qualify, so it stays rare
enough to read as a hit rather than a flicker.

## 3. The room is empty above bench height

Fog hides the far wall but there is nothing between the benches and the
ceiling, so the scene reads as objects on a table rather than a room.

- Dust motes: a few hundred slowly drifting points catch the spot cones and
  make the air visible. This is what sells volumetric light without paying
  for volumetrics, and it is cheap
- Cable runs leaving frame, implying the rig continues past the shot
- A dim ceiling fixture or two, matching the spots

## 4. Small, high value

- Bloom on the LEDs and heated pins ONLY. A pin at 100% is bright but does
  not bleed, so it reads as a lit object rather than as something hot
- Vignette, to pull the eye to the centre row
- Shallow depth of field on the far chambers, which reads as photographed
  rather than rendered

## Deliberately not doing

- Screen space reflections: the surfaces are rough, so the cost buys nothing
- More geometry on the flies: 272k triangles is already past what a 3072
  shadow map resolves
- True volumetric lighting: dust motes fake it convincingly at a fraction of
  the cost
