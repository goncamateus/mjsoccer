# Drop the roller robot (tier 1); keep only the kinematic chassis

The roller robot (tier 1: four wheels through 80 slipping rollers, 94 DoF per robot) was the
primary physics model and the one the GPU port was meant to accelerate. Neither GPU engine
ports it well: MJX-JAX cannot collide its dribbler cylinder with the ball at all, and MuJoCo
Warp warns that single mechanisms beyond ~60 DoF are still slow. We removed it from the repo
rather than keep an unmaintained second physics, and made the kinematic chassis (tier 0) the
only robot, driven by the go-to-point navigation controller of Planning-the-path-with-rl, so
one model runs on the CPU, MJX-JAX and MJX-Warp alike.

## Consequences

- Supersedes restriction 4 of `.scratch/physics-fidelity/map.md` ("tier 0 is kept as a debug
  tier with tier 1's envelope"): the chassis keeps the hardware-derived envelope numbers, but
  no longer as a copy of a second model.
- The word "tier" leaves the language; there is one robot model.
- The history (MJCF, kick tables, acceptance runs) stays in git.
