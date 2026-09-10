# Documentation: record the numbers and fix the drift

Type: task
Status: open
Blocked by: 04, 05, 08, 10, 11, 12, 13

## Question

Last ticket. Nothing to decide; write down what the map settled, and fix what is already wrong.

**Existing drift, independent of every other ticket:**
1. All six environment pages state the tipping step pays `-10.0`
   (`docs/environments/go_to_ball.md:136` and the same line in the other five). The constant was
   10.0 when `3e19dc4` landed and is **100.0** today, changed in `cd495d7`.
2. `docs/index.md` never mentions tipping: absent from the "terminates on" column of the
   environment index (`docs/index.md:37-47`) and from the Timing and Observation sections. Add it
   at `docs/index.md:46`.
3. `docs/index.md:135-137` claims observation values "sit in roughly [−1, 1]" and attributes
   excursions to "a tier-1 chassis pushed by another robot". False today: tier 1's own drive
   exceeds `MAX_V` and `MAX_W` (`L9`). Whether the sentence becomes true or gets corrected
   depends on `11`.
4. The tier-1 wheel-to-chassis matrix is **never stated anywhere** — `docs/index.md:69-77` gives
   only the `ctrlrange`. Add it, with the measured `r_eff = 30.089` mm and `R = 0.0796` m, the
   wheel angles 60/135/225/300°, and a note that the sign convention is the mirror of rSim's
   (`W2`) so a ported wheel vector drives backwards.

**New material from this map:** the derived drivetrain envelope and the acceleration it implies
(`01`, `04`) stated alongside the action table, as `docs/index.md:79-82` promises; the contact
coefficients and their source class (`05`); the dribbler's action semantics (`08`); the turnover
handler and what `info` reports (`10`); the observation normalisers and the ball-height datum
(`11`); tier 0's matched envelope and what it does *not* match (`12`); the timestep rationale and
the real control rate (`13`).

**Corrections owed to the audit documents themselves** (found while resolving `01`):
- `docs/research/rsoccer-vs-mjsoccer.md` `W3` and the §7 limits table give rSim's acceleration
  ceiling as 4.67 m/s². That assumes all four wheels project at unit force; the 60/135/225/300°
  geometry caps the projection at `Σ|sin α| = 3.14626` forward and `Σ|cos α| = 2.41421` lateral.
  The real figure is **3.68 m/s² forward / 2.82 lateral** — which vindicates rSim's own "reduced
  to 3m/s2 accel" comment on the weak axis. `W3`'s conclusion is unaffected.
- The same projection factors apply to any future statement of *this* robot's acceleration, so
  state them once alongside the wheel matrix rather than per-number.
- `B9` calls rSim's 0.043 kg ball "a 7 % mass difference". `02` shows it is a **transcription
  error** — rulebook §2.2's diameter in metres pasted into the mass field, with `ballRadius`
  derived by halving it. Re-label.
- `P4` implies mjsoccer's ball is too lively against rSim's 0.49 m/s². `02` traces that 0.49 to
  ER-Force's `hackFactor = 1.4` Bullet workaround and puts the measured band at 0.26–0.31, so
  **mjsoccer's 0.293 is right and rSim is wrong**. Invert the framing.
- `C2` presents the bounce difference neutrally. `02` measures COR on carpet at 0.47–0.56, so
  rSim's `bounce = 0.5` is its one defensible contact constant and **the zero-COR ball here is the
  fidelity gap**. Invert.
- `docs/analysis/ssl-simulator-rsoccer-comparison.md`'s CoM comparison (see the map's Notes).

Also decide where the two audit documents live long-term: `docs/research/` and `docs/analysis/`
are both untracked today. They are the justification for every number in this map, so they should
be committed, and the analysis doc's CoM comparison corrected (see the map's Notes).
