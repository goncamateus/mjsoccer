# Physical constants: the ball, the carpet, and the rulebook

Type: research
Status: resolved
Blocked by: —

## Question

What are the real, citable physical constants for an SSL ball on SSL carpet, and which rulebook
limits constrain the simulator?

The two simulators disagree on every one of these and neither cites a source:

| quantity | mjsoccer | rSim |
|---|---|---|
| ball mass | 0.046 kg (`models/ball.xml:7`) | 0.043 kg (`sslconfig.h:123`) |
| ball radius | 0.0215 m | 0.0215 m |
| ball↔floor sliding friction | 0.35 (`models/scene_rollers.xml:23`) | 1.0 (`sslworld.cpp:154`) |
| restitution | **none** — `solref="0.006 1"`, critically damped | `bounce = 0.5` above `bounce_vel = 0.1` |
| rolling deceleration | ≈0.29 m/s², from `μ_roll = 0.0009` (derived, `P4`) | 0.49 m/s² constant Coulomb + a hard kill below 0.01 m/s |
| floor friction | 0.8 (`models/field.xml:42`) | 1.0 |

Deliver, each cited to a primary source:
1. The **regulation** ball: the SSL rulebook's specified type, and that ball's real mass and
   diameter. A golf ball is 45.93 g / 42.67 mm by the R&A and USGA rules — establish whether the
   rulebook actually specifies it, and therefore whether 0.046 or 0.043 kg is right.
2. Its **coefficient of restitution** on a hard floor, and what that becomes on the carpet SSL
   plays on. A golf ball has a published COR; the carpet is the unknown. rSim's 0.5 needs
   checking against anything measured — published SSL/grSim calibration work counts, a blog post
   does not.
3. The ball's **rolling deceleration** on SSL carpet in m/s², from any measured source (team
   calibration papers, grSim/ER-Force/TIGERs published figures, SSL vision logs).
4. **Rulebook limits** that bind the simulator: maximum kick speed (flat and chip), maximum
   robot dimensions, dribbler ball-coverage limit, and whether robot speed is capped at all.
5. The carpet's **sliding friction** against a rubber omni-wheel roller, if any source exists.

State clearly which of the five have no primary source, because those become "deliberate
approximation, recorded" rather than "justified" — that distinction is the whole destination.

## Answer

Resolved 2026-09-09. Full workings: [`research/02-physical-constants.md`](../research/02-physical-constants.md).

**No deliverable came back fully UNSOURCED.** Two sub-questions did, flagged in place: oblique and
ball-off-robot restitution (only unbacked config values exist, and the two published redirect
figures differ by 3×), and an SSL *rubber* roller on SSL *felt* carpet specifically.

| quantity | model today | recommended | class |
|---|---|---|---|
| ball mass | 0.046 kg | **keep** — R&A Rule 4.2 caps at 45.93 g | RULEBOOK |
| ball radius | 0.0215 m | **keep** — R&A Rule 4.3 floors at 42.67 mm | RULEBOOK |
| ball COR on carpet | **0** (critically damped) | **0.50** (measured band 0.47–0.56) | MEASURED |
| ball rolling decel | ≈0.293 m/s² | **keep** `μ_roll = 0.0009` — band 0.26–0.31 | MEASURED |
| ball↔carpet sliding μ | 0.35 | **keep** — matches ER-Force's empirical 0.35 | MEASURED |
| ball inertia β = I/mr² | **0.4** (MuJoCo solid-sphere default) | **0.5** | MEASURED |
| max kick speed | uncapped | **6.5 m/s, flat and chip alike** (§8.4.2) | RULEBOOK |
| dribbler coverage | unmodelled | **≤20 % ⇒ ≤10.9 mm hull penetration** (§3.2.4) | RULEBOOK |
| roller↔floor μ | **1.0 effective** | **0.25** | MEASURED |

**Two model numbers turn out to be right and were never justified**: ball↔floor sliding friction
0.35 and `μ_roll = 0.0009`. Both now have a source. Keep them and cite them.

**Three findings that invert what the audits said.**
1. rSim's `ballMass = 0.043` is a **transcription error**, not a modelling choice: rulebook §2.2's
   *diameter in metres* pasted into the mass field — and its `ballRadius = 0.0215` is that same
   0.043 halved. `B9` should be re-labelled from "a 7 % mass difference" to "upstream typo".
2. rSim's 0.49 m/s² ball deceleration is a **traceable hack**: ER-Force's `simball.cpp` computes
   `hackFactor = 1.4; rollingDeceleration = hackFactor * 0.35` under a comment calling it "quite a
   hack" — a Bullet stability workaround. rSim copied the result, and the 0.01 m/s kill branch,
   into ODE where the instability does not exist. `P4` implied mjsoccer's ball is too lively;
   the measured band says **mjsoccer is right and rSim is wrong**.
3. `bounce = 0.5` is rSim's **one defensible contact constant**. `C2` inverts: the zero-COR ball
   here is the fidelity gap, not upstream's bounce.

**Cross-ticket contradiction with `01`, verified.** The only measured roller-on-carpet coefficient
is **0.25**, not the 1.0 `01` assumed for its friction budget (the model declares roller
`friction="1.0"` against a floor of 0.8, and MuJoCo takes the elementwise max, so 1.0 is what is
in force today). At μ = 0.25 the budget is `0.25 · 2.69207 · 9.81 = 6.602` N against `01`'s
recommended 8.16 N of forward ground force. So:

- `01`'s **`forcerange` stands** — it comes from the motor, not from the floor.
- `01`'s **"torque-limited, 31 % of budget" conclusion does not.** The robot would be
  *friction*-limited at `μ·g = 2.45` m/s², not torque-limited at 2.98.
- The envelope's **purpose still holds**: 2.45 m/s² is further below the 7.86 m/s² tip threshold
  than 2.98 was. Straight-line acceleration could not tip the robot at all.
- This makes `04` and `05` a **joint decision**: which of torque or friction binds is not
  determined by either ticket alone.
