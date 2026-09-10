# Decision: contact parameters that are defensible rather than tuned-to-taste

Type: grilling
Status: open
Blocked by: —

## Question

Which contact coefficients change, to what, and on what authority?

The candidates, with what is wrong with each today:

| contact | today | concern |
|---|---|---|
| roller ↔ floor | `friction="1.0 0.005 0.0001"`, `condim="4"`, `solref="0.004 1"` (`models/robot_rollers.xml:311`) | μ = 1.0 is at the high end for rubber on carpet; it sets the whole acceleration ceiling (9.81 m/s², above the 7.86 tip threshold). rSim's tangential figure is 0.8 |
| ball ↔ floor | `friction="0.35 0.35 0.005 0.0009 0.0009"`, `solref="0.006 1"` → **no bounce** (`models/scene_rollers.xml:23`) | a real ball bounces (`C2`); rolling decel ≈0.29 m/s² against rSim's 0.49 (`P4`); sliding μ 0.35 against rSim's 1.0 (`C3`) |
| ball ↔ wall | `solref="0.004 1"`, no restitution (`models/field.xml:44-55`) | same: a missed shot dies against the wall instead of rebounding |
| walls, bumpers | `solref="0.004 1"` | 4 ms is a very stiff contact for a plastic bumper; the model default is `0.02 1` |
| ball ↔ dribbler | `friction="2.5 …"`, `solref="0.004 0.5"` (underdamped) (`models/scene_rollers.xml:21-22`) | μ = 2.5 is deliberately super-unity to make the bar hold. Is there a physical retention story, or is this a tuned fudge that should be recorded as one? |
| robot ↔ robot / wall | 0.5 / 0.6, unsourced | see the map's "Not yet specified" |

**`02` has now sourced most of this. The remaining decisions are narrower than the table above
suggests:**

| row | `02`'s verdict | what is left to decide |
|---|---|---|
| roller ↔ floor μ | measured **0.25** (Williams et al., IEEE T-RA 18(3) 2002, tilt table) | whether to adopt 0.25 against a 2002 measurement whose roller compound and carpet are both unspecified — the only such measurement that exists. Adopting it makes the robot friction-limited at 2.45 m/s² and re-opens `01`'s binding constraint |
| ball ↔ floor sliding μ | **0.35 is right** — matches ER-Force's empirical μ_k | nothing. Keep, and cite it |
| ball rolling decel | **`μ_roll = 0.0009` is right** — implies 0.293 m/s², band is 0.26–0.31 | nothing. Keep, and cite it. rSim's 0.49 is a copied Bullet workaround |
| ball COR | model has **0**, measured band **0.47–0.56** | the `solref` damping ratio that yields COR ≈ 0.50, and whether ball↔wall gets the same |
| ball inertia | model has **β = 0.4** (MuJoCo's solid-sphere default, verified), measured **0.5** | whether to declare `diaginertia` explicitly on `models/ball.xml:7`. One attribute; affects spin transfer, backspin retention and every kick |
| walls, bumpers `solref` | unsourced | unchanged — still a judgement call |
| ball ↔ dribbler μ = 2.5 | unsourced | unchanged — decide whether it is a physical retention story or a recorded fudge. Rulebook §3.2.4 caps ball coverage at 20 %, which is a real constraint on the mechanism |
| robot ↔ robot / wall | unsourced | unchanged |

Decide, per row: the value, the source class (measured / rulebook / vendor / **deliberate
approximation, recorded**), and whether it is a `<pair>` or left to geom mixing — `C9` notes most
ball/robot contacts are emergent averages today, only the two dribbler pairs are explicit.

Watch the coupling: lowering roller μ lowers the acceleration ceiling, which interacts with `04`;
raising ball↔floor friction and adding restitution both change every kick, pass and dribble, which
is why `09` waits on this. The ball's moment of inertia belongs here too even though it is a body property, not a
contact one: it is the same decision — a measured value replacing an unexamined default.
`impratio` (currently 1) also belongs here — raising it makes friction
harder and slip less abrupt.

Constraint from the acceptance gates: possession duration on `Dribbling-v0` must not degrade more
than 5% against `docs/index.md:199-212`. A contact change that fixes tipping by ruining retention
is not a fix.
