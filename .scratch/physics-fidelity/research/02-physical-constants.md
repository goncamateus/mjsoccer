# 02 — Physical constants: the ball, the carpet, and the rulebook

Ticket: `.scratch/physics-fidelity/issues/02-physical-constants.md`. Resolved 2026-09-09.
Rulebook quoted from the `ssl-rules` chapter sources on `master` and from the rendered rule book
(both fetched 2026-09-09); ball conformance limits from the R&A/USGA Equipment Rules; everything
else from peer-reviewed measurement, published SSL team papers, or the source of the league's own
simulators. Repo values cited `path:LINE`; upstream to `/tmp/rsim-upstream` (`b413932`).

Per the map's standing constraint 1, rSim's numbers are treated as a differential reference only.
Two of them turn out to have a traceable provenance, and §3 names it.

## Summary

| quantity | current model | recommended | source class | citation |
|---|---|---|---|---|
| ball mass | 0.046 kg (`models/ball.xml:7`) | **0.046 kg — keep** | RULEBOOK | SSL rules §2.2; R&A Rule 4.2 max 45.93 g |
| ball radius | 0.0215 m (`models/ball.xml:7`) | **0.0215 m — keep** | RULEBOOK | SSL rules §2.2 (0.043 m dia.); R&A Rule 4.3 min 42.67 mm |
| ball COR, hard floor | n/a (`solref="0.006 1"`, critically damped) | **0.89 on steel, 0.68 on wood** (report only; MuJoCo has no COR) | MEASURED | Ismail & Stronge, IOP Conf. Ser. MSE 36 012038 |
| ball COR, SSL carpet | **0** (critically damped) | **0.47–0.56** | MEASURED | ER-Force `simulator.cpp:46`; TIGERs `BallParameters.java:91` |
| ball rolling deceleration | ≈0.293 m/s² (`μ_roll=0.0009`, `scene_rollers.xml:23`) | **0.26–0.31 m/s²; keep 0.0009** | MEASURED | Rojas & Simon 0.305; ER-Force ETDP 2016 0.30; TIGERs 0.26 |
| ball sliding friction (ball↔carpet) | 0.35 (`scene_rollers.xml:23`) | **0.26–0.35; keep 0.35 or drop to 0.31** | MEASURED | ER-Force `simulator.cpp:47` (μ_k = 0.35); a_slide 2.5–3.47 m/s² |
| ball inertia β = I/mr² | 0.4 (uniform sphere, implied by `mass=`) | **0.5** (measured; unrepresentable without an explicit `diaginertia`) | MEASURED | TIGERs `BallParameters.java:49`; Rojas & Simon k=0.5805 |
| max ball speed off a robot | uncapped | **6.5 m/s, flat and chip alike** | RULEBOOK | SSL rules §8.4.2 *Ball Speed* |
| max chip speed | uncapped | **6.5 m/s — no separate chip rule exists** | RULEBOOK | SSL rules §8.4.2; "chip" absent from the rule book |
| max robot envelope | — | **Ø0.18 m × 0.15 m cylinder** | RULEBOOK | SSL rules §3.2.2 *Shape* |
| dribbler ball coverage | — | **≤20 % of top-view area ⇒ ≤10.9 mm of hull penetration** | RULEBOOK | SSL rules §3.2.4 *Dribbling Device* |
| robot top speed cap | — | **none** (1.5 m/s applies only during Stop and to crash closing speed) | RULEBOOK | SSL rules §8.4.2 *Crashing*; established by `01` |
| roller↔floor sliding friction | 1.0 effective (`robot_rollers.xml:311` geom 1.0 vs floor 0.8, MuJoCo takes the max) | **0.25** | MEASURED | Williams, Carter, Gallina & Rosati, IEEE T-RA 2002, Table I |
| floor sliding friction (`field.xml:42`) | 0.8 | no independent basis; see roller row | MEASURED | as above |

Verdicts by deliverable: **1 RULEBOOK · 2 MEASURED · 3 MEASURED · 4 RULEBOOK · 5 MEASURED**.
No deliverable came back with nothing. Two carry sub-questions that did — §2's *carpet-specific
tangential* COR and §5's *SSL-carpet, rubber-compound* number — and both are flagged in place.

---

## 1. The regulation ball

**The rule book specifies a golf ball, by name, and gives the numbers.** SSL rules §2.2 *Ball*:

> The ball is a standard orange golf ball. It weights approximately 0.046 kilograms and its
> diameter measures 0.043 meters.
> — [`chapters/playingenvironment.adoc:80-81`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc)

So the ticket's premise holds: a golf ball is specified, and the R&A/USGA Equipment Rules own its
conformance. Those rules give **limits, not a value**:

| R&A/USGA rule | text | kind |
|---|---|---|
| Rule 4.2 *Weight* | "The weight of the ball must not be greater than 1.620 ounces avoirdupois (45.93 g)." | **maximum**; no minimum |
| Rule 4.3 *Size* | "The diameter of the ball must not be less than 1.680 inches (42.67 mm)." | **minimum**; no maximum |

([R&A, Rules of Equipment, Part 4 — Conformance of Balls](https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls);
the USGA co-publishes identical text as Part 4 Rules 2 and 3 of the
[Equipment Rules](https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/equipment-rules.html).)

### 1.1 So is 0.046 or 0.043 kg right? 0.046 — and rSim's 0.043 is a transcription error

The arithmetic settles it three ways:

```
R&A conformance maximum      45.93 g   = 0.04593 kg
SSL rule book §2.2           0.046 kg              (= the R&A maximum, rounded to 2 s.f.)
a real ball, weighed         46 g                  (IOP drop-test study, Table 1)
```

The IOP drop-test paper of §2 independently weighed and measured its golf ball at **46 g and
42.80 mm** (Table 1) — a real ball sitting hard against both conformance limits, as competition
balls do, because a maximum-mass, minimum-diameter ball flies furthest. `models/ball.xml:7`'s
`mass="0.046"` is therefore correct to 0.15 % of the conformance limit and exact against the
rule book. **Keep it.**

rSim's `ballMass = 0.043` (`/tmp/rsim-upstream/src/robosim/sslconfig.h:123`) is not a different
ball. It is **§2.2's diameter in metres pasted into the mass field**: the very next line,
`ballRadius = 0.0215` (`sslconfig.h:121`), is 0.043/2, so the same rule-book sentence supplied
both, and one of the two was read off the wrong noun. There is no ball of any specification
weighing 43 g and measuring 43 mm; that combination exists only as a typo. `B9`'s "ball mass
differs" should be recorded as **rSim being wrong**, not as a disagreement — and its 7 % figure
is a 6.5 % error in rSim, not a modelling choice.

### 1.2 Radius

`size="0.0215"` (43.0 mm diameter) is the rule book's figure exactly. It is 0.77 % above the R&A
minimum of 42.67 mm and 0.47 % above the measured 42.80 mm ball. A simulator should pick the
rule book's 0.0215 m: it is what the league states, it is what every other SSL simulator uses, and
the sub-1 % spread is far below the fidelity of anything else in the contact model. **Keep it.**

**Source class: RULEBOOK.**

---

## 2. Coefficient of restitution

MuJoCo has no restitution coefficient — bounce falls out of `solref`'s damping ratio — so this
section reports the physical COR only and does not touch the mapping.

### 2.1 On a hard floor

Peer-reviewed normal drop test, high-speed video at 250 fps, drops from 1.3–1.7 m (impact
5.05–5.77 m/s), three repeats per configuration, ball weighed at 46 g / 42.80 mm:

| target | 1.3 m | 1.4 m | 1.5 m | 1.6 m | 1.7 m |
|---|---|---|---|---|---|
| steel plate | 0.897 | 0.896 | 0.894 | 0.892 | 0.889 |
| wood plate | 0.681 | 0.677 | 0.676 | 0.675 | 0.666 |

(M. R. M. Ismail & W. J. Stronge et al., "Coefficient of restitution of sports balls: A normal
drop test", *IOP Conf. Ser.: Mater. Sci. Eng.* **36** 012038 (2012),
[doi:10.1088/1757-899X/36/1/012038](https://iopscience.iop.org/article/10.1088/1757-899X/36/1/012038/pdf),
Tables 1, 3 and 4.)

Two properties to carry forward. **COR falls as impact speed rises** — 0.897 → 0.889 across a
14 % speed increase, and the paper's own conclusion states the trend. And **COR falls sharply as
the target softens** — 0.90 on steel becomes 0.68 on wood, a 24 % drop for a target that is still
rigid by carpet standards. Neither of these is a hard floor under 6 mm of felt.

Note this 0.89 sits above the 0.70–0.85 band the golf industry quotes at its own standard inbound
speed of 125 ft/s (38.1 m/s) — consistent with the speed dependence, since the drop test's
5–6 m/s is six to seven times slower.

### 2.2 On the carpet SSL plays on

Two SSL-derived figures, from the two simulators the league actually uses, both for the vertical
(normal) direction:

| source | value | what it is |
|---|---|---|
| ER-Force `simulator.cpp:40-47`, comment headed "(empirical measurments)" | **0.56** | `Ball vs. Floor: Restitution: sqrt(h'/h) = sqrt(0.314) = 0.56` — a drop test on the SSL floor: the ball returned 31.4 % of its drop height |
| TIGERs Sumatra `BallParameters.java:86-91` | **0.47** | `chipDampingZ`, "Chip kick velocity damping factor in Z direction" — the per-hop vertical velocity ratio of a chipped ball, i.e. a normal COR by construction |

([ER-Force framework `src/amun/simulator/simulator.cpp`](https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simulator.cpp);
[TIGERs Sumatra `BallParameters.java`](https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java).)

`chipDampingZ` is not an arbitrary tuning knob: it is one of the three fields of
`SSL_BallModelChipFixedLoss` in the league's own geometry protocol
([`messages_robocup_ssl_geometry.proto:118-127`](https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto)),
which SSL-Vision broadcasts to every team so that they can predict a chipped ball's hops. A team
that got it wrong would miss chip receptions all tournament.

**A simulator should pick 0.50, and say why.** The two SSL measurements bracket it (0.47 and
0.56); it is the midpoint to within 2 %; and it is the value at which a chip landing at 3 m/s
rebounds to 1.5 m/s and dies in three hops, which is what SSL replays show. Anything in
0.47–0.56 is defensible; outside it is not.

The two figures differ because they measure different things — ER-Force dropped a ball vertically
at low speed, TIGERs fitted a whole chip trajectory whose first landing is faster and oblique —
and the speed dependence of §2.1 predicts exactly that ordering (faster impact, lower COR).

### 2.3 Is rSim's 0.5 defensible? Yes — uniquely among its numbers

`ballBounce = 0.5` (`sslconfig.h:124`) lands inside the measured 0.47–0.56 band, 6 % below
ER-Force's drop test and 6 % above TIGERs' chip fit. It is the one contact constant in rSim that
survives contact with a source, and `C2`'s reading should be revised: **the bounce is not the
ODE workaround, the absence of bounce is the fidelity gap.** `models/scene_rollers.xml:23`'s
critically damped `solref="0.006 1"` gives the ball a COR of 0, i.e. a chip that lands is dead
and a shot off the back wall stops — behaviour with no support in any source found here.

Two caveats on rSim's version even so. `ballBounceVel = 0.1` is a threshold below which no bounce
occurs at all, which nothing measured here supports. And rSim applies the *same* 0.5 to the walls
(`sslworld.cpp:151-156,168-169`), where the surface is a hard board and not carpet; §2.1's
steel-vs-wood contrast says a wall's COR should be materially higher than a carpet's.

### 2.4 What is not sourced here

The **tangential** COR — the horizontal velocity a ball keeps through an oblique bounce — has an
SSL number (`chipDampingXYFirstHop = 0.8`, `chipDampingXYOtherHops = 0.85`, same file) but no
measurement behind it that could be found, and the same is true of ball-off-robot restitution
(TIGERs `redirectRestitutionCoefficient = 0.2`; ER-Force's "Ball vs. Robot: Restitution: about
0.60"). Those two disagree by 3×, which is itself evidence that neither is measured. **If the
project needs an oblique-bounce or a redirect model, that sub-question is UNSOURCED** and should
be recorded as a deliberate approximation.

**Source class: MEASURED** (for the normal COR, on both surfaces).

---

## 3. Rolling deceleration on SSL carpet

The best-supported number in this ticket. Three independent measurements, on carpet, by three
different groups, over two decades:

| source | surface | a_roll (m/s²) | a_slide (m/s²) | k_switch | method |
|---|---|---|---|---|---|
| Rojas & Simon (FU-Fighters) | "our carpet" | **−0.305** | −3.47 | 0.5805 ± 0.042 | 26 shots captured by their computer-vision system; two-line least-squares fit with the breakpoint chosen by minimal cost |
| ER-Force ETDP 2016 §5.2 | the playing field | **−0.3** | −2.5 | 0.6 (theory: 5/7) | "Using our plotter software, we measured…" from a v-t trace of a ball shot at 3.3 m/s |
| TIGERs Sumatra, `ROBOCUP` specialization | RoboCup competition carpet | **−0.26** | −3.0 | 0.64 | production ball-prediction parameters, per-venue |

([Rojas, R. & Simon, M., *Like a Rolling Ball*, Freie Universität Berlin](https://robocup.mi.fu-berlin.de/buch/rolling.pdf), §3;
[ER-Force Extended Team Description Paper 2016](https://ssl.robocup.org/wp-content/uploads/2019/01/2016_ETDP_ER-Force.pdf) §5.1–5.2;
[TIGERs `BallParameters.java:24-41`](https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java).)

These are not three copies of one number: Rojas & Simon predate the other two, ER-Force cite them
as `[6]` but measured independently and got a different `k_switch`, and TIGERs' figure is a
separate per-venue calibration. The league's own geometry protocol standardises exactly this
model — `SSL_BallModelStraightTwoPhase{acc_slide, acc_roll, k_switch}` — and its comment names
the ER-Force 2016 ETDP as the definition
([`messages_robocup_ssl_geometry.proto:100-116`](https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto)).

**The band is 0.26–0.31 m/s² on carpet.** ER-Force state the necessary caveat: "The exact time
points and decelerations are contingent on the playing field" (§5.2), and TIGERs carry six
different venue specializations for the same parameter. So a simulator should pick the centre of
the band and treat it as venue-dependent, not as a constant of nature.

### 3.1 Is the model's 0.29 defensible? Yes. Is rSim's 0.49? No.

Re-deriving the model's figure from the declarations. MuJoCo's rolling-friction coefficient has
dimensions of length and bounds the rolling-resistance **torque** at `μ_roll · f_n`. With
`friction="0.35 0.35 0.005 0.0009 0.0009"` (`models/scene_rollers.xml:23`) and
`mass="0.046"` (`models/ball.xml:7`):

```
f_n   = m g            = 0.046 · 9.81                  = 0.451260 N
τ_max = μ_roll · f_n   = 0.0009 · 0.451260             = 4.06134e-4 N·m
a     = τ_max / (m r (1 + β)),  β = I/mr² = 0.4        (uniform sphere; MuJoCo's default from `mass=`)
      = 4.06134e-4 / (0.046 · 0.0215 · 1.4)
      = 4.06134e-4 / 1.38460e-3                        = 0.2933 m/s²
```

`P4`'s 0.29 reproduces. **0.293 m/s² sits in the middle of the 0.26–0.31 measured band** — 4 %
below Rojas & Simon, 2 % below ER-Force, 13 % above TIGERs. It is defensible as it stands and
`μ_roll = 0.0009` needs no change; for reference, 0.00080 would give 0.26 and 0.00095 would give
0.31, so the whole measured band lives inside a 0.0008–0.00095 window.

rSim's 0.49 is **1.6× to 1.9× the measured band** and is not defensible. Its provenance is
traceable, and it is not a measurement:

```
rSim:      fk = ballFriction · m · g,  ballFriction = 0.05     (sslconfig.h:125, sslworld.cpp:286)
           ⇒ a = 0.05 · 9.81 = 0.4905 m/s²

ER-Force:  const btScalar hackFactor = 1.4;
           const btScalar rollingDeceleration = hackFactor * 0.35;   ⇒ 0.49 m/s²
           preceded by:  // \mu_r = -a / g = 0.0357 (while rolling)   ⇒ 0.350 m/s²
           and:          // rollingFriction in bullet is too unstable to be useful
           and:          // this is quite a hack as it's always applied
                                        (framework/src/amun/simulator/simball.cpp:53-88)
```

rSim's 0.49 is ER-Force's physical 0.35 multiplied by a **`hackFactor` of 1.4 that ER-Force's own
comment calls "quite a hack"** — a Bullet-stability compensation, carried into an ODE simulator
where the instability it compensates for does not exist. rSim's structure is copied wholesale:
the same "kill the ball below 0.01 m/s" branch appears in both (`sslworld.cpp:295-298` against
`simball.cpp:73-77`), and ER-Force's version carries the comment "the real ball snaps to a
dimple". Even ER-Force's un-hacked 0.35 is 13 % above the top of the measured band; the 0.49 is
an artefact of an artefact. This is the map's constraint 1 in its purest form.

The kill threshold itself, unusually, *is* supported. Rojas & Simon observed the ball "suddenly
dies" below "around 2 cm/s", because "if the ball rolls too slowly, it cannot 'climb' the carpet"
(§2). 0.01 m/s is the right order of magnitude for a real effect, even though `P4`'s complaint
that mjsoccer's ball creeps instead of parking remains fair.

### 3.2 A by-product: ball↔carpet sliding friction

The same three measurements fix the sliding coefficient, since `a_slide = μ_k g`:

```
Rojas & Simon    3.47 / 9.81 = 0.354      (the paper states 35.4 % of g directly)
TIGERs           3.00 / 9.81 = 0.306
ER-Force ETDP    2.50 / 9.81 = 0.255
```

and ER-Force's simulator states it as a measured coefficient outright:
`Ball vs. Floor: Friction: \mu_k = -a / g (while slipping) = 0.35` (`simulator.cpp:47`).

**Band 0.26–0.35.** `models/scene_rollers.xml:23`'s `0.35` sits at the top of it and matches
ER-Force's stated figure exactly. rSim's `1.0` (`sslworld.cpp:154`) is **3–4× the measured
value** and has no support. `C3` should be re-read the same way as `C2`: mjsoccer is right and
rSim is the outlier, not "a backspun ball grips almost three times harder upstream" as a neutral
difference. A simulator could equally pick 0.31, the band centre; 0.35 is defensible.

### 3.3 A flagged mismatch: the ball's moment of inertia

`k_switch` is a direct measurement of the ball's inertia distribution: for `I = β m r²`, sliding
ends at `v_switch = v₀/(1 + β)`, which is 5/7 = 0.714 for a uniform sphere (β = 0.4).

```
Rojas & Simon   k = 0.5805  ⇒ β = 0.723      "the golf balls used in our experiments do not have
ER-Force        k = 0.6     ⇒ β = 0.667       a constant density and their moment of inertia
TIGERs          k = 0.64    ⇒ β = 0.563       differs from (2/5) m R²"  — Rojas & Simon §3
TIGERs, stated directly:  inertiaDistribution = 0.5, "between 0.4 (massive sphere) and 0.66 (hollow sphere)"
```

Every measurement lands **above** 0.4. `models/ball.xml:7` declares only `mass="0.046"` on a
sphere geom, so MuJoCo computes β = 0.4 and the model's ball is 20–45 % too easy to spin up.
This changes how long a kicked ball slides before it rolls, and it is measured, not guessed.
Fixing it needs an explicit inertia declaration; that is a model edit and belongs to whichever
ticket owns `models/ball.xml`, not here.

### 3.4 Rolling friction or an applied force?

MuJoCo can express rolling deceleration either through the contact pair's rolling-friction term
(what `scene_rollers.xml:23` already does) or as an explicit applied force (what rSim and
ER-Force both do). **The contact-pair term is the natural one**: it is a constraint bound
proportional to the actual normal force, so it applies only while the ball is on the ground and
vanishes automatically in flight, whereas an applied force needs the "am I touching the floor?"
test that both other simulators had to hand-write — and which rSim gets wrong by applying its
drag to the ball's full 3-D velocity, `z` component included (`sslworld.cpp:288-291`). The
choice of fix is another ticket's.

**Source class: MEASURED.**

---

## 4. Rulebook limits that bind the simulator

All from the SSL rule book, quoted from the chapter sources on `master` and cross-checked against
the rendered edition for section numbers.

| § | rule | limit |
|---|---|---|
| 8.4.2 | *Ball Speed* (non-stopping foul) | "A robot must not accelerate the ball faster than 6.5 meters per second in 3D space." |
| 3.2.2 | *Shape* | "A robot must fit inside a 0.18 meters wide and 0.15 meters high cylinder at any point in time." Team height spread ≤ 0.02 m. |
| 3.2.4 | *Dribbling Device* | "80% of the area of the ball when viewed from above has to be outside the convex hull around the robot. This limitation applies as well to all kicking devices, even if such infringement is momentary." Plus: must not elevate the ball; another robot must be able to remove it; must not remove all its degrees of freedom. |
| 8.4.1 | *Excessive Dribbling* | not further than 1 m from where dribbling started |
| 8.4.2 | *Crashing* | closing speed component along the line joining two opposing robots > 1.5 m/s is a foul |
| 2.1.3 | *Field Markings* | "Distances from a robot are measured from its nearest side to the respective measurement point, with an assumed radius of 0.09 m." |
| 2.1.2 | *Field Surface* | "The playing surface is a green felt mat or carpet. The floor under the carpet is level, flat, and hard." |

([`chapters/offenses.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc),
[`chapters/robots.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc),
[`chapters/playingenvironment.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc),
rendered at [robocup-ssl.github.io/ssl-rules](https://robocup-ssl.github.io/ssl-rules/sslrules.html).)

### 4.1 Kick speed: one cap, and it covers the chip

**There is no separate chip-kick speed limit, because there is no chip-kick rule at all.** The
string `chip` does not occur anywhere in the rule book's chapter sources. §8.4.2 says "6.5 meters
per second **in 3D space**" — the magnitude of the full 3-D velocity vector, which is precisely
the phrasing needed to make one number bind a flat kick and a chip identically. A chipped ball
launched at 45° at 6.5 m/s total is legal; the same chip at 6.5 m/s *horizontal* is not.

This is worth stating because the league's simulation protocol *does* separate them —
`RobotSpecs.max_linear_kick_speed` and `max_chip_kick_speed`
([`ssl_simulation_config.proto:48-51`](https://github.com/RoboCup-SSL/ssl-simulation-protocol/blob/master/proto/ssl_simulation_config.proto))
— but those are *hardware* declarations per robot, not rule limits. A simulator should pick 6.5
m/s as a single 3-D magnitude cap on the ball's post-kick speed and apply it to both devices.

Note also that 6.5 m/s is a **foul threshold, not a physical ceiling**. The rule exists because
robots can and do exceed it; a hardware-fidelity simulator should model what the kicker can
deliver and let 6.5 m/s be a referee event, not clamp the physics at it.

### 4.2 Robot dimensions

Ø0.18 m × 0.15 m is a bounding cylinder the robot must fit inside "at any point in time" — an
envelope, not a specification, so any robot geometry inside it conforms. The rules add two
derived figures a simulator can use directly: an **assumed radius of 0.09 m** for every
rule-distance measurement (§2.1.3), and a vision-pattern flat top guaranteed to fit a 0.085 m
radius circle cut off 0.055 m in front of centre (§3.2.3).

### 4.3 Dribbler coverage, in metres

§3.2.4's 80 % is an area fraction of the ball's top-view disc. Converting it to the linear
penetration a simulator can actually check: for a circular segment of area fraction 0.2, the
central angle solves `θ − sin θ = 0.4π`, giving `θ = 2.1131` rad, and the segment depth is

```
h = r (1 − cos(θ/2)) = 0.0215 · (1 − cos 1.0566) = 0.0215 · 0.50814 = 0.010925 m
```

**The ball may be at most 10.9 mm inside the robot's convex hull**, and "even if such
infringement is momentary" means this is an instantaneous bound, not a time average — which is a
constraint on how deep a dribbler pocket may be, not just on where the ball rests.

### 4.4 Robot speed

Not re-derived: `01` established the rule book caps no robot speed, acceleration or motor power,
and that the 1.5 m/s figures are a Stop-state limit and a collision closing-speed limit rather
than a top-speed cap. Nothing found here changes that, and no rule constrains the drivetrain
except §3.2.2's cylinder, which bounds wheel diameter.

**Source class: RULEBOOK.**

---

## 5. Carpet against a rubber omni-wheel roller

One measurement exists, and it is peer-reviewed. Williams, Carter, Gallina & Rosati built a
three-wheeled omni robot for the Ohio University RoboCup team, hit slip in testing, and measured
the coefficients on a tilt table: the wheels were fixed at an angle such that **only the rollers**
touched the surface, a board covered with the test material was pivoted up until the vehicle slid,
and the coefficient was taken as the tangent of the slide angle, averaged over several trials.

| surface | μ′_W max (roller, drive direction) | μ′_T max (roller, transverse) | μ″_W max / μ″_T max (hub between rollers) |
|---|---|---|---|
| paper | 0.26 | 0.09 | 0.47 |
| **carpet** | **0.25** | **0.15** | **0.56** |

(R. L. Williams II, B. E. Carter, P. Gallina & G. Rosati, "Dynamic Model with Slip for Wheeled
Omni-Directional Robots", *IEEE Transactions on Robotics and Automation* **18**(3), 285–293, June
2002; final manuscript at
[people.ohio.edu/williams/html/PDF/IEEETRA02.pdf](https://people.ohio.edu/williams/html/PDF/IEEETRA02.pdf),
§III.A.2 and Table I.)

**μ′_W = 0.25 is the number this deliverable asks for**: the roller sliding on carpet along the
wheel's drive direction, which is the traction budget. μ′_T = 0.15 is the transverse direction,
where the roller rolls rather than slides. μ″ = 0.56 is the rigid hub material in the gaps between
rollers, and the paper's headline finding is that those gaps — 10 % of each angular sector on
their wheel — dominate slip behaviour "even at low rates and accelerations", because the contact
coefficient more than doubles as each gap passes through the contact patch.

**What a simulator should pick, and the honest caveat.** 0.25 for the roller contact. But two
mismatches must be recorded rather than papered over. The roller **compound is not stated** — the
wheels were off-the-shelf units "developed for material handling applications", not the moulded
rubber rollers a modern SSL robot carries — and the **carpet is not identified**, certainly not
the green felt mat §2.1.2 specifies. So while *an omniwheel roller on carpet* is measured, the
narrower question — *an SSL rubber roller on SSL felt* — has **no source**: no SSL team publishes
a wheel/carpet coefficient, and a search of the ETDP archive found only the qualitative remark
that the carpet's friction changes over a tournament as the robots wear it, forcing gain
retuning. That sub-question should be recorded as a deliberate approximation.

### 5.1 What this says about the model

`models/robot_rollers.xml:311` gives every roller collision geom `friction="1.0 …"` against
`models/field.xml:42`'s floor at `0.8`; MuJoCo mixes unpaired geoms by taking the element-wise
maximum, so the **effective roller↔floor sliding coefficient is 1.0** — four times the only
measured figure. Neither 1.0 nor 0.8 has a basis; 0.8 in particular appears nowhere in any source
consulted here.

This matters beyond this ticket. `01` sized the drivetrain against a friction budget of
`μ·m·g = 26.4` N computed at μ = 1.0. At μ = 0.25 that budget is `0.25 · 2.69207 · 9.81 = 6.60` N
against the 8.16 N of forward ground force `01` recommends — per wheel, `0.25 · 6.602 = 1.65` N of
capacity against 2.59 N demanded. **The robot would slip rather than be torque-limited**, which
inverts `01`'s central conclusion. `01`'s `forcerange` recommendation does not change (it comes
from the motor, not the floor), but its "31 % of the friction budget, so slip is a rare event"
finding is contingent on a coefficient that has no source. This is a cross-ticket flag, not a
re-derivation: whichever ticket sets the method for justifying a contact coefficient should carry
it, and the map already lists robot↔robot and robot↔wall friction as pending on the same question.

**Source class: MEASURED** (for an omniwheel roller on carpet; the SSL-specific rubber-on-felt
figure is UNSOURCED, §5 above).

---

## Unresolved, and stated assumptions

1. **The COR band 0.47–0.56 is normal-direction only.** Tangential restitution and ball-off-robot
   restitution are unsourced (§2.4), and the two published SSL figures for the latter differ by
   3×.
2. **ER-Force's 0.56 and μ_k = 0.35 are labelled "(empirical measurments)" in a source comment,**
   not in a paper. The method is stated (`sqrt(h'/h)`, `−a/g`) but the raw data is not published.
   They are treated as measurements here because they are the calibration of the league's own
   reference simulator, which the ticket names as acceptable, and because both are independently
   corroborated — 0.56 by TIGERs' 0.47, and 0.35 by Rojas & Simon's 0.354.
3. **All three rolling-deceleration measurements are venue-specific.** ER-Force say so explicitly
   and TIGERs carry six venue specializations. 0.26–0.31 is a band, not a constant, and a
   simulator that wants one number is choosing.
4. **Which carpet.** No source consulted identifies the carpet by product. §2.1.2 says only "green
   felt mat or carpet". Every carpet-dependent number here therefore inherits an unquantified
   venue spread.
5. **Williams et al. is a 2002 measurement on unspecified rollers and unspecified carpet** (§5).
   It is the only measurement found; it is not an SSL-carpet measurement.
6. **The ball's moment of inertia** (§3.3) is measured at β ≈ 0.5–0.7 against the model's implied
   0.4, and is not fixable without an explicit inertia declaration. Recorded, not fixed.
7. **`ballBounceVel = 0.1`** in rSim — a velocity below which no bounce occurs — has no support in
   anything found here, unlike the 0.01 m/s roll-death threshold, which does (§3.1).
8. **The 6.5 m/s ball cap is a foul threshold, not a physical limit** (§4.1). Treating it as a
   clamp on the kicker would be a rules model, not a hardware model.

## Sources

League (RoboCup Small Size League), fetched 2026-09-09:
- Rules of the RoboCup Small Size League, rendered — https://robocup-ssl.github.io/ssl-rules/sslrules.html (§2.1.2 Field Surface, §2.1.3 Field Markings, §2.2 Ball, §3.2.2 Shape, §3.2.4 Dribbling Device, §8.4.1 Excessive Dribbling, §8.4.2 Ball Speed / Crashing)
- Chapter sources on `master` — https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc, .../robots.adoc, .../offenses.adoc
- SSL-Vision geometry protocol, `SSL_BallModelStraightTwoPhase` and `SSL_BallModelChipFixedLoss` — https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto
- SSL simulation protocol, `RobotSpecs` — https://github.com/RoboCup-SSL/ssl-simulation-protocol/blob/master/proto/ssl_simulation_config.proto
- ER-Force Extended Team Description Paper 2016, §5 "Ball model" — https://ssl.robocup.org/wp-content/uploads/2019/01/2016_ETDP_ER-Force.pdf
- RobôCIn Extended Team Description Paper 2024 — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf

Golf ball conformance (R&A / USGA, joint owners of the Equipment Rules):
- R&A, Rules of Equipment, Part 4 — Conformance of Balls, Rule 4.2 (Weight) and Rule 4.3 (Size) — https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls
- USGA Equipment Rules, Part 4 (identical text, Rules 2 and 3) — https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/equipment-rules.html

Measurement work:
- Rojas, R. & Simon, M., *Like a Rolling Ball*, Freie Universität Berlin — https://robocup.mi.fu-berlin.de/buch/rolling.pdf (26 vision-tracked shots on carpet; a_slide −3.47 m/s², a_roll −0.305 m/s², k = 0.5805 ± 0.042)
- Ismail, M. R. M. & Stronge, W. J. et al., "Coefficient of restitution of sports balls: A normal drop test", *IOP Conf. Ser.: Mater. Sci. Eng.* **36** 012038 (2012), doi:10.1088/1757-899X/36/1/012038 — https://iopscience.iop.org/article/10.1088/1757-899X/36/1/012038/pdf (golf ball 46 g / 42.80 mm; COR 0.889–0.897 on steel, 0.666–0.681 on wood, 1.3–1.7 m drops)
- Williams II, R. L., Carter, B. E., Gallina, P. & Rosati, G., "Dynamic Model with Slip for Wheeled Omni-Directional Robots", *IEEE Transactions on Robotics and Automation* **18**(3), 285–293, June 2002 — https://people.ohio.edu/williams/html/PDF/IEEETRA02.pdf (tilt-table roller friction on carpet: μ′_W 0.25, μ′_T 0.15, hub 0.56)

Simulator calibration (the league's two reference simulators):
- ER-Force `framework`, `src/amun/simulator/simulator.cpp:40-65` — the "(empirical measurments)" block: ball/floor restitution 0.56 from `sqrt(h'/h) = sqrt(0.314)`, ball/floor friction μ_k = 0.35, ball/robot restitution 0.60 and friction 0.18 — https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simulator.cpp
- ER-Force `framework`, `src/amun/simulator/simball.cpp:36-88` — `\mu_r = -a/g = 0.0357`, `hackFactor = 1.4`, `rollingDeceleration = 0.49`, the 0.01 m/s snap-to-dimple branch, and the citation of Rojas & Simon — https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simball.cpp
- TIGERs Mannheim `Sumatra`, `modules/moduli-geometry/.../BallParameters.java` — accSlide −3000 mm/s², accRoll −260 mm/s², kSwitch 0.64, inertiaDistribution 0.5, chipDampingZ 0.47, chipDampingXY 0.8/0.85, redirectRestitutionCoefficient 0.2, each with per-venue specializations including `ROBOCUP` — https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java

Source (this repo, local revision):
- `models/ball.xml:7` (ball geom: mass, radius, friction, `solref`)
- `models/scene_rollers.xml:21-23` (ball/dribbler and ball/floor contact pairs)
- `models/field.xml:42` (floor plane, `friction="0.8 …"`), `:44` (wall `solref`)
- `models/robot_rollers.xml:311` (roller collision geom, `friction="1.0 …"`)
- `docs/research/rsoccer-vs-mjsoccer.md` §1 (`P4`, `P5`), §3 (`B9`), §5 (`C2`, `C3`)
- `.scratch/physics-fidelity/research/01-drivetrain-envelope.md` §3.1, §5

Upstream (`/tmp/rsim-upstream`, `b413932`):
- `src/robosim/sslconfig.h:119-128` (`ballRadius` 0.0215, `ballMass` 0.043, `ballFriction` 0.05, `ballBounce` 0.5, `ballBounceVel` 0.1, plus the dead `ballSlip`/damping fields of `P5`)
- `src/robosim/sslworld.cpp:151-182` (ball/wall and ball/ground surfaces, `mu = 1`, bounce 0.5)
- `src/robosim/sslworld.cpp:279-300` (the constant Coulomb drag applied to the full 3-D velocity, and the 0.01 m/s kill)
