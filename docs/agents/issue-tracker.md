# Issue tracker: Local Markdown

Issues, specs and research for this repo live as markdown under `.scratch/`, one directory per
effort: `.scratch/<feature-slug>/`.

## Layout

```
.scratch/<feature-slug>/
├── spec.md          # spec-style efforts (gpu-port, mjlab-port)
├── map.md           # map-style efforts (gpu-port, physics-fidelity): see Wayfinding below
├── decisions.md     # optional: decisions out of a grilling session (mjlab-port)
├── issues/
│   ├── NN-<slug>.md         # one ticket per file, numbered from 01, never one combined file
│   ├── NN-<slug>.study.md   # optional companion study guide (Type: study, Source: link)
│   └── NN-<name>/           # optional: scripts and logs a ticket produced
├── research/
│   ├── NN-<slug>.md         # written-up findings, linked from the ticket that asked
│   └── NN-<name>/           # its raw assets (scripts, logs, CSVs)
└── bench/, prototype/, baseline/, candidate-NN/, NN-bench/, ...   # free-form artifacts
```

Not every slug has every file. `gpu-port-impl` and `rl-learning` have no spec or map;
`gpu-port-impl`'s tickets link the spec of `gpu-port` (`Spec: [..](../../gpu-port/spec.md)`).

## Ticket formats

- **Spec-style** (`gpu-port-impl`, `mjlab-port`): a `Spec: [..](../spec.md)` line, then
  `**What to build:**`, `**Blocked by:** NN`, `**Status:** done` and a checklist of
  acceptance criteria. Progress and conversation append under `## Comments` at the bottom.
- **Map-style** (`gpu-port`, `physics-fidelity`, `rl-learning`): `Type:`
  (`research`/`grilling`/`task`), `Status:` and `Blocked by:` lines at the top, the question
  under `## Pergunta`, the result under `## Answer`.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/issues/` (creating the directory if needed),
numbered after the last ticket there.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (Destino, Notes, Decisions so far, Not yet specified, Fora de escopo).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top (`—` when none). A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions so far in `map.md`.
