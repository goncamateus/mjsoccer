"""Ticket 12: eval return by decisions, Brax PPO (MJX-Warp) x RSL-RL (mjlab), from the
train12 logs. Writes research/07-rsl-rl-vs-brax.png and prints a per-epoch table.
    uv run --group mjlab python .scratch/mjlab-port/bench/plot12.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SERIES = {"mjx": ("Brax PPO, MJX-Warp", "#2a78d6"), "mjlab": ("RSL-RL, mjlab", "#eb6834")}
# ppo_mjx: "step 655,360 dps ... (train N) eval return R"; ppo_mjlab: "decisions N dps D eval return R"
LINE = re.compile(r"(?:step|decisions)\s+([\d,]+)\s+dps\s+(\S+)(?:\s+\(train\s+(\S+)\))?"
                  r"\s+eval return\s+(\S+)\s+len\s+(\S+)")


def parse(path):
    rows = []
    for m in LINE.finditer(path.read_text()):
        step, dps, train, ret, length = m.groups()
        rows.append((int(step.replace(",", "")), float(train or dps), float(ret), float(length)))
    return rows


def main():
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, (label, color) in SERIES.items():
        rows = parse(HERE / "train12" / f"{key}.log")
        xs, ys = [r[0] / 1e6 for r in rows], [r[2] for r in rows]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=3, label=label)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#333")
        dps = [r[1] for r in rows if r[0] > 0]
        print(f"{label}: final {ys[-1]:.3f}, best {max(ys):.3f}, train dps median "
              f"{sorted(dps)[len(dps) // 2]:.0f}")
        for r in rows:
            print(f"  {r[0]:>11,}  {r[2]:7.3f}  len {r[3]:6.1f}")
    ax.set(xlabel="decisions (M)", ylabel="eval return (128 worlds, deterministic)",
           title="ContestedPossession-v0, seed 0, 8 192 worlds")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = HERE.parent / "research" / "07-rsl-rl-vs-brax.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
