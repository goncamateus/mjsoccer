#!/usr/bin/env bash
# Train one RL method on mjsoccer envs x both tiers, N runs at a time.
#
#   examples/sweep.sh sac all 4
#   examples/sweep.sh ppo GoToBall-v0,Dribbling-v0 2 --timesteps 200000
#
# Args past the third go to examples/<algo>_agent.py. Each run saves its policy and full
# output to runs/sweep/<algo>_<timestamp>/<env>_t<tier>.{zip,log}; meant to run under screen.
set -euo pipefail
cd "$(dirname "$0")/.."

usage="usage: $0 sac|ppo all|ENV[,ENV...] JOBS [agent args...]"
algo=${1:?$usage}
envs=${2:?$usage}
max=${3:?$usage}
shift 3
extra=("$@")
case $algo in sac | ppo) ;; *) echo "$usage" >&2; exit 2 ;; esac

[[ $envs == all ]] && envs=$(uv run --group rl python -c 'import mjsoccer; print(",".join(mjsoccer.ENV_IDS))')

out=runs/sweep/${algo}_$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
echo "$0 $algo $envs $max ${extra[*]}" >"$out/cmd.txt"

export PYTHONUNBUFFERED=1 # logs fill live, not at exit
# ponytail: split the cores evenly so N torch runs don't oversubscribe; override by exporting it.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / max > 0 ? $(nproc) / max : 1 ))}

run() { # env tier
	local name=$1_t$2
	if uv run --group rl python "examples/${algo}_agent.py" --env "$1" --tier "$2" \
		--save "$out/$name" "${extra[@]}" >"$out/$name.log" 2>&1; then
		echo "$(date +%T) done $name"
	else
		echo "$(date +%T) FAIL $name (see $out/$name.log)"
	fi
}

echo "sweep -> $out"
for env in ${envs//,/ }; do
	for tier in 0 1; do
		while (($(jobs -rp | wc -l) >= max)); do wait -n || true; done
		echo "$(date +%T) start ${env}_t$tier"
		run "$env" "$tier" &
	done
done
wait
echo "sweep finished -> $out"
