#!/usr/bin/env bash
# Train PPO on mjsoccer envs on a GPU backend, one run at a time.
#
#   examples/sweep_gpu.sh mjx all
#   examples/sweep_gpu.sh mjlab GoToBall-v0,Dribbling-v0 --epochs 50
#
# Args past the second go to examples/ppo_<backend>.py. Each run saves its run directory
# (args.json + checkpoints) to runs/sweep/<backend>_<timestamp>/<env>/ and its output to
# <env>.log beside it; render them with examples/visualize_sweep.py. Meant to run under screen.
#
# One run at a time: the GPU holds one. Each run starts at the script's own --num-envs default
# and, on a GPU out-of-memory, restarts from scratch at half the worlds (down to 1024); the
# failed attempt's log stays as <env>.oom-<n>.log. A --num-envs in the args is used as given,
# no retry.
set -euo pipefail
cd "$(dirname "$0")/.."

usage="usage: $0 mjx|mjlab all|ENV[,ENV...] [ppo args...]"
backend=${1:?$usage}
envs=${2:?$usage}
shift 2
extra=("$@")
case $backend in mjx) group=gpu ;; mjlab) group=mjlab ;; *) echo "$usage" >&2; exit 2 ;; esac

[[ $envs == all ]] && envs=$(uv run python -c 'import mjsoccer; print(",".join(mjsoccer.ENV_IDS))')

out=runs/sweep/${backend}_$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
echo "$0 $backend $envs ${extra[*]}" >"$out/cmd.txt"

export PYTHONUNBUFFERED=1 # logs fill live, not at exit
fixed=0
for a in "${extra[@]}"; do [[ $a == --num-envs* ]] && fixed=1; done

train() { # env [num_envs]
	uv run --group "$group" python "examples/ppo_$backend.py" --env "$1" --save "$out/$1" \
		${2:+--num-envs "$2"} "${extra[@]}" >"$out/$1.log" 2>&1
}

echo "sweep -> $out"
for env in ${envs//,/ }; do
	echo "$(date +%T) start $env"
	if ((fixed)); then
		if train "$env"; then echo "$(date +%T) done $env"; else echo "$(date +%T) FAIL $env (see $out/$env.log)"; fi
		continue
	fi
	n=$(uv run python -c "import sys; sys.path.append('examples'); import ppo_$backend as p; print(p.parse_args(['--env', '$env']).num_envs)")
	while :; do
		if train "$env" "$n"; then echo "$(date +%T) done $env (--num-envs $n)"; break; fi
		# JAX/XLA, torch and Warp each word a GPU out-of-memory their own way.
		if ((n > 1024)) && grep -qiE 'RESOURCE_EXHAUSTED|out of memory|OutOfMemory|failed to allocate' "$out/$env.log"; then
			mv "$out/$env.log" "$out/$env.oom-$n.log"
			rm -rf "${out:?}/$env"
			n=$((n / 2))
			echo "$(date +%T) OOM $env, retry at --num-envs $n"
		else
			echo "$(date +%T) FAIL $env (--num-envs $n, see $out/$env.log)"
			break
		fi
	done
done
echo "sweep finished -> $out"
