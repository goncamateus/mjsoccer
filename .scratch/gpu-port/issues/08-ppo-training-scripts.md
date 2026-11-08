# Qual derivado do PPO, e como se estruturam os scripts de treino

Type: grilling
Status: resolved
Blocked by: 02, 04

## Pergunta

Com o levantamento do 02 e um ambiente que roda (04):

- Que derivado do PPO, e sobre que base de código (adotar uma stack, vendorizar um arquivo único como
  o `examples/sac_cleanrl.py` fez para o SAC, ou escrever do zero)?
- Um script por task ou um script com a task como argumento? Onde ficam (`examples/`, `scripts/`)?
- Como o treino escolhe o `nworld` (limitado pelo pool de contatos do Warp, 05) e o `repeat_action`
  (o motor é só o MJX-Warp, ADR 0002).
- Logging, checkpoint, e avaliação: avaliar no backend CPU periodicamente, ou só no MJX?

## Answer

Resolvido em 2026-11-08 (grilling).

- **Base**: `brax ppo.train` como dependência do grupo `gpu`, sem vendorizar. Fallback: se o `pmap`
  incondicional do Brax quebrar com Warp numa GPU, vendoriza-se só o `ppo/train.py` com `pmap` → `jit`
  (verificação em [Rodar o `brax ppo.train` sobre o MJX-Warp nesta máquina](12-brax-ppo-warp-smoke.md)).
- **Consequência para o ticket 10**: a API funcional do lado MJX é a interface `brax.envs.Env`
  (`reset(rng)`, `step(state, action)`, `State` com `obs/reward/done/metrics/info`).
- **Derivado**: PPO vanilla do Brax; HPs como defaults de flag (estilo Playground); variantes (crítico
  assimétrico, LR por KL, distribucional) ficam como flags do Brax, sem código nosso.
- **Script**: um só, `examples/ppo_mjx.py --env <id>`, espelho do `examples/ppo_agent.py` (`--env`,
  `--seed`, `--save`, `--play`), saída em `runs/ppo_mjx_<env>/`, rodado com `uv run --group gpu`.
- **`nworld`**: `--num-envs` com default 8 192 (tasks de 2 robôs) e 1 024 (7 robôs), folga sobre o
  pico do 05; overridable; sem autotune. `naconmax`/`njmax` dimensionados pelo env, não pelo script.
- **Decisão**: `--repeat-action` vai como kwarg do env (default 1); `action_repeat=1` no Brax. A
  decisão tem um dono só, o env.
- **Logging/checkpoint**: `progress_fn` imprime no stdout (passos, decisões/s, retorno de eval);
  checkpoint Orbax do Brax em `runs/`. Sem tensorboard/wandb.
- **Avaliação**: durante o treino só o eval embutido do Brax no MJX-Warp. `--play` carrega o
  checkpoint e roda a política (inferência JAX na CPU) no **backend CPU**, com o render/viewer de lá.
  Sem eval periódico na CPU.
- **Wrapper de treino**: o script usa o equivalente do `wrap_for_brax_training` (com `Data.where` e
  `full_reset`) que mora no pacote do lado MJX; o desenho dele fica com o ticket 10 e a fog "Reset em lote".
