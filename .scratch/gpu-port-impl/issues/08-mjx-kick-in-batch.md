# 08: Kick no lote do MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Os mundos que disparam rodam o passo de disparo com 32 iterações a `dt/4` + 16 a `dt`; os ociosos rodam 24 a `dt` e ficam congelados com `timestep = 0` nas outras 24, via `opt.timestep` por mundo (`in_axes`), sem `Data.where` por passo. `plunger_lock` `int32 (n_robots, 2)` e `fired` no estado por mundo, decrementado por passo de controle, cooldown 10 lido do CPU. A `KICK_TABLE_MS` vem do CPU.

**Blocked by:** 07 (Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick)

**Status:** done

- [x] ContestedPossession no MJX dispara kick e chip
- [x] Task-logic parity cobre `plunger_lock`, `fired` e o cooldown
- [x] Warp smoke: um mundo dispara e um ocioso mantém `qpos`/`qvel` bit a bit nas iterações a `timestep = 0`

## Comments

**Feito em 2026-11-14** (branch `gpu-port-08-mjx-kick`).

- **Estado por mundo**: `plunger_lock` `int32 (n_robots, 2)` e `fired` `bool (n_robots,)` em
  `State.info["world"]`, zerados no `init_state`. `control` faz o `BaseEnv._actuate`: `a > 0` com
  o plunger destravado põe `jnp.interp(KICK_SPEED_MAX * a, KICK_TABLE_MS[j], KICK_TABLE_N)` no
  atuador, trava com `kick_cooldown` (lido de `cpu._kick_cooldown`, 10 a 40 Hz) e marca `fired`;
  `settle` decrementa a trava (mesmo ponto do CPU: depois da física, antes da obs).
- **Agenda da física**: `timesteps(fire)` devolve os `(L,)` timesteps de um passo de controle de
  um mundo: disparando, `FIRING_FINE_STEPS` a `dt/FIRING_SUBSTEPS` + o resto a `dt` (32 + 16);
  ocioso, `decimation` a `dt` + zeros (24 + 24). `L` é o mesmo para os dois (48 a 40 Hz), então o
  lote roda um `fori_loop` só. `substep(data, timestep)` é um `mjx.step` com
  `sys.replace(opt=opt.replace(timestep=ts))`: sob o `vmap` do env o `timestep` vira campo
  batcheado do modelo (o mesmo `in_axes` do ticket 15), sem `Data.where` por passo.
  `physics(data, fire)` usa a agenda; `physics(data, n=...)` (keyword-only) segue sendo os `n`
  passos simples do `seat_ball`. `step` passa `world["fired"].any()`.
- **Custo**: com o esquema, todo passo de controle roda 48 iterações no lote, dispare alguém ou
  não (um `cond` com predicado por mundo vira `select` sob `vmap`). É o 2× do K = 32 medido no
  ticket 15; o ganho de pular as 24 extras quando ninguém no lote dispara fica para quem medir
  (precisaria do predicado fora do `vmap`).
- **Testes**: no task-logic parity, `_action` deixa kick/chip aleatórios (≈ metade dos passos
  pede), `BASE_STATE` ganhou `plunger_lock` e `fired`, o `ctrl` comparado agora inclui as forças
  dos plungers, e o rollout exige ao menos um disparo e um pedido bloqueado pela trava. Novo
  `test_control_step_timesteps` (agenda 32×dt/4 + 16×dt / 24×dt + 24×0, soma = 25 ms). Mutação:
  decrementar a trava de 2 ou tirar o `lock == 0` derruba os 5 rollouts. Warp smoke novo
  `test_warp_kick_chip_and_frozen_idle_worlds` (4 mundos, bola assentada no azul): mundo 0 chuta,
  1 dá chip, 2-3 ociosos; dentro do passo de disparo, iteração a iteração, os ociosos mantêm
  `qpos`/`qvel` bit a bit nas 24 iterações a `timestep = 0`. Medido no Warp: kick 6.83 m/s, chip
  6.33 m/s e 0.47 m de altura; o mesmo cenário no CPU dá 6.82 / 6.34 / 0.46. O smoke antigo
  passou a mandar kick/chip aleatórios também.
- **Bug pego no caminho**: o `seat_ball` chamava `physics(d, n)` posicional e, com a assinatura
  nova, o `n` caía em `fire`; o smoke do kick pegou (bola ainda rolando a 0.9 m/s). `n` virou
  keyword-only.
