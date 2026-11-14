# 09: Falhas por mundo no MJX-Warp

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Um mundo com `qpos`/`qvel` não finito, ou com qualquer bit do `overflow` do Warp, termina (terminal, não truncation), com recompensa e obs zeradas, e o auto-reset do Brax/Playground o restaura; sem abort dentro do jit. `State.metrics` ganha `diverged`, `overflow` e `reset_overlap` (0/1 por passo). Os testes MJX checam os contadores no host e abortam com os bits de overflow por tipo na mensagem. O CPU mantém o `RuntimeError`.

**Blocked by:** 07 (Primeiro passo do MJX: grupo `gpu`, `mjsoccer.mjx.make` e ContestedPossession sem kick)

**Status:** done

- [x] Mundo forçado a NaN termina, sai com recompensa/obs 0 e volta pelo auto-reset sem derrubar os outros
- [x] Overflow forçado (buffer subdimensionado) trata o mundo igual ao NaN
- [x] Os três contadores aparecem em `State.metrics`
- [x] Task-logic parity e smoke falham com mensagem clara se `diverged` ou `overflow` > 0

## Comments

**Feito em 2026-11-14** (branch `gpu-port-09-mjx-failures`).

- **Onde**: tudo no `settle` (`mjx/base.py`), o fim de cada passo de controle. `diverged` =
  `qpos`/`qvel` não finito; `overflow` = `_impl.overflow != 0` (qualquer bit, lido por
  `overflow_bits(d)`, 0 para `Data` fora do Warp, i.e. os snapshots JAX-CPU do parity). Falha →
  `terminated` (nunca `truncation`), `obs` e `reward` por `jnp.where` para 0. Com
  `repeat_action > 1` o mundo que falhou fica parado pelo `Data.where` já existente, e as
  métricas dele são as do passo que falhou.
- **`State.metrics`**: `diverged`, `overflow`, `reset_overlap`, float32 0/1, já no `init_state`
  (o `EpisodeWrapper` do Brax só soma as chaves que vê no reset). `reset_overlap` fica 0 até uma
  task colocar por `sample_pose` (10/11).
- **Overflow é pegajoso**: o Warp só zera a máscara no `reset_data`; nosso reset parte de
  `make_data`, então o auto-reset (que troca o `Data` inteiro pelo inicial em cache) a limpa.
  Bits medidos (`OverflowType`): com o buffer default, 4 mundos × 200 passos com ações
  aleatórias → nenhum bit, nem `ITERATIONS`/`LS_ITERATIONS` (que também são bits da máscara e
  terminariam o mundo; se aparecerem com 7 robôs, o 10 decide se mascara).
  `nacon_per_world=2` → `BROADPHASE` em todos e `NEFC|ITERATIONS` em um, já no reset.
- **Host**: `mjsoccer.mjx.check_failures(state)` (lote ou não) levanta `RuntimeError`
  "MJX: k of N world(s) diverged …; m of N world(s) overflowed Warp (BROADPHASE in 4, …):
  naconmax/njmax too small?". O parity chama depois do `init_state` e de cada `settle`; o smoke
  depois do reset e de cada passo. O `examples/ppo_mjx.py` (13) usa o mesmo.
- **Knob novo**: `make(..., nacon_per_world=N)` sobrescreve a cota por mundo do buffer de contato
  (`njmax` segue 4×). Serve ao teste do overflow e a quem quiser dimensionar à mão.
- **Testes**: task-logic (JAX-CPU) +3: NaN em `qpos`/`qvel` no `settle` → done terminal,
  reward/obs 0, `diverged` 1; `check_failures` passa limpo e levanta com a contagem. Smoke (GPU,
  subprocesso) +3: NaN no mundo 0 com `repeat_action` 1 e 2 → só ele termina zerado e marcado,
  os outros finitos e sem flag; sob `brax.envs.wrappers.training.wrap` (`repeat_action` 1) o
  mesmo passo devolve o mundo 0 ao estado inicial (`qpos`/`qvel`/obs bit a bit), o
  `episode_metrics["diverged"]` soma 1 e mais 5 passos saem limpos; overflow forçado → todos
  terminam zerados com `overflow` 1 e a mensagem cita `BROADPHASE`. Mutação conferida: anular o
  `failed` derruba os 5 testes de falha.
- **Achado (para o 13) — `repeat_action > 1` sob o `EpisodeWrapper` do Brax**: `jit(wrap(env).step)`
  com `repeat_action=2` dá `UnexpectedTracerError` (um `DataWarp` segura `contact__*` como
  `BatchTracer` do `vmap` dentro do `scan` do `EpisodeWrapper`). Já acontece no commit do 07, sem
  este diff, e também com o laço do `repeat_action` desenrolado em Python (não é o `fori_loop`).
  Sem `wrap` (`jit(vmap(step))`) funciona. Por isso a metade auto-reset do teste roda só com
  `repeat_action=1`. O 13 precisa resolver (ou usar `action_repeat` do wrapper com
  `repeat_action=1` no env).
- **Achado (para o 13/17) — o `AutoResetWrapper` do Brax só restaura `pipeline_state` e `obs`**:
  `info["world"]` (`steps`, `grace`, `prev`, …) segue do episódio que acabou. O Playground não
  está no grupo `gpu`; o wrapper do 13 tem de resetar o `info["world"]` junto.
- **Merge com o 08**: o smoke passou a sortear kick/chip também (o 08 tirou o repouso); NaN e
  overflow rodam com essas ações. O probe dos bits acima foi com kick em repouso.
- **Verificação** (depois do merge com `gpu-port-impl`, 08 dentro): `uv sync && uv run pytest`
  95 passed, 2 skipped; `uv sync --group gpu` + `systemd-run --user --scope -p MemoryMax=10G uv
  run pytest` 124 passed. Nenhum processo na GPU depois.
