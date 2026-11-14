# 10: GoToBall no MJX (cena de 7 robôs) e relatório de decisões/s

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** GoToBall roda no MJX-Warp com a cena de 7 robôs, com `naconmax` dimensionado pelos candidatos do broadphase. Medir decisões/s ponta a ponta (env completo: task, controlador, reset) de ContestedPossession e GoToBall contra a base CPU 4.0 do ticket 02. É relatório, não portão: serve para pegar regressão grosseira da lógica JAX.

**Blocked by:** 08 (Kick no lote do MJX-Warp), 09 (Falhas por mundo no MJX-Warp)

**Status:** done

- [x] GoToBall no task-logic parity e no Warp smoke
- [x] `overflow == 0` com o `naconmax` escolhido
- [x] Decisões/s das duas tasks (MJX × CPU 4.0) anotadas neste ticket

## Comments

**Feito em 2026-11-14** (branch `gpu-port-10-mjx-gotoball`).

- **`GoToBallEnv`** (`mjx/tasks.py`): recompensa 1 e fim a < 0.2 m da bola, sem estado de task.
  Bola uniforme em ±(HALF − 0.1), depois os 7 robôs pelo `scatter`.
- **Decisão `_scatter` × `sample_pose`**: o GoToBall segue a regra do CPU (`_scatter`: caixa da
  task, `≥ 0.2 m`, heading uniforme), não a do `sample_pose` (margem 0.3, `> 0.25`). Núcleo comum
  `mjx.base.first_free(cand, taken, min_dist, strict)`; o `sample_pose` virou um caso dele.
  Por robô, `SCATTER_CANDIDATES = 8` pares (x, y) + o heading, sequencial entre robôs (laço
  Python, 7 fixo). Com 7 robôs e a bola num campo de 104 m² cada sorteio é rejeitado ~1 % das
  vezes, então os 8 falham ~1e-16 — não pus 200 (o spec falava em 200 para o `sample_pose`,
  cujo CPU limita em 200; o `_scatter` não tem limite e 8 já dá a mesma distribuição na prática).
  Nenhum livre → fica o de maior distância mínima e o primeiro estado marca `reset_overlap`.
- **`place` devolve `(data, overlap)`** em todas as tasks; `reset` passa o `overlap` ao
  `init_state(data, rng, overlap=False)`, que o põe em `metrics["reset_overlap"]` (0 nos passos).
  `MjxEnv.place_ball` novo (`BallIds.place`, em repouso).
- **Parity** (`test_mjx_task_logic.py`): GoToBall em `TASKS` (init, 5 rollouts até o fim, place).
  O place do GoToBall usa um `np_random` falso por blocos (`_Scattered`): o CPU consome um
  número variável de candidatos por robô e o sorteio do heading pula o resto do bloco. Os 2
  primeiros candidatos do robô 0 caem na bola, então o caminho de rejeição roda. Teste novo:
  todos os candidatos do robô 0 na bola → `overlap`. Mutação: `first_free` sempre no candidato 0
  derruba 7 testes.
- **Achado — o snapshot do parity custava 0.33 s por passo com 7 robôs**: o `Data` do MJX-JAX
  (`put_data(impl="jax")`) tem `_impl` com buffers densos de todos os pares de geoms (`efc_J`
  1.4 M × 55: 833 MB com 7 robôs, 33 MB com 2), e o `jit(control)`/`jit(settle)` copiava isso a
  cada chamada (60 s por seed). O `_put` agora guarda `.replace(_impl=None)`: a lógica só lê
  campos públicos (o `overflow_bits` já caía em 0 sem `overflow`). Suíte do parity: 39 testes em
  ~25 s.
- **Warp smoke**: GoToBall em `TASKS` (repeat_action 1 e 2). O smoke agora devolve os bits de
  overflow OR-ados nos 10 passos (tem que ser 0) e `reset_overlap` (0); a posse do amarelo no
  reset fica só para o ContestedPossession; no GoToBall confere as 8 posições (7 robôs + bola) a
  ≥ 0.2 m no reset. O teste do kick ficou só no ContestedPossession (é sobre o kick, e a cena
  dele assume o robô na origem).
- **`naconmax`**: mantido `max(64, 24 × n_robots)` = 168/mundo com 7 robôs. O que enche o pool é
  `ncollision` (candidatos do broadphase, total do lote; o `nacon` mantido é ~1/mundo): pico de
  **120.8/mundo** no GoToBall (40.3 no ContestedPossession, cota 64), também com a perseguição
  scripted dando 447 fins de episódio em 1 024 mundos. O pool é do lote, então o que importa é
  a média; os ~316/mundo de colisão forçada do 05 em um mundo não estouram enquanto a média
  ficar abaixo de 168. `nefc` pico 102 por mundo contra `njmax` 672 (dá para baixar, não mexi).
- **Pedido do 09 (`ITERATIONS`/`LS_ITERATIONS` com 7 robôs)**: nenhum bit de overflow em nenhum
  run abaixo (OR de todos os mundos e passos = 0), incluindo os dois de perseguição. Nada a
  mascarar.
- **Decisões/s** (`.scratch/gpu-port-impl/10-bench/dps_mjx.py`, saída em `results.txt`): env
  inteiro sob `brax.envs.wrappers.training.wrap` (auto-reset com estado em cache), 40 Hz,
  `repeat_action` 1, ações uniformes no device (kick/chip disparam a cada re-arme, como no bench
  do CPU), `fori_loop` de 20 decisões por chamada, melhor de 3 × 10 s, um processo por tamanho,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, RTX 3060 12 GB. CPU 4.0 do ticket 02 (24/12 envs
  `AsyncVectorEnv`, melhor de 3 × 20 s):

  | task | CPU 4.0 | MJX-Warp | nworld | × |
  |---|---|---|---|---|
  | ContestedPossession (2 robôs) | 4 862 (24 envs async) | 7 599 | 256 | 1.6 |
  | | | 13 408 | 1 024 | 2.8 |
  | | | 15 371 | 2 048 | 3.2 |
  | | | **16 427** | 4 096 | **3.4** |
  | | | OOM | 8 192 | |
  | GoToBall (7 robôs) | 1 802 (12 envs async) | 2 029 | 256 | 1.1 |
  | | | 2 185 | 512 | 1.2 |
  | | | **2 211** | 1 024 | **1.2** |
  | | | 2 206 | 2 048 | 1.2 |
  | | | 2 131 | 4 096 | 1.2 |

  Perseguição scripted (CHASE=1, 1 024 mundos): 12 214 e 2 273. O GoToBall satura em ~2.2 k já
  a 512 mundos. Coerente com o 05: física pura 111 k SPS com 7 robôs = 4.6 k decisões/s a 24
  iterações, e o esquema de disparo do 08 roda 48 iterações em todo passo de controle, dispare
  alguém ou não → ~2.3 k. Para 2 robôs, 918 k SPS ÷ 48 = 19 k, e medimos 16.4 k. Ou seja: a lógica
  JAX não tem regressão grosseira; o custo dominante é o 2× do passo de disparo. O ganho óbvio
  (rodar 24 iterações quando nenhum mundo do lote dispara, predicado fora do `vmap`) fica para o
  ticket que for medir treino; com uma política treinada o disparo é raro.
- **OOM**: com a pré-alocação default do XLA (75 % da GPU) o Warp fica sem memória para os
  grafos já a 8 192 (2 robôs) / 2 048 (7 robôs); com `XLA_PYTHON_CLIENT_PREALLOCATE=false` vai a
  4 096 / 4 096. Vários tamanhos no mesmo processo também acumulam memória (o 4 096 falha depois
  do 2 048): um processo por tamanho.
- **Verificação** (depois de `git merge gpu-port-impl`, já em dia): `uv sync && uv run pytest`
  95 passed, 2 skipped; `uv sync --group gpu` + `systemd-run --user --scope -p MemoryMax=10G uv
  run pytest` 142 passed. Nenhum processo na GPU depois.
