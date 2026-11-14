# 02 — Protótipo mjlab: ContestedPossession com kick (resultados)

Medido em 2026-11-14, branch `prototype/mjlab` (a partir de `main` `a63f858`), RTX 3060 12 GB, driver
595.99.02. Escopo e o que medir: [`../decisions.md`](../decisions.md) itens 7-13. **Sem gate
numérico**: este arquivo só informa.

Código (descartável, só no branch):

- `src/mjsoccer/mjlab_prototype.py`: o env. `Task` (torch) espelha `mjx/base.py` +
  `mjx/tasks.ContestedPossessionEnv` (controlador de navegação, kick/chip com trava, posse por
  sensor + grace, recompensa/fim do `_ShootTask`, overflow/NaN → terminal com recompensa 0). Os
  termos de manager só chamam o `Task`.
- `src/mjsoccer/scene.py`: `build` dividido em `build_spec` + compile (a cena entra inteira no
  `SceneCfg.spec_fn`). Os 253 testes rápidos passam (`pytest`, sem `-m validation`).
- `pyproject.toml`/`uv.lock`: `mujoco` 4.0.1, `mujoco-mjx[warp]` 4.0.1, grupo `mjlab` (mjlab git
  `6ab9846` + torch ≥ 2.14 do índice `pytorch-cu130`).
- Scripts e números brutos: `.scratch/mjlab-port/prototype/` — `dps_mjlab.py` → `results.txt`;
  `idle_world.py` + `overflow.py` → `idle_overflow.txt`; o bench MJX de referência re-rodado em
  4.0.1 → `results_mjx_401.txt`.

Como as decisões 10-12 viraram código:

- **Passo de disparo (10):** `decimation=48`, `MujocoCfg(timestep=1/960, cone="elliptic")`. O
  `Task.__init__` troca `sim.wp_model.opt.timestep` por `wp.array(nworld)` e chama
  `sim.create_graph()`. O `apply_actions` escreve in-place, a cada substep, `sched[fire, k]`
  (32 × dt/4 + 16 × dt se o mundo disparou; 24 × dt + 24 × 0 se não). O controlador roda uma vez
  por decisão (`process_actions`). Horizonte, recompensa e time-out são nossos
  (`scale_rewards_by_dt=False`, `episode_length_s` infinito, time-out por `Task.max_steps`).
- **Cena (11):** `attach_unprefixed(spec, scene.build_spec(1, 1))` no `spec_fn`, nomes iguais aos do
  CPU (`scene._resolve` roda sobre o modelo compilado pelo mjlab). O action term aponta para uma
  entidade-âncora vazia (`EntityCfg(spec_fn=mujoco.MjSpec)`). Contra o CPU: `nq` 21, `nv` 20 e `nu`
  12 iguais; `nbody` 13 contra 12 (o corpo da âncora).
- **Reset (12):** `Task.seat_pool()` faz, em lote, o `place` + `seat_ball(amarelo)` do MJX (336
  passos a dt com o dribbler ligado) e guarda `qpos`/`qvel`/`prev`/`prev_goal`/`grace` por mundo. O
  evento `reset` copia o slot do mundo de volta. Nos 64 mundos do smoke, todos terminaram com a bola
  no dribbler (`grace` = 2 no amarelo).

## Tabela: mjlab × MJX-Warp (ContestedPossession, ações aleatórias, kick disparando)

Decisões/s ponta a ponta, melhor de 3 × 10 s, como no `dps_mjx.py`. O MJX foi re-rodado neste
mesmo venv (mujoco 4.0.1). A coluna "MJX 4.0.0" vem do `10-bench/results.txt`.

| nworld | mjlab dec/s | só física (48 launches) | overhead | MJX 4.0.1 | MJX 4.0.0 | mjlab: memória do device | Warp mempool (pico) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 7 455 | 9 149 | 18,5 % | — | 7 599 | 666 MB | 40 MB |
| 1 024 | 13 302 | 15 116 | 12,0 % | 13 503 | 13 408 | 1 146 MB | 99 MB |
| 2 048 | 15 344 | 16 904 | 9,2 % | — | 15 371 | 1 788 MB | 179 MB |
| 4 096 | **16 588** | 17 910 | 7,4 % | **16 550** | 16 427 | 3 102 MB | 339 MB |
| 8 192 | **17 223** | 18 350 | 6,1 % | OOM (bench) | OOM (bench); 16,0–16,4 k no treino | 5 698 MB | 659 MB |
| 16 384 | 17 524 | 17 912 | 2,2 % | — | — | 10 916 MB | 1 298 MB |
| 32 768 | OOM (`Failed to allocate 2147483648 bytes`) | | | | | | |

- Fração de decisões com disparo: 17,4–17,6 % dos mundos por decisão (kick ou chip, ações
  uniformes, trava de 10 passos). Overflow: 0 mundos em todos os tamanhos (`nconmax` 64, `njmax`
  256 por mundo, o mesmo dimensionamento do MJX).
- O pool de primeiros estados custa 0,22 s a 256 mundos e 3,6 s a 8 192. O `reset` do MJX, que
  assenta mundo a mundo dentro do jit, leva 4,5 s a 4 096.
- O pico de memória do torch é desprezível (8–23 MB). Quase tudo é Warp.

## Respostas

| pergunta (decisão 13) | resposta |
|---|---|
| decisões/s contra ~16,4 k do MJX-Warp | **Empate.** 16 588 contra 16 550 a 4 096 mundos, na mesma versão. O mjlab sobe para 17,2 k a 8 192 e 17,5 k a 16 384, tamanhos que o bench do MJX não alcança. |
| memória GPU e `nworld` máximo estável | **16 384 roda** (10,9 GB usados de 12 GB), e 32 768 dá OOM. O MJX dá OOM a 8 192 no bench e treina a 8 192. "Estável" aqui significa ~40 s de passos com 0 overflow, **não** um treino longo. |
| mundo ocioso bit a bit sob o grafo capturado | **Sim.** Em 1 024 mundos × 40 decisões, 38 912 mundo-decisões ociosos tiveram `qpos`, `qvel`, `act`, `time` e `ctrl` idênticos (`==`) entre a iteração 24 e a 48. 2 048 mundo-decisões com disparo se moveram, ou seja, a agenda por mundo chega ao grafo. |
| `Data.overflow` limpo pelo `reset_data` do mjlab | **Sim.** `sim.reset(env_ids)` zera `overflow` só nos mundos mascarados e mantém os bits dos outros (`a_reset_cleared`, `a_others_kept`). |
| termination própria lendo `overflow` | **Funciona.** Com `nconmax=8`, apareceram os bits NEFC, BROADPHASE e ITERATIONS. Todo mundo com bit setado terminou (como `terminated`, não `time_out`), com recompensa 0, auto-reset e contador de episódio zerado, e o `forward()` pós-reset não religou os bits. Ressalva: nesse tamanho, **todos** os 512 mundos estouravam a cada passo. O teste não separa mundos bons de ruins num lote misto. |
| `uv lock` com torch + JAX num env só | **Resolve, sem `conflicts`.** Levou 7,8 s. Ficaram juntos torch 2.14.1+cu130, jax/jaxlib 0.12.0.1 (cuda12), mujoco/mujoco-core/mujoco-mjx/mujoco-warp 4.0.1 e warp-lang 1.18.0. `uv sync --group gpu --group mjlab` instala, e torch (CUDA) e JAX (`CudaDevice`) importam no mesmo processo. O bench MJX roda no venv novo. |

## Surpresas

1. **O overhead do manager é pequeno aqui**: 6–7 % a 4–8 k mundos, contra 62–76 % no nightly do
   mjlab. Os 48 substeps por decisão pesam muito mais que o Python por termo, então o passo é
   limitado pela física. A pesquisa (§7.2) temia o contrário. Ponto em aberto: a coluna "só física"
   roda com a última agenda escrita, não com a mistura de disparo de cada decisão, então o
   overhead é aproximado.
2. **Memória do device ≫ mempool do Warp.** A 16 384 mundos, 10,9 GB estão usados no device, mas o
   pico usado do mempool é só 1,3 GB. Não investiguei a diferença. Candidatos: memória reservada
   pelo mempool, os grafos capturados duas vezes (no `__init__` do mjlab e no nosso
   `create_graph()`), o contexto CUDA de torch + Warp. É isso que limita o `nworld` máximo, e vale
   medir antes do spec.
3. **Streams**: o mjlab escreve pelo `TorchArray` no stream do Warp, mas um tensor torch comum roda
   no stream corrente do torch, e o mjlab não sincroniza os dois. Por precaução, todo o protótipo
   roda dentro de `torch.cuda.stream(ExternalStream(wp.get_stream()))`
   (`mjlab_prototype.warp_stream()`). Não testei sem isso. Um port de verdade deve decidir.
4. **O `attach` avisa conflito de `<option>`** ("parent has 0.002 (default), child has 0.00104167,
   keeping parent value"). É inofensivo, porque o `MujocoCfg` sobrescreve depois, mas fica um aviso
   por build. O `zfar` do `<visual>` também fica o do mjlab (50 contra 60).
5. **Ordem de carga**: o `Task` precisa existir antes dos managers (o termo de obs é dimensionado no
   load, e o mjlab já chama reset no init). Resolvi com um hook em `load_managers`. O reset do
   init chega antes do pool e é ignorado.
6. **Frescor da observação ≠ MJX**: o mjlab chama `forward()` antes da obs, então a obs vê `xpos`
   fresco. O MJX observa o `Data` saído do último `mjx.step`, defasado de um substep nos mundos que
   dispararam (nos ociosos, as iterações a dt=0 recalculam a cinemática). É item de paridade, e
   aqui não foi medido.
7. **Efeito colateral do lock**: o lock universal tem um só torch, então o grupo `rl` (SB3) também
   passa para 2.14.1+cu130. Os testes rápidos do repo (253) passam no mujoco 4.0.1, o que não
   substitui a paridade CPU↔MJX que a decisão 8 pede ao subir o `main`.

## Não medido (fora do escopo das decisões 9 e 13)

Paridade CPU × mjlab, GoToBall (7 robôs, #1263 ITERATIONS), o caminho 24+24 com `step()` próprio,
treino RSL-RL × Brax, diversidade do pool contra `seat_ball` por reset.
