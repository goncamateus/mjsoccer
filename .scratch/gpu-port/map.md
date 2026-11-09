# Mapa: port GPU do mjsoccer

## Destino

Uma **especificação resolvida e ordenada** de três mudanças:
1. a remoção do tier 1 ([ADR 0001](../../docs/adr/0001-drop-the-roller-robot.md));
2. o robô dirigido pelo **controlador de navegação** do Planning-the-path-with-rl, no backend CPU e no MJX;
3. o **port GPU** das seis tasks como uma implementação MJX única, que roda no MJX-Warp sobre o
   modelo base ([ADR 0002](../../docs/adr/0002-mjx-warp-only.md)), com o backend CPU como oráculo e
   scripts de treino derivados do PPO em JAX.

O mapa está pronto quando não sobrar nada a decidir e `/to-spec` puder colapsá-lo sem reabrir
nenhuma pergunta.

## Notes

**Domínio.** Robô omni da Divisão A da RoboCup SSL em MuJoCo (`mjsoccer`), 6 tasks Gymnasium.
Vocabulário em [`CONTEXT.md`](../../CONTEXT.md) — use os termos de lá (backend, MJX-Warp, target pose, controlador de navegação, passo de controle, decisão, paridade).
Referências primárias:
- Planning-the-path-with-rl, commit `2287531`: `envs/navigation.py` (`go_to_point_new`) e
  `envs/enhanced.py` (ação, taxas). Clone de leitura em
  `/tmp/claude-1000/-home-mateus-doc-ssl-rl/f508caab-f682-434d-ae16-16ae10db361f/scratchpad/ptp-src/repo`
  (volátil — reclone de `github.com/goncamateus/Planning-the-path-with-rl` se sumir).
- MJX: `doc/mjx.rst` do `google-deepmind/mujoco` (tabela "Feature Parity", seções MJX-Warp e
  "Sharp Bits"). MuJoCo Warp: `doc/mjwarp/index.rst` e o README de `google-deepmind/mujoco_warp`.

**Skills que toda sessão deve consultar.** `mattpocock-skills:grilling` e
`mattpocock-skills:domain-modeling` para tickets `grilling`; `mattpocock-skills:research` para
tickets `research`. CodeGraph (`codegraph_*`) antes de ler código-fonte à mão.

**Restrições permanentes, resolvidas no mapeamento (2026-11-08).**
1. **Motivação**: velocidade de treino, PPO massivamente paralelo e domain randomization em escala.
2. **Backend CPU convive** com o MJX e é o oráculo da física; o port não o substitui.
3. **Tier 1 removido do repo** (ADR 0001). Um robô só, o chassi cinemático, nas 6 tasks. Isso
   derruba a restrição 4 do `physics-fidelity`. A remoção é mecânica e entra direto no spec.
4. **Hardware**: RTX 3060 12 GB local (driver 595). Cluster futuro não é requisito.
5. **Controle** (CPU e MJX): a ação é uma target pose mais atuadores,
   `(x, y, sin θ, cos θ, kick, chip, dribble)`, 7 dims em todas as tasks; `x, y` escalados por
   6.0 × 4.5 m; velocidade-alvo sempre 0. O `go_to_point_new` roda **fixo a 40 Hz** (24 passos de
   física; sai o `control_hz`). A política decide a cada `repeat_action` passos de controle (default 1).
   O controlador entra **verbatim sobre o envelope de hardware atual** (3.30 m/s, 1.30 m/s²,
   37.1 rad/s, 55 rad/s²): satura na aceleração, freia seguro (planeja com 1 m/s²).
6. **Uma implementação MJX em JAX sobre o MJX-Warp** (`impl="warp"`). O MJX-JAX saiu (ADR 0002, ticket da
   variante).
7. **Modelo base congelado** na descrição física (geometria, massas, inércias, contato, atuadores,
   `timestep`). A numérica pode diferir (float32, `iterations`/`tolerance`, `FIRING_SUBSTEPS`) se
   passar nos critérios de aceitação. *(A variante de modelo do MJX-JAX saiu com o ADR 0002: o
   MJX-Warp roda o base exato.)*
8. **Paridade** é estatística, pelos critérios de aceitação, numa checagem só: MJX-Warp × CPU no
   modelo base (diferença de motor; a variante saiu com o ADR 0002). Sem
   critério de transferência de política — depende da política, não só do simulador.
9. **Treino**: scripts novos, derivados do PPO, em JAX.
10. **DR**: o spec decide a capacidade (quais campos variam por mundo, e como), não quais nem ranges.
11. **Dependências**: grupo opcional `gpu` (`mujoco-mjx[warp]`, JAX CUDA, `brax` pinado no git
    `c860529` — ticket 12; o `mujoco-warp` separado é desnecessário, pesquisa 01); a base continua
    `gymnasium`, `mujoco`, `numpy`.
12. **Alvo MuJoCo 4.0** (decidido 2026-11-08): o port e o backend CPU sobem juntos de 3.11.0 para
    `mujoco 4.0.0` / `mujoco-mjx[warp]==4.0.0` (→ `warp-lang==1.18.0`; MuJoCo Warp embutido 3.17),
    porque oráculo e MJX precisam da mesma versão para a paridade valer. O spec inclui o upgrade do
    CPU com uma rodada dos testes de fidelidade antes do port (o 4.0 muda derivadas de atuador
    saturado em `implicitfast` e o `nativeccd`/`multiccd`). Avaliar a base depender só de `mujoco-core`.
    A colisão do MJX-JAX é igual à da 3.11. Confirmado no 4.0 (instalação): `jax 0.12.0.1` (`jax[cuda12]`) com o FFI do Warp.

**Fatos medidos (2026-11-08, remedidos no `mujoco 4.0.0` pela instalação; modelo cinemático).**

| cena | robôs | `nv` | DoF/robô | `ncon` médio | meshes que colidem | `nmesh` | mesh↔mesh do MJX: robô↔robô + robô↔campo |
|---|---|---|---|---|---|---|---|
| GoToBall, StaticDefenders | 7 | 55 | 7 | ~1 | 637 (≤32 vértices) | 1 088 | 105 861 + 13 916 |
| ContestedPossession (e as de 2 robôs) | 2 | 20 | 7 | ~1 | 202 | 423 | 5 041 + 3 976 |

- Meshes duplicados por robô (133 cada). Por robô colidem 87 meshes; no campo, 28 (paredes, gols).
- `<pair>` explícitos: 1 plane↔sphere (`condim` 6, bola × piso), 12 mesh↔sphere (`condim` 4, **bola ×
  parede** `col_f_walls_1_c*`; a bola × chassi vai por contype contra os 87 meshes), 2 cylinder↔sphere por robô (`condim` 4): `r_col_barDribbler` r 9 mm × 72 mm e
  `r_col_axisDribbler` r 5 mm × 94 mm (`models/robot_kinematic.xml:336-337`).
- `<option>`: Newton, 100 iterações, 50 de line search, cone elíptico, `implicitfast`, 1/960 s.
- A doc do MJX diz que o MJX-JAX **não colide sphere↔cylinder**, mas o código da 3.11 tem um SDF
  desde out/2024 (pesquisa 01). O MJX-JAX avalia a lista estática de pares sem broadphase real; os pares
  mesh↔mesh são o **bloqueio** dele: RAM do host no compile XLA (OOM a 8 GB com 2 robôs, mesmo sem
  robô↔robô — instalação), não só desempenho.
- MJX-Warp roda o modelo base nas duas cenas; `naconmax` se dimensiona pelos candidatos do broadphase
  (~120/mundo com 7 robôs), não pelo `max(ncon)`.
- MJX-Warp: sem autodiff; buffers fixos (`naconmax`, `njmax`) com overflow indefinido se não
  checado; contatos só no `_impl` privado (a doc manda usar sensor de contato); reset parcial com
  `jnp.where` "does not work out-of-the-box"; recaptura de grafo CUDA degrada o SPS pela ponte JAX.
- A posse lê `data.contact` (`src/mjsoccer/base.py:237`); o disparo do kick muda
  `model.opt.timestep` por env (`src/mjsoccer/base.py:137-147`).

**Mode.** Planejamento. Os tickets produzem decisões; as edições acontecem depois que o mapa
fechar. Os tickets `task` rodam num env isolado (fora do `uv` do projeto, **em disco, nunca em `/tmp`**, que é
tmpfs; todo job MJX/JAX com `MemoryMax`) e não mexem em `src/`,
`models/` nem `pyproject.toml`. Research vai em `.scratch/gpu-port/research/NN-<slug>.md`, sem
branch nem push (convenção do `physics-fidelity`).

## Decisions so far

<!-- uma linha por ticket fechado: [título](issues/NN-slug.md): gist -->

- [O MJX roda o nosso modelo? Versões e suporte, nos dois motores](issues/01-mjx-feature-support.md): sim, com `mujoco-mjx[warp]==3.11.0`; o bloqueio do MJX-JAX é chassi↔chassi (~310 MB/mundo com 7 robôs), não o cilindro; posse por sensor `contact` geom-geom; `naconmax` por `max(ncon)` medido.
- [Que implementações de PPO em JAX rodam sobre o MJX?](issues/02-jax-ppo-stacks.md): só o Brax PPO (via wrapper do Playground `rss2025`) roda documentado com Warp; precisa de reset por `Data.where`; recomendação é `MjxEnv` + `brax ppo.train`.
- [Como randomizar o modelo por mundo pela API MJX?](issues/03-mjx-domain-randomization.md): `vmap` com `in_axes` sobre o `mjx.Model` serve aos dois motores; contatos da bola pedem `pair_*`; derivados (`body_invweight0`) ficam velhos sem `mj_setConst` no host (0.28 ms/variante).
- [Instalar o MJX nesta máquina e rodar o modelo base e a variante](issues/04-mjx-install-smoke.md): env em `~/.venvs/mjsoccer-gpu` com `mujoco`/`mjx` 4.0.0, `warp-lang 1.18.0`, `jax 0.12.0.1`; o Warp roda o base nas duas cenas (`naconmax` ≥ 128/mundo); o MJX-JAX dá OOM no compile até sem robô↔robô, então a variante tira todo mesh de robô da colisão.
- [Benchmark bruto: MJX-JAX, MJX-Warp e CPU](issues/05-mjx-benchmark.md): pico Warp 918 k / 111 k SPS (2 / 7 robôs) contra 245 k / 80 k da CPU com 12 threads (3.7× / 1.4×); o MJX-JAX na variante fica 40–50× abaixo da CPU; no Warp o pool de contatos (~7 KB/slot, dimensionado pelo broadphase) limita o `nworld`.
- [O que a variante do MJX-JAX simplifica, e qual a meta de speedup](issues/06-variant-content-and-speedup.md): o MJX-JAX e a variante saem (ADR 0002), o port roda só no Warp; sem limiar de speedup; ContestedPossession e GoToBall portados primeiro, com decisões/s ponta a ponta medidas contra a base CPU (1.5 k / 0.3 k DPS).
- [O port do controlador de navegação: o que vai verbatim e o que se adapta](issues/07-navigation-controller-port.md): quirks verbatim (`MAX_DV` por passo, frenagem a 1 m/s², chatter, `ω` sem limite), só o `entry` vira puro; `max_velocity` 3.3; entrada pelo `qpos`/`qvel`; PassEndurance zera `vx, vy` depois do controlador; scripted e observação como hoje; `goto_ball_policy` em target pose, com um caminho de velocidade crua só para os testes de envelope.
- [Qual derivado do PPO, e como se estruturam os scripts de treino](issues/08-ppo-training-scripts.md): `brax ppo.train` vanilla no grupo `gpu` (a API MJX vira `brax.envs.Env`); um `examples/ppo_mjx.py --env`; `--num-envs` 8 192 / 1 024 (revisto no 13); `repeat_action` só no env; stdout + Orbax; eval no MJX, `--play` no backend CPU.
- [Que critérios de aceitação sobrevivem, e com que tolerâncias de paridade](issues/09-acceptance-and-parity.md): tilt/tombamento saem com o tier 1; ficam envelope (por velocidade crua), kick/chip, bola, posse, `spin`/`kick_recoil`/`random`/`competent`; o controlador ganha chegada (< 5 cm, < 2.5°, ≤ 1.25× trapézio), chatter e teto de `ω`; paridade = mesmos limiares + ≤ 2 % (≤ 5 % tempo) nos determinísticos e mediana MJX no [p25, p75] da CPU nos com seed; seeds 0-4 simétricos; throughput fora.
- [Uma implementação da lógica das tasks para os dois backends, ou duas?](issues/10-shared-task-logic.md): duas (numpy CPU intacto, JAX em `src/mjsoccer/mjx/`), seguradas por um teste de task-logic parity por snapshot (`put_data`, JAX-CPU, sem GPU); constantes e config lidas do CPU; posse pelo sensor `contact` nos dois; só `brax.envs.Env`; `mjsoccer.mjx.make` sobre o `ENV_IDS`.
- [Que capacidade de domain randomization o port oferece](issues/11-dr-capacity.md): só no MJX, uma variante por slot no início do run; `sampler(mj_model, rng)` do usuário com `mj_setConst` no host e `in_axes` por diff; `opt.timestep` recusado; ruído/latência na task, desligados; eval no nominal.
- [Rodar o `brax ppo.train` sobre o MJX-Warp nesta máquina](issues/12-brax-ppo-warp-smoke.md): roda com o Brax do `main` (`c860529`; o 0.14.2 do PyPI quebra no `jax 0.12`), `pmap` de 1 GPU ok, sem vendorizar; `WARP` dá 88–91 % da física bruta mas recaptura por época e estoura a GPU (8 192 e 4 096); `WARP_STAGED`/`NONE` estáveis a ~metade; DR por `in_axes` sem custo.
- [Modo de grafo do Warp e `--num-envs` default do treino](issues/13-graph-mode-and-num-envs.md): `WARP_STAGED` default no treino e na eval, kwarg `graph_mode`/`--graph-mode`; `--num-envs` 4 096 (2 robôs); 7 robôs pela regra "maior estável 30 épocas, ≥ 2 GB de folga", medido no ticket 14; `WARP` vira default só se o 14 o estabilizar.

## Not yet specified

- **Kick num lote.** Hoje o disparo roda o passo de controle a `timestep / FIRING_SUBSTEPS`
  mudando o `MjModel` compartilhado; num lote, cada mundo dispara numa hora. `opt.timestep` por mundo é representável no motor, mas `refsafe` limita `timeconst` ≥ 2×`timestep` — efeito numérico não testado; a DR não mexe no `timestep` (ticket 11). Substep o lote todo,
  `timestep` por mundo, ou outro mecanismo — depende do que o 01 e o 05 mostrarem.
- **Tabela de kick em float32**: se o MJX-Warp precisa de uma tabela de kick recalibrada própria (o gatilho agora é o [ticket de aceitação e paridade](issues/09-acceptance-and-parity.md): velocidade de saída fora de 2 % da CPU).
- **Reset em lote**: `sample_pose` é amostragem por rejeição com `raise`; `seat_ball` assenta a bola
  por um loop de passos; auto-reset por mundo.
- **Estado por mundo**: posse com graça (`POSSESSION_GRACE`), timers, contadores das tasks.
- **Falhas por mundo**: NaN (hoje `RuntimeError`), overflow do Warp — resetar o mundo, marcar, ou
  abortar.
- **Render / viewer / GIF** a partir do estado MJX (copiar um mundo para um `MjData`?). O `--play` do treino já roda no backend CPU (ticket 08); sobra só se alguém precisar ver um mundo do lote.
- **Testes do backend MJX-Warp**: ele precisa de GPU CUDA; onde moram as checagens do [ticket de aceitação e paridade](issues/09-acceptance-and-parity.md) (`validation_sweep.py --backend`? `-m validation`?); o que entra na suíte padrão (pulado sem
  GPU) e o que fica `-m validation`. O teste de task-logic parity ([ticket 10](issues/10-shared-task-logic.md)) roda em JAX-CPU; falta só decidir se entra na suíte padrão (pulado sem o grupo `gpu`).

## Fora de escopo

- Qualquer coisa do tier 1 além de removê-lo (ADR 0001).
- MuJoCo Warp nativo / torch: o port é uma implementação MJX em JAX.
- Simulação diferenciável (APG, SHAC) e identificação de sistema: contatos discretos, só no
  MJX-JAX, e o PPO não usa.
- Transferência de política como critério de correção.
- Contribuir upstream para o MJX ou o MuJoCo Warp.
- Cluster / multi-GPU.
- Tuning de hiperparâmetros e resultados de treino.
- Escolha dos parâmetros e ranges de DR.
- Portar os scripts SB3 / cleanRL; algoritmos fora da família PPO.
- Sim-to-real.
- MJX-JAX e a variante de modelo: 40–50× abaixo da CPU no 05, e o contato da barra pior ([ADR 0002](../../docs/adr/0002-mjx-warp-only.md), [ticket da variante](issues/06-variant-content-and-speedup.md)).
