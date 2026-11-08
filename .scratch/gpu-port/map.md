# Mapa: port GPU do mjsoccer

## Destino

Uma **especificação resolvida e ordenada** de três mudanças:
1. a remoção do tier 1 ([ADR 0001](../../docs/adr/0001-drop-the-roller-robot.md));
2. o robô dirigido pelo **controlador de navegação** do Planning-the-path-with-rl, no backend CPU e no MJX;
3. o **port GPU** das seis tasks como uma implementação MJX única, que roda no MJX-JAX (com a sua
   **variante de modelo**) e no MJX-Warp, com o backend CPU como oráculo e scripts de treino
   derivados do PPO em JAX.

O mapa está pronto quando não sobrar nada a decidir e `/to-spec` puder colapsá-lo sem reabrir
nenhuma pergunta.

## Notes

**Domínio.** Robô omni da Divisão A da RoboCup SSL em MuJoCo (`mjsoccer`), 6 tasks Gymnasium.
Vocabulário em [`CONTEXT.md`](../../CONTEXT.md) — use os termos de lá (backend, MJX-JAX, MJX-Warp,
variante de modelo, target pose, controlador de navegação, passo de controle, decisão, paridade).
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
6. **Uma implementação MJX em JAX**, trocando de motor por `impl="jax"` / `impl="warp"`.
7. **Modelo base congelado** na descrição física (geometria, massas, inércias, contato, atuadores,
   `timestep`). A numérica pode diferir (float32, `iterations`/`tolerance`, `FIRING_SUBSTEPS`) se
   passar nos critérios de aceitação. O MJX-JAX roda uma **variante de modelo** gerada por transform
   declarado no `scene.build` (não um segundo MJCF à mão); mínimo: cilindros do dribbler → cápsulas
   de mesmo raio e comprimento total. A tabela de kick é recalibrada para a variante.
8. **Paridade** é estatística, pelos critérios de aceitação, em duas checagens: variante × base no
   CPU (custo de fidelidade da variante) e MJX × CPU no mesmo modelo (diferença de motor). Sem
   critério de transferência de política — depende da política, não só do simulador.
9. **Treino**: scripts novos, derivados do PPO, em JAX.
10. **DR**: o spec decide a capacidade (quais campos variam por mundo, e como), não quais nem ranges.
11. **Dependências**: grupo opcional `gpu` (`mujoco-mjx`, JAX CUDA, `mujoco-warp`); a base continua
    `gymnasium`, `mujoco`, `numpy`.

**Fatos medidos no mapeamento (2026-11-08, `mujoco 3.11.0`, modelo cinemático).**

| cena | robôs | `nv` | DoF/robô | `ncon` médio | meshes que colidem | pares mesh↔mesh candidatos |
|---|---|---|---|---|---|---|
| GoToBall, StaticDefenders | 7 | 55 | 7 | ~1 | 637 (≤32 vértices) | 105 861 |
| ContestedPossession (e as de 2 robôs) | 2 | 20 | 7 | ~1 | 202 | 5 041 |

- `<pair>` explícitos: 1 plane↔sphere (`condim` 6, bola × piso), 12 mesh↔sphere (`condim` 4, chassi ×
  bola), 2 cylinder↔sphere por robô (`condim` 4): `r_col_barDribbler` r 9 mm × 72 mm e
  `r_col_axisDribbler` r 5 mm × 94 mm (`models/robot_kinematic.xml:336-337`).
- `<option>`: Newton, 100 iterações, 50 de line search, cone elíptico, `implicitfast`, 1/960 s.
- MJX-JAX **não colide sphere↔cylinder** (`mjx.rst`, nota [3] da Feature Parity); MJX-Warp: "All".
  O MJX-JAX avalia a lista estática de pares sem broadphase real — os 105 mil pares mesh↔mesh são o
  risco de desempenho dele.
- MJX-Warp: sem autodiff; buffers fixos (`naconmax`, `njmax`) com overflow indefinido se não
  checado; contatos só no `_impl` privado (a doc manda usar sensor de contato); reset parcial com
  `jnp.where` "does not work out-of-the-box"; recaptura de grafo CUDA degrada o SPS pela ponte JAX.
- A posse lê `data.contact` (`src/mjsoccer/base.py:237`); o disparo do kick muda
  `model.opt.timestep` por env (`src/mjsoccer/base.py:137-147`).

**Mode.** Planejamento. Os tickets produzem decisões; as edições acontecem depois que o mapa
fechar. Os tickets `task` rodam num env isolado (fora do `uv` do projeto) e não mexem em `src/`,
`models/` nem `pyproject.toml`. Research vai em `.scratch/gpu-port/research/NN-<slug>.md`, sem
branch nem push (convenção do `physics-fidelity`).

## Decisions so far

<!-- uma linha por ticket fechado: [título](issues/NN-slug.md): gist -->

## Not yet specified

- **Kick num lote.** Hoje o disparo roda o passo de controle a `timestep / FIRING_SUBSTEPS`
  mudando o `MjModel` compartilhado; num lote, cada mundo dispara numa hora. Substep o lote todo,
  `timestep` por mundo, ou outro mecanismo — depende do que o 01 e o 05 mostrarem.
- **Tabela de kick em float32 e na variante**: se a recalibração por motor é necessária ou só por
  modelo.
- **Reset em lote**: `sample_pose` é amostragem por rejeição com `raise`; `seat_ball` assenta a bola
  por um loop de passos; auto-reset por mundo.
- **Estado por mundo**: posse com graça (`POSSESSION_GRACE`), timers, contadores das tasks.
- **Falhas por mundo**: NaN (hoje `RuntimeError`), overflow do Warp — resetar o mundo, marcar, ou
  abortar.
- **Render / viewer / GIF** a partir do estado MJX (copiar um mundo para um `MjData`?).
- **Testes dos backends MJX**: o MJX-JAX roda em CPU, então dá para testar sem GPU; o que entra na
  suíte padrão e o que fica `-m validation`.
- **Versões pinadas** de `mujoco`, `mujoco-mjx`, `mujoco-warp` e JAX.

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
