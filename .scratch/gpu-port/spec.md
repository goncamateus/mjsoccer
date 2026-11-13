# Spec: port GPU do mjsoccer

Status: ready-for-agent
Origem: [mapa do port GPU](map.md) (fechado em 2026-11-13, 21 tickets resolvidos).
ADRs: [0001 — remover o tier 1](../../docs/adr/0001-drop-the-roller-robot.md),
[0002 — só MJX-Warp](../../docs/adr/0002-mjx-warp-only.md). Vocabulário: [`CONTEXT.md`](../../CONTEXT.md).

## Problem Statement

Treinar políticas nas seis tasks do mjsoccer é lento. O pipeline atual roda o MuJoCo na CPU via
`AsyncVectorEnv` e chega a ~1.5 k decisões/s com 2 robôs e ~0.3 k com 7. Com isso não dá para fazer
PPO massivamente paralelo nem domain randomization em escala numa RTX 3060 local. O repo também
mantém duas físicas de robô (o roller robot de 94 DoF e o chassi cinemático), e nenhum motor de GPU
porta bem a primeira. A ação da política hoje é velocidade crua, não a target pose do controlador
de navegação do Planning-the-path-with-rl, o que separa o mjsoccer do trabalho de origem.

## Solution

Três mudanças em ordem, depois de subir o MuJoCo para 4.0:

1. **Remover o tier 1.** Fica um robô só, o chassi cinemático, nas 6 tasks.
2. **Dirigir o robô pelo controlador de navegação**, nos dois backends. A ação passa a ser uma
   target pose mais kick/chip/dribble, com 7 dims em todas as tasks. O controlador roda fixo a 40 Hz.
3. **Fazer o GPU port**: uma segunda implementação das seis tasks em JAX funcional sobre o MJX-Warp,
   no modelo base exato, exposta como `brax.envs.Env`. O backend CPU fica como oráculo da física.
   A implementação JAX é segurada por um teste de task-logic parity e por uma checagem estatística
   de paridade. Junto vêm o script de treino `brax ppo.train` e a capacidade de domain randomization
   por mundo.

O usuário treina com `uv run --group gpu examples/ppo_mjx.py --env <id>` a dezenas de milhares de
decisões/s (medido: ~29 k com 2 robôs, ~4 k com 7). Ele assiste à política treinada com `--play` no
backend CPU, no viewer `mujoco.studio`.

## User Stories

### Pesquisador treinando políticas

1. Como pesquisador, quero treinar PPO nas seis tasks no MJX-Warp com um script só (`--env <id>`), para não manter um script por task.
2. Como pesquisador, quero que o `--num-envs` default seja o maior `nworld` estável na minha GPU (8 192 com 2 robôs, 2 048 com 7), para não ter de descobrir o tamanho do lote por tentativa e erro.
3. Como pesquisador, quero poder sobrescrever o `--num-envs`, para experimentar outros tamanhos de lote.
4. Como pesquisador, quero que os hiperparâmetros do PPO sejam defaults de flag, no estilo Playground, para ajustar sem editar código.
5. Como pesquisador, quero usar as variantes do PPO que o Brax já tem (crítico assimétrico, LR por KL, distribucional) como flags, sem código nosso.
6. Como pesquisador, quero o progresso no stdout (passos, decisões/s, retorno de eval, taxas de `diverged`/`overflow`/`reset_overlap`), para acompanhar o treino sem tensorboard.
7. Como pesquisador, quero checkpoints Orbax em `runs/ppo_mjx_<env>/`, para retomar e avaliar depois.
8. Como pesquisador, quero `--play` para carregar o checkpoint e rodar a política no backend CPU com o viewer, para ver o comportamento na física de referência.
9. Como pesquisador, quero `--seed`, `--save` e `--play` com a mesma semântica do `examples/ppo_agent.py`, para não reaprender a CLI.
10. Como pesquisador, quero `--repeat-action` como kwarg do env (default 1), para a decisão ter um dono só e o Brax rodar com `action_repeat=1`.
11. Como pesquisador, quero `--graph-mode` (default `WARP`, com `WARP_STAGED` como opção), para cair num modo mais conservador se a GPU apertar.
12. Como pesquisador, quero que o treino limpe o cache de grafos do Warp a cada época, para a memória da GPU ficar plana em runs longos.
13. Como pesquisador, quero que um mundo que diverge (NaN/inf) ou estoura buffer do Warp seja terminado e resetado sem derrubar o treino, para um mundo ruim não matar um run de horas.
14. Como pesquisador, quero que recompensa e observação de um mundo divergido sejam zeradas antes de chegar ao gradiente, para o NaN não contaminar a política.
15. Como pesquisador, quero ver a taxa de `overflow` no log, para saber quando o `naconmax`/`njmax` está mal dimensionado.
16. Como pesquisador, quero que o eval standalone aborte com mensagem clara (bits de overflow por tipo) se algum mundo divergir ou estourar, para não confiar em números de avaliação corrompidos.
17. Como pesquisador, quero que o auto-reset reamostre os estados iniciais a cada avaliação (`num_resets_per_eval`), para os mundos não ficarem presos sempre no mesmo start.
18. Como pesquisador, quero que o eval rode no modelo nominal por default e sob DR com `--eval-randomize`, para separar desempenho nominal de robustez.

### Pesquisador usando domain randomization

19. Como pesquisador, quero passar um `sampler(mj_model, rng)` para `mjsoccer.mjx.make(id, randomize=sampler)`, para escolher eu mesmo quais campos variam e com que ranges.
20. Como pesquisador, quero que cada mundo receba a sua variante uma vez no início do run, determinística pela seed de treino, para os runs serem reprodutíveis.
21. Como pesquisador, quero que o framework recalcule os derivados (`mj_setConst`) por variante, para massas e inércias alteradas não deixarem `body_invweight0` e afins velhos.
22. Como pesquisador, quero que o framework descubra sozinho quais campos mudaram (diff contra o nominal) e monte o `in_axes`, para não declarar campos à mão.
23. Como pesquisador, quero erro imediato se o meu sampler mudar `opt.timestep`, para não quebrar o disparo do kick sem perceber.
24. Como pesquisador, quero que o docstring diga que os contatos da bola são `<pair>` (mexer em `pair_*`, não em `geom_*`), para minha DR de atrito da bola ter efeito.
25. Como pesquisador, quero ruído de observação e latência de ação como opções do wrapper de treino, desligadas por default, para ligá-las sem misturá-las com a DR do modelo.
26. Como pesquisador, quero que sem sampler (`randomize=None`) o MJX rode o modelo nominal, para o default ser o mesmo modelo do CPU.

### Usuário das tasks (Gymnasium, CPU)

27. Como usuário das tasks, quero que as seis tasks continuem com os mesmos ids, via `gymnasium.make`, para meu código não quebrar.
28. Como usuário das tasks, quero que a ação seja `(x, y, sin θ, cos θ, kick, chip, dribble)` em todas as tasks, com `x, y` escalados por 6.0 × 4.5 m, para a política comandar onde o robô vai, não a que velocidade.
29. Como usuário das tasks, quero que o controlador de navegação rode fixo a 40 Hz e que `control_hz` saia, para o passo de controle ser sempre 25 ms.
30. Como usuário das tasks, quero que a política decida a cada `repeat_action` passos de controle (default 1), para controlar a frequência de decisão sem mexer na física.
31. Como usuário das tasks, quero que o PassEndurance ignore o `x, y` da ação e zere `vx, vy` depois do controlador, para a task continuar só de giro com a mesma ação de 7 dims.
32. Como usuário das tasks, quero que os robôs scripted continuem como hoje (amarelos com `ctrl = 0`, companheiro do PassEndurance só com o dribbler ligado), para as tasks não mudarem de dificuldade.
33. Como usuário das tasks, quero que a observação não mude, para políticas e wrappers existentes continuarem lendo os mesmos campos.
34. Como usuário das tasks, quero `render_mode="human"` no viewer novo do `mujoco.studio`, com a câmera de hoje no primeiro frame e depois livre, para inspecionar episódios.
35. Como usuário das tasks, quero escolher `viewer_gfx` (`opengl_headless` default, ou `web`), e ser avisado no docstring de que o `web` escuta em todas as interfaces sem autenticação.
36. Como usuário das tasks, quero que fechar a janela do viewer não derrube o episódio e que o próximo `render()` a reabra.
37. Como usuário das tasks, quero que `rgb_array` e os GIFs continuem no `mujoco.Renderer`/EGL, para gravar rápido.
38. Como usuário das tasks, quero que `import mjsoccer` nunca importe JAX nem `mjsoccer.mjx`, para a instalação base continuar leve (`gymnasium`, `mujoco`, `numpy`).
39. Como usuário das tasks, quero que o backend CPU continue a levantar `RuntimeError` em divergência, para o oráculo falhar alto.

### Pesquisador de física (oráculo e paridade)

40. Como pesquisador de física, quero um robô só (o chassi cinemático) e nenhuma menção a "tier", para não manter duas físicas.
41. Como pesquisador de física, quero que o CPU e o MJX rodem a mesma versão do MuJoCo (4.0.0), para a paridade comparar só a diferença de motor.
42. Como pesquisador de física, quero rodar de novo os testes de fidelidade no CPU 4.0 antes do port, para pegar mudanças do 4.0 (derivadas de atuador saturado no `implicitfast`, `nativeccd`/`multiccd`).
43. Como pesquisador de física, quero que o passo de disparo do kick use K = 32 (32 passos a `dt/4` mais 16 a `dt`) nos dois backends, para o mecanismo ser o mesmo e o lote não pagar 3.8×.
44. Como pesquisador de física, quero uma `KICK_TABLE_MS` só, indexada por plunger, recalibrada no CPU 4.0 com K = 32 a 40 Hz e lida pelo MJX, para não existir tabela float32 própria.
45. Como pesquisador de física, quero que a posse seja lida por um sensor `contact` geom-geom nos dois backends, para a mesma definição valer no CPU e no MJX.
46. Como pesquisador de física, quero que o envelope (3.30 m/s, 1.30 m/s², yaw < 37.1 rad/s, 55 rad/s²) continue testado por um caminho interno de velocidade crua, fora do `action_space`.
47. Como pesquisador de física, quero critérios de aceitação novos para o controlador (chegada < 5 cm e < 2.5°, tempo ≤ 1.25× o trapézio, chatter RMS < 0.05 m/s no último segundo, `ω` comandado ≤ 37.1), medidos primeiro na CPU, com limiar no valor medido mais margem.
48. Como pesquisador de física, quero que o MJX-Warp passe cada critério com os mesmos limiares da CPU, para o port não afrouxar a fidelidade.
49. Como pesquisador de física, quero paridade determinística (≤ 2 % em envelope, kick e COR; ≤ 5 % no tempo de chegada; ≤ 1 cm / 1° no erro de chegada), com o mesmo estado inicial escrito em `qpos`/`qvel` nos dois backends.
50. Como pesquisador de física, quero paridade com seed (mediana MJX dentro do [p25, p75] da CPU, seeds 0-4, amostras simétricas) para `random`, `competent`, taxa de posse e retorno.
51. Como pesquisador de física, quero rodar `scripts/validation_sweep.py --backend mjx`, com os cenários definidos uma vez e uma coluna `backend` nos CSVs, para comparar backends no mesmo formato.
52. Como pesquisador de física, quero que tilt, tombamento, `head_on`, `wall` e `dribble_turn` saiam junto com o tier 1, para os critérios refletirem o chassi planar.

### Mantenedor do código

53. Como mantenedor, quero que o lado MJX importe constantes e config do CPU (`base`, `scene`, `tasks`, atributos de classe), para nenhum número ser copiado.
54. Como mantenedor, quero um teste de task-logic parity por snapshot (snapshot `MjData` → `put_data` → mesma ação → mesmas obs, recompensa, término, posse, saída do controlador e estado por mundo), rodando em JAX-CPU sem GPU, para as duas implementações não divergirem em silêncio.
55. Como mantenedor, quero que mudar uma task obrigue a mexer nos dois lados, com o task-logic parity falhando se eu esquecer um.
56. Como mantenedor, quero um Warp smoke das 6 tasks (make, reset, ~10 passos, um kick) que pula sozinho sem `gpu`/CUDA, para `uv run pytest` rodar em qualquer máquina.
57. Como mantenedor, quero `mjsoccer.mjx.make(id, **kw)` lendo o mesmo `ENV_IDS`, com ids e horizontes iguais, para os dois backends terem um catálogo só.
58. Como mantenedor, quero o controlador de navegação como função pura, com os quirks do original verbatim, para portá-lo para JAX linha a linha.
59. Como mantenedor, quero o estado por mundo num pytree em `State.info` (grace, plunger lock, steps, chave RNG, estado da task), para o reset e o task-logic parity tratarem tudo junto.
60. Como mantenedor, quero os comandos de teste documentados (base, `gpu`, `-m validation`, sweep sob `MemoryMax`) nos docstrings e no `docs/index.md`, para rodar a camada certa sem adivinhar.
61. Como mantenedor, quero o grupo opcional `gpu` com `mujoco-mjx[warp]==4.0.0`, JAX CUDA e `brax` pinado no git `c860529`, para a GPU não entrar na base.
62. Como mantenedor, quero que o port comece por ContestedPossession e GoToBall, com decisões/s ponta a ponta registradas contra a base CPU, para pegar cedo uma regressão grosseira da lógica JAX.

## Implementation Decisions

A ordem das fases é parte da decisão. Cada fase deixa a suíte verde.

### Fase 0 — MuJoCo 4.0 no backend CPU

- A base sobe para `mujoco 4.0.0`. Avaliar se a base pode depender só de `mujoco-core`.
- O passo de disparo muda para K = 32 nesta fase: `FIRING_SUBSTEPS = 4` fica e entra
  `FIRING_FINE_STEPS = 32` em `scene`. O passo de disparo cai de 96 para 48 iterações e continua
  durando 25 ms.
- A `KICK_TABLE_MS` é recalibrada no CPU 4.0 float64, com K = 32 a 40 Hz, pelo `calibrate.py` do
  candidate-09 adaptado. A chave passa a ser só por plunger.
- Os testes de fidelidade do `physics-fidelity` rodam de novo, incluindo kick/chip, antes de qualquer
  código MJX.

### Fase 1 — Remover o tier 1 (ADR 0001)

- Saem o MJCF do roller robot e a cena dele, o `wheel_command`, os critérios de tilt e tombamento,
  `test_tipover_ends_episode`, `pretip.csv` e os cenários `head_on`, `wall` e `dribble_turn`.
- A palavra "tier" sai do código e da doc. A restrição 4 do `physics-fidelity` cai.
- A remoção é mecânica, sem decisão pendente.

### Fase 2 — Controlador de navegação no backend CPU

- **Ação**: `(x, y, sin θ, cos θ, kick, chip, dribble)`, 7 dims em todas as tasks. `x, y` são
  escalados por 6.0 × 4.5 m, e a velocidade-alvo é sempre 0.
- **Taxa**: o controlador roda fixo a 40 Hz (24 passos de física de 1/960 s) e o `control_hz` sai.
  `repeat_action` (default 1) define quantos passos de controle uma decisão dura.
- **Controlador**: `go_to_point_new` (Planning-the-path-with-rl, commit `2287531`) vira função pura.
  Os quirks ficam verbatim:
  - `MAX_DV = 0.1` por passo;
  - frenagem planejada com `MAX_ACCEL = 1`;
  - sem banda morta de posição;
  - zona morta de 2.5° no giro;
  - `ω = 2·Δθ` sem limite próprio;
  - `max_velocity` por eixo no referencial rotacionado.

  Só duas coisas mudam: sai a mutação de `entry`, e `max_velocity` passa a ser 3.3 m/s.
- **Entradas e saída**: as entradas vêm de `qpos`/`qvel` das juntas `robot_x/y/yaw`, não da
  observação. A saída, no referencial do corpo, passa pela rotação body→world que já existe até os
  servos de velocidade, com o clamp por eixo do `ctrlrange` mantido.
- **Escopo**: o controlador roda só sobre os `n_controlled`. Os robôs scripted ficam como hoje.
  No PassEndurance, `vx, vy` vão a 0 depois do controlador.
- **Velocidade crua**: o caminho de velocidade crua do `_drive_robot` atual fica como caminho
  interno, fora do `action_space`, só para os testes de envelope.
- **Policies scripted**: `goto_ball_policy` devolve uma target pose (alvo = bola, heading = ângulo
  robô→bola). Mudam junto os testes de env, `make_gifs` e `validation_sweep`.
- **Posse**: o CPU deixa de varrer `data.contact` e passa a ler um sensor `contact` geom-geom.
  O sensor não é descrição física, então o modelo base continua congelado.
- **Viewer**: `render_mode="human"` migra para `mujoco.studio` (`launch_native.launch` +
  `handle.sync`), sem fallback para `mujoco.viewer`.
  - Kwarg `viewer_gfx`: `opengl_headless` (default) ou `web`. O docstring avisa que o `web` escuta
    em todas as interfaces sem autenticação.
  - O `_frame_field` vira um `ViewerPlugin` de disparo único, com a pose de câmera de hoje.
  - O env ignora o `(model, data)` que o `sync` devolve. Janela fechada não derruba o episódio, e o
    próximo `render()` a reabre.
  - `rgb_array`, `make_gifs` e `visualize_sweep` ficam no `mujoco.Renderer`.
- **Critérios novos do controlador** (ver Testing Decisions): medir primeiro na CPU, depois fixar
  os limiares.

### Fase 3 — Port MJX-Warp (ADR 0002)

- **Layout**: subpacote `mjsoccer.mjx` com três módulos.
  - `base`: env base, controlador, posse, observação comum.
  - `tasks`: as 6 tasks.
  - `wrappers`: config de treino, ruído/latência, DR.

  `import mjsoccer` nunca importa `mjsoccer.mjx`.
- **Interface**: só `brax.envs.Env` (`reset(rng)`, `step(state, action)`, `State` com
  `obs/reward/done/metrics/info`). Não há classe Gymnasium vetorizada.
- **Entrada**: `mjsoccer.mjx.make(id, *, repeat_action=1, graph_mode="WARP", randomize=None, **kw)`
  sobre o mesmo `ENV_IDS`, sem `envs.register_environment` do Brax.
- **Motor**: `impl="warp"` no modelo base exato. A numérica pode diferir (float32,
  `iterations`/`tolerance`) se passar nos critérios. `naconmax` e `njmax` são dimensionados pelo env
  a partir dos candidatos do broadphase (~120/mundo com 7 robôs; ≥ 128 medido), não pelo
  `max(ncon)`, e não pelo script.
- **Fonte única de números**: constantes de módulo e config de task (`n_yellow`, `energy_steps`, ...)
  são importadas do lado CPU. Nada é copiado.
- **Estado por mundo**: pytree em `State.info`:
  - `grace (n_robots,)`;
  - `plunger_lock (n_robots, 2) int32`, decrementado por passo de controle, cooldown 10 lido do CPU;
  - `fired` (bool por robô);
  - `steps` em passos de controle;
  - a chave RNG do mundo;
  - o estado da task (`prev`, `prev_goal`, `count`, `last_ball`, `stopped`).

  O env é dono de `steps` e `truncation`. O `EpisodeWrapper` recebe
  `episode_length = max_steps / repeat_action` só para concordar.
- **Kick no lote**: `opt.timestep` por mundo via `in_axes`. Os mundos que disparam rodam 32
  iterações a `dt/4` e 16 a `dt`. Os ociosos rodam 24 a `dt` e ficam congelados com `timestep = 0`
  nas outras 24. Sem `Data.where` por passo. A tabela de kick vem do lado CPU.
- **Reset**:
  - Auto-reset do Brax/Playground (`wrap_for_brax_training` com `mjx.Data.where` no `where_done`),
    estado inicial em cache por mundo, reamostrado a cada eval por `num_resets_per_eval`. Sem
    `full_reset` e sem wrapper de reset nosso.
  - `seat_ball` roda o mesmo loop físico (0.35 s = 336 passos, dribbler girando) dentro do `reset`.
  - `sample_pose`: 200 candidatos vetorizados por objeto, sequencial entre objetos, primeiro válido
    por `argmax`, o que dá a mesma distribuição do CPU. Se nenhum candidato for válido, fica o de
    maior distância mínima aos já colocados e soma 1 em `reset_overlap`.
- **Falhas por mundo**:
  - Um mundo termina quando `qpos`/`qvel` deixa de ser finito ou quando algum bit de
    `_impl.overflow` está ligado. O término é terminal (não truncation), recompensa e obs são
    zeradas por `jnp.where`, e o auto-reset restaura o mundo. Não há abort dentro do jit.
  - `State.metrics` ganha três contadores 0/1 por passo: `diverged`, `overflow` e `reset_overlap`.
  - O eval standalone e os testes checam esses contadores no host e abortam se algum for > 0.
  - O CPU mantém o `RuntimeError`.
- **Domain randomization** (só no MJX):
  - O usuário passa `randomize=sampler` com `sampler(mj_model, rng) -> None`, que muta o modelo.
  - Para cada slot, o framework muta o `MjModel` em cache in place, roda `mj_setConst`, copia os
    campos variados e os derivados (`body_invweight0`, `dof_invweight0`, `actuator_acc0`, ...), e
    restaura o nominal (~0.28 ms/slot). Nunca copia o `MjModel`.
  - O diff contra o nominal vira o `in_axes`. O resultado é adaptado ao
    `randomization_fn(sys, rng) -> (sys, in_axes)` do `brax ppo.train`. O RNG é derivado da seed
    de treino, determinístico por slot.
  - Campos estruturais (meshes, `contype`, `*_limited`, inteiros do `opt`) ficam fora da capacidade.
    O framework recusa um sampler que mude `opt.timestep`.
  - Ruído de observação (via `info["rng"]`) e latência de ação (buffer de N passos de controle no
    `State`) ficam na task, desligados por default e fora do task-logic parity.
- **Ordem**: ContestedPossession e GoToBall primeiro (uma task por cena). As decisões/s ponta a
  ponta são registradas contra a base CPU remedida no 4.0 (era 1 535 / 299 DPS no 3.11). É
  relatório, não portão. Depois vêm as outras quatro.

### Fase 4 — Treino

- O script `examples/ppo_mjx.py --env <id>` espelha o `examples/ppo_agent.py` (`--seed`, `--save`,
  `--play`). A saída vai para `runs/ppo_mjx_<env>/`. Roda com `uv run --group gpu`.
- Usa o `brax ppo.train` vanilla, sem vendorizar. O `pmap` com 1 GPU funciona. Os hiperparâmetros
  são defaults de flag.
- `--num-envs`: 8 192 com 2 robôs e 2 048 com 7 robôs. A regra é o maior estável por 30 épocas com
  ≥ 2 GB de folga, revalidada na task real (a folga foi medida num env de brinquedo).
- `--graph-mode`: default `WARP`, com `WARP_STAGED` como flag. `WARP_STAGED_EX` não entra. No modo
  `WARP`, o fim de cada época (`progress_fn`) roda `wp.clear_jax_callable_graph_cache()`,
  `gc.collect()` e `wp.synchronize()`, no treino e na eval.
- `--repeat-action` é repassado ao env, e o Brax roda com `action_repeat=1`.
- Logging: `progress_fn` no stdout (passos, decisões/s, retorno de eval, taxas dos três
  contadores). Checkpoint Orbax. Sem tensorboard/wandb.
- Eval: o eval embutido do Brax roda no MJX, no modelo nominal (`--eval-randomize` avalia sob DR).
  O `--play` faz inferência JAX na CPU e roda no backend CPU com o viewer Studio.

### Dependências

- A base continua `gymnasium`, `mujoco`, `numpy`.
- O grupo opcional `gpu` traz `mujoco-mjx[warp]==4.0.0` (→ `warp-lang 1.18.0`), `jax[cuda12]`
  (0.12.0.1 testado) e
  `brax @ git+https://github.com/google/brax@c8605294264a7b7546268adbb4beabc9eb624000`, até sair uma
  release do Brax > 0.14.2. O `mujoco-warp` separado não entra.

## Testing Decisions

**Um bom teste** verifica comportamento externo: o que o env devolve (obs, recompensa, término,
métricas) e o que a física faz (velocidades, posições, contatos) a partir de uma ação. Não verifica
detalhes internos. A paridade da física é estatística, pelos critérios de aceitação, nunca
trajetória por trajetória. Só a task-logic parity é exata, estado por estado.

**Seams** (decididas no [ticket dos testes](issues/19-mjx-test-placement.md)). Há três; não surgiu
nenhuma outra.

1. **API pública do env**: `gymnasium.make` no CPU e `mjsoccer.mjx.make` no MJX. Os testes de env
   existentes continuam nesta seam e só trocam a ação para target pose.
2. **Sweep de validação**: `episode(..., backend=)` do `validation_sweep`, uma seam já existente que
   ganha o parâmetro `backend`. Toda checagem de critério e de paridade passa por ela.
3. **Snapshot `MjData` → `put_data`** para o task-logic parity. É a única seam nova, necessária
   porque as duas implementações têm de ser comparadas no mesmo estado físico.

**Camadas**:

- **Suíte padrão (`uv run pytest`)**. O MJX é pulado sem o grupo `gpu`.
  - Testes de env existentes, adaptados à ação de 7 dims, ao controlador, à posse por sensor e ao
    viewer Studio.
  - **Task-logic parity**: `importorskip` de `jax`/`mujoco.mjx`, com `JAX_PLATFORMS=cpu` forçado no
    topo do módulo.
    - Os snapshots vêm de rollouts CPU com seeds 0-4. Com a mesma ação, compara obs, recompensa,
      término, posse, saída do controlador e o pytree de estado por mundo contra os atributos do
      CPU, na tolerância de float32.
    - Cobre `_place` com candidatos injetados e `plunger_lock`/`fired`/cooldown 10.
    - Cobre as 6 tasks. Aborta se algum contador de falha for > 0.
  - **Warp smoke**: `make` + `reset` + ~10 passos nas 6 tasks, com `nworld` pequeno e um kick
    disparado. Exige `diverged == overflow == 0` e obs finita. O mundo ocioso tem de manter
    `qpos`/`qvel` bit a bit nas iterações a `timestep = 0`. É pulado sem CUDA.
- **`-m validation`** (`tests/test_validation.py`). Não há marker `gpu`: os casos MJX pulam sem CUDA.
  - Os testes de critério ganham o parâmetro `backend`:
    - envelope por velocidade crua;
    - kick/chip × `KICK_TABLE_MS`;
    - COR e inércia da bola;
    - posse (inclusive no giro de 2 rad/s);
    - `spin` e `kick_recoil`;
    - os critérios do controlador (alvos a 0.5 / 2 / 6 m, giros de 0 / 90 / 180°).
  - `test_parity_deterministic`: ≤ 2 % em envelope, kick e COR; ≤ 5 % no tempo de chegada;
    ≤ 1 cm / 1° no erro de chegada.
  - `test_parity_seeded`: mediana MJX dentro do [p25, p75] da CPU, `--seeds 0-4`, um episódio/mundo
    por seed por env. Sem testes de distribuição.
  - As tolerâncias podem ser revistas depois da primeira medição.
- **Fora dos testes**: o throughput do MJX é relatório (decisões/s), nunca critério. O "dentro de
  10 % do arquivo" do `docs/index.md` continua valendo só para o CPU.

**Prior art**:

- os testes de env e de validação existentes;
- os testes de kick/chip e de posse do `physics-fidelity`;
- os scripts de medição em `.scratch/gpu-port/research/` (`12-brax/`, `14-warp/`, `15-kick/`): o env
  de brinquedo do Brax sobre o Warp, a limpeza de grafo por época e o kick com `timestep` por mundo;
- o `calibrate.py` do candidate-09.

**Execução**:

- Não há CI; tudo roda local.
- Os comandos ficam documentados nos docstrings e no `docs/index.md`. Os jobs MJX e o sweep rodam
  sob `systemd-run --user --scope -p MemoryMax=...`; o `MemoryMax` é comando documentado, não
  código no teste.

## Out of Scope

- Qualquer coisa do tier 1 além de removê-lo.
- MJX-JAX e a variante de modelo, que rodam 40–50× abaixo da CPU (ADR 0002).
- MuJoCo Warp nativo ou torch: o port é uma implementação MJX em JAX.
- Simulação diferenciável (APG, SHAC) e identificação de sistema.
- Transferência de política como critério de correção.
- Contribuições upstream para o MJX ou o MuJoCo Warp.
- Cluster e multi-GPU.
- Tuning de hiperparâmetros e resultados de treino.
- Escolha dos parâmetros e ranges de DR. Reamostragem de DR por episódio. DR no backend CPU.
- Portar os scripts SB3/cleanRL. Algoritmos fora da família PPO.
- Sim-to-real.
- CI.
- Espectador de um mundo do lote MJX-Warp no Studio. Uma classe Gymnasium vetorizada sobre o MJX.
- Pool de M estados pré-amostrados no reset (fica como upgrade se a diversidade de starts medir mal).

## Further Notes

- **Ressalvas medidas**:
  - O `WARP` recaptura ~3 grafos por época e o SPS de treino cai 3–6 % em 30 épocas. O dev aceitou
    isso.
  - A folga de 2.4 GB a 8 192 mundos foi medida num env de brinquedo. Revalidar na task real pela
    mesma regra, e baixar o default se ela não fechar.
  - O treino com 7 robôs satura em ~4.25 k decisões/s já a 512 mundos.
- **Fonte das medições**: `.scratch/gpu-port/research/` (benchmark, smoke do Brax, estabilidade do
  `WARP`, kick no Warp, Studio). O env isolado de medição está em `~/.venvs/mjsoccer-gpu`.
- **Inconsistência a resolver na implementação**: o ticket dos testes fala em
  `uv sync --extra gpu`, mas o `pyproject.toml` usa `[dependency-groups]` e o ticket do treino usa
  `uv run --group gpu`. Seguir o grupo (`--group gpu`) e documentar só esse comando.
- **Detalhe de cada decisão**: está no ticket correspondente do [mapa](map.md). Este spec só os
  consolida.
