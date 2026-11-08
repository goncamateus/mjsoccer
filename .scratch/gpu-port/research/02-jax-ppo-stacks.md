# 02 — Que implementações de PPO em JAX rodam sobre o MJX?

Ticket: `.scratch/gpu-port/issues/02-jax-ppo-stacks.md`. Pesquisado em 2026-11-08, só em fonte
primária (código-fonte, docs oficiais e paper dos autores). Clones rasos de leitura em
`/tmp/claude-1000/-home-mateus-doc-ssl-rl/716cbd60-1120-41e0-842f-d53c5add287a/scratchpad/`.
Nada foi instalado nem executado. **Nenhum número de throughput abaixo foi medido aqui**: todos são
reportados pelos autores, em A100/RTX 4090/H100, e nenhum em RTX 3060.

Commits lidos (os `arquivo:linha` abaixo referem-se a eles):

| repo | commit | data |
|---|---|---|
| [google/brax](https://github.com/google/brax/tree/c8605294264a7b7546268adbb4beabc9eb624000) | `c860529` | 2026-10-20 |
| [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground/tree/e7c53bc214727333b17de3ca9890ce438ba6a2ca) | `e7c53bc` | 2026-11-08 |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco/tree/371c2b444582552c977bc696d135d552ddcda613) (`doc/mjx.rst`, `mjx/`) | `371c2b4` | 2026-11-07 |
| [luchris429/purejaxrl](https://github.com/luchris429/purejaxrl/tree/31756b197773a52db763fdbe6d635e4b46522a73) | `31756b1` | 2024-09-09 |
| [EdanToledo/Stoix](https://github.com/EdanToledo/Stoix/tree/8fff19c94a478c1e1ff75fbf10396d488aa25d70) | `8fff19c` | 2026-03-18 |
| [EdanToledo/Stoa](https://github.com/EdanToledo/Stoa/tree/b3a084ed05893c301e079a8f527de90513896d7e) (wrappers do Stoix) | `b3a084e` | 2025-10-06 |
| [keraJLi/rejax](https://github.com/keraJLi/rejax/tree/d7510951b7a89ca8463cb2a6d18e21750ec454ff) | `d751095` | 2026-06-10 |
| [goncamateus/Planning-the-path-with-rl](https://github.com/goncamateus/Planning-the-path-with-rl/tree/2287531e87b31dacbc60f482321d3fc3640d4246) | `2287531` | — |

Abreviações: `BRAX/` = `google/brax@c860529/brax/`; `PG/` = `mujoco_playground@e7c53bc/`;
`MJ/` = `mujoco@371c2b4/`.

## Resposta curta

- **Só uma pilha PPO em JAX está documentada pelos autores como rodando com `impl="warp"`: o Brax
  PPO dentro do wrapper do MuJoCo Playground (`rss2025/`).** O README manda rodar
  `train-jax-ppo --env_name CartpoleBalance --impl warp` (`PG/rss2025/README.md`, seção "Running from
  CLI"), e desde o Playground 0.2.0 (2026-03-16) o default de todos os envs passou a ser MuJoCo Warp
  (`PG/rss2025/CHANGELOG.md:7-11`; ex.: `PG/rss2025/mujoco_playground/_src/dm_control_suite/cartpole.py:53-55`,
  `impl="warp", naconmax=0, njmax=2`).
- **Atenção à mudança de rumo do Playground**: na `main` atual a raiz do repo foi reescrita em
  **PyTorch** sobre `reaf-sim` + MuJoCo Warp, com RSL-RL como treinador (`PG/README.md:9`,
  `PG/AGENTS.md`, seção "Architecture"). A versão JAX/MJX + Brax PPO ficou congelada no
  subdiretório `rss2025/` ("Looking for the original RSS 2025 JAX/MJX environments? They remain
  available under rss2025", `PG/README.md`, nota após a tabela de demos). Logo, o "treinamento do
  Playground" em JAX **é** o Brax PPO; não existe um PPO JAX próprio do Playground.
- **Brax PPO puro** (sem o wrapper do Playground) roda sobre qualquer env no protocolo
  `brax.envs.Env`; o auto-reset dele faz `jax.tree.map(jnp.where)` cego sobre o estado, o que a doc do
  MJX diz que **não funciona out-of-the-box com MJX-Warp** (`MJ/doc/mjx.rst:135-146`). Com MJX-JAX
  funciona.
- **PureJaxRL e rejax** não têm adaptador MJX; usam envs do `brax.envs` (backend `positional` no
  PureJaxRL), que o próprio Brax declarou não mantidos (`BRAX/../README.md:4`). Adaptar = escrever
  um wrapper Gymnax sobre o env MJX. Ambos ignoram truncamento no GAE.
- **Stoix** tem adaptador para envs do Playground (via Stoa) e seis variantes de PPO, mas fixa
  `jax<0.6.0` (`Stoix/pyproject.toml:145`) e o auto-reset do Stoa é `lax.cond` por env sob `vmap`;
  **compatibilidade com `impl="warp"` não verificada** (ver §4).
- **Base mais simples de adaptar para as 6 tasks**: o Brax PPO (`brax.training.agents.ppo.train`)
  alimentado por envs no padrão `MjxEnv` do Playground e embrulhados por um `wrap_for_brax_training`
  próprio (cópia das ~45 linhas do auto-reset do Playground, trocando o `where_done` por
  `mjx.Data.where`). É a única combinação que já traz tudo que o ticket pede (normalização,
  crítico assimétrico, tanh-normal em [-1, 1], action repeat, checkpoint, logging por callback) e
  que tem um caminho de `impl="warp"` exercitado pelos autores. Ver §6.

## 1. Brax PPO (`brax/training/agents/ppo`)

### Backend e auto-reset

- O `train()` não conhece o motor: recebe um `envs.Env` e o embrulha com `wrap_env_fn` ou, por
  default, `envs.training.wrap` (`BRAX/training/agents/ppo/train.py:80-113`). Não há menção a
  `impl` nem a Warp no `train.py` (busca por `warp`/`impl` sem resultados no arquivo).
- Auto-reset default (`BRAX/envs/wrappers/training.py:129-158`): no `reset` guarda
  `first_pipeline_state`/`first_obs`; no `step`, depois do passo, troca as folhas por elas onde
  `done`, com `jp.where` folha a folha. O guarda `if done.shape[0] != x.shape[0]: return y`
  (linhas 146-152) deixa passar folhas sem dimensão de lote. **Reset sempre para o mesmo estado
  inicial em cache** por env (não re-amostra a pose). Re-amostrar exige `num_resets_per_eval > 0`,
  que faz um reset completo no host entre avaliações (`train.py:278-279`, `816-830`).
- Estado por mundo: o `VmapWrapper` faz `jax.vmap(env.reset/step)` (`training.py:59-73`); DR por
  mundo via `DomainRandomizationVmapWrapper`, que vmapeia sobre um `System` em lote
  (`training.py:223-260`; o lote é criado em `train.py:95-103` com `num_envs // device_count`
  chaves).
- `training_epoch` é sempre `jax.pmap` (`train.py:669-676`), mesmo com uma GPU. O anúncio do
  MJX-Warp no Playground diz que "pmap support is not yet available with MJX-Warp"
  ([discussion #197](https://github.com/google-deepmind/mujoco_playground/discussions/197)); como o
  Playground roda `ppo.train` com `--impl warp` numa GPU, a leitura é que *multi-device* não
  funciona, uma GPU sim. **Inferência, não verificada**; a data do post não foi conferida.

### Throughput reportado

- O `train()` calcula e loga `training/sps` por época (`train.py:693-700`), mas o repo do Brax não
  publica uma tabela de SPS para o PPO. Os números vêm do paper do Playground (§2).

### Recursos

| recurso | onde |
|---|---|
| normalização de observação | `normalize_observations` (`train.py:201`), modos `welford`/`ema`, `normalize_until_count` (`train.py:202-204`), via `acme.running_statistics` |
| crítico assimétrico | `make_ppo_networks(policy_obs_key='state', value_obs_key='state')` (`BRAX/training/agents/ppo/networks.py:91-100`): com obs em dict, política e crítico leem chaves diferentes |
| ações em [-1, 1] | `distribution_type='tanh_normal'` é o default (`networks.py:100`, `120-130`) → `NormalTanhDistribution` (`BRAX/training/distribution.py:142-160`) |
| action repeat | `action_repeat` (`train.py:187`) → `EpisodeWrapper` faz `lax.scan` de `env.step` com a mesma ação e soma a recompensa (`training.py:98-104`) |
| truncamento | `EpisodeWrapper` marca `info['truncation']` (`training.py:109-112`), e o GAE separa término de truncamento (`losses.py:212-222`); `bootstrap_on_timeout` opcional (`train.py:212`, `294-298`) |
| checkpoint | `save_checkpoint_path`/`restore_checkpoint_path` (`train.py:237-239`), Orbax `PyTreeCheckpointer` (`BRAX/training/checkpoint.py:146-148`), salva config da rede junto (`train.py:844-853`) |
| logging | só callbacks: `progress_fn(step, metrics)` e `policy_params_fn` (`train.py:234-235`); quem loga em wandb/TB é o script chamador |
| avaliação | `num_evals`, `num_eval_envs`, `deterministic_eval` (`train.py:226-230`) |

### Derivados de PPO

Num único `train()`, por flags (`BRAX/training/agents/ppo/losses.py:144-300`):
clipping da razão (`clipping_epsilon`, linha 242), **clipping do valor** opcional
(`clipping_epsilon_value`, linhas 260-268), normalização de vantagem on/off (`normalize_advantage`,
linhas 232-233), **crítico distribucional** com quantile Huber (`use_distributional_critic`,
linhas 103-140 e 246-255), **LR adaptativo por KL** no estilo RSL-RL (`LRSchedule.ADAPTIVE_KL`,
`BRAX/training/agents/ppo/optimizer.py:24-59`), PPO de visão (`networks_vision.py`, `vision=True`,
com aumento de pixels `augment_pixels`). Sem PPO-penalty nem PPO recorrente. Outros agentes do
mesmo pacote: SAC, APG, ARS, ES, BC (`BRAX/training/agents/`).

## 2. MuJoCo Playground (`rss2025/`): Brax PPO + wrapper próprio

### Backend e auto-reset

- `impl` vira flag (`--impl {jax,warp}`, `PG/rss2025/learning/train_jax_ppo.py:79`) e é repassada como
  override do config do env (`train_jax_ppo.py:285`). O env cria o modelo com
  `mjx_env.put_model(..., impl=...)` e o `Data` com `mjx_env.make_data(..., impl, naconmax, njmax)`
  (`PG/rss2025/mujoco_playground/_src/mjx_env.py:130-172`). Para Warp, o `put_model` desliga os avisos
  de overflow de iterações do solver e do line search (linhas 138-152). Flag extra
  `--warp_kernel_cache_dir` (`train_jax_ppo.py:178-181`, `224-227`).
- O treino é **exatamente o `brax.training.agents.ppo.train`** com
  `wrap_env_fn=wrapper.wrap_for_brax_training` (`train_jax_ppo.py:385-398`).
- `wrap_for_brax_training` (`PG/rss2025/mujoco_playground/_src/wrapper.py:87-132`) = `VmapWrapper` (ou
  `BraxDomainRandomizationVmapWrapper`) → `EpisodeWrapper` do Brax → `BraxAutoResetWrapper` próprio.
- `BraxAutoResetWrapper` (`wrapper.py:161-248`):
  - default: reset para um estado inicial em cache por env, como o Brax;
  - `full_reset=True`: chama `env.reset` dentro de cada `step` e escolhe com `where` (custa tempo
    de parede, linhas 108-110, 203-206). É o único caminho, entre as pilhas lidas, que **re-amostra
    a pose inicial a cada episódio dentro do JIT**; preserva `info['AutoResetWrapper_preserve_info']`
    para curriculum (linhas 172-174, 241-243);
  - guarda uma chave RNG por env (`AutoResetWrapper_rng`) e um contador `done_count`.
- O `where_done` (`wrapper.py:222-228`) é o mesmo do Brax: `jp.where` por folha, pulando folhas
  cuja dimensão 0 ≠ `num_envs`. O `mjx.Data.where` oficial, que a doc manda usar com Warp
  (`MJ/doc/mjx.rst:135-146`), consulta a tabela `_BATCH_DIM` dos campos do Warp e mantém os campos
  não-loteados (`MJ/mjx/mujoco/mjx/_src/types.py:1188-1218`). **Inferência**: o guarda por formato
  do Playground imita isso por coincidência de shape; um campo não-loteado com dimensão 0 igual a
  `num_envs` seria trocado indevidamente. Não verificado se existe tal campo.
- DR por mundo: `randomization_fn(mjx_model) -> (model_v, in_axes)`, depois
  `vmap(reset/step, in_axes=[in_axes, 0])` trocando `env._mjx_model` num context manager
  (`wrapper.py:251-280`).
- `naconmax` é o total de contatos **somando todos os mundos**; `njmax` é por mundo
  (`MJ/doc/mjx.rst:117-123`). Ex.: Go1 getup usa `naconmax=30 * 8192`
  (`PG/rss2025/mujoco_playground/_src/locomotion/go1/getup.py:66`).

### Throughput reportado (paper do Playground, arXiv [2502.08844](https://arxiv.org/html/2502.08844), consultado em 2026-11-08, versão HTML corrente)

Números do Brax PPO em **MJX-JAX** (o paper é anterior ao MJX-Warp), passos de env/s, A100:

| task | SPS (A100) | tempo de treino citado |
|---|---|---|
| CartpoleBalance (estado) | 718 626 ± 6 894 (Tab. IV) | "most environments training in under 10 minutes on a single GPU" (§IV-A) |
| Go1 Joystick plano / rugoso | 417 451 / 291 060 (Tab. VII) | plano < 5 min em **2×** RTX 4090 |
| Berkeley Humanoid plano / rugoso | 120 145 / 30 393 (Tab. VII) | < 15 min em 2× RTX 4090 |
| LEAP cube reorient | 76 354 (Tab. IX) | ~2080 s em **1×** RTX 4090, 8192 envs (Fig. 5) |
| PandaPickCube (estado) | 140 386 (Tab. IX) | — |

- `num_envs` dos configs publicados: 2048 no DM Control (`PG/rss2025/mujoco_playground/config/dm_control_suite_params.py:43`),
  8192 na locomoção (`.../config/locomotion_params.py:43`).
- **MJX-Warp**: o post de anúncio fala em "1.5-2x higher throughput with Warp" em vários envs de
  manipulação e regressões leves em PandaPick*/AlohaSinglePeg
  ([discussion #197](https://github.com/google-deepmind/mujoco_playground/discussions/197)); os
  gráficos do post não foram lidos (imagens). A doc do MJX dá SPS de física, não de treino:
  Humanoid 2.96 M e Aloha Pot 2.33 M com JAX FFI em `GraphMode.WARP`, caindo para 0.80 M / 0.65 M
  se o grafo CUDA for recapturado a cada passo (`MJ/doc/mjx.rst:183-205`); GPU e lote não
  informados nesse trecho.
- MJX-JAX, só física: humanoide único, A100, lote 8192 → ~950 K passos/s (`MJ/doc/mjx.rst:662-672`).
- **Inferência para o nosso caso, não medida**: SPS de física ÷ 24 passos de física por passo de
  controle (40 Hz a 1/960 s, restrição 5 do mapa) ≈ SPS de decisão com `repeat_action=1`. Uma RTX
  3060 tem bem menos banda e SMs que uma A100; nenhuma fonte dá número para ela.

### Recursos

| recurso | onde |
|---|---|
| normalização de observação | `normalize_observations=True` nos configs (`dm_control_suite_params.py:35`) |
| crítico assimétrico | locomoção/manipulação usam `value_obs_key="privileged_state"` (`locomotion_params.py:61-62` e seguintes); flags `--policy_obs_key`/`--value_obs_key` (`train_jax_ppo.py:147-150`) |
| ações em [-1, 1] | herda o `tanh_normal` do Brax |
| action repeat | `--action_repeat` (`train_jax_ppo.py:117`) → `EpisodeWrapper` do Brax; com `--vision` o Brax recusa `action_repeat != 1` (`BRAX/training/agents/ppo/train.py:340-344`) |
| checkpoint | `logdir/checkpoints`, `--load_checkpoint_path` com retomada do último (`train_jax_ppo.py:335-355`) |
| logging | `--use_wandb` e `--use_tb` (tensorboardX) no `progress_fn` (`train_jax_ppo.py:93-100`, `321-333`, `405-413`); rscope para inspecionar trajetórias |
| DR | `--domain_randomization` usa o randomizer registrado do env (`train_jax_ppo.py:101`, `378-381`) |

### Derivados de PPO

Os do Brax (§1). O mesmo diretório também tem `train_rsl_rl.py` (RSL-RL em PyTorch via
`wrapper_torch.py`), que não é JAX.

## 3. PureJaxRL

### Backend e auto-reset

- **Sem MJX.** O único adaptador contínuo é `BraxGymnaxWrapper(env_name, backend="positional")`,
  que chama `brax.envs.get_environment` e aplica `EpisodeWrapper(episode_length=1000)` +
  `AutoResetWrapper` do Brax (`purejaxrl/wrappers.py:117-146`). Rodar sobre MJX exige escrever um
  wrapper Gymnax próprio em volta do env MJX. Último commit em 2024-09-09.
- Auto-reset = o do Brax (estado inicial em cache). O GAE usa só `done`, **sem distinguir
  truncamento** (`ppo_continuous_action.py:153-160`: `delta = r + γ·V'·(1-done) - V`), então trata o
  limite de episódio como término.
- O README diz: "we implement the *entire training pipeline in JAX*, including the environment" e
  que o treino inteiro é jitável/vmapeável/pmapeável (`purejaxrl/README.md`, parágrafo inicial).

### Throughput reportado

Só comparações relativas: "over 1000x faster than standard PyTorch RL implementations" com muitos
agentes em paralelo, e "2048 PPO agents in half the time it takes to train a single PyTorch PPO
agent" (`README.md`, seção "Performance"), em CartPole/MinAtar. Nenhum SPS absoluto e nenhuma tarefa
MuJoCo/MJX. Default do `ppo_continuous_action.py`: `NUM_ENVS=2048`, `NUM_STEPS=10`, hopper
(`ppo_continuous_action.py:292-309`).

### Recursos

| recurso | situação |
|---|---|
| normalização de observação | `NormalizeVecObservation` + `NormalizeVecReward` (`wrappers.py:226`, `304`; ligados por `NORMALIZE_ENV`, `ppo_continuous_action.py:80-82`) |
| crítico assimétrico | não; ator e crítico leem o mesmo `x` (`ppo_continuous_action.py:20-56`) |
| ações em [-1, 1] | gaussiana diagonal **sem tanh** + `ClipAction` (`ppo_continuous_action.py:42`; `wrappers.py:174-184`); o espaço declarado é [-1, 1] (`wrappers.py:141-146`) |
| action repeat | fixo em 1 no wrapper (`wrappers.py:120`) |
| checkpoint | nenhum |
| logging | `jax.debug.callback` com print (flag `DEBUG`); `wandb` aparece em `requirements.txt`, não no script contínuo |

### Derivados de PPO

Arquivos separados, um por variante: `ppo.py` (discreto), `ppo_continuous_action.py`, `ppo_rnn.py`
(GRU), `ppo_minigrid.py`, `dpo_continuous_action.py` (Discovered Policy Optimisation, *drift* no
lugar do clipping, `dpo_continuous_action.py:215-220`). Todos com clipping de valor
(`ppo_continuous_action.py:190-196`) e normalização de vantagem.

## 4. Stoix (+ Stoa)

### Backend e auto-reset

- Adaptador para envs do Playground: `make_playground_env` carrega o env pelo
  `mujoco_playground.registry` com `config_overrides=config.env.kwargs` e o embrulha em
  `MuJoCoPlaygroundToStoa` (`Stoix/stoix/utils/make_env.py:387-415`). Logo `impl` poderia ir em
  `env.kwargs` (os configs de exemplo usam `kwargs: {}`,
  `stoix/configs/env/mjc_playground/dm_control/cartpole_balance.yaml`). Não usa o wrapper Brax do
  Playground.
- DR: o adaptador aplica o randomizer **por env no `reset`**, desfazendo o lote do modelo
  (`Stoa/stoa/env_adapters/playground.py:80-110`).
- Auto-reset: `AutoResetWrapper`/`CachedAutoResetWrapper`/`OptimisticResetVmapWrapper` do Stoa,
  aplicados **antes** do `VmapWrapper` (`make_env.py:44-60`); o `step` faz `jax.lax.cond(done, ...)`
  por env (`Stoa/stoa/core_wrappers/auto_reset.py:153`, `226`). Sob `vmap`, um `cond` vira `select`
  em todas as folhas, equivalente ao `tree.map(where)` que a doc diz não funcionar com Warp
  (`MJ/doc/mjx.rst:135-141`). **Inferência; compatibilidade com `impl="warp"` não verificada.**
- Truncamento tratado: GAE truncado com valor de bootstrap guardado no auto-reset
  (`stoix/systems/ppo/anakin/ff_ppo_continuous.py:108-120`, `170-178`).
- Dependências: `jax>=0.4.25,<0.6.0` e `playground>=0.0.5` (`Stoix/pyproject.toml:145`, `161`), ao
  passo que o Playground com Warp exige `mujoco>=3.6.0`, `mujoco-mjx>=3.6.0`, `warp-lang>=1.18.0`
  (`PG/rss2025/pyproject.toml:25-34`). **Não verificado** se essas versões resolvem juntas sob
  `jax<0.6`.

### Throughput reportado

Nenhum número absoluto no README; calcula `steps_per_second` por rollout
(`ff_ppo_continuous.py:610`). README: "fully compiled with JAX's jit thus making training and
environment execution very fast" (`Stoix/README.md:35`).

### Recursos

| recurso | onde |
|---|---|
| normalização de observação | `system.normalize_observations`, com aquecimento de estatísticas (`ff_ppo_continuous.py:359-395`, `524-528`) |
| crítico assimétrico | não encontrado (busca por `privileged`/`asymmetric`/`critic_obs` em `stoix/` sem resultados) |
| ações em [-1, 1] | `NormalAffineTanhDistributionHead` (`stoix/configs/network/mlp_continuous.yaml`) |
| action repeat | não encontrado para o Playground (`frame_skip` só para outro adaptador, `make_env.py:225`) |
| checkpoint | Orbax (`stoix/utils/checkpointing.py:7`, `61`) |
| logging | console, wandb, neptune, JSON, TensorBoard (`stoix/configs/logger/logger.yaml:10-58`) |
| distribuição | Anakin (tudo em JAX, `pmap`) e Sebulba (atores/aprendiz separados) (`README.md:44-46`) |

### Derivados de PPO

`stoix/systems/ppo/README.md`: `ff_ppo` (discreto), `ff_ppo_continuous`, `ff_ppo_penalty` e
`ff_ppo_penalty_continuous` (penalidade de KL em vez de clipping), `ff_dpo_continuous`, `rec_ppo`
(recorrente), e `sebulba/ff_ppo`. Configuração por Hydra, um arquivo por sistema.

## 5. rejax

- PPO contínuo e discreto num `PPO` com mixins de normalização de observação e de recompensa
  (`rejax/src/rejax/algos/ppo.py:35-40`). Sem tanh: amostra e **clipa** a ação no espaço do env
  (`ppo.py:126-137`). GAE só com `done`, sem truncamento (`ppo.py:174-178`; `last_val` zerado se
  `last_done`, linha 99).
- Envs pelo protocolo Gymnax; adaptador `brax2gymnax.create_brax(env_name)` para envs nomeados do
  Brax (`src/rejax/compat/brax2gymnax.py:12-55`). **Sem adaptador MJX/Playground**, sem checkpoint
  (não encontrado), sem crítico assimétrico, sem action repeat.
- Foco declarado: vmapear o treino inteiro sobre seeds/hiperparâmetros (`README.md`, "Vectorize
  training"). Benchmark só como gráfico de speedup sobre o CleanRL em hopper/breakout, A100
  (`README.md`, mesma seção). Último commit 2026-06-10.

## 6. Comparação e base mais simples para as 6 tasks

| | Brax PPO | Playground `rss2025` (Brax PPO) | PureJaxRL | Stoix | rejax |
|---|---|---|---|---|---|
| fala MJX | sim, via env `brax.envs.Env` | sim, nativo (`MjxEnv`) | não (Brax `positional`) | sim, via adaptador Stoa | não |
| `impl="warp"` documentado | não | **sim** (`--impl warp`, default desde 0.2.0) | não | não verificado | não |
| auto-reset | `tree.map(where)` p/ 1º estado | idem, ou `full_reset` com re-amostragem | o do Brax | `lax.cond` por env (Stoa) | Gymnax |
| truncamento no GAE | sim | sim | não | sim | não |
| normalização de obs | sim | sim | sim (+ recompensa) | sim | sim (+ recompensa) |
| crítico assimétrico | sim (`value_obs_key`) | sim (`privileged_state`) | não | não encontrado | não |
| [-1, 1] | tanh-normal | tanh-normal | gauss + clip | tanh afim | gauss + clip |
| action repeat | sim | sim | não | não p/ Playground | não |
| checkpoint | Orbax | Orbax + retomada | não | Orbax | não |
| logging | callback | wandb, TB | print | 5 backends | — |
| derivados | clip, clip-valor, KL-LR adaptativo, distribucional, visão | os do Brax | PPO, RNN, DPO | PPO, PPO-KL, DPO, RNN, Sebulba | PPO |
| SPS publicado (1 GPU) | — | 30 K–720 K env-SPS, A100, MJX-JAX | só relativo | — | só relativo |

**Recomendação (inferência deste estudo).** O caminho de menor atrito é escrever as 6 tasks no
padrão `MjxEnv` do Playground (`reset(rng) -> State`, `step(State, action) -> State`, obs em dict com
`state` e `privileged_state`) e treinar com `brax.training.agents.ppo.train` passando um
`wrap_env_fn` nosso, copiado do `wrap_for_brax_training`
(`PG/rss2025/mujoco_playground/_src/wrapper.py:87-248`) com duas mudanças:

1. no auto-reset, trocar o `where_done` por `state.data.where(done, reset_data)` (o método oficial,
   `MJ/mjx/mujoco/mjx/_src/types.py:1188`), que vale para `impl="jax"` e `impl="warp"`;
2. usar `full_reset=True` (ou equivalente), porque as tasks precisam de pose inicial re-amostrada por
   episódio; com o reset em cache do Brax, cada env repetiria sempre o mesmo estado inicial.

Isso dá um único código de treino para as 6 tasks (só muda o env), mantém a restrição 6 do mapa
(`impl` por flag) e herda normalização, crítico assimétrico, tanh-normal, checkpoint e os derivados
de PPO por flag. Não instale o pacote `playground` como dependência: o que se reaproveita são ~200
linhas de wrapper e o padrão de env; o pacote puxa a Menagerie e o resto dos envs.

Se for preciso ter o PPO no repo (single-file, hackável), o `ppo_continuous_action.py` do PureJaxRL
(~330 linhas) é o menor ponto de partida, mas faltam truncamento no GAE, crítico assimétrico,
tanh, action repeat e checkpoint; tudo isso o Brax já tem.

Riscos em aberto, que dependem de outros tickets: o `pmap` incondicional do Brax PPO com Warp em
mais de uma GPU (irrelevante para a RTX 3060); recaptura de grafo CUDA pela ponte JAX↔Warp
(`MJ/doc/mjx.rst:149-205`), que pode ser agravada por `full_reset` e pelo kick que muda `timestep`
por env; e o `naconmax` somado entre mundos.

## 7. Contraste: como o Planning-the-path-with-rl treinava (não é alvo)

Commit `2287531`.

- **SAC em PyTorch**, CPU/GPU via `args.cuda` (`methods/sac.py:150-186`): dois Q MLP 256-256
  (`sac.py:22-50`), política gaussiana com tanh e reescala para o `action_space`
  (`sac.py:96-110`), alvo por média móvel (`sac.py:122-147`), temperatura automática só sem CAPS
  (`sac.py:197-200`).
- **Regularização CAPS** (suavidade temporal e espacial da ação): `lambda_temporal` e
  `lambda_spacial` somados à perda da política, com ruído `caps_epsilon` no estado
  (`sac.py:274-282`).
- **Ambientes em CPU, 16 processos**: `gym.vector.AsyncVectorEnv` com `num_envs` envs
  (`train.py:21-22`), `num_envs: 16` e `total_timesteps: 150000` em todos os experimentos
  (`experiments.yml`). Cada iteração dá um passo nos 16 envs, grava no replay buffer e faz um update
  (`train.py:31-66`). Variantes `-v1` com *frameskip* (`experiments.yml`, `FRAMESKIP`/`FSCAPS`).
- Logging wandb com `charts/SPS` (`train.py:72-87`); checkpoint `actor.pt`/`critic.pt` a cada 9999
  passos (`train.py:88-89`, `sac.py:301-309`).
- Contraste: off-policy, poucos envs, física em CPU e um gradiente por passo vetorial. O alvo do
  port é on-policy (PPO), milhares de mundos em GPU e o laço inteiro jitado.

## Itens não verificados

1. Se o `pmap` do Brax PPO funciona com MJX-Warp em uma GPU: inferido de o Playground documentar
   `--impl warp` com `ppo.train`; não executado.
2. Se o `where_done` por formato do Playground é sempre equivalente a `mjx.Data.where` com Warp.
3. Se o Stoix roda com `impl="warp"`, e se as dependências fecham com `jax<0.6`.
4. Throughput de qualquer pilha numa RTX 3060: nenhuma fonte publica; precisa ser medido (ticket de
   benchmark).
5. Números de SPS do MJX-Warp em treino: só o "1.5-2x" do post #197; os gráficos não foram lidos.
