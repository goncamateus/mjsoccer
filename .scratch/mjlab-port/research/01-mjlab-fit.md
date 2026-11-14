# 01 — O mjlab hospeda as seis tasks do mjsoccer? Custo e ganho contra o port MJX-Warp + Brax PPO

Pesquisado em 2026-11-14, só em fonte primária: código-fonte, docs, `pyproject.toml`/`uv.lock`,
changelog, workflows e issues do mjlab; código do MuJoCo Warp na tag que o mjlab pina; metadados do
PyPI; o JSON de throughput publicado pelo próprio mjlab. **Nada foi instalado, compilado nem
executado** (os repositórios foram só clonados em `/home/mateus/doc/_ext/`). O único comando local
fora de leitura foi `nvidia-smi` (RTX 3060, driver 595.99.02), para saber qual build de torch serve.

Convenção: **[fonte]** = fato lido no arquivo/linha ou na URL citada. **[inferência]** = conclusão
minha a partir das fontes, não verificada por execução. Tudo que só um protótipo responde está em §8.

## Fontes

| repo | ref | data | abreviação |
|---|---|---|---|
| mujocolab/mjlab, `main` (clone completo) | `6ab9846` (`v1.6.0-81-g6ab9846`) | 2026-11-14 | `ML/` = `https://github.com/mujocolab/mjlab/blob/6ab9846ea068714678fd0e9f791f898c6b697b20/` |
| mujocolab/mjlab, tag `v1.6.0` (última release) | `v1.6.0` | 2026-08-08 | `ML160/` |
| google-deepmind/mujoco_warp, tag `v4.0.1` (a que o mjlab `main` pina) | `2db59f4` | 2026-11-11 | `MW/` = `https://github.com/google-deepmind/mujoco_warp/blob/v4.0.1/` |
| Nightly do mjlab, `https://mujocolab.github.io/mjlab/nightly/throughput_data.json` | commit `45b1e26` (última entrada) | 2026-10-17 | `NIGHTLY` |
| Issues do mjlab (`gh issue view/search`) | — | lido em 2026-11-14 | `#n` |
| PyPI JSON API (`https://pypi.org/pypi/<pkg>/<ver>/json`) | — | lido em 2026-11-14 | `PyPI` |
| arXiv 2601.22074 (paper do mjlab, v2) | v2 | 2026-02-25 | `PAPER` |
| Este repo | `a63f858` | 2026-11-14 | sem prefixo |

---

## Resposta curta

1. **Hospeda, sim, sem bloqueio duro de física** [inferência sobre fatos]. O mjlab roda o
   `mujoco_warp` direto (não o MJX), 4.0.1 no `main`. Este é o mesmo motor do nosso port, uma patch
   acima. Ele aceita uma cena arbitrária por `MjSpec` (`SceneCfg.spec_fn`), robôs sem freejoint,
   `<pair>`, sensores `contact` nativos, `nconmax`/`njmax` por mundo, e expõe `qpos`, `ctrl`,
   `sensordata`, `overflow` e o modelo como tensores torch sem cópia. Também se pode escrever torch
   (ou kernels Warp) livremente dentro do passo.
2. **Mas o passo de disparo do kick não cabe no `decimation` fixo** [inferência]. O env assume
   `step_dt = timestep × decimation` constante (`ML/src/mjlab/envs/manager_based_rl_env.py:269-271`)
   e deriva disso o horizonte (`:283-286`) e a escala da recompensa (`:157-163`). O nosso esquema de
   48 iterações com `opt.timestep` por mundo (32×dt/4 + 16×dt, ou 24×dt + 24×0) funciona no
   `mujoco_warp` (`MW/mujoco_warp/_src/types.py:1074`: `timestep: array("*", float)`, indexado por
   `worldid % shape[0]`). O mjlab, porém, não expande `opt.timestep` por mundo (§2.4), e o
   `step_dt` que ele reporta sairia errado. Dá para contornar com código nosso: sobrescrever o
   `step()` ou ser dono do tempo, da recompensa e do horizonte. Não é bloqueio, mas é o maior
   pedaço de código "contra o framework".
3. **Adotar o mjlab é uma terceira implementação da lógica das tasks, em torch** [inferência]. Os
   ~1 100 linhas de JAX em `src/mjsoccer/mjx/` (base 580, tasks 272, wrappers 133, randomize 141)
   viram torch dentro de termos de manager. O task-logic parity (snapshot → `put_data`) tem de ser
   refeito para torch. O spec atual pôs "MuJoCo Warp nativo ou torch" explicitamente fora de escopo
   (`.scratch/gpu-port/spec.md`, Out of Scope) e o ADR 0002 decidiu "through the `mujoco-mjx` JAX
   API". Adotar o mjlab pede um ADR novo.
4. **Ganhos concretos** [fonte]: DR por evento na GPU com `set_const` por grupo (inclui reamostrar
   por episódio, hoje fora de escopo), ruído/atraso/histórico de observação e atraso de atuador
   nativos, crítico assimétrico por `obs_groups`, export ONNX, viewer nativo/viser, multi-GPU. Some
   também a dor do cache de grafos JAX↔Warp: o mjlab captura o grafo uma vez no init.
5. **Custos concretos** [fonte + inferência]: a última release (v1.6.0, 2026-08-08) pina
   `mujoco~=3.11`, incompatível com o nosso `mujoco==4.0.0`. Só o `main` não lançado está no 4.0.1,
   o que obriga um pin git e subir o CPU para 4.0.1. O RL é RSL-RL (torch), single-agent. A taxa de
   overhead do manager publicada é 62–76 % do tempo do passo (NIGHTLY). O nosso passo de controle
   tem 48 iterações de física, cada uma com Python por substep no loop do mjlab. Que o mjlab chegue
   perto dos ~16 k decisões/s de hoje é **a** pergunta que só o protótipo responde.

---

## 1. Arquitetura

### 1.1 Env, managers e passo

- **Managers** [fonte]: `ManagerBasedRlEnvCfg` tem `observations`, `actions`, `events`, `rewards`,
  `terminations`, `commands`, `curriculum`, `metrics` e `recorders`
  (`ML/src/mjlab/envs/manager_based_rl_env.py:51-163`). A ordem de carga é event → command →
  action → observation → termination → reward → … (`:302-357`).
- **O passo** (`:387-507`) [fonte]: `process_action` uma vez por passo do env (`:436`), depois o
  loop `for _ in range(decimation)`: `apply_action` → `scene.write_data_to_sim` → `sim.step` →
  `scene.update` (`:438-444`). Depois vêm terminações, recompensa e eventos `step`/`interval`. O
  auto-reset dos mundos com `done` (`:469-473`) é seguido de **um** `sim.forward()` (`:479`), dos
  comandos e da observação. Termination e reward veem grandezas derivadas atrasadas de um substep,
  por projeto (`:396-411`).
- **`apply_action` roda em todo substep** [fonte]: `ML/src/mjlab/managers/action_manager.py:177-184`.
  Um `ActionTerm` próprio implementa `process_actions` (uma vez por decisão) e `apply_actions` (por
  substep) (`:40-68`).
- **Auto-reset** [fonte]: `_reset_idx` chama `sim.reset(env_ids)`, que é `mjwarp.reset_data` com
  máscara (`ML/src/mjlab/sim/sim.py:507-519`). Depois vêm `scene.reset`, os eventos `reset` e o reset
  de cada manager (`manager_based_rl_env.py:581-619`).
- **Recompensa** [fonte]: por default é multiplicada por `step_dt` (`scale_rewards_by_dt=True`,
  `:157-163`). Para manter a recompensa do CPU é preciso desligar.
- **Horizonte** [fonte]: `max_episode_length = ceil(episode_length_s / step_dt)` (`:283-286`), e
  `is_finite_horizon` decide se o time-out é terminal ou truncation (`:134-142`). O wrapper do
  RSL-RL repassa `time_outs` para bootstrap (`ML/src/mjlab/rl/vecenv_wrapper.py:86-92`).
- **"DirectEnv" não é suportado nem planejado** [fonte]: na issue #100, um mantenedor escreveu
  "DirectEnv feature should not be supported!!!". Mas `ManagerBasedRlEnv` é uma classe Python comum,
  e nada impede uma subclasse que sobrescreva `step()` [inferência].

### 1.2 Cena e entidade

- **Cena** [fonte]: `Scene` parte de um `scene.xml` só de visual (`ML/src/mjlab/scene/scene.py:26,67`;
  `ML/src/mjlab/scene/scene.xml`). Anexa cada `EntityCfg` com `spec.attach(prefix="<nome>/")`
  (`:226-253`), depois os sensores, e por fim chama `SceneCfg.spec_fn(spec)`: "Optional callback to
  modify the `MjSpec` after entities and sensors have been added but before compilation" (`:52-54,73-74`).
- **Entidade** [fonte]: `EntityCfg.spec_fn: Callable[[], MjSpec]` (`ML/src/mjlab/entity/entity.py:99-102`).
  No máximo **uma** freejoint por entidade (`:190-197`). Entidades sem freejoint são embrulhadas
  num corpo **mocap** (`auto_wrap_fixed_base_mocap`, `ML/src/mjlab/utils/spec.py:190-235`), e só um
  evento de reset as posiciona (FAQ, `ML/docs/source/faq.rst`, "Why are my fixed-base robots all
  stacked…").
- **`<option>` do XML é ignorado** [fonte]. O `attach` não propaga `<option>`. O mjlab aplica
  `MujocoCfg` ao modelo compilado (`ML/src/mjlab/sim/sim.py:98-153`, chamado em `:261,264`) e avisa
  se a entidade tiver option não-default (`scene.py:245-251`; FAQ "My XML `<option>` flags are not
  taking effect"). Os defaults do `MujocoCfg` **diferem do nosso `scene.xml:4`**: `timestep=0.002`
  (nós 1/960), `cone="pyramidal"` (nós `elliptic`) (`sim.py:103-117`). `iterations=100`,
  `ls_iterations=50`, `implicitfast` e Newton coincidem. O `MujocoCfg` só expõe `euler` e
  `implicitfast` (`:63-66`), o que nos basta.
- **Mapeamento para o mjsoccer** [inferência]. A nossa cena já sai de `MjSpec`
  (`src/mjsoccer/scene.py:235-270`: `from_file`, `attach` com prefixo, `add_pair`, `add_sensor`).
  Há dois caminhos:
  - **(a) Cena inteira no `SceneCfg.spec_fn`**, sem entidades. É o modelo base exato: sem
    embrulho mocap e com os mesmos nomes do CPU. Em troca, não há `Entity`; ações e observações
    leem `env.sim.data`/`env.sim.model` direto. Os `ActionTermCfg` exigem `entity_name`
    (`action_manager.py:26-27,47-50`), então o nosso term teria de contornar isso (por exemplo
    apontar para uma entidade-âncora vazia).
  - **(b) Um robô por entidade e a bola como entidade flutuante.** Ganham-se os helpers
    (`write_root_state_to_sim`, `XmlActuatorCfg` sobre os atuadores do XML,
    `ML/src/mjlab/actuator/xml_actuator.py:1-41`). Em troca, cada robô ganha um corpo mocap pai e
    os `<pair>` bola×dribbler cruzando entidades vão para o `spec_fn`. Fica um modelo que não é
    mais byte-igual ao do CPU, e a parity teria de cobrir isso.

  Recomendo (a) para manter "um modelo base só" do ADR 0002.

### 1.3 Quanto da nossa lógica por mundo cabe

[inferência, baseada em §1.1–1.2]. O estado do mjlab é **tensor torch mutável por mundo, dentro
de objetos de manager**, não um pytree funcional. Peça a peça:

| peça nossa (`State.info["world"]`, `mjx/base.py`) | onde mora no mjlab |
|---|---|
| `grace (n_robots,)`, `plunger_lock (n_robots,2) int32`, `fired`, `action_queue` | buffers `torch` no nosso `ActionTerm` (zerados em `reset(env_ids)`) |
| `steps` em passos de controle, truncation | contador nosso + termination custom (o `episode_length_buf` conta passos do env, não de controle; ver §2.4) |
| estado da task (`prev`, `prev_goal`, `count`, `last_ball`, `stopped`) | buffers num termo de reward/termination ou num `CommandTerm` |
| controlador de navegação (função pura) | função torch chamada no `apply_actions` a cada 48 substeps (ou a cada substep do passo de controle, se `decimation` = 1 passo de controle) |
| robôs scripted (amarelos `ctrl=0`, companheiro só com dribbler) | escrita direta em `sim.data.ctrl` no mesmo term |
| `seat_ball` (336 passos de física **dentro** do reset) | **não cabe como está**: `sim.step` avança todos os mundos (`sim.py:498-505`). Precisa de pool de estados pré-assentados gerado num evento `startup`, ou do assentamento como parte do episódio |
| `sample_pose` (200 candidatos, `argmax`) | evento `reset` em torch, direto |
| RNG por mundo (`info["rng"]`) | o mjlab usa o RNG global do torch (`manager_based_rl_env.py:534-540`); não há chave por mundo. A reprodutibilidade bit a bit já não existe no Warp (FAQ "Why aren't my training runs reproducible…", `mujoco_warp#562`) |

**Torch/Warp custom no passo** [fonte]: `sim.data.*` e `sim.model.*` são `TorchArray`, views torch
sem cópia dos arrays Warp, no stream do Warp (`ML/src/mjlab/sim/sim_data.py:1-75`). Escrever
in-place não invalida o grafo capturado. Substituir um array exige `create_graph()`
(`sim.py:207-227,356-388`). Um kernel `wp.launch` próprio também funciona, porque os arrays são
`wp.array` [inferência].

## 2. Física

### 2.1 Motor e versão

- [fonte] O mjlab importa `mujoco_warp` direto (`ML/src/mjlab/sim/sim.py:11`), sem MJX e sem JAX.
  `main` pina `mujoco-warp~=4.0.1`, `mujoco-core[rendering]~=4.0.1` e `warp-lang>=1.15.0`
  (`ML/pyproject.toml:38-40`); o `uv.lock` resolve `mujoco-warp 4.0.1`, `mujoco-core 4.0.1` e
  `warp-lang 1.18.0`. A release **v1.6.0** pina `mujoco-warp~=3.11.0` e `mujoco~=3.11.0`
  (`ML160/pyproject.toml:38-40`; PyPI `mjlab/1.6.0`). A troca para 4.0.1 está em "Upcoming version
  (not yet released)" do `ML/docs/source/changelog.rst`.
- [fonte] O nosso CPU está em `mujoco==4.0.0` (`pyproject.toml:12`) e o MJX-Warp em
  `mujoco-mjx[warp]==4.0.0`, que carrega uma cópia própria do MuJoCo Warp
  (`.scratch/gpu-port/research/01-mjx-feature-support.md`, Fontes). `mujoco 4.0.1` e
  `mujoco-mjx 4.0.1` existem no PyPI desde 2026-11-11.
- [inferência] Para a parity CPU × GPU continuar "mesma versão" (user story 41), o CPU teria de
  subir para 4.0.1, com nova rodada dos testes de fidelidade e de `KICK_TABLE_MS` (como na Fase 0).

### 2.2 Recursos do nosso modelo

| recurso | status no mjlab/mujoco_warp 4.0.1 | base |
|---|---|---|
| vários robôs por mundo (2 e 7) | native: é só MJCF. Com entidades, cada robô é uma `EntityCfg` com prefixo | `scene.py:226-253` [fonte] |
| juntas `robot_x/y` slide + `robot_yaw` hinge, sem freejoint | native como "fixed articulated" (vem embrulhado em mocap, §1.2) | `entity.py:139-160`, `spec.py:190-235` [fonte] |
| bola com freejoint | native ("floating non-articulated") | `entity.py:139-160` [fonte] |
| meshes de colisão convexos, `<pair>` explícitos | o `mujoco_warp` 4.0.1 é o mesmo motor do nosso port, que já roda o modelo base exato | ADR 0002 [fonte]; mesma família de versão [inferência] |
| plunger kicker/chipper (`motor` em slide limitado) | native (XML). `XmlActuatorCfg` embrulha `motor`/`velocity` do XML | `xml_actuator.py:1-41` [fonte] |
| passo de disparo com `opt.timestep` por mundo | **custom**: o motor suporta (`MW/…/types.py:1074`, `constraint.py:575`: `opt_timestep[worldid % opt_timestep.shape[0]]`), mas o `expand_model_fields` do mjlab só percorre campos de topo do `Model` (`ML/src/mjlab/sim/randomization.py:51-56`), não `opt.*` | [fonte] + workaround [inferência] |
| sensor `contact` geom-geom (posse) | native: `ContactSensorCfg` sobre o sensor `contact` do MuJoCo (`ML/src/mjlab/sensor/contact_sensor.py:1-15,…`), com `history_length` por substep (`:141,157`; FAQ "My contact sensor misses collisions…"). Sensores já no spec viram `BuiltinSensor` (`scene.py:286-288`) | [fonte] |
| `nconmax`/`njmax` | native, **por mundo** (`sim.py:156-167`, passados a `put_data` em `:335-341`). Não expõe `naconmax` total, que o nosso env usa (`mjx/base.py:256-257`); o `put_data` do Warp aceita `naconmax` (`MW/mujoco_warp/_src/io.py:2474,2490`) | [fonte] |
| modos de grafo | captura **um** grafo Warp por `step`/`forward`/`reset` no init, se houver mempool e driver ≥ 12.4 (`sim.py:351-388,576-597`); `graph_conditional` opcional (`:175-179,203-204`). Não há `WARP_STAGED` | [fonte] |
| NaN por mundo | `mdp.terminations.nan_detection` termina o mundo (`ML/src/mjlab/envs/mdp/terminations.py:47-49`) pelo `NanGuard.detect_nans` em `qpos/qvel/qacc/qacc_warmstart/sensordata` (`ML/src/mjlab/utils/nan_guard.py:89-114`). `nan_guard` despeja estados para debug (`:16-23`, desligado por default). Obs ganha `nan_policy` (`observation_manager.py:100`) | [fonte] |
| overflow do Warp por mundo | **custom**: o mjlab não lê `overflow` em lugar nenhum (`grep overflow src/mjlab` só acha CSS). O `mujoco_warp` tem `Data.overflow: array("nworld", int)` e o bitmask `OverflowType` (`MW/…/types.py:161-185,2626,2804`); uma termination nossa lê `env.sim.data.overflow`. A issue #1263 (aberta) mostra os avisos de `LS_ITERATIONS` no próprio Go1 do mjlab | [fonte] + [inferência] |
| chip kick | mesmo mecanismo do kick (segundo plunger) | [inferência] |

### 2.3 Paridade e "nosso modelo base"

[inferência] Com o caminho (a) de §1.2, o modelo compilado é o do `scene.build` mais o `scene.xml`
visual do mjlab (só `<visual>`/`<statistic meansize>`), com `MujocoCfg` configurado igual ao nosso
`<option>`. A física é a mesma do port atual, um patch acima (4.0.0 → 4.0.1). Mesmo assim, as
paridades estatísticas (ticket 12) e o `hold_turn` (ticket 16) precisam rodar de novo.

### 2.4 O passo de controle de tempo variável

[inferência, detalhando §Resposta 2]. Há três formas, da menos para a mais invasiva:

1. **`decimation = 48 × repeat_action`**, com `timestep = 1/960` no `MujocoCfg`, e um
   `opt.timestep` por mundo trocado à mão:
   `sim.wp_model.opt.timestep = wp.array(nworld)`, depois `sim.create_graph()`. A agenda é escrita
   in-place a cada substep pelo `apply_actions`. Fica o mesmo custo de 48 iterações do port atual
   (ticket 08). O `step_dt` do mjlab sai 50 ms em vez de 25 ms, então horizonte, recompensa e
   `dt` dos comandos passam a ser nossos (`scale_rewards_by_dt=False`, time-out custom).
2. **Subclasse com `step()` próprio**: um loop de 24 substeps e mais 24 opcionais só quando algum
   mundo dispara (o ganho que o ticket 08 deixou "para quem medir"). Aqui o predicado fica fora de
   qualquer `vmap`, porque o loop é Python.
3. **Mudar o mecanismo do kick** está fora de questão (spec, user story 43).

## 3. RL

- **Biblioteca** [fonte]: `rsl-rl-lib==5.5.2` (`ML/pyproject.toml:49`), PyTorch. O runner é
  `MjlabOnPolicyRunner(OnPolicyRunner)` (`ML/src/mjlab/rl/runner.py:12`).
- **PPO** [fonte]: `RslRlPpoAlgorithmCfg` com épocas, mini-batches, `schedule="adaptive"` (LR por
  KL, `desired_kl=0.01`), entropia e clip (`ML/src/mjlab/rl/config.py`). Modelos MLP, CNN ou RNN
  (`class_name`, `rnn_type`). `num_steps_per_env=24` e `logger="wandb"` são os defaults
  (`config.py:88,105`), e o logger aceita tensorboard ou classe própria (changelog Upcoming).
- **Assimetria** [fonte]: `obs_groups = {"actor": ("actor",), "critic": ("critic",)}`
  (`config.py:92`), com grupos de observação independentes no env.
- **Multi-agente: não** [fonte + inferência]. A recompensa é um tensor `(num_envs,)`
  (`manager_based_rl_env.py:457`, `vecenv_wrapper.py:84-95`), e a busca por "multi-agent",
  "multiagent" e "multi agent" nas issues voltou vazia (2026-11-14). **Não nos afeta**: as seis
  tasks controlam um robô só (`n_controlled = 1`, `src/mjsoccer/base.py:53`; nenhuma task
  sobrescreve).
- **Checkpoint/export** [fonte]: `.pt` do RSL-RL (`load_checkpoint="model_.*.pt"`) e ONNX com
  metadados (`runner.py:45-126`; `export_onnx_on_save`).
- **Eval/play/viewer** [fonte]: `play` com `--agent zero|random|trained` e viewer `native`
  (`mujoco.viewer`, `ML/src/mjlab/viewer/native/viewer.py:138`) ou `viser` (web)
  (`ML/src/mjlab/scripts/play.py:35,50,288-301`). Não há eval embutido em época, como o do Brax
  [inferência: não achei no runner]. O nosso `--play` no backend CPU com `mujoco.studio` continuaria
  nosso, rodando a política torch ou ONNX na CPU [inferência].
- [inferência] As variantes do Brax que o spec expõe como flag (crítico assimétrico, LR por KL)
  existem no RSL-RL. Não verifiquei a "distribucional".

## 4. DR, ruído de observação e latência

- **DR** [fonte]: por `EventTermCfg` com `mode ∈ {"startup","reset","interval","step"}`
  (`ML/src/mjlab/managers/event_manager.py:24`), sobre funções `mdp.dr.*`: geom, body, joint,
  pair, actuator, tendon, câmera, luz e material (`ML/src/mjlab/envs/mdp/dr/__init__.py`). Os
  campos usados são expandidos por mundo no init (`manager_based_rl_env.py:312`). Os derivados são
  recalculados na GPU por `mujoco_warp.set_const` com o grupo inferido do campo (`body_mass → ALL`,
  `geom_size → FIXED`, … `event_manager.py:27-84`; `sim.py:481-488`).
  - **Ganho**: DR reamostrada por episódio (`mode="reset"`), hoje fora de escopo no spec, sem
    `mj_setConst` no host por slot.
  - **Lacuna**: para o par só existe `dr.pair_friction` (`ML/src/mjlab/envs/mdp/dr/pair.py`). Os
    nossos contatos da bola são `<pair>` (`models/ball.xml:7-26`), e mexer em
    `pair_solref`/`pair_solimp` pede um evento nosso sobre `env.sim.model.pair_*`, depois de
    expandir o campo [inferência]. Não há guarda contra mudar `opt.timestep` porque o mjlab não
    randomiza `opt` [fonte: nenhuma função em `mdp/dr` toca `opt`].
- **Ruído de observação** [fonte]: `ObservationTermCfg.noise` (`NoiseCfg | NoiseModelCfg`) por
  termo, ligado por grupo com `enable_corruption` (`observation_manager.py:26,87`;
  `ML/src/mjlab/utils/noise/noise_cfg.py`).
- **Atraso de observação** [fonte]: `delay_min_lag`/`delay_max_lag`, `delay_per_env` e
  `delay_hold_prob` por termo, mais `history_length` (`observation_manager.py:35-60`).
- **Atraso de ação** [fonte]: só no **atuador**, em passos de física, com `delay_min_lag`,
  `delay_max_lag` e `delay_hold_prob` (`ML/src/mjlab/actuator/actuator.py:67-112,323-347`). Atrasa
  o comando do atuador, que no nosso caso é a saída do controlador de navegação. A nossa latência é
  da **decisão** (target pose) em passos de controle, antes do controlador (ticket 15), e continua
  custom no `ActionTerm` [inferência].

## 5. Instalação

- **Python** [fonte]: `>=3.11,<3.15` (`ML/pyproject.toml:31`), e `.python-version` = 3.13. O nosso
  pede `>=3.12`, então são compatíveis.
- **torch/CUDA** [fonte]: `torch>=2.14.0`, com extras `cu130`, `cu126` e `cpu` mutuamente
  exclusivos e índices PyTorch explícitos (`ML/pyproject.toml:36,93-99,134-139`). O changelog diz
  que o `cu126` é para drivers < 580, e o nosso é 595, então `cu130` serve. O `warp-lang` vem do
  índice NVIDIA (`:134`). A FAQ recomenda CUDA ≥ 12.4 para grafos condicionais e "RTX 40-series
  GPUs (or newer)"; a nossa é uma 3060 (Ampere), que não está na recomendação.
- **Coexistência com o grupo `gpu`** [fonte + inferência]:
  - mjlab `main` exige `mujoco-core~=4.0.1`; nosso base exige `mujoco==4.0.0` → `mujoco-core==4.0.0`
    (PyPI `mujoco/4.0.0`). **Conflito**; resolve subindo o base para `mujoco==4.0.1` e o `gpu` para
    `mujoco-mjx[warp]==4.0.1` (pede `warp-lang==1.18.0`, igual ao lock do mjlab).
  - O `torch` do nosso grupo `rl` está em 2.13.0 (`uv.lock`), e o mjlab pede ≥ 2.14. Teria de subir.
  - `jax[cuda12]` traz wheels `nvidia-*-cu12`; o torch `cu130` traz CUDA 13. São nomes de pacote
    diferentes e devem coexistir num venv, mas não verifiquei. Num mesmo **processo**, o JAX
    pré-aloca 75 % da GPU por default; os dois stacks teriam de rodar em processos separados ou com
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` [inferência].
  - A release PyPI do mjlab (1.6.0) não serve (mujoco 3.11); seria `mjlab @ git+…@<commit>`.
- **Licença** [fonte]: Apache-2.0 (`ML/pyproject.toml:8`, `LICENSE`). `src/mjlab/utils/lab_api/` é
  BSD-3 (Isaac Lab) (`ML/README.md`, Third-Party Code).

## 6. Maturidade

- **Releases** [fonte: `git log` das tags]: v0.1.0 (2025-09-29), v1.0.0 (2026-01-28), v1.1.0
  (02-12), v1.2.0 (03-06), v1.3.0 (04-14), v1.4.0 (05-26), v1.5.0 (06-28) e v1.6.0 (08-08), mais
  patches 1.1.1 e 1.5.1–1.5.3. Isso dá ~1 minor por mês até agosto e **nenhuma release em 3
  meses**; o `main` tem 81 commits depois da v1.6.0. O classifier é "Production/Stable"
  (`ML/pyproject.toml`). CI diário com dependências na última versão (`ML/.github/workflows/nightly.yml`).
- **Issues relevantes** [fonte]:
  - #1263 (aberta, 2026-10-24): avisos repetidos de `ITERATIONS`/`LS_ITERATIONS` no env embutido.
    O mesmo bitmask que o nosso `overflow` usa para terminar o mundo (ticket 09).
  - #784 (fechada): penetração por overflow do buffer de contato só a 2048/4096 mundos.
  - #675 (fechada): sobre interop com JAX, o mantenedor respondeu "To be quite frank with you, I
    have no idea."
  - #100: o DirectEnv foi rejeitado.
- **Limitações documentadas** [fonte]: não há determinismo, porque o MuJoCo Warp não o garante
  (FAQ, `mujoco_warp#562`); o MuJoCo Warp é "still in beta" (FAQ "My training crashes with NaN
  errors"); `device="cpu"` ainda cria contexto CUDA (FAQ, #949).
- **Throughput publicado** [fonte: NIGHTLY, última entrada 2026-10-17, commit `45b1e26`, RTX 5090,
  4 096 envs, `decimation=4`, `num_steps=200`]. O método está em
  `ML/scripts/benchmarks/measure_throughput.py:83-119`, e "physics SPS" ali é **passos de env**
  por segundo:

  | task | physics (env-steps/s) | env (env-steps/s) | overhead |
  |---|---|---|---|
  | Mjlab-Velocity-Flat-Unitree-Go1 | 1 035 405 | 246 244 | 76.2 % |
  | Mjlab-Tracking-Flat-Unitree-G1 | 740 172 | 212 181 | 71.3 % |
  | Mjlab-Lift-Cube-Yam | 1 892 705 | 725 359 | 61.7 % |

  O PAPER (abstract, v2) não traz números de throughput. A FAQ só diz "on par or faster" que o
  Isaac Lab.
- [inferência] Esses números não são comparáveis aos nossos: outra GPU (5090 vs 3060), modelos
  sem meshes de 87 peças por robô, e 4 substeps por passo contra os nossos 48. O que se tira deles
  é que o overhead do manager (Python + torch por termo) come **62–76 %** do passo do env em
  tasks simples.

## 7. Veredito de encaixe

### 7.1 Por recurso

| recurso do mjsoccer | mjlab | nota |
|---|---|---|
| 6 tasks, cenas de 2 e 7 robôs, bola, campo | **native** | cena inteira por `SceneCfg.spec_fn` (§1.2a) |
| modelo base exato (sem variante) | **native** com (a); custom-ish com (b), por causa do mocap | §1.2 |
| `<option>` do `scene.xml` | **native** via `MujocoCfg` (setar `timestep`, `cone="elliptic"`) | defaults diferentes, §1.2 |
| ação de 7 dims → controlador de navegação a 40 Hz | **custom code** (`ActionTerm` torch) | §1.3 |
| kick/chip com plunger, `KICK_TABLE_MS`, trava | **custom code** | §1.3 |
| passo de disparo (`timestep` por mundo, 48 iterações) | **custom code** contra o framework (`opt.timestep` à mão + `create_graph`, ou `step()` próprio; `step_dt` errado) | §2.4; o maior risco de manutenção |
| `repeat_action` | **native** como `decimation`, mas acoplado ao item acima | §2.4 |
| posse por sensor `contact` + grace | sensor **native**, grace **custom** | §2.2 |
| robôs scripted | **custom code** (escrever `ctrl`) | §1.3 |
| `seat_ball` no reset | **custom code**, por redesenho: pool de estados assentados no `startup`, ou assentar no início do episódio | §1.3; `sim.step` não é por subconjunto de mundos |
| `sample_pose` + `reset_overlap` | **custom code** (evento `reset`) | |
| obs/recompensa/término das 6 tasks | **custom code** (termos torch), reescrevendo `mjx/tasks.py` | terceira implementação da lógica |
| NaN por mundo → terminal + reset | **native** (`nan_detection`); zerar obs/recompensa é custom ou `nan_policy="sanitize"` | §2.2 |
| overflow por mundo → terminal + contador | **custom code** (lê `sim.data.overflow`) | §2.2 |
| `naconmax` total | **custom** (o mjlab expõe só `nconmax` por mundo; dimensionar por mundo) | §2.2 |
| métricas `diverged`/`overflow`/`reset_overlap` | **native** (`MetricsManager`) | |
| DR por mundo com `set_const` | **native**, melhor que o atual (GPU, por episódio) | `pair_solref`/`solimp` custom |
| ruído de obs | **native** | |
| latência de ação em passos de controle | **custom code** (o atraso nativo é do atuador, em passos de física) | §4 |
| PPO com crítico assimétrico, LR por KL | **native** (RSL-RL) | |
| checkpoint, export | **native** (`.pt` + ONNX) | substitui Orbax |
| `--play` no backend CPU com Studio | **custom** (continua nosso) | |
| task-logic parity CPU × GPU | **custom**, refeito em torch | |
| coexistir com o grupo `gpu` (JAX) | **custom/decisão**: subir CPU e MJX para 4.0.1, torch ≥ 2.14, pin git do mjlab | §5 |
| multi-agente | não precisamos | §3 |
| **bloqueio** | **nenhum** encontrado no código | |

### 7.2 Custo × ganho

[inferência]

- **Ganha**: DR por episódio na GPU; ruído/atraso/histórico de obs prontos; crítico assimétrico;
  ONNX; viewer de lote (viser); multi-GPU; nada de cache de grafos JAX por época (o mjlab captura
  uma vez); torch para quem prefere; um ecossistema ativo, com o time do MuJoCo Warp por trás
  (README, Acknowledgments).
- **Perde/paga**: reescrever ~1 100 linhas JAX e o parity em torch, ou manter três
  implementações; lutar contra o `decimation` fixo no passo de disparo; o overhead de Python por
  substep em 48 substeps; dependência de um `main` não lançado; reabrir o ADR 0002 e o "Out of
  Scope" do spec; perder o eval embutido do Brax.
- **Resposta à pergunta**: o mjlab **consegue** hospedar as seis tasks, mas não como "config de
  manager". A maior parte do valor do mjsoccer (controlador, kick, posse, reset assentado) vira
  código nosso dentro de um `ActionTerm`/eventos, e o framework contribui com DR, RL e tooling. Só
  vale trocar se o protótipo de §8 mostrar decisões/s ≥ o atual, ou se DR por episódio e
  ruído/atraso nativos pesarem mais que a reescrita.

## 8. O que só um protótipo responde

Base de comparação (RTX 3060): ContestedPossession 16.4 k dec/s a 4 096 mundos no bench e
16.0–16.4 k no treino a 8 192; GoToBall 2.2 k dec/s (`.scratch/gpu-port-impl/10-bench/results.txt`;
issue 13 `## Comments`). O teto só-física é 918 k / 111 k passos/s com 2 / 7 robôs
(`.scratch/gpu-port/research/05-mjx-benchmark.md`, tabela principal).

1. **Decisões/s ponta a ponta no mjlab** (ContestedPossession e GoToBall, mesmos `nworld`), com
   `decimation=48` e o `ActionTerm` torch, separando física (`measure_physics_sps`) e env
   (`measure_env_sps`) como o script do mjlab faz.
2. **Overhead por substep**: quanto custam 48 × (`apply_actions` + `write_data_to_sim` +
   `scene.update` + graph launch) por decisão. Testar também o caminho 2 de §2.4 (24 + 24 só
   quando alguém dispara).
3. **Memória GPU e `nworld` máximo estável** com torch + Warp na 3060 (hoje 8 192 / 2 048).
   Separar o default do caching allocator do torch e o mempool do Warp.
4. **`opt.timestep` por mundo** sob o grafo capturado do mjlab: o mundo ocioso fica bit a bit nas
   iterações a `timestep=0`, como o smoke do ticket 08 exige?
5. **Parity**: os critérios de aceitação e as paridades determinística e com seed (ticket 12),
   mais o `hold_turn` (ticket 16), no `mujoco_warp` 4.0.1 do mjlab contra o CPU 4.0.1.
6. **Overflow por mundo**: o `Data.overflow` é limpo pelo `reset_data` mascarado do mjlab? (O
   ticket 09 diz que o Warp só zera a máscara no `reset_data`; o mjlab reseta por `reset_data`,
   `sim.py:507-519`.) E os bits `ITERATIONS`/`LS_ITERATIONS` disparam nas cenas de 7 robôs (#1263)?
7. **Instalação**: o `uv lock` com o mjlab no git, `mujoco 4.0.1`, `mujoco-mjx[warp] 4.0.1`,
   `jax[cuda12]` e torch `cu130` resolve num env só?
8. **Pool de estados assentados vs `seat_ball`**: a diversidade de starts fica igual à do CPU?
   (Esse pool já é o "upgrade" listado no Out of Scope do spec.)
9. **Qualidade de treino**: RSL-RL vs Brax PPO com o mesmo orçamento de decisões, no retorno de
   eval das duas tasks de referência.
