# Como randomizar o modelo por mundo pela API MJX? (research 03)

Ticket: [`issues/03-mjx-domain-randomization.md`](../issues/03-mjx-domain-randomization.md).
Data: 2026-11-08. Escopo: **capacidade** (que campos do `Model` podem variar por mundo, e como), não
parâmetros nem ranges.

## Fontes (clones de leitura, commits fixos)

| repo | commit | data do commit |
|---|---|---|
| `google-deepmind/mujoco` (inclui `mjx/` e `doc/`) | `371c2b4` | 2026-11-07 |
| `google-deepmind/mujoco_warp` | `4d1c64b` | 2026-11-06 |
| `google-deepmind/mujoco_playground` | `e7c53bc` | 2026-11-08 |
| `google/brax` | `c860529` | — |

Abreviações nas citações: `MJ` = `https://github.com/google-deepmind/mujoco/blob/371c2b4`,
`MJW` = `https://github.com/google-deepmind/mujoco_warp/blob/4d1c64b`,
`PG` = `https://github.com/google-deepmind/mujoco_playground/blob/e7c53bc`,
`BX` = `https://github.com/google/brax/blob/c860529`.

**Ressalva de versão.** As fontes são o `main` de novembro de 2026; o projeto roda `mujoco 3.11.0`
(medido no mapa). Nomes de campos e a tabela `_BATCH_DIM` do MJX-Warp são gerados automaticamente e
mudam entre versões. Reconferir tudo aqui contra as versões que o ticket de pins escolher.

---

## 1. MJX-JAX: `vmap` sobre campos do `mjx.Model`

### Mecanismo

- O `mjx.Model` é um pytree. **Campos `jax.Array` aceitam dimensão de lote**; a doc diz que é o
  jeito natural de expressar DR (`MJ/doc/mjx.rst:377-379`). **Campos `np.ndarray` são estruturais**:
  entram na chave do JIT, e mudá-los recompila. Exemplo da doc: `jnt_limited` (numpy) contra
  `jnt_range` (JAX, muda em runtime) (`MJ/doc/mjx.rst:380-384`).
- O padrão canônico, usado igual no Playground e no Brax, tem quatro passos:
  1. `jax.vmap` sobre um vetor de chaves gera `N` cópias dos campos escolhidos;
  2. `model.tree_replace({...})` põe esses arrays com eixo líder `N` no modelo;
  3. monta um pytree `in_axes` que vale `None` em tudo e `0` nos campos randomizados;
  4. `jax.vmap(step, in_axes=[in_axes, 0, 0])(model_v, state, action)`.

  Fontes: `PG/rss2025/mujoco_playground/_src/locomotion/go1/randomize.py:24-113`,
  `PG/rss2025/mujoco_playground/_src/wrapper.py:251-289`,
  `BX/brax/envs/wrappers/training.py:223-260`.
- **Quando se amostra:** o wrapper chama `randomization_fn` **uma vez, no `__init__`**
  (`PG/rss2025/.../wrapper.py:260`, `BX/brax/envs/wrappers/training.py:232`). O PPO do Brax passa
  `num_envs // device_count` chaves (`BX/brax/training/agents/ppo/train.py:99-103`). Cada slot de env
  fica com **um modelo fixo pelo treino todo**, e o auto-reset não reamostra. Para reamostrar por
  episódio, os campos teriam de viver no `State` e ser trocados no reset com `jnp.where`. *Não
  verificado:* não achei esse padrão nas stacks lidas.

### Quais campos podem variar (classificação por tipo em `MJ/mjx/mujoco/mjx/_src/types.py`, `class Model` na linha 558)

Campos `jax.Array` (podem ir no lote), entre eles os comuns em DR:

- `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`, `body_pos`, `body_quat`, `body_gravcomp`
- `geom_friction`, `geom_solref`, `geom_solimp`, `geom_solmix`, `geom_margin`, `geom_gap`, `geom_size`,
  `geom_pos`, `geom_quat`, `geom_rbound`, `geom_aabb`
- `pair_friction`, `pair_solref`, `pair_solreffriction`, `pair_solimp`, `pair_margin`, `pair_gap`
- `dof_frictionloss`, `dof_armature`, `dof_damping`
- `jnt_stiffness`, `jnt_range`, `jnt_solref`, `jnt_solimp`
- `actuator_gainprm`, `actuator_biasprm`, `actuator_gear`, `actuator_ctrlrange`, `actuator_forcerange`,
  `actuator_dynprm`
- `qpos0`
- os derivados `body_invweight0`, `dof_invweight0`, `tendon_invweight0`, `body_subtreemass`,
  `actuator_acc0`
- em `opt`: `timestep`, `gravity`, `density`, `viscosity`, `wind`, `tolerance`, `impratio`
  (`types.py:504-523`)

Campos `np.ndarray` (estruturais, **não** vão no lote): tipos, ids, endereços e flags (`*_limited`,
`geom_type`, `geom_contype`/`conaffinity`, `geom_condim`, `geom_dataid`), além de **toda a geometria
de mesh** (`mesh_vert`, `mesh_face`, …, `types.py:792`). Os inteiros de `opt` (`iterations`,
`ls_iterations`, `solver`, `cone`, `integrator`) também são estáticos.

Pré-cálculos privados do MJX-JAX feitos no `put_model`, que **não acompanham** mudanças em runtime
(`MJ/mjx/mujoco/mjx/_src/io.py:424-459`):

- `dof_hasfrictionloss = dof_frictionloss > 0` (numpy). Um `dof_frictionloss` que começa em zero não
  passa a ter efeito se for randomizado para > 0 (linha 425).
- `mesh_convex`: os hulls das meshes que colidem são pré-computados (linha 458). Escala ou forma de
  mesh não é randomizável no MJX-JAX.
- `geom_rbound_hfield`: cópia de `geom_rbound` usada em heightfield (linha 427).

**Relevante para o `mjsoccer`.** Os contatos bola × piso, chassi × bola e dribbler × bola são
`<pair>` explícitos (Notes do mapa). Pela semântica do MuJoCo, um `<pair>` "specify all their
properties explicitly … and the properties of the individual geoms are not used"
(`MJ/doc/XMLreference.rst:4280-4283`). Logo, DR de atrito e de `solref`/`solimp` nesses contatos tem
de mexer em **`pair_*`**, não em `geom_*`. Os dois são `jax.Array` no MJX-JAX (`types.py:834`) e
batcháveis no MJX-Warp (`'pair_friction': True`, `MJ/mjx/mujoco/mjx/warp/types.py:2283`).

### Quantidades derivadas que o `mj_setConst` recalcula

- A tabela oficial do que é seguro mudar está em `MJ/doc/programming/simulation.rst:756-905`
  ("mjModel changes"). As linhas mais comuns em DR:
  - `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`: "Safe with mj_setConst"; massa e inércia
    costumam ser escaladas juntas.
  - `body_pos`, `body_quat`: "Safe with mj_setConst", mas inseguro para corpos estáticos (BVH).
  - `dof_armature`, `geom_pos`, `geom_quat`, `geom_size`, `geom_margin`, `geom_gap`, `qpos0`: "Safe
    with mj_setConst".
  - `actuator_gainprm`, `actuator_biasprm`: "Safe" sem `mj_setConst`, mas o damping vindo de
    `dampratio` fica com o valor compilado.
  - `mesh_vert` e afins: "Unsafe for colliding meshes".
  - Atrito, `solref` e `solimp` (geom ou pair) não aparecem como exceção, então pela regra geral "real
    valued … safe" (`simulation.rst:770-772`) são seguros sem recálculo.
- O que o `mj_setConst` recalcula (lista do `set_const` do Warp, que replica o C,
  `MJW/mujoco_warp/_src/set_const.py:761-830`):
  - FIXED: `body_subtreemass`, `geom_aabb`, `geom_rbound`.
  - KIN0: `tendon_length0`, `eq_data` (âncoras), referências de câmera e luz.
  - INERT0: `dof_invweight0`, `body_invweight0`, `tendon_invweight0`, `actuator_acc0`.
- **O MJX (JAX ou Warp) não expõe `set_const`.** Não há `set_const`/`setconst` em
  `MJ/mjx/mujoco/mjx/` (grep no commit `371c2b4`). Quem randomiza massa no MJX-JAX fica com os
  `*_invweight0` **velhos**, a menos que os calcule por fora.
- Onde o derivado velho pesa no MJX-JAX: o `body_invweight0` entra na linha de restrição de **todo
  contato** (`MJ/mjx/mujoco/mjx/_src/constraint.py:480`), e também em equality e friction (linhas
  134, 217, 268, 382). Ele define a aproximação diagonal usada na regularização (`R`), ou seja, a
  "maciez" efetiva do contato. Massa ±10 % sem atualizar o `invweight0` muda essa maciez de forma
  inconsistente com a massa nova. *Não medido:* o tamanho do efeito no `mjsoccer`.
  `geom_rbound` só é lido no cull de broadphase (`collision_driver.py:427`). Não achei leitura de
  `body_subtreemass` em `_src/` fora de `types.py` (grep).
- **Prática das stacks:** o Playground (Go1, Berkeley humanoid, G1, T1) escala `body_mass` ±10 %, soma
  massa ao torso, mexe em `body_ipos` e `dof_armature`, e **não** recalcula `body_inertia` nem
  `*_invweight0` (`PG/rss2025/.../go1/randomize.py:47-66`). O T1 randomiza `actuator_gainprm[:,0]` e
  ajusta `actuator_biasprm[:,1] = -kp` (`PG/rss2025/.../t1/randomize.py:70-76`). Ou seja, o padrão de
  fato aceita derivados velhos.
- **Alternativa correta e barata (medida):** para cada variante, no host, escrever os campos
  primários no próprio `MjModel`, chamar `mujoco.mj_setConst(m, d)`, copiar os derivados e empilhar.
  Os arrays empilhados entram por `in_axes` como qualquer outro campo. Medido no Ryzen 5 5600X,
  `mujoco 3.11.0`, modelo cinemático, script em
  `/tmp/claude-1000/-home-mateus-doc-ssl-rl/716cbd60-1120-41e0-842f-d53c5add287a/scratchpad/scripts/cost2.py`:
  - cena de 7 robôs: **0,28 ms por variante** (≈ 1,1 s para 4096);
  - cena de 2 robôs: 0,10 ms por variante.

  Copiar o `MjModel` é que é caro (`copy.copy` = 170 ms na cena de 7 robôs, por causa das 1088
  meshes): faça in-place num modelo só e restaure no fim. *Ressalva:* o `mj_setConst` acorda árvores
  dormentes e usa o `MjData` como workspace (`simulation.rst:820-823`). Inofensivo aqui, porque o
  `MjData` é descartável.

---

## 2. MJX-Warp: campos com dimensão de lote

### Como o lote funciona no MuJoCo Warp nativo

- Muitos campos do `mjw.Model` têm eixo líder de lote, **1 por padrão**, e são indexados por
  `field[worldid % field.shape[0]]` (`MJ/doc/mjwarp/index.rst:512-525`). Isso aparece em 482 pontos de
  kernel em `MJW/mujoco_warp/_src/*.py`, por exemplo `collision_convex.py:89-95` e `bvh.py:203`.
- `mjw.put_model(mjm, batch_sizes={"campo": K})` aloca o eixo líder `K`. A validação só aceita campos
  cujo spec de array começa com `"*"` (`MJW/mujoco_warp/_src/io.py:415-439`, `io.py:1684-1688`).
- **Consequência do módulo:** `K` não precisa ser igual a `nworld`. Com `K` variantes e `N` mundos, o
  mundo `w` usa a variante `w % K`.

### Pela ponte MJX (`impl="warp"`)

- `mjx.put_model(..., impl="warp", batch_sizes=...)` repassa `batch_sizes` ao MJW. Campos batcháveis
  *fora* de `batch_sizes` têm o eixo 1 removido, para o `mjx.Model` ter as mesmas shapes do MJX-JAX
  (`MJ/mjx/mujoco/mjx/_src/io.py:470-520`, em especial 500-517). O teste
  `MJ/mjx/mujoco/mjx/_src/io_test.py:380-414` cobre `batch_sizes={'mat_texid': nworld}`.
  `geom_dataid` em lote não é suportado no MJX (`io.py:506-508`).
- Na chamada FFI, cada campo é expandido até o `ndim` esperado pelo Warp
  (`MJ/mjx/mujoco/mjx/warp/ffi.py:265-280`): um campo sem lote vira `shape[0] == 1`.
  - Com `vmap`, o `marshal_custom_vmap` (`ffi.py:411-480`) achata as dimensões de lote do modelo para
    o eixo líder (`_flatten_batch_dim`, `ffi.py:322-332`).
  - Por isso **o mesmo `randomize` com `in_axes` do MJX-JAX funciona no MJX-Warp**: os campos com
    `in_axes=0` chegam ao Warp com eixo líder `N`, e o resto com 1.
  - Há também um segundo padrão, só Warp: passar o campo já com eixo líder e `in_axes=None`. O teste
    `MJ/mjx/mujoco/mjx/warp/smooth_test.py:307-357` dá a `geom_pos` shape `(7, ngeom, 3)` e chama
    `vmap(kinematics, in_axes=(None, 0))`. Esse padrão quebra o MJX-JAX, então não serve para uma
    implementação única.
- **Quais campos têm lote no MJX-Warp.** Tabela gerada `_BATCH_DIM['Model']` em
  `MJ/mjx/mujoco/mjx/warp/types.py:1781-2391`: 133 campos `True` e 476 `False` (contagem por script).
  - `True` inclui tudo de DR comum:
    - massa e inércia: `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`, `body_pos`,
      `body_quat`, `body_subtreemass`, `body_invweight0`;
    - contato: `geom_friction`, `geom_solref`, `geom_solimp`, `geom_solmix`, `geom_margin`,
      `geom_gap`, `geom_size`, `geom_rbound`, `geom_aabb`, e `pair_friction`, `pair_solref`,
      `pair_solreffriction`, `pair_solimp`, `pair_margin`, `pair_gap`;
    - juntas: `dof_armature`, `dof_damping`, `dof_frictionloss`, `dof_invweight0`, `jnt_range`,
      `jnt_stiffness`, `jnt_solref`, `jnt_solimp`;
    - atuadores: `actuator_gainprm`, `actuator_biasprm`, `actuator_gear`, `actuator_ctrlrange`,
      `actuator_forcerange`, `actuator_dynprm`, `actuator_acc0`, `actuator_delay`;
    - `qpos0`, e em `opt`: `opt__timestep`, `opt__gravity`, `opt__tolerance`, `opt__ls_tolerance`,
      `opt__impratio_invsqrt`, `opt__density`, `opt__viscosity`, `opt__wind`.
  - `False`: ids, tipos, endereços, flags, `jnt_limited` (linha 2095), `mesh_vert` (2165), e os
    inteiros de `opt` (`iterations`, `ls_iterations`, `solver`, `cone`, `integrator`).

### O que muda na GPU sem recompilar, e o que exige recompilar o `MjSpec`

- **Muda sem recompilar o `MjSpec`:** qualquer campo `True` acima. Os valores são buffers comuns na
  chamada FFI, e trocar valores não muda a assinatura.
  - Trocar a **shape** (de 1 para `N`, ou outro `K`) muda a assinatura e força retrace do JIT/FFI. A
  chave do cache do callable é estrutural (`ffi.py:139-157`). *Inferido do código, não medido.*
  - No modo de grafo `WARP` (default), o grafo CUDA é recapturado se os ponteiros de buffer do XLA
  mudarem, e "excessive graph recaptures … will degrade performance". `WARP_STAGED` e
  `WARP_STAGED_EX` evitam isso com buffers de staging (`MJ/doc/mjx.rst:151-181`). Se os campos
  randomizados forem reamostrados por episódio como novos arrays JAX, *não verificado* se o XLA mantém
  os ponteiros estáveis. Medir com o modo `WARP_STAGED` se o SPS cair.
- **Derivados no Warp nativo:** o `mjw.set_const(m, d, groups=...)` existe e roda na GPU
  (`MJW/mujoco_warp/_src/set_const.py:761-865`). Ele cobre FIXED, KIN0 e INERT0. SPRING não é
  suportado, e `dof_length`, `flexedge_*` não têm lote (docstring, linhas 818-821). A docstring lista
  os campos seguros: `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`, `body_pos`, `body_quat`,
  `dof_armature`, `actuator_armature`, `actuator_gear`, `geom_size`, `qpos0`, entre outros (linhas
  773-791).
- **Issue 893** (`https://github.com/google-deepmind/mujoco_warp/issues/893`, "mj_setConst"): aberta
  em 2025-12-09, **fechada** em 2026-06-24, com checklist `setFixed`/`set0` (#1086), `setStat`
  (#1122) e `setSpring` marcados (lido via `gh issue view`).
  - *Discrepância:* a docstring do `set_const` no `4d1c64b` ainda diz "SPRING group unsupported in
    mjwarp" (`set_const.py:821`, `types.py:965`).
  - **Esse `set_const` não está exposto pela ponte MJX** (grep em `MJ/mjx/mujoco/mjx/`). Pela API MJX,
    derivados continuam sendo calculados no host (via `mj_setConst`, seção 1) ou ignorados.
- **O que exige recompilar o `MjSpec` no host:** tudo que é estrutural (`False`). Isso inclui
  geometria de mesh, número de geoms ou corpos, `contype`/`conaffinity` para colidir onde antes não
  colidia (`simulation.rst:862-866`), e campos que o compilador deriva de convenções (`dampratio`,
  `springdamper`, estatísticas, `simulation.rst:811-818`).
  - A doc recomenda "modify the corresponding mjSpec and then compile … 1 call per new field
    instance, i.e., nworld host calls" (`MJ/doc/mjwarp/index.rst:527-537`).
  - Para variar mesh por mundo no Warp nativo há o fluxo "per-world assets": compilar `K` variantes,
    compilar uma base com todos os assets, e sobrescrever `geom_dataid`, `geom_size`, `geom_aabb`,
    `geom_rbound`, `geom_pos`, `body_mass`, `body_subtreemass`, `body_inertia`, `body_invweight0`,
    `body_ipos`, `body_iquat` com arrays por mundo. "MJWarp … does not provide utilities for
    construction of dependent per-world field variants" (`index.rst:539-831`).
  - **Esse fluxo não passa pela ponte MJX**, porque `geom_dataid` em lote não é suportado
    (`io.py:506-508`).
- **Custo de recompilar o `MjSpec` por variante (medido,** mesmo host e versão; script `cost3.py` no
  scratchpad**):**

  | cena | `spec.compile()` (recompilação) | 4096 variantes, 1 core |
  |---|---|---|
  | 7 robôs | ~277 ms (primeira compilação: 492 ms) | ~19 min |
  | 2 robôs | ~106 ms | ~7 min |

  A isso se soma o `put_model` de cada variante, *não medido*. Para campos reais, a alternativa
  "editar o `MjModel` in-place + `mj_setConst`" custa ~0,28 ms por variante, uns três ordens de
  grandeza menos.

---

## 3. Como o MuJoCo Playground implementa DR com `impl="warp"`

- **Estado do repo:** o Playground atual foi reescrito em **PyTorch sobre `reaf-sim` + MuJoCo Warp**.
  Os ambientes JAX/MJX originais "remain available under `rss2025`" (`PG/README.md:9` e `:53`, nota após a
  tabela). O que interessa a um port em JAX é o `rss2025/`.
- **`rss2025` (JAX):** um único `domain_randomize(model, rng) -> (model_v, in_axes)` serve aos dois
  motores.
  - O Go1 Joystick declara `impl="warp"`, `naconmax=4*8192` e `njmax=40`
    (`PG/rss2025/.../go1/joystick.py:92-94`) e usa o mesmo `go1_randomize.domain_randomize`, que é só
    `vmap` + `tree_replace` + `in_axes` (`go1/randomize.py:24-113`).
  - O script de treino tem a flag `--impl {jax,warp}` (`PG/rss2025/learning/train_jax_ppo.py:79`) e
    pega o randomizer do registry (linha 379).
  - O `BraxDomainRandomizationVmapWrapper` não sabe qual é o motor (`PG/rss2025/.../wrapper.py:251-289`).
  - **Não há nada específico de Warp no DR do `rss2025`.** A ponte MJX (seção 2) é que transforma o
    eixo do `vmap` em eixo líder do Warp.
  - Sem `set_const`, e com amostragem única no `__init__`.
- **Playground atual (torch, para contraste, fora do escopo do port):** o
  `reaf_sim/.../common/model_randomization.py` randomiza por padrão de nome com `env_mask`, ou seja,
  reamostra por mundo no reset, na GPU (linhas 31-113). Ele tem pós-processadores:
  - `geom_size` recalcula `rbound` e `aabb` só para primitivas (linhas 165-227);
  - `body_mass` escala `body_inertia` junto (linhas 230-252).

  O `TorchWrapper` faz duas coisas (`reaf_sim/.../core/torch_wrapper.py:14-20, 534-552`):
  - na primeira escrita, faz o *tile* do campo de 1 para `nworld` e **dispara a recaptura do grafo
    CUDA**;
  - marca o `SetConstLevel` pendente, que o `Simulator.set_const()` executa com `mjw.set_const`
    (`reaf_sim/.../core/simulator.py:304-335`).

  Em resumo: no Warp nativo, a stack de referência usa `set_const` na GPU e paga uma recaptura de grafo
  só na expansão do lote.

---

## 4. Latência de atuação e ruído de observação (DR do lado da task)

- **Ruído de observação:** no `rss2025`, cada task soma ruído uniforme às observações dentro do
  `_get_obs`, escalado por `noise_config.level` × `scales.<termo>`. A chave vem de
  `state.info["rng"]`, dividida a cada passo (`PG/rss2025/.../go1/joystick.py:42-43, 315-347`). Fica
  fora do `Model` e é JAX puro, então vale igual para `jax` e `warp`.
- **Latência de atuação:** não achei latência de ação nos ambientes `rss2025` lidos (grep por
  `latency|delay` em `rss2025/`). O que há é `last_act`/`last_last_act` em `info`, usado só em
  recompensas de suavidade (por exemplo `spot/joystick.py:194-195, 258-259, 489-490`).
  - O Playground atual (torch) implementa as duas latências como processadores da task, num buffer
    circular por env com atraso sorteado em `[min_lag, max_lag]` substeps (`DelayBuffer`,
    `reaf_sim/.../common/delay_buffer.py:15-60`):
    - comando: `StochasticSubstepCommands` com `delay=(0.005, 0.010)` s ("serial bus latency");
    - medida: `StochasticSubstepMeasurements`.

    Uso em `PG/mujoco_playground/tasks/manipulation/pick_cube/trossen_wxai/task.py:96-124`.
  - Em JAX, o equivalente é um ring buffer no `State.info` com índice por mundo. *Não verificado:* não
    há implementação de referência em JAX nas fontes lidas.
- **Alternativa nativa (registrar, não decidir):** o MuJoCo agora tem atraso de atuador no próprio
  modelo, `actuator/@nsample` (ring buffer do `ctrl`) e `@delay` (`MJ/doc/XMLreference.rst:5700-5729`).
  - No MJX-Warp, `actuator_delay` é campo com lote (`MJ/mjx/mujoco/mjx/warp/types.py:1814`;
    `MJW/mujoco_warp/_src/types.py:2174`; implementação em `MJW/mujoco_warp/_src/history.py`).
  - No MJX-JAX, `actuator_delay` **não existe** no `mjx.Model` (ausente de `_src/types.py`), e nenhum
    código em `MJ/mjx/mujoco/mjx/_src/*.py` fora dos testes o referencia. *Não verificado* se o
    `put_model(impl="jax")` rejeita um modelo com `nsample > 0` ou ignora o atraso.
  - Por isso, para uma implementação única nos dois motores (restrição 6), a latência fica do lado da
    task.

---

## Achados laterais para outros itens do mapa

- **Kick num lote (Not yet specified):** `opt.timestep` é `jax.Array` no MJX-JAX
  (`MJ/mjx/mujoco/mjx/_src/types.py:523`) e `opt__timestep` tem lote no MJX-Warp
  (`warp/types.py:2276`). Um `timestep` **por mundo** via `in_axes` é suportado pelas duas
  representações.
  - *Não verificado:* o comportamento numérico disso. O `refsafe` do MJX-JAX limita
    `timeconst >= 2*timestep` por linha de restrição (`_src/constraint.py:65-66`), então o `solref`
    efetivo muda junto com o substep.
- **Reset parcial:** a ressalva de `jnp.where` no MJX-Warp vale para `mjx.Data`
  (`MJ/doc/mjx.rst:135-140`). O erro do FFI sugere `Data.where` em vez de `jax.tree.map`
  (`warp/ffi.py:376-405`). Campos do `Model` não são lidos e escritos pelo Warp; *não verificado* se
  trocá-los com `jnp.where` no reset tem a mesma restrição.

## Resumo para o spec (capacidade)

1. **Mecanismo único:** `randomize(model, rng) -> (model_v, in_axes)` + `jax.vmap(..., in_axes=...)`.
   Funciona igual em `impl="jax"` e `impl="warp"`, sem `batch_sizes`.
2. **Campos candidatos:** os reais que são `jax.Array` no MJX-JAX **e** `True` no `_BATCH_DIM` do
   MJX-Warp. Para contatos do `mjsoccer`, isso significa `pair_*`.
3. **Derivados** (`*_invweight0`, `body_subtreemass`, `geom_rbound`/`aabb`, `actuator_acc0`): ou
   ignorar, como o Playground, ou calcular no host com `mj_setConst` in-place (~0,3 ms por variante)
   e empilhar. O MJX não expõe `set_const`.
4. **Estruturais** (mesh, contype, número de geoms, inteiros de `opt`): só por recompilação do
   `MjSpec` (~0,1–0,3 s por variante). Isso fica fora da capacidade de DR por mundo pela API MJX.
5. **Latência e ruído:** do lado da task, em `State.info`.
