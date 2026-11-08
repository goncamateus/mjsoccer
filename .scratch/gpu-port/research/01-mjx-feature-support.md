# 01 — O MJX roda o nosso modelo? Versões e suporte, nos dois motores

Ticket: `.scratch/gpu-port/issues/01-mjx-feature-support.md`. Pesquisado em 2026-11-08 só em fonte
primária: doc e código-fonte do MuJoCo/MJX, código do MuJoCo Warp, issues e PRs upstream e metadados
do PyPI. **Nada foi instalado, compilado nem executado.** Os números de memória abaixo são **contas
feitas a partir do código-fonte, não medições**. Tudo que precisa de medição está listado no fim (§10)
como entrada para os tickets de instalação e de benchmark.

## Fontes

| repo | ref | data | abreviação |
|---|---|---|---|
| google-deepmind/mujoco, **tag `3.11.0`** (a versão instalada), lido por `gh api …/contents?ref=3.11.0` | `3.11.0` | 2026-07-27 | `T/` = `https://github.com/google-deepmind/mujoco/blob/3.11.0/` |
| google-deepmind/mujoco, `main` (clone raso) | `371c2b4` | 2026-11-07 | `MJ/` = `https://github.com/google-deepmind/mujoco/blob/371c2b4/` |
| google-deepmind/mujoco_warp, `main` (clone raso) | `4d1c64b` | 2026-11-06 | `MJW/` = `https://github.com/google-deepmind/mujoco_warp/blob/4d1c64b/` |
| google-deepmind/mujoco_playground (`rss2025/`) | `e7c53bc` | 2026-11-08 | `PG/` |
| PyPI JSON API (`https://pypi.org/pypi/<pkg>/<ver>/json`) | — | lido em 2026-11-08 | — |

O MJX-Warp **não usa o pacote `mujoco-warp`**: o `mujoco-mjx` carrega uma cópia própria do MuJoCo
Warp em `mujoco/mjx/third_party/mujoco_warp` e importa essa cópia (`T/mjx/mujoco/mjx/warp/__init__.py:36-38`).
Na tag `3.11.0` a cópia é o `mujoco-warp` **3.10.0.1**
(`T/mjx/mujoco/mjx/third_party/mujoco_warp/pyproject.toml:7`). Por isso o comportamento do MJX-Warp
abaixo vem dessa cópia (`T/mjx/mujoco/mjx/third_party/mujoco_warp/_src/…`, abreviado `TW/`), e não do
`main` do repositório `mujoco_warp`.

Fatos do modelo, conferidos à mão nos arquivos (não reproduzem medição):
`models/scene.xml:4` (`<option>`: `implicitfast`, Newton, 100/50 iterações, `elliptic`, sem
`tolerance`, que fica no default 1e-8), `models/ball.xml` (freejoint, inércia isotrópica, 13 pares),
`models/scene.xml:20-23` (2 pares bola × dribbler), `models/robot_kinematic.xml:150-381`.

---

## Resposta curta (bloqueios primeiro)

1. **Bloqueio do MJX-JAX: a colisão mesh↔mesh entre robôs, não o cilindro.** O MJX-JAX aloca um
   contato fixo para cada par candidato, 4 contatos por par mesh↔mesh e linhas de restrição para
   todos eles. Também gera uma chamada separada por **par de ids de mesh**. Como o `scene.build` cria
   cada robô com um `MjSpec.from_file` novo e prefixado, os assets de mesh devem ser duplicados por
   robô (*inferido, não verificado*). Isso daria ~105 mil grupos na cena de 7 robôs e ~5 mil na de 2.
   Pela conta do código, o `efc_J` chegaria a ~310 MB **por mundo** com 7 robôs e a ~9 MB por mundo
   com 2 robôs (§4). Nos dois casos a cena não cabe na 3060 com milhares de mundos, e o compile em XLA
   provavelmente explode a RAM do host. A variante de modelo do MJX-JAX precisa cortar as colisões
   chassi↔chassi (por `contype`/`conaffinity` ou trocando-as por primitivas), não só trocar os
   cilindros do dribbler.
2. **A premissa "MJX-JAX não colide sphere↔cylinder" está desatualizada no código.** A doc ainda
   diz que não (`T/doc/mjx.rst:539`). Mas desde out/2024 o `collision_driver` tem
   `(SPHERE, CYLINDER): sphere_cylinder` (`T/mjx/mujoco/mjx/_src/collision_driver.py:98`). A função é
   uma otimização SDF com 10 passos de gradiente, não analítica (`T/…/collision_sdf.py:214-218`,
   `:198`). O `put_model` só rejeita pares sem função (`T/…/_src/io.py:353-359`). Logo, o modelo base
   *deve* carregar no MJX-JAX. A precisão desse contato no nosso caso (bola contra a lateral de um
   cilindro de r = 9 mm) não foi testada upstream: o único teste é uma esfera sobre a face plana. A
   variante cápsula continua sendo a opção segura, mas virou escolha e não obrigação.
3. **Versões:** `mujoco 3.11.0` ↔ `mujoco-mjx 3.11.0` (exige `mujoco>=3.11.0.dev0`), e o extra
   `[warp]` **fixa `warp-lang==1.14.0`**. O `mujoco-warp` separado é desnecessário. O JAX não tem pin.
   O driver 595 e a sm_86 servem a CUDA 12 e 13, tanto no JAX quanto no Warp. Atenção: o
   `pyproject.toml` do `rss2025` no `main` pede `warp-lang>=1.18.0`, o que conflita com o pin. O
   `playground 0.2.0` publicado no PyPI pede `warp-lang>=1.11` e não conflita (§1).
4. **`<pair>` condim 4/6 + elíptico, Newton, juntas slide/hinge/free, atuadores `motor`/`velocity`,
   sensores `jointpos`/`jointvel`/`actuatorfrc`:** suportados nos dois motores. Não usamos `condim` 1
   em lugar nenhum, só 3, 4 e 6 (§2, §3).
5. **`implicitfast`:** a ressalva do MJX-Warp ("midpoint") é sobre o tratamento especial de corpos
   livres isolados, o que no nosso caso é **a bola**. No `mujoco 3.11.0` a CPU trocou o midpoint por
   derivadas giroscópicas, que nem o MJX-JAX nem o MJX-Warp 3.11 implementam. Para uma esfera de
   inércia isotrópica com o CoM na origem o termo giroscópico é zero, então o efeito esperado é nulo
   (*inferido*) (§2).
6. **float32:** o MJX-Warp é sempre float32. O MJX-JAX é float32 por default e vai a float64 com
   `jax_enable_x64`. Com `tolerance` 1e-8, o Newton do MJX-JAX em float32 provavelmente nunca para
   cedo e roda as 100 iterações (*inferido*). O Warp limita a tolerância a ≥ 1e-6 (§5).
7. **Posse:** um sensor `contact` com casamento `geom1=ball`/`geom2=<dribbler>` serve aos dois motores.
   O MJX-JAX **não aceita casamento por `body`** (`T/…/_src/io.py:333-337`), então são 2 sensores por
   robô (barra e eixo), com `data="found"` (§6).
8. **`naconmax`/`njmax`:** só existem no Warp. O `naconmax` é a soma de todos os mundos, e o `njmax`
   vale por mundo. O default sem argumento é ~48 contatos no total, o que é inviável. Dimensionar a
   partir de `ncon`/`nefc` **máximos** medidos na CPU. O custo de memória no Warp é pequeno: dezenas
   de MB para 4096 mundos, pela conta (§7).
9. **Reset parcial no Warp:** `data.where(done, reset_data)` (`T/…/_src/types.py:1170`). O
   `where_done` do Playground acerta por formato (§8).
10. **`opt.timestep` por mundo:** é aceito nos dois motores (`T/…/warp/types.py:1894`; kernels do
    Warp leem `opt_timestep[worldid % shape]`, `TW/forward.py:69`) (§9; reaproveita o 03).

---

## 1. Versões compatíveis

**PyPI, 2026-11-08** (`requires_dist` de cada release):

| pacote | versão | requer |
|---|---|---|
| `mujoco-mjx` | 3.11.0 (2026-07-28) | `mujoco>=3.11.0.dev0`, `jax`, `jaxlib` (sem pin), `scipy`, `trimesh`, `etils[epath]`; `[warp]`: **`warp-lang==1.14.0`**; Python ≥ 3.10 |
| `mujoco-mjx` | 3.12.0 / 3.13–3.15 / 3.16–4.0.0 | `warp-lang==1.16.0` / `==1.17.0` / `==1.18.0`; 4.0.0 passa a depender de `mujoco-core` e Python ≥ 3.11 |
| `mujoco-warp` | 3.11.0 | `mujoco>=3.9.0`, `warp-lang>=1.14`; `[cuda]` = `jax[cuda12]` |
| `warp-lang` | 1.14.0 (2026-06-01) | só `numpy`; wheel `manylinux_2_28_x86_64` |
| `jax` | 0.11.0 (2026-07-16) … 0.12.0.1 (2026-10-21) | — |
| `playground` (PyPI) | 0.2.0 (2026-03-16) | `brax>=0.14.2`, `jax`, `mujoco-mjx>=3.6.0`, `mujoco>=3.6.0`, **`warp-lang>=1.11`**, Python ≥ 3.11 |
| `playground` (PyPI) | 1.0.0 (2026-11-06) | `mujoco-warp>=4.0.0`, `mujoco[sysid]>=4.0.0`, `warp-lang>=1.18.0,<1.18.1` (a reescrita em torch, ver 02) |

- **Combinação para `mujoco 3.11.0`:** `mujoco==3.11.0`, `mujoco-mjx[warp]==3.11.0` (que puxa
  `warp-lang==1.14.0`), e `jax[cuda12]` ou `jax[cuda13]`. O `mujoco-mjx` e o `mujoco` andam em
  lockstep de versão: o pin `>=3.11.0.dev0` aceita `mujoco` mais novo, mas o `mjx` foi escrito contra o
  mesmo número. Manter os dois iguais.
- **JAX: sem pin upstream.** Não achei registro de qual JAX o CI do `mjx 3.11.0` usou. **Não
  verificado** se o `mjx 3.11.0` ou o FFI JAX do `warp-lang 1.14.0` funcionam com o `jax 0.12.x`.
  Proposta para o ticket de instalação: começar pelo JAX contemporâneo (`jax 0.11.0`, publicado 12
  dias antes do `mjx 3.11.0`) e só subir se testar.
- **Driver 595 e RTX 3060 (sm_86, 12 GB; `nvidia-smi` local).** JAX: CUDA 12 pede driver ≥ 525 e
  SM ≥ 5.2; CUDA 13 pede driver ≥ 580 e SM ≥ 7.5 ([docs.jax.dev/installation](https://docs.jax.dev/en/latest/installation.html),
  lido em 2026-11-08). Warp: builds `+cu12` pedem driver R525+; builds CUDA 13 pedem R580+ e Turing ou
  mais nova ([Warp installation guide](https://nvidia.github.io/warp/stable/user_guide/installation.html),
  página da 1.19). O wheel PyPI da 1.14.0 remete os "CUDA 13 builds" para o guia (descrição PyPI da
  1.14.0), então o wheel default da 1.14 é provavelmente CUDA 12 (*inferido*). As duas variantes rodam
  com o driver 595.
- **`mujoco-warp` separado não é dependência.** Ver "Fontes". Instalá-lo não muda o MJX-Warp. Só
  faria sentido para usar a CLI `mjwarp-testspeed --measure_alloc` no dimensionamento do §7, e nesse
  caso `mujoco-warp==3.11.0` aceita `warp-lang>=1.14`.
- **Conflito do Playground.** O `PG/rss2025/pyproject.toml:34` no `main` pede `warp-lang>=1.18.0`.
  Isso conflita com o `mujoco-mjx[warp]==3.11.0`, e o ticket 02 citou esse arquivo. O release PyPI
  `playground==0.2.0`, que é o `rss2025` publicado, pede `warp-lang>=1.11` e fecha com a 1.14.0. Para
  copiar o wrapper (recomendação do 02) não importa. Para instalar o pacote, usar o
  `playground==0.2.0` do PyPI, e não o checkout.
- **Python:** o projeto pede ≥ 3.12 (`pyproject.toml:9`), o que é compatível com todos os pacotes acima.

## 2. Contato, cone, condim, integrador e solver

### `<pair>` com condim 4 e 6, cone elíptico

- A tabela de Feature Parity (`T/doc/mjx.rst:478-499`) lista Cone `PYRAMIDAL`/`ELLIPTIC` nos dois
  motores; Condim "All" no Warp e "1, 3, 4, 6 (1 is not supported with `ELLIPTIC`)" no JAX.
- **No código do MJX-JAX:** o `make_data` levanta `NotImplementedError` se o cone for elíptico e
  existir algum contato com `dim == 1` (`T/mjx/mujoco/mjx/_src/io.py:676-679`). Os pares explícitos
  usam `pair_dim` e os dinâmicos usam o `condim` do geom de maior `priority`, ou o máximo dos dois
  (`T/…/_src/collision_driver.py:221-229`). As linhas elípticas são `condim` linhas por contato
  (`T/…/_src/constraint.py:661-662`).
- **Não usamos `condim` 1.** Nos três arquivos só há `condim` 3, 4 e 6: `field.xml` 29×3;
  `robot_kinematic.xml` 71×3 e 18×4; `ball.xml` 2×4 (geom da bola e classe `ball_wall`) e 1×6 (par
  bola × piso); `scene.xml` 2×4 (pares do dribbler). Os contatos dinâmicos herdam max(3, 4) = 4 para
  bola × chassi/kicker e 3 para robô × robô/parede/piso. Conjunto final: {3, 4, 6}.
- **Pares:** o MJX-JAX consome `<pair>` primeiro e não duplica o par dinâmico equivalente
  (`T/…/collision_driver.py:138-145`, `:203-205`). Os parâmetros do contato vêm de `pair_*`
  (`MJ/mjx/mujoco/mjx/_src/collision_driver.py:275`). No MJX-Warp os pares entram no `nxn_pairid`
  (`TW/io.py:590-599`).
- **Margem:** o MJX-JAX rejeita `margin`/`gap` ≠ 0 em pares com mesh (`T/…/_src/io.py:361-370`).
  Não usamos margem (nenhum `margin`/`gap` em `models/` nem em `scene.py`).

### Newton

Suportado nos dois: o JAX tem `CG`, `NEWTON`; o Warp tem "All except `PGS`, `noslip`"
(`T/doc/mjx.rst:498-500`).

### `implicitfast` e a ressalva "midpoint"

- A tabela: no Warp, "All except `IMPLICITFAST` midpoint integrator feature"; no JAX, `IMPLICITFAST`
  "not supported with fluid drag" (`T/doc/mjx.rst:489-491`). O README do MuJoCo Warp repete a frase
  (`MJW/README.md:120`). Não temos fluido (`density`/`viscosity` = 0), então a restrição do JAX
  (`T/…/_src/io.py:269-272`) não se aplica.
- **O que é o "midpoint".** No MuJoCo 3.7.0 entrou a "midpoint integration for standalone free
  bodies in `implicit` and `implicitfast`": regra do ponto médio implícita na rotação de corpos livres
  sem filhos (`MJ/doc/changelog.rst:1806-1810`). No 3.8.1 ela foi restrita ao `implicitfast` e
  desligada com fluido (`:1704-1708`). **No 3.11.0 a CPU substituiu o midpoint por derivadas
  giroscópicas**, "making `implicitfast` identical to `implicit` for such bodies", o que amortece de
  leve o tombamento (`MJ/doc/changelog.rst:1380-1386`, seção 3.11.0 em `:1348`). O PR do midpoint no
  Warp foi fechado sem merge ([mujoco_warp#1316](https://github.com/google-deepmind/mujoco_warp/pull/1316),
  fechado em 2026-04-30). O termo giroscópico só entrou no Warp em 2026-10-12
  ([#1669](https://github.com/google-deepmind/mujoco_warp/pull/1669); `MJW/mujoco_warp/_src/forward.py:2117`,
  `implicit_free_body` em `:2286`), depois da cópia que vem no `mjx 3.11.0`. Essa cópia não tem nada de
  midpoint nem de giroscópico (busca por `gyroscop|free_body|midpoint` em `TW/forward.py` sem
  resultado).
- **O MJX-JAX também não tem:** o `implicit` monta `M − h·qDeriv` só com as derivadas de atuador
  (bias/gain afim) e de amortecimento (`T/…/_src/forward.py:412`, `T/…/_src/derivative.py:34-73`,
  `# TODO(team): rne derivative` em `:71`).
- **Isso nos afeta?** O único corpo livre isolado é a bola: `freejoint`, sem filhos, `diaginertia`
  isotrópica `1.063175e-05` × 3 e `inertial pos="0 0 0"` (`models/ball.xml:11-14`). O robô não é
  corpo livre (slide + slide + hinge). Para inércia isotrópica, ω × (Iω) = c·(ω × ω) ≡ 0 e a derivada
  em ω também é zero. Então o bloco giroscópico da CPU 3.11 vira só `M − h·D`, igual ao que o MJX
  faz. **Efeito esperado nulo** (*inferido da matemática, não testado*). As outras derivadas que o
  `implicitfast` usa (o `kv` dos atuadores `velocity` e o `damping` das juntas) os dois motores
  implementam.

## 3. Juntas, atuadores, `equality`/`tendon` e sensores dos três MJCF

| o que o modelo declara | onde | MJX-JAX | MJX-Warp |
|---|---|---|---|
| juntas `slide` (robot_x, robot_y, kicker_slide, chipper_slide), `hinge` (robot_yaw, dribbler_pivot, dribbler_spin), `free` (bola) | `robot_kinematic.xml:152-154,329,332,341,352`; `ball.xml:12` | ✓ `FREE, BALL, SLIDE, HINGE` (`T/doc/mjx.rst:463-465`) | ✓ All |
| limites de junta (`range` + `solreflimit`), `stiffness`/`springref`, `damping`, `armature` | idem | ✓ constraint `LIMIT_JOINT` (`T/doc/mjx.rst` tabela Constraint); mola passiva `jnt_stiffness` (`MJ/mjx/mujoco/mjx/_src/passive.py:74`); `dof_armature` em `smooth.py:314` | ✓ |
| `<motor>` (kicker, chipper) → dyn `NONE`, gain `FIXED`, bias `NONE`, trn `JOINT` | `robot_kinematic.xml:371-372` | ✓ (`T/doc/mjx.rst:466-477`) | ✓ |
| `<velocity>` (dribbler, vel_x/y/yaw) → gain `FIXED`, bias `AFFINE`, trn `JOINT`; `forcerange` | `robot_kinematic.xml:370,373-375` | ✓; clamp de `forcerange` em `MJ/mjx/mujoco/mjx/_src/forward.py:278-283` | ✓ |
| `ctrlrange` | idem | ✓ (`forward.py:170`) | ✓ |
| `equality`, `tendon` | **nenhum** nos três arquivos | — | — |
| sensores `jointpos`, `jointvel`, `actuatorfrc` | `robot_kinematic.xml:378-380` | ✓ os três estão na nota [2] (`T/doc/mjx.rst:533-538`) | ✓ "All except `PLUGIN`" |
| geoms `mesh`, `plane`, `sphere`, `cylinder` | — | ver §4 e "Resposta curta" 2 | ✓ "All" |

O `put_model` do JAX confere por enum os tipos de bias, dyn, gain, trn, eq, sensor e wrap e levanta
erro se faltar algum (`T/…/_src/io.py:372-388`). Pela tabela, nada do que declaramos cai fora.

## 4. Meshes convexos de ≤ 32 vértices no MJX-JAX (SAT)

- **Suporte:** `(PLANE, MESH)`: `plane_convex`, `(SPHERE, MESH)`: `sphere_convex`, `(MESH, MESH)`:
  `convex_convex` (`T/…/_src/collision_driver.py:91,101,112`). A doc recomenda menos de ~200 vértices
  para mesh↔primitiva e "**fewer than 32 vertices**" para convexo↔convexo (`T/doc/mjx.rst:636-650`).
  O nosso `maxhullvert="32"` (`robot_kinematic.xml:6…`, `field.xml:6…`) fica **no limite** da
  recomendação, que é "roughly". `ncon` por par: `plane_convex` 4, `sphere_convex` 1,
  `convex_convex` 4 (`T/…/_src/collision_convex.py:137,227,943`).
- **Como o MJX-JAX dimensiona isso (o bloqueio).**
  1. `geom_pairs` enumera **estaticamente** todo par que passa no filtro de corpo/peso/
     `contype`/`conaffinity`. Não há broadphase real (`T/…/collision_driver.py:127-205`).
  2. `_geom_groups` agrupa por `FunctionKey(types, data_ids, condim)`, onde `data_ids` são os
     **ids de mesh** (`:219-231`). Cada grupo vira uma chamada da função de colisão no grafo XLA
     (`:405-438`).
  3. `make_condim` reserva `ncon × nº de pares` contatos (`:346-403`). Todos entram como linhas de
     restrição, com `condim` linhas cada no cone elíptico (`constraint.py:619-665`). O `efc_J` é
     `(nefc, nv)` denso (`MJ/mjx/mujoco/mjx/_src/io.py:736`).
- **Pares no nosso modelo.** Os 71 geoms `contype=1 conaffinity=1` de cada chassi colidem com os
  chassis dos outros robôs: 21 × 71² = **105 861** pares com 7 robôs, 71² = **5 041** com 2. Os números
  batem com os do mapa. Somam-se chassi × paredes (12 hulls) e chassi × traves (16 hulls, `contype=1`):
  +7 × 71 × 28 = 13 916 com 7 robôs e +3 976 com 2 (*conta minha*). Kicker e chipper (`contype=2
  conaffinity=0`) só colidem com a bola.
- **Assets duplicados por robô.** O `scene.build` faz `MjSpec.from_file` + `spec.attach(child,
  prefix=…)` por robô (`src/mjsoccer/scene.py:253-258`). A doc diz "All assets from the child model
  will be copied in … if the parent and the child are not the same mjSpec"
  (`MJ/doc/programming/modeledit.rst:300-301`). Logo, cada robô tem os seus próprios ids de mesh e
  **cada par chassi↔chassi vira um grupo de 1 par**: ~105 mil chamadas `convex_convex` desenroladas no
  XLA com 7 robôs e ~5 mil com 2 (*inferido; conferir `model.nmesh` na CPU*). Com grupos de 1 par, o
  `max_geom_pairs`, que corta **por grupo**, não ajuda (`collision_driver.py:368-376`, `:420-430`). O
  `max_contact_points` limita contatos por condim **depois** de rodar todas as funções
  (`:378-397`, `:449-455`), então reduz memória mas não compute nem compile.
- **Ordem de grandeza da memória (conta, float32, sem medição).** 7 robôs: ~120 mil pares mesh↔mesh
  × 4 contatos × 3 linhas ≈ 1,4 M linhas; `efc_J` = 1,4 M × `nv` 55 × 4 B ≈ **310 MB por mundo**. 2
  robôs: ~9 mil × 4 × 3 ≈ 108 mil linhas × `nv` 20 × 4 B ≈ **8,6 MB por mundo**, ou ~35 GB para 4096
  mundos, só no `efc_J`. Os dois são inviáveis em 12 GB. O tempo e a RAM do host para compilar ~10⁵
  grupos provavelmente são o que derrubou a máquina na tentativa anterior (*hipótese*).
- **Saídas documentadas (decisão do spec, não deste ticket):** marcar a lista de contatos com `<pair>`
  e desligar o resto por `contype`/`conaffinity` ("Enabling only an explicit list of valid contacts
  can have a dramatic effect", `T/doc/mjx.rst:578-584`); `max_contact_points`/`max_geom_pairs` como
  custom numeric, como no exemplo da shadow hand (`MJ/mjx/mujoco/mjx/test_data/shadow_hand/right_hand.xml:9-10`);
  primitivas no lugar dos hulls do chassi na variante. A bola × hull (sphere↔mesh, `ncon=1`) e o
  chassi × piso (`plane_convex`) custam bem menos. Ainda assim, também são um grupo por mesh.
- **No MJX-Warp isso não acontece:** o broadphase é NxN sobre a lista filtrada quando ela tem menos de
  250 mil pares, ou SAP acima disso (`TW/io.py:644-651`); os contatos vão para um buffer global de
  `naconmax`. O custo de compile do Warp cresce com o número de **tipos** de colisor convexo
  (`T/doc/mjwarp/index.rst:1050-1056`), e no nosso modelo são dois: sphere↔mesh e mesh↔mesh. O
  sphere↔cylinder e o plane↔mesh são primitivos no Warp
  (`TW/collision_primitive.py:1283-1297`).

## 5. float32 contra o float64 da CPU, e x64 no MJX-JAX

- **MJX-Warp: sempre float32.** "MJWarp utilizes floats in contrast to MuJoCo's default double …
  Solver settings, including iterations, collision detection, and small friction values may be
  sensitive" (`T/doc/mjwarp/index.rst:1005-1011`). O Warp limita `tolerance` a ≥ 1e-6 "to avoid the
  solver spending iterations needlessly" (`TW/io.py:413-416`; doc em `T/doc/mjwarp/index.rst:1102`).
- **MJX-JAX: float32 por default, float64 se `jax_enable_x64`.** O dtype dos buffers é
  `jp.zeros(1, float).dtype` (`T/…/_src/io.py:581,639,672`), que segue o flag x64 do JAX. Os testes
  upstream rodam alguns casos com `jax.enable_x64(True)` (`MJ/mjx/mujoco/mjx/_src/math_test.py:217,241-249`).
  **Não verificado:** que o pipeline completo de step (colisão SAT/SDF e solver) passe em x64, e
  quanto custa em throughput e memória numa GPU de consumo. Ver §10.
- **O que muda para nós:**
  - O MJX-JAX **não limita** a `tolerance`. Com o default 1e-8 em float32, os critérios
    `improvement < tolerance` e `gradient < tolerance` (`T/…/_src/solver.py:562-568`) provavelmente
    nunca disparam. Sob `vmap`, o `while_loop` (`:602`) roda até o último mundo terminar, ou seja, as
    100 iterações de `models/scene.xml:4` (*inferido*). A restrição 7 do mapa já permite mexer em
    `tolerance`/`iterations`.
  - Atritos pequenos (torção e rolagem de 1e-4 a 9e-4 nos pares da bola, `ball.xml:24`,
    `scene.xml:21-22`) estão na classe que a doc do Warp aponta como sensível.
  - `solref` mínimo 0,004 ≥ 2 × dt (`dt` = 1/960 ≈ 0,00104 s), então o `refsafe` não corta nada no
    timestep nominal. No substep do kick (dt/`FIRING_SUBSTEPS`) a desigualdade também vale, porque o
    dt é menor.
  - A resolução de posição em float32 a 6 m é ~5e-7 m (ulp), irrelevante para o campo. O `time`
    acumulado com dt de 1e-3 tem erro relativo ~1e-7 em episódios de minutos (*conta*).
  - O GPU também não é determinístico ("No. There may be ordering or small numerical differences",
    `T/doc/mjwarp/index.rst` FAQ "Is MJWarp on GPU deterministic?"). Isso reforça a paridade
    estatística, e não bit a bit, da restrição 8.

## 6. Ler contatos no MJX-Warp: a posse por sensor de contato

- **Hoje:** `_update_possession` varre `data.contact[:ncon]` e marca o robô cujo `dribbler_body`
  toca a bola (`src/mjsoccer/base.py:231-245`). O corpo `dribbler` tem como geoms colidíveis só os dois
  cilindros, `r_col_axisDribbler` e `r_col_barDribbler` (`robot_kinematic.xml:331-338`).
- **Por que sensor:** no MJX os contatos ficam em `mjx.Data._impl`; "We encourage users to read out
  contacts solely through contact sensors" (`T/doc/mjx.rst:128-130`). No Warp o buffer é global,
  `(naconmax, …)` com `worldid` (`TW/types.py:1938-1952`), e o `contact__*` não tem eixo de lote
  (`T/…/warp/types.py`, `_BATCH_DIM['Data']`). Varrê-lo sob `vmap` é inviável.
- **O que cada motor aceita no sensor `contact`:**
  - MJX-JAX: casamento `none-none` e `geom-geom` (também só `geom1` ou só `geom2`); redução
    `mindist`/`maxforce`; data "all" (`T/doc/mjx.rst:533-538`, nota [2]). O `put_model` levanta erro
    para casamento por `site`, `body` ou `subtree` e para redução `netforce`
    (`T/…/_src/io.py:318-351`). O contato conta como achado se `dist − includemargin < 0`
    (`T/…/_src/sensor.py:541-543`), que é a mesma condição de um contato com margem 0 na CPU.
  - MJX-Warp: "All except `PLUGIN`" (`T/doc/mjx.rst:510-512`), incluindo `body`. Parâmetro extra
    `opt.contact_sensor_maxmatch`, default 64 e configurável por `<custom><numeric
    name="contact_sensor_maxmatch">` (`TW/io.py:424-428`; `T/doc/mjwarp/index.rst:300-320`). É
    estático, sem lote (`T/…/warp/types.py:1877`). Matches acima do limite são **ignorados**, e o
    estouro marca `OverflowType.CONTACT_MATCH` (`TW/types.py:140-160`).
- **Forma comum aos dois:** por robô, dois sensores `<contact geom1="ball" geom2="{prefix}r_col_barDribbler"
  data="found" num="1"/>` e o mesmo para `r_col_axisDribbler`. Toque = `found > 0` em qualquer um. É
  1 float por sensor no `sensordata`, que tem lote nos dois motores (`T/…/warp/types.py:1465`). Com
  `geom-geom` especificado o casamento é exato e cabe em `maxmatch` pequeno; a doc recomenda
  justamente "specify `geom1` and `geom2`" (`T/doc/mjwarp/index.rst:1039-1042`). A semântica do
  `found` (0 ou número de contatos casados) está em `MJ/doc/XMLreference.rst:9095-9170`.
- **Onde os sensores entram:** no `scene.build`, por `spec.add_sensor` junto dos pares do dribbler
  (`scene.py:260-265`). Sensor não muda a física (restrição 7), mas muda `nsensordata`. Na variante
  cápsula, os nomes dos geoms se mantêm se o transform preservar o nome.
- **Pegadinha do Warp que não nos afeta:** o FAQ "Why are contacts reported when there are no
  collisions?" fala de **sensores de colisão** (distance/normal/fromto), que geram 1 contato por par
  mesmo sem toque. Sensores `contact` "will report the correct information for contacts affecting
  the physics" (`T/doc/mjwarp/index.rst`, FAQ Features).

## 7. `naconmax` / `njmax`: semântica, dimensionamento e memória

- **Semântica:** "`naconmax` defines the maximum number of contacts for all worlds combined. `njmax`
  defines the maximum number of constraints per world … Scale `naconmax` by the number of
  environments" (`T/doc/mjx.rst:118-124`). Um mundo pode passar de `nconmax` se o total couber, mas
  `njmax` é rígido por mundo (`T/doc/mjwarp/index.rst:146-160`). Os valores de `<size>` do MJCF são
  ignorados (`:161-164`). Estouro dá comportamento indefinido e é marcado em `Data.overflow`
  (bitmask por mundo: `NEFC`, `NARROWPHASE`, `CCD`, `CONTACT_MATCH`, …; `TW/types.py:140-160`, campo
  `(nworld,)` em `:2181`).
- **Default é armadilha:** o `mjx.make_data(impl="warp")` cria o `Data` com `nworld=1`
  (`T/…/_src/io.py:765-773`). Sem argumento, o Warp chuta `nconmax` = 48 e `njmax` = 64 para cenas
  sem hfield, flex nem SDF (`TW/io.py:1178-1201`), e o `naconmax` vira 48 para **todos** os mundos.
  É obrigatório passar `naconmax`.
- **Como dimensionar (método, sem números medidos):**
  1. Na CPU (`mujoco 3.11.0`, que é o oráculo), rodar rollouts de cada task, com política aleatória e
     com resets cobrindo colisões robô↔robô e robô↔parede, e registrar **`max(d.ncon)`** e
     **`max(d.nefc)`** por passo de física. O `ncon médio ~1` do mapa não serve para dimensionar.
  2. `nconmax` ≈ 1,5 × max(ncon), arredondado para o próximo tamanho da escada do Warp (16, 24, 32,
     48, …; `TW/io.py:1184`). `naconmax = nconmax × nworld`. `njmax` ≈ 1,5 × max(nefc).
  3. Conferir no Warp com `Data.overflow` (ou `mjwarp-testspeed --measure_alloc`,
     `T/doc/mjwarp/index.rst:266-279`). A CPU e o Warp podem gerar contagens diferentes em mesh↔mesh:
     o `multiccd` e o colisor `PLANE<>MESH` dão 4 contatos no Warp contra 3 na CPU
     (`MJ/doc/mjwarp/index.rst`, "Maximum number of contacts per collider").
  - Para referência, o Playground usa `naconmax = 30 × 8192` e `njmax = 40` no Go1 (ver 02, §"Data").
  - **Piso analítico por mundo (conta):** bola × piso, 1 contato e 6 linhas; até 2 contatos bola ×
    dribbler por robô que segura, 4 linhas cada; limites ativos de `dribbler_pivot`/kicker/chipper
    (até 3 por robô, 1 linha cada). Isso dá `njmax` ≳ 6 + 8 + 3·N antes de qualquer colisão de chassi.
    Um choque chassi↔chassi pode somar vários contatos por par de hulls, 3 linhas cada; é o termo que
    só a medição fecha.
- **Custo de memória no Warp (conta, não medido):** cada contato tem ~45 escalares de 4 B (campos em
  `TW/types.py:1938-1952`, com `efc_address` de até `nmaxpyramid`), ~200 B. Com 4096 mundos ×
  `nconmax` 48, isso dá ~40 MB. As linhas de restrição por mundo, `njmax` × (`nv` + ~10), dão com
  `njmax` 128 e `nv` 55 ~33 KB por mundo, ~135 MB para 4096 mundos. Somam-se o buffer de CCD
  (`naccdmax`, cresce com `ccd_iterations`; `T/doc/mjwarp/index.rst:322-360`) e o resto do `Data`
  (~O(`nworld` × (`nbody` + `ngeom` + `nv`²)) escalares; o `ngeom` de ~640 com 7 robôs pesa nos
  `geom_xpos`/`xmat`). Ordem de grandeza: centenas de MB, folgado nos 12 GB. **Medir com
  `testspeed --memory`.**
- **Memória compartilhada JAX × Warp:** o JAX pré-aloca a maior parte da GPU por default
  ([GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), `XLA_PYTHON_CLIENT_PREALLOCATE`
  / `MEM_FRACTION`). Os modos `WARP_STAGED*` alocam buffers de staging do lado Warp
  (`T/doc/mjx.rst:151-181`). *Não verificado* como os dois dividem a GPU na prática; é item do ticket
  de instalação.

## 8. Reset parcial de mundos no MJX-Warp e o auto-reset do Playground

- A doc: `jax.tree.map(jnp.where, done, reset_data, data)` funciona no JAX, "does not work
  out-of-the-box for MJX-Warp due to internal implementation details", e a forma unificada é
  `data = data.where(done, reset_data)` (`T/doc/mjx.rst:135-147`). O método existe na 3.11.0
  (`T/…/_src/types.py:1170`).
- **O que são os "detalhes internos":** os campos `contact__*` e `efc__*` do Warp não têm eixo de
  mundo; a dimensão 0 deles é `naconmax`/`njmax`. O `Data.where` consulta `_BATCH_DIM['Data']` e só
  aplica `where` nos campos com lote, mantendo o resto (`MJ/mjx/mujoco/mjx/_src/types.py:1186-1200`).
  O FFI confere essas formas e sugere `Data.where` quando falham (`MJ/mjx/mujoco/mjx/warp/ffi.py:370-405`).
  Manter o buffer de contato "do estado atual" é correto, porque a colisão o reescreve no início do
  próximo passo (*inferido*).
- **Como o Playground faz com `impl="warp"`:** o `BraxAutoResetWrapper` não chama `Data.where`. O
  `where_done` dele aplica `jp.where` folha a folha e **devolve a folha atual quando
  `x.shape[0] != done.shape[0]`** (`PG/rss2025/mujoco_playground/_src/wrapper.py:222-230`). Isso
  preserva por acidente de formato os mesmos campos sem lote que o `Data.where` preserva. A troca de
  `where_done` por `data.where` que o 02 recomenda continua valendo: o caso degenerado
  `naconmax == num_envs` quebraria o atalho por formato (*inferido*). Reuso do 02, §2 e §6.

## 9. `opt.timestep` por mundo (kick)

Reaproveitado do 03 (`.scratch/gpu-port/research/03-mjx-domain-randomization.md`, "Achados
laterais", `:301-307`) e conferido na 3.11.0:

- MJX-JAX: `Option.timestep: jax.Array` (`T/…/_src/types.py:512`), então aceita `vmap`/`in_axes`.
- MJX-Warp: `'opt__timestep': True` em `_BATCH_DIM['Model']` (`T/…/warp/types.py:1894`; spec
  `timestep: array("*", float)`, `TW/types.py:851`). Os kernels leem
  `opt_timestep[worldid % opt_timestep.shape[0]]` (`TW/forward.py:69,130`).
- Ressalvas já registradas no 03:
  - o `refsafe` (`timeconst ≥ 2·timestep`) é avaliado com o `timestep` do mundo;
  - um `Model` com campo batelado que muda a cada passo pode mudar ponteiros de buffer e forçar
    recaptura de grafo CUDA no modo `WARP` (`T/doc/mjx.rst:151-207`).
- O que muda em relação à CPU: hoje o kick multiplica o **número** de passos por `FIRING_SUBSTEPS`
  num só env (`base.py:137-147`). Num lote, o número de iterações do `scan` é estático. Então ou todo
  o lote roda `n × FIRING_SUBSTEPS` substeps com `timestep` por mundo (os mundos sem disparo usariam
  `dt`, `dt`, … e depois ficariam parados?), ou todo mundo usa `dt/FIRING_SUBSTEPS` no passo de
  controle em que algum mundo dispara. Esse desenho fica para o spec ("Kick num lote" no mapa). A
  pergunta do ticket ("o campo aceita lote?") é **sim nos dois**.

## 10. Itens para os tickets de instalação e de benchmark (não medidos aqui)

**Instalação (env isolado, fora do `uv` do projeto, em disco e não em `/tmp`):**
1. Resolver `mujoco==3.11.0`, `mujoco-mjx[warp]==3.11.0` (→ `warp-lang==1.14.0`) e
   `jax[cuda12]==0.11.0`; depois testar `jax 0.12.0.1`. Registrar se o FFI do Warp 1.14 importa com
   cada JAX. Conferir o CUDA do wheel `warp-lang 1.14.0`.
2. Se o Playground for instalado como pacote, usar `playground==0.2.0` (PyPI), não o checkout do
   `main`, por causa do pin `warp-lang>=1.18.0`.
3. Conferir na CPU, sem MJX, só `scene.build`: `model.nmesh`, para confirmar a duplicação de assets
   por robô do §4, e o número de pares que o `geom_pairs` do MJX enumeraria.
4. Configurar `XLA_PYTHON_CLIENT_PREALLOCATE`/`MEM_FRACTION` antes de qualquer teste com Warp, e
   limitar a RAM do processo (ulimit/cgroup) antes de qualquer `put_model`/`jit` do MJX-JAX.

**Benchmark/medição:**
5. **Compile do MJX-JAX com a cena de 2 robôs antes da de 7**, com limite de memória do host, e
   medir tempo e pico de RAM. Repetir com a variante que corta chassi↔chassi. A cena de 7 robôs com o
   modelo base não deve ser tentada sem esse corte.
6. `max(ncon)` e `max(nefc)` na CPU por task, para fechar `nconmax`/`naconmax`/`njmax`, e depois
   `Data.overflow` no Warp.
7. Memória total do Warp por `nworld` (`testspeed --memory`) para 2 e 7 robôs.
8. Precisão do `sphere_cylinder` SDF do MJX-JAX contra a CPU no contato bola × barra do dribbler
   (posição, normal, `dist`), comparada com a variante cápsula.
9. Nº de iterações do Newton por passo no MJX-JAX com `tolerance` 1e-8 e 1e-6 em float32, e o SPS
   correspondente.
10. MJX-JAX com `jax_enable_x64`: se o step completo roda, e a que custo.
11. Compile do MJX-Warp: quantos kernels CCD únicos e o tempo de primeira chamada.

## Não verificado (resumo)

- Duplicação dos assets de mesh por robô no `scene.build` (inferida da doc do `attach`).
- O número de ~10⁵ grupos de colisão no MJX-JAX e as contas de memória do §4 e do §7.
- Compatibilidade de `jax 0.11/0.12` com `mujoco-mjx 3.11.0` e o FFI do `warp-lang 1.14.0`.
- CUDA do wheel `warp-lang 1.14.0` (inferido CUDA 12).
- Robustez do `sphere_cylinder` SDF no nosso contato.
- O efeito nulo do termo giroscópico na bola (inferido da inércia isotrópica).
- O solver do MJX-JAX roda as 100 iterações com `tolerance` 1e-8 em float32.
- Divisão de memória entre JAX e Warp na mesma GPU.
