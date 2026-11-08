# O MJX roda o nosso modelo? Versões e suporte, nos dois motores

Type: research
Status: resolved
Blocked by: —

## Pergunta

Com `mujoco 3.11.0` instalado e a RTX 3060 no driver 595, quais versões de `mujoco-mjx`,
`mujoco-warp` e JAX com CUDA são compatíveis entre si? E o MJX-JAX e o MJX-Warp suportam tudo que o
**modelo base** e a **variante do MJX-JAX** (cilindros do dribbler → cápsulas) usam?

Responder, item a item e com fonte (doc, código-fonte ou issue upstream):

- `<pair>` explícitos com `condim` 4 e 6, cone elíptico (o MJX-JAX não aceita `condim` 1 com
  elíptico — confirmar que não usamos 1).
- `implicitfast` + Newton; o que é a ressalva do MJX-Warp "All except `IMPLICITFAST` midpoint
  integrator feature" e se nos afeta.
- Os tipos de atuador, juntas (slide/hinge), `equality`/`tendon` e sensores que
  `models/robot_kinematic.xml`, `models/ball.xml` e `models/field.xml` declaram.
- Meshes convexos de até 32 vértices em mesh↔mesh e mesh↔sphere no MJX-JAX (SAT).
- float32: o que muda em relação ao float64 da CPU, e se o MJX-JAX permite x64.
- Ler contatos no MJX-Warp: a posse hoje varre `data.contact` (`src/mjsoccer/base.py:237`); como fazer
  com sensor de contato (`CONTACT`, matching `geom-geom`) nos dois motores.
- `naconmax` / `njmax`: como dimensionar para as cenas de 2 e de 7 robôs, e o custo de memória.
- Reset parcial de mundos no MJX-Warp ("`jnp.where` does not work out-of-the-box") — como o
  MuJoCo Playground faz auto-reset com `impl="warp"`.
- Mudar `opt.timestep` por mundo (o kick hoje muda o `timestep` do `MjModel` no passo de disparo,
  `src/mjsoccer/base.py:137-147`): o campo aceita lote nos dois motores?

## Onde fica a resposta

`.scratch/gpu-port/research/01-mjx-feature-support.md`, com cada afirmação citando a fonte.

## Answer

Detalhe e fontes em [`research/01-mjx-feature-support.md`](../research/01-mjx-feature-support.md)
(lido na tag `mujoco 3.11.0`; `main` 371c2b4, `mujoco_warp` 4d1c64b, `playground` e7c53bc). Só
leitura: nada instalado nem rodado. Resumo, bloqueios primeiro:

- **Bloqueio do MJX-JAX: colisões entre chassis, não o cilindro.** O MJX-JAX reserva 4 contatos com
  linhas de restrição por par mesh↔mesh candidato e gera uma chamada XLA por par de ids de mesh.
  Estimativa (não medida): ~10⁵ grupos e `efc_J` de ~310 MB **por mundo** com 7 robôs, ~9 MB com 2.
  Inviável em 12 GB; causa provável do OOM da primeira tentativa (hipótese). A variante tem que cortar
  chassi↔chassi; trocar só os cilindros não basta.
- **sphere↔cylinder existe no MJX-JAX** desde out/2024 (SDF de 10 passos, não analítico), apesar da
  doc. O modelo base deve carregar; precisão no nosso contato não testada. A troca por cápsulas vira
  escolha, não obrigação.
- **Versões**: `mujoco==3.11.0` + `mujoco-mjx[warp]==3.11.0` (fixa `warp-lang==1.14.0`, MuJoCo Warp
  embutido — `mujoco-warp` separado é desnecessário). JAX sem pin upstream: testar `jax 0.11.0`
  primeiro. Driver 595 / sm_86 servem CUDA 12 e 13. `playground==0.2.0` do PyPI, não o `main`
  (`rss2025` pede `warp>=1.18`).
- **Suportado nos dois motores**: pares condim 4/6, cone elíptico, Newton, juntas slide/hinge/free,
  atuadores motor/velocity, sensores jointpos/jointvel/actuatorfrc. Sem equality/tendon; nunca condim 1.
- **`implicitfast` midpoint no Warp** afeta só corpos livres (a bola); termo giroscópico nulo pela
  inércia isotrópica (inferido).
- **float32**: Warp sempre, tolerância ≥ 1e-6; MJX-JAX aceita x64. Com `tolerance` 1e-8 o Newton do
  MJX-JAX deve rodar as 100 iterações (inferido).
- **Posse**: 2 sensores `contact` por robô, geom-geom (bola × barra, bola × eixo), `data="found"`;
  MJX-JAX rejeita matching por `body`; no Warp ajustar `contact_sensor_maxmatch` (default 64).
- **`naconmax`** é total de todos os mundos, `njmax` por mundo; default (~48 no total) inviável.
  Dimensionar por `max(ncon)`/`max(nefc)` medidos na CPU e checar `Data.overflow`.
- **Reset parcial**: `data.where(done, reset_data)` (3.11). O `where_done` do Playground quebra se
  `naconmax == num_envs`.
- **`opt.timestep` por mundo** aceito nos dois motores (confirma a pesquisa de DR).
- §10 do arquivo lista 11 itens de medição — copiados para os tickets de instalação e benchmark.

> **Revisão (2026-11-08):** as versões acima ficam superadas pela restrição 12 do mapa (alvo MuJoCo 4.0.0). A colisão do MJX-JAX não mudou entre 3.11.0 e 4.0.0 (`git diff 3.11.0 4.0.0 -- mjx/mujoco/mjx/_src/collision_driver.py`: só anotações de tipo), então o bloqueio chassi↔chassi continua.
