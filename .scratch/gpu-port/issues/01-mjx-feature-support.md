# O MJX roda o nosso modelo? Versões e suporte, nos dois motores

Type: research
Status: open
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
