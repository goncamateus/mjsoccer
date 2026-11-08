# Como randomizar o modelo por mundo pela API MJX?

Type: research
Status: resolved
Blocked by: —

## Pergunta

Pela API MJX, como cada mundo de um lote recebe valores próprios de campos do `Model` (massa,
atrito, `solref`/`solimp`, ganho de atuador, e o que mais for comum em DR), nos dois motores?

Responder com fonte:

- MJX-JAX: `vmap` sobre campos do `mjx.Model` (`in_axes`) — quais campos podem variar, e quais
  dependem de quantidades derivadas que o `mj_setConst` recalcula (massa → `body_invweight`, etc.).
- MJX-Warp: campos com dimensão de lote (`batch_sizes`, indexação `worldid % shape[0]`), quais mudam
  na GPU sem recompilar, quais exigem recompilar o `MjSpec` no host por variante (issue 893 do
  `mujoco_warp`), e o custo disso.
- Como o MuJoCo Playground implementa DR com `impl="warp"`.
- Latência de atuação / ruído de observação: são DR do lado da task (fora do `Model`) — só registrar
  como as stacks costumam fazer.

O spec decide a **capacidade**, não quais parâmetros nem ranges.

## Onde fica a resposta

`.scratch/gpu-port/research/03-mjx-domain-randomization.md`.

## Answer

Detalhe e fontes em [`research/03-mjx-domain-randomization.md`](../research/03-mjx-domain-randomization.md)
(commits pinados: mujoco 371c2b4, mujoco_warp 4d1c64b, mujoco_playground e7c53bc, brax c860529). Resumo:

- **Um mecanismo nos dois motores**: `randomize(model, rng) -> (model_v, in_axes)` e
  `jax.vmap(step, in_axes=[in_axes, 0, 0])`. Com `impl="warp"` a ponte (`warp/ffi.py`) vira o eixo do
  vmap em eixo líder do Warp (`field[worldid % shape[0]]`); sem `batch_sizes`.
- **Podem variar**: campos `jax.Array` no MJX-JAX e `True` em `_BATCH_DIM` no Warp (133 campos):
  massa, inércia, atrito, `solref`/`solimp`, ganho/bias de atuador, `qpos0`, `opt.timestep`.
  Campos estruturais (meshes, `contype`, `*_limited`, `opt` inteiros) exigem recompilar.
- **Nossos contatos da bola são `<pair>`**: DR de atrito/`solref` tem que mexer em `pair_*`, não `geom_*`.
- **Derivados ficam velhos**: nenhum motor expõe `set_const` pelo MJX; massa nova deixa
  `body_invweight0` velho (entra na regularização de contato do MJX-JAX). O Playground ignora.
- **Correção barata (medida, 3.11.0, cena de 7 robôs)**: editar o `MjModel` in place + `mj_setConst`
  e empilhar os derivados: 0.28 ms/variante (~1.1 s para 4096). Recompilar `MjSpec`: ~277 ms/variante
  (~19 min para 4096). Copiar o `MjModel`: 170 ms (1088 meshes) — editar in place.
- **Warp nativo**: `mjw.set_const` na GPU (issue 893 fechada 2026-06-24; grupo SPRING sem suporte).
  Meshes por mundo não passam pelo MJX.
- **Grafo CUDA (Warp)**: mudar shape de campo retraça; modo WARP default recaptura com ponteiros XLA
  instáveis; `WARP_STAGED` evita. **Não verificado** se reamostrar a cada reset mantém ponteiros.
- **Playground amostra uma vez** no `__init__` do wrapper (cada slot fica com um modelo o run todo).
  Reamostrar por episódio exigiria os campos no `State` — sem código de referência em JAX.
- **Latência/ruído são da task**: ruído uniforme em `_get_obs` com `info["rng"]`; rss2025 sem latência
  de ação (Playground torch usa `DelayBuffer`). `actuator_delay` nativo existe no Warp, não no MJX-JAX —
  não serve a uma implementação única.
- **`timestep` por mundo** é representável nos dois motores; efeito numérico não testado (`refsafe`
  limita `timeconst` ≥ 2×`timestep`). Fontes são `main` de nov/2026 — reverificar nas versões pinadas.

Scripts de medição: `/tmp/claude-1000/-home-mateus-doc-ssl-rl/716cbd60-1120-41e0-842f-d53c5add287a/scratchpad/scripts/cost{,2,3}.py` (voláteis).
