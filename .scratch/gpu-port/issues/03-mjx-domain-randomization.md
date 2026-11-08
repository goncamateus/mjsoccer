# Como randomizar o modelo por mundo pela API MJX?

Type: research
Status: open
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
