# Que implementações de PPO em JAX rodam sobre o MJX?

Type: research
Status: open
Blocked by: —

## Pergunta

Que implementações de PPO e derivados em JAX treinam sobre ambientes MJX hoje (Brax PPO, o
treinamento do MuJoCo Playground, PureJaxRL e outras que as fontes primárias apontarem)?

Para cada uma, com fonte:

- Roda com `impl="warp"`, ou só com `impl="jax"`? Como faz auto-reset e lida com o estado por mundo.
- Throughput reportado numa GPU só, e com que tamanho de lote (`num_envs`).
- Recursos: normalização de observação, crítico assimétrico, ações contínuas limitadas em [-1, 1],
  *action repeat*, checkpoint, logging.
- Que derivados de PPO elas oferecem (PPO clássico, variantes com clipping/normalização de
  vantagem diferentes, etc.) e qual é a base de código mais simples de adaptar para 6 tasks.
- Como o Planning-the-path-with-rl treinava (SAC em torch, `methods/sac.py`) só como contraste — não
  é alvo.

## Onde fica a resposta

`.scratch/gpu-port/research/02-jax-ppo-stacks.md`.
