# Que implementações de PPO em JAX rodam sobre o MJX?

Type: research
Status: resolved
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

## Answer

Detalhe e fontes em [`research/02-jax-ppo-stacks.md`](../research/02-jax-ppo-stacks.md). Resumo:

- **Só o Brax PPO** (via wrapper do MuJoCo Playground, `train-jax-ppo --impl warp`) é documentado
  rodando com `impl="warp"`; Warp é default no Playground desde 0.2.0. O `main` do Playground virou
  torch/RSL-RL; a versão JAX está congelada em `rss2025/` e treina com `brax ppo.train`.
- **Auto-reset**: o do Brax faz `jnp.where` cego em toda folha, reseta para um primeiro estado em
  cache — é o que o MJX diz que não funciona com Warp. A correção oficial é `mjx.Data.where`
  (`types.py:1188`).
- **Recursos do Brax PPO**: normalização de obs, crítico assimétrico (`value_obs_key`), tanh-normal
  em [-1, 1], action repeat, truncamento no GAE, checkpoint Orbax, logging por callback. Variantes por
  flag: value clipping, LR por KL adaptativo, crítico distribucional, visão.
- **Throughput**: só A100/MJX-JAX (paper do Playground, 30K–720K passos/s, 2048–8192 envs); Warp só
  "1.5-2x" (discussion #197). Nada para RTX 3060.
- **PureJaxRL / rejax**: sem adaptador MJX, sem truncamento no GAE, sem checkpoint nem crítico
  assimétrico. **Stoix**: 6 variantes de PPO via Stoa, mas pina `jax<0.6`; reset por `lax.cond` em
  `vmap`, sem verificação com Warp.
- **Recomendação (inferência, não testada)**: tasks no padrão `MjxEnv` do Playground, treino com
  `brax ppo.train` e uma cópia própria do `wrap_for_brax_training` com `Data.where` e `full_reset`.
- **Não verificado**: `pmap` incondicional do Brax com Warp numa GPU; Stoix com Warp; SPS na 3060.
- **Contraste** Planning-the-path: SAC torch + CAPS, 16 envs CPU, 150k passos.
