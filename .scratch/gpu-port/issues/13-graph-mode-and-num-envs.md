# Modo de grafo do Warp e `--num-envs` default do treino

Type: grilling
Status: resolved
Blocked by: 12

## Pergunta

O [smoke do Brax sobre o MJX-Warp](12-brax-ppo-warp-smoke.md) mostrou que o `graph_mode` `WARP`
(o mais rápido, 31 k decisões/s a 4 096 mundos) recaptura o grafo a cada época e estoura a GPU
(época 3 a 8 192, época 6 a 4 096); `WARP_STAGED` (16 k) e `NONE` (18 k, eval 4× mais lenta) são
estáveis a 4 096, e só `NONE` roda a 8 192. Isso derruba o `--num-envs` 8 192 do
[ticket do PPO](08-ppo-training-scripts.md).

Decidir:
- que `graph_mode` o `mjsoccer.mjx.make` usa por default no treino (e na eval do Brax), e se é kwarg;
- o `--num-envs` default das tasks de 2 robôs (e se o de 7 robôs, 1 024, precisa de verificação de
  treino, já que só a física bruta foi medida);
- se vale uma investigação (ticket `task`) para tornar o `WARP` estável — endereços de buffer fixos
  (doação, uma chamada `jit` para o treino inteiro) — ou se o spec aceita ~metade do SPS.

## Answer

Resolvido em 2026-11-08 (grilling), sobre os números do
[smoke do Brax sobre o MJX-Warp](12-brax-ppo-warp-smoke.md).

- **Modo default**: `WARP_STAGED` no treino e na eval do Brax. Melhor ponta a ponta medido e
  estável (13.0 k decisões/s a 4 096, 5.7 GB). `NONE` perde na eval (4×); misto não medido, sai.
- **Kwarg**: `mjsoccer.mjx.make(..., graph_mode=...)` e `--graph-mode` no `examples/ppo_mjx.py`,
  default `WARP_STAGED`.
- **`--num-envs` 2 robôs**: 4 096 (8 192 dá OOM em `WARP_STAGED`).
- **`--num-envs` 7 robôs**: regra fixa — o maior `nworld` estável por 30 épocas no modo default com
  ≥ 2 GB de folga na GPU. O número sai do ticket
  [Estabilizar o `WARP` e medir o treino de 7 robôs](14-warp-stability-and-7-robot-train.md) e entra
  no spec sem novo grilling.
- **`WARP`**: upgrade condicional. O mesmo ticket 14 tenta endereços estáveis (time-box de uma
  sessão); se `WARP` ficar plano por 30 épocas, vira o default e os dois `--num-envs` se recalculam
  pela regra acima. Se falhar, fica `WARP_STAGED` sem mais investigação.
