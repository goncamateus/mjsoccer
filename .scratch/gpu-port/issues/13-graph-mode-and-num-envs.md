# Modo de grafo do Warp e `--num-envs` default do treino

Type: grilling
Status: open
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
