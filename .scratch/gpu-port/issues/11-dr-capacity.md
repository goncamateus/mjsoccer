# Que capacidade de domain randomization o port oferece

Type: grilling
Status: open
Blocked by: 01, 10

## Pergunta

Dado o que a pesquisa de DR mostrou (`vmap` com `in_axes` sobre o `mjx.Model`, `pair_*` para os
contatos da bola, derivados velhos sem `set_const`), que **capacidade** o spec dá — sem escolher
parâmetros nem ranges (restrição 10)?

- Quando se amostra: uma vez por slot no início do run (padrão Playground) ou por episódio (campos no
  `State`, sem referência em JAX).
- Derivados: recalcular no host com `mj_setConst` in place e empilhar, ignorar como o Playground, ou
  restringir a DR a campos sem derivados.
- Interface: uma função `randomize(model, rng)` por task, um registro declarativo de campos, ou outra.
- Onde ficam ruído de observação e latência de ação (lado da task, nos dois motores igual).
- No Warp: modo de grafo (`WARP_STAGED`?) se a DR reamostrar arrays.
