# Que capacidade de domain randomization o port oferece

Type: grilling
Status: resolved
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

## Answer

Resolvido em grilling (2026-11-08).

1. **DR só no lado MJX.** O backend CPU continua determinístico (modelo nominal, cache do
   `scene.py` intacto). A task-logic parity (ticket 10) e a paridade estatística (ticket 09) rodam
   com DR desligada; o `--play` no CPU usa o nominal.
2. **Amostragem uma vez por slot** no início do run (padrão Playground): o slot i fica com a sua
   variante o run todo. Reamostrar por episódio fica fora (sem referência em JAX, sem `mj_setConst`
   no host). A estabilidade do grafo CUDA (`WARP_STAGED`, ponteiros com `vmap`/`in_axes`) virou item
   do ticket 12.
3. **Derivados no host**: para cada slot, editar o `MjModel` em cache in place, `mj_setConst`,
   copiar campos variados + derivados (`body_invweight0`, `dof_invweight0`, `actuator_acc0`, ...)
   para uma pilha e restaurar o nominal no fim. ~0.28 ms/variante (~2.3 s para 8 192). Nunca copiar
   o `MjModel` (170 ms). Campos estruturais (meshes, `contype`, `*_limited`, inteiros do `opt`) ficam
   fora da capacidade.
4. **Interface**: um callable do usuário `sampler(mj_model, rng) -> None` que muta o modelo,
   passado em `mjsoccer.mjx.make(id, randomize=sampler)`; default `None` = sem DR. O framework faz
   `setConst`, diff contra o nominal (vira o `in_axes`), empilha e chama `put_model`. RNG derivado
   da seed de treino, determinístico por slot. Adapta-se ao `randomization_fn(sys, rng) -> (sys,
   in_axes)` do `brax ppo.train` (seed numpy derivada da chave). Docstring avisa que os contatos da
   bola são `<pair>` (mexer em `pair_*`, não `geom_*`). Sem registro declarativo nem DR por task.
5. **`opt.timestep` rejeitado**: o framework recusa se o sampler mudar o `timestep` — o disparo do
   kick e o `refsafe` não dependem da DR.
6. **Ruído de observação e latência de ação** são do lado da task em `src/mjsoccer/mjx/` (ruído
   via `info["rng"]`; latência como buffer de N passos de controle no `State`), config do wrapper de
   treino, desligados por default, fora da task-logic parity. Sem `actuator_delay` nativo.
7. **Eval**: o `examples/ppo_mjx.py` avalia no modelo nominal por default; `--eval-randomize`
   avalia sob DR.

Termo novo no `CONTEXT.md`: **domain randomization (DR)** / **variant**.
