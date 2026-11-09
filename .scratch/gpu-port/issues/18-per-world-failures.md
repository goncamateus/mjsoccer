# Falhas por mundo: NaN e overflow do Warp

Type: grilling
Status: open
Blocked by: 17

## Pergunta

No CPU uma divergência levanta `RuntimeError` (`src/mjsoccer/base.py:153-159`). Num lote de
milhares de mundos, um mundo que diverge não pode derrubar o treino, e o Warp tem overflow de
`naconmax`/`njmax` que é máscara em `Data.overflow` e não para a simulação
([benchmark](05-mjx-benchmark.md)). Decidir, sobre o reset do
[ticket de reset em lote](17-batch-reset-and-world-state.md):
- NaN/inf num mundo: resetar o mundo (com que recompensa/`done`), marcar numa métrica, ou abortar;
- overflow do Warp: abortar o run (buffers mal dimensionados), resetar o mundo, ou só contar;
- o que o `State.metrics` expõe para o log do treino (contadores por época);
- se o eval e os testes de paridade tratam diferente (abortar onde o treino reseta).
