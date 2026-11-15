# 05: Streams torch × Warp e memória do device × mempool

Spec: [backend mjlab](../spec.md) — Fase 2 (Streams), Further Notes

**What to build:** duas incógnitas do protótipo viram decisões medidas. (1) O mjlab não sincroniza o stream do torch com o do Warp; o protótipo rodou tudo no stream do Warp. Medir se rodar fora dele corrompe dados e documentar a decisão. (2) A 16 384 mundos o device usa 10,9 GB e o mempool do Warp só 1,3 GB de pico; explicar a diferença (mempool reservado, grafo capturado duas vezes, contexto CUDA torch + Warp) e cortar o desperdício se der.

**Blocked by:** 04

**Status:** done

- [x] Decisão de stream tomada, com a medição que a sustenta, documentada no código e no relatório
- [x] Diferença de memória decomposta por fonte, com números
- [x] Se houver desperdício evitável (ex.: grafo capturado duas vezes), removido e o `nworld` máximo remedido

## Comments

Done (ticket 05). Relatório: [research/04-streams-and-memory.md](../research/04-streams-and-memory.md).

- **Stream**: fica o torch no stream do Warp (wrapper do `SoccerEnv`). Sem ele, o stream
  default do torch é seguro (o do Warp é bloqueante), mas um caller num side stream corrompe
  todo mundo (`data.time` ≠ soma dos timesteps escritos). Custo no ruído. Teste:
  `test_a_caller_on_a_side_stream_does_not_race_warp`.
- **Memória**: a diferença era memória de grafo CUDA, 7,2 GB a 16 384 mundos, toda scratch
  do EPA dimensionada por `naccdmax` (default do Warp = `naconmax`). Não era mempool
  reservado nem contexto; a captura dupla não fica residente, mas a captura do `__init__` do
  mjlab falhava a 32 768. `scene.NACCD_PER_ROBOT = 1`, passado já no `put_data`: 16 384
  mundos de 8,9 para 2,0 GB; maior `nworld` que roda de 16 384 para 98 304 (o maior testado), mesmo dec/s.
- **Achado para depois**: o MJX-Warp usa o mesmo default de `naccdmax`; provável causa do
  OOM do bench MJX a 8 192.
