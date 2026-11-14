# 05: Streams torch × Warp e memória do device × mempool

Spec: [backend mjlab](../spec.md) — Fase 2 (Streams), Further Notes

**What to build:** duas incógnitas do protótipo viram decisões medidas. (1) O mjlab não sincroniza o stream do torch com o do Warp; o protótipo rodou tudo no stream do Warp. Medir se rodar fora dele corrompe dados e documentar a decisão. (2) A 16 384 mundos o device usa 10,9 GB e o mempool do Warp só 1,3 GB de pico; explicar a diferença (mempool reservado, grafo capturado duas vezes, contexto CUDA torch + Warp) e cortar o desperdício se der.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Decisão de stream tomada, com a medição que a sustenta, documentada no código e no relatório
- [ ] Diferença de memória decomposta por fonte, com números
- [ ] Se houver desperdício evitável (ex.: grafo capturado duas vezes), removido e o `nworld` máximo remedido
