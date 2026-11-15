# 01: ADR 0003 "MJX-Warp e mjlab" + glossário

Spec: [backend mjlab](../spec.md)

**What to build:** a decisão de ter três backends (CPU oráculo, MJX-Warp, mjlab) fica registrada. A ADR 0003 substitui a 0002 ("só MJX-Warp") com o contexto do protótipo (empate de decisões/s, 16 384 mundos) e as consequências (terceira implementação da lógica em torch, mesmo contrato, mesma paridade). O glossário ganha os termos novos.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] ADR 0003 escrita, citando decisões 1-16 e os resultados do protótipo; a 0002 marcada como substituída
- [x] `CONTEXT.md` define backend mjlab e pool de primeiros estados, sem conflitar com termos existentes
