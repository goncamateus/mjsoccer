# 11: `--num-envs` default por cena (treino longo)

Spec: [backend mjlab](../spec.md) — Fase 3

**What to build:** o default do `--num-envs` do script mjlab é o maior `nworld` estável num treino longo (não os ~40 s do protótipo), medido para a cena de 2 robôs e a de 7, com 0 overflow e memória plana.

**Blocked by:** 05, 06, 08

**Status:** done

- [x] Treino longo medido com 2 e 7 robôs em candidatos de `nworld` (incluindo 8 192 e 16 384 com 2 robôs)
- [x] Default gravado no script por cena, com a medição registrada

## Comments

Done (ticket 11). Relatório: [research/06-num-envs.md](../research/06-num-envs.md); logs em
`bench/train11/`. `NUM_ENVS = {2: 8192, 7: 16384}` em `examples/ppo_mjlab.py`, que agora
imprime por época as falhas do treino e a memória do device.

- Memória plana nas 7 execuções (CP 8 192-32 768, ~32 M decisões; GoToBall 4 096-32 768,
  ~5 M). GoToBall: 0 falhas em todos; 32 768 roda mas usa 10,6/12 GB.
- **Desvio do "0 overflow"**: na CP o treino teve overflow raro (13 em 31 M a 8 192; 3 + 1
  diverged a 16 384; 0 a 32 768), dependente da política, não do `nworld`. Só o mundo
  afetado termina. Tipo do bit não medido (ver relatório).
- 2 robôs fica em 8 192 e não 32 768 (o maior estável) por decisão do usuário: dps satura e
  o lote maior aprende menos por decisão.
