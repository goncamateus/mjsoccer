# 12: Paridade MJX-Warp × CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `validation_sweep.py --backend {cpu,mjx}`: cenários definidos uma vez; no MJX cada (cenário, param, seed) é um mundo do lote; CSVs com coluna `backend`; import do MJX preguiçoso. Em `test_validation` os testes de critério ganham o parâmetro `backend` (MJX passa os mesmos limiares da CPU) e entram `test_parity_deterministic` (≤ 2 % em envelope, kick e COR; ≤ 5 % no tempo de chegada; ≤ 1 cm / 1° no erro de chegada, mesmo estado inicial em `qpos`/`qvel`) e `test_parity_seeded` (mediana MJX no [p25, p75] da CPU, seeds 0-4, amostras simétricas). Casos MJX pulados sem CUDA. Comandos documentados no `docs/index.md`, com o sweep sob `systemd-run --user --scope -p MemoryMax=...`.

**Blocked by:** 05 (Critérios de aceitação do controlador de navegação), 11 (As quatro tasks restantes no MJX)

**Status:** ready-for-agent

- [ ] `validation_sweep.py --backend mjx --seeds 0-4` roda e grava CSVs com coluna `backend`
- [ ] Todos os critérios passam no MJX com os limiares da CPU
- [ ] `test_parity_deterministic` e `test_parity_seeded` verdes nesta máquina
- [ ] Comandos de teste por camada documentados nos docstrings e no `docs/index.md`
