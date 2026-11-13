# Onde moram os testes do backend MJX-Warp

Type: grilling
Status: resolved
Blocked by: 16

## Pergunta

O MJX-Warp precisa de GPU CUDA e do grupo opcional `gpu`; o teste de task-logic parity
([ticket 10](10-shared-task-logic.md)) roda em JAX-CPU. Decidir, para as checagens do
[ticket de aceitação e paridade](09-acceptance-and-parity.md) e as do kick
([ticket do kick no lote](16-kick-in-batch.md)):
- o que entra na suíte padrão (pulado sem `gpu`/sem CUDA) e o que fica `-m validation`;
- se a paridade MJX × CPU vira `scripts/validation_sweep.py --backend mjx` ou um script próprio;
- se o teste de task-logic parity entra na suíte padrão (pulado sem o grupo `gpu`);
- como o CI (se houver) e a máquina local rodam cada camada.

## Answer

Resolvido em grilling (2026-11-13).

1. **Sem CI.** Toda camada roda na máquina local; montar CI fica fora de escopo (runner hospedado
   não tem GPU). A divisão por camada deixa um CI só-CPU possível depois.
2. **Suíte padrão (`uv run pytest`)**, MJX pulado sem o grupo `gpu`:
   - `tests/test_mjx_task_logic.py`: o task-logic parity, `pytest.importorskip` do `jax`/`mujoco.mjx`,
     `JAX_PLATFORMS=cpu` forçado no topo do módulo (nunca toca a GPU); aborta com contador de falha
     > 0 (ticket 18).
   - `tests/test_mjx_warp_smoke.py`: `make` + `reset` + ~10 passos nas 6 tasks, `nworld` pequeno,
     com um kick disparado; asserta `diverged == overflow == 0` e obs finita. Pulado sem `gpu`/CUDA
     (`jax.devices("gpu")` falha → skip).
3. **Paridade pelo sweep**: `scripts/validation_sweep.py --backend {cpu,mjx}`, cenários definidos uma
   vez; no MJX cada (cenário, param, seed) é um mundo do lote; CSVs ganham coluna `backend`. Import do
   MJX preguiçoso, só no ramo `mjx`. Sem script próprio.
4. **Asserções em `tests/test_validation.py` (`-m validation`)**: os testes de critério ganham
   parâmetro `backend` (MJX passa os mesmos limiares da CPU), mais `test_parity_deterministic`
   (≤ 2 %, ≤ 5 % no tempo de chegada, ≤ 1 cm / 1°) e `test_parity_seeded` (mediana MJX no
   [p25, p75] da CPU, seeds 0-4), via `episode(..., backend=)` do sweep. Casos MJX pulados sem CUDA.
5. **Sem marker `gpu`.** A divisão continua só padrão / `validation`; a camada GPU se seleciona por
   disponibilidade (skip). Nesta máquina `-m validation` roda CPU + MJX.
6. **Kick no lote, por camada**: `_plunger_lock`, `fired` e cooldown 10 no task-logic parity; no Warp
   smoke, um mundo dispara e um ocioso tem de manter `qpos`/`qvel` bit a bit nas iterações a
   `timestep = 0`; velocidade de saída × `KICK_TABLE_MS` (≤ 2 % MJX × CPU) no `validation`, mais os
   testes de kick/chip do `physics-fidelity` no CPU 4.0.
7. **Comandos documentados** nos docstrings dos testes e no `docs/index.md`: `uv run pytest` (base);
   `uv sync --extra gpu && uv run pytest` (+ task-logic parity e smoke); `uv run pytest -m validation`
   e `validation_sweep.py --backend mjx --seeds 0-4` sob `systemd-run --user --scope -p MemoryMax=...`.
   O `MemoryMax` é comando documentado, não código no teste.
