# 05: Critérios de aceitação do controlador de navegação

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** Cenários novos no sweep e no `-m validation`, do repouso, alvos a 0.5 / 2 / 6 m e giros de 0 / 90 / 180°: chegada (erro final < 5 cm e < 2.5°), tempo (≤ 1.25× o trapézio analítico: 3.3 m/s, sobe a 1.30, freia a 1.0), chatter no alvo (RMS de velocidade no último segundo < 0.05 m/s) e teto (`ω` comandado ≤ 37.1). Os `spin` e `kick_recoil` são reescritos em target pose ou velocidade crua. Os números são ponto de partida: medir na CPU primeiro e fixar o limiar no medido mais margem.

**Blocked by:** 04 (Ação em target pose e controlador de navegação no backend CPU)

**Status:** ready-for-agent

- [ ] Sweep roda a grade de alvos e giros e grava os CSVs
- [ ] Limiares fixados a partir da medição CPU (registrar medido e margem neste ticket)
- [ ] `test_validation` asserta chegada, tempo, chatter e teto; `spin` e `kick_recoil` ativos
- [ ] `uv run pytest -m validation` verde
