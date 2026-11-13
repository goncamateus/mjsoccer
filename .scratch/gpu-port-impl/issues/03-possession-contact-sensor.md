# 03: Posse pelo sensor `contact` no backend CPU

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** A posse deixa de varrer `data.contact` e lê um sensor `contact` geom-geom (bola × barra do dribbler), a mesma definição que o MJX vai ler. O sensor não altera a descrição física do modelo base.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** ready-for-agent

- [ ] Posse calculada só a partir do sensor `contact`; nenhuma varredura de `data.contact` na lógica da task
- [ ] Geometria, massas, contato e atuadores do modelo base inalterados
- [ ] Testes de posse (bola assentada, giro de 2 rad/s) verdes
