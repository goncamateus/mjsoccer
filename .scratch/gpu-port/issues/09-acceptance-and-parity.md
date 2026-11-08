# Que critérios de aceitação sobrevivem, e com que tolerâncias de paridade

Type: grilling
Status: open
Blocked by: 05

## Pergunta

Sem o tier 1 e com o controlador de navegação entre a política e o chassi:

1. Quais critérios de aceitação de hoje (`tests/test_validation.py`, `scripts/validation_sweep.py`,
   `docs/index.md`) continuam valendo para o robô cinemático, quais caem com o tier 1, e quais
   critérios novos o controlador pede (ex.: tempo e erro de chegada a uma target pose)?
2. Para cada critério que fica, a **tolerância de paridade** nas duas checagens: variante × base no
   CPU, e MJX × CPU no mesmo modelo — à luz do ruído de float32 e das diferenças de motor que o 05
   mostrar.
3. Com que amostra (seeds, episódios) cada checagem roda. Sweeps de validação usam `--seeds 0-4`,
   nunca mais que 5 seeds.
