# O que a variante do MJX-JAX simplifica, e qual a meta de speedup

Type: grilling
Status: open
Blocked by: 05

## Pergunta

Com os números do 05:

1. A **variante de modelo** do MJX-JAX fica só com a troca cilindro → cápsula, ou também troca os
   meshes de colisão robô↔robô e robô↔parede por um casco convexo por robô? O que mais (iterações do
   solver, `max_geom_pairs`, `max_contact_points`), lembrando que a descrição física do modelo base
   é congelada e a variante paga o custo de fidelidade medido pelos critérios de aceitação?
2. Qual a **meta de speedup** por motor (sobre a CPU com 12 threads), abaixo da qual aquele motor não
   compensa ser mantido no spec?
