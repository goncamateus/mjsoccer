# O que a variante do MJX-JAX simplifica, e qual a meta de speedup

Type: grilling
Status: resolved
Blocked by: 05

## Pergunta

Com os números do 05:

1. A **variante de modelo** do MJX-JAX: a instalação mostrou que **nenhum mesh de robô** pode ficar
   em par de colisão (o robô↔campo sozinho estoura 8 GB no compile com 2 robôs), e a pesquisa 01 que
   a troca cilindro → cápsula é **opcional** (o MJX-JAX tem sphere↔cylinder SDF). Com que geometria
   substituir os 87 meshes de colisão do robô contra robô, campo e bola (casco convexo por robô,
   primitivas, poucos cascos)
   e se a troca por cápsulas ainda compensa pela precisão medida no 05? O que mais (iterações do
   solver, `max_geom_pairs`, `max_contact_points`), lembrando que a descrição física do modelo base
   é congelada e a variante paga o custo de fidelidade medido pelos critérios de aceitação?
2. Qual a **meta de speedup** por motor (sobre a CPU com 12 threads), abaixo da qual aquele motor não
   compensa ser mantido no spec?

*Contexto do 05 (2026-11-08):* com a colisão de robô inteira trocada por um prisma convexo, o MJX-JAX
fica 40–50× **abaixo** da CPU com 12 threads, e o SDF do cilindro perde 1/3 dos contatos laterais da
barra. O MJX-Warp roda o base exato, mas só 3.7× (2 robôs) e 1.4× (7 robôs) acima da CPU. A pergunta 2
inclui, portanto, se cada motor fica no destino do mapa (restrições 6 e 7).


## Answer

Resolvido em 2026-11-08 (grilling). Registrado no
[ADR 0002](../../../docs/adr/0002-mjx-warp-only.md).

1. **O MJX-JAX sai do destino, e a variante de modelo sai junto.** O 05 mostra o MJX-JAX 40–50×
   abaixo da CPU mesmo com toda colisão de robô trocada por um prisma convexo, e o SDF do cilindro
   perde 1/3 dos contatos laterais bola × barra. Nenhuma geometria fecha esse fator, então a pergunta 1
   (que geometria usar) morre. O port roda só no MJX-Warp, no modelo base, pela API JAX do
   `mujoco-mjx`.
2. **A meta de speedup é medida em decisões/s ponta a ponta** (env completo: task, controlador,
   reset) contra o pipeline CPU atual, não física pura contra o `mujoco.rollout`.
   Base CPU medida (tier 0, 40 Hz, `AsyncVectorEnv`, ação aleatória, `mujoco` 3.11): **1 535 DPS**
   com 2 robôs (ContestedPossession, 24 envs) e **299 DPS** com 7 robôs (GoToBall, 12 envs). Com 1 env
   sync: 531 / 205. A física pura do Warp (05) é 38 k / 4.6 k DPS: folga de ~25× / ~15×.
3. **Sem limiar.** O usuário já sabe que vai ter ganho: as seis tasks vão para o MJX-Warp, e nenhuma
   fica só na CPU.
4. **Ordem com medição.** O spec porta primeiro ContestedPossession e GoToBall (uma task por cena), com
   o controlador de navegação, e registra as decisões/s ponta a ponta contra a base acima (remedida no
   4.0). É relatório, não portão: serve para pegar cedo uma regressão grosseira da lógica em JAX.
