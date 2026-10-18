# Decisão: o chassi deve colidir com o piso?

Type: grilling
Status: resolved
Blocked by: —

## Pergunta

Hoje ele não consegue. As geoms do chassi são `contype="1" conaffinity="1"` e o piso é
`contype="4" conaffinity="4"` (`models/field.xml:42`), e o MuJoCo exige
`(contype1 & conaffinity2) | (contype2 & conaffinity1) ≠ 0` — que dá zero. O rSim tem a mesma lacuna
por outro motivo: `sslworld.cpp:171-197` nunca cria uma superfície chassi/chão (`B5`).

**Só as rodas tocam o piso, nos dois simuladores.** Então um robô tombado não tem onde cair: fica
pendurado no aro de uma roda ou o chassi atravessa o carpete.

Com um destino de fidelidade ao hardware isso parece um defeito, não uma escolha — um chassi real
deitado de lado está em contato com o carpete. Mas:

1. O comportamento pós-queda de um robô tombado é algo que este simulador precisa representar? Hoje
   o episódio termina em 60° (`TO1`), então esse estado nunca dura mais que uma fração de segundo.
   Se a terminação ficar, a queda nunca é simulada e a lacuna sai de graça.
2. Quanto custa habilitar? 209 cascos (hulls) do chassi × um teste contra o plano a cada step, por
   robô, até 7 robôs. A linha de base de tempo de relógio do `03` limita a resposta, e o `07` pode
   podar os cascos de qualquer forma.
3. Existe um meio-termo barato: um único casco convexo ou uma primitiva única como proxy de colisão
   *só com o piso* para o chassi, em seu próprio bit de contype, deixando os 209 cascos para
   robô↔robô?

Este ticket está desbloqueado porque é uma questão de escopo, não de ajuste — e a resposta molda o
design da máscara de geoms do `07`.

## Answer

**Sim: habilitar, com a colisão completa.** Piso `contype="5" conaffinity="5"` (era 4/4,
`models/field.xml:42`): o bit 1 passa a casar com os cascos do chassi e dos cubos de roda. Sem proxy,
sem geometria nova. Aplicado junto com o `04`/`05` (mesma exceção registrada no mapa), com o teste
`test_chassis_rests_on_carpet`. Medições em [`../candidate-06/`](../candidate-06/), contra os modelos
pós-`05`.

1. **Importa mesmo com o término a 60°.** Girando o robô sobre os roletes, o primeiro vértice do
   chassi chega ao carpete a **9.5° no mínimo, 25.5° na mediana** (59 de 72 direções de
   inclinação; nas outras 13 os roletes continuam sendo o ponto mais baixo até 90°)
   (`touch_current.txt`). Ou seja, o chassi deveria encostar dentro da faixa 10-60° em que o `10`
   decide o que é recuperável, e hoje o robô continua pivotando sobre os roletes. Também: robôs não
   controlados nunca terminam o episódio. Queda lateral, 2 s parado: sem colisão o robô atravessa e
   termina **de costas** (180°, CoM a 1.6 mm do carpete); com colisão fica de lado (89°, CoM a 87 mm).
2. **Custo: ruído.** O ticket supunha 209 cascos; os candidatos novos contra o piso são **111 por
   robô** (87 do chassi + 24 dos cubos de roda, `clearance.txt`) — o resto são roletes, que já tocam o piso. Com uma
   folga de 3-4 mm em repouso (`clearance.txt`) e p99 de inclinação de 4° (`../candidate-04/summary.csv`), o broadphase descarta todos: zero contatos
   chassi↔piso em 31 250 steps de ação aleatória (GoToBall + StaticDefenders, 10 seeds cada) e em 2 s
   de comando cheio em x e y. Vazão (melhor de 3, 2000 steps, 3 processos em paralelo): GoToBall
   772 → 763 steps/s, StaticDefenders 136.8 → 135.5 (−1.0 a −1.3 %, dentro do ruído; `measure_*.txt`). Os números absolutos ficam abaixo da linha de base do
   `03` (948.6 / 153.3, `../baseline/throughput.csv`) porque lá era um processo só; só a diferença
   entre variantes da mesma rodada vale. Tombamentos,
   `v` final de direção (1.92 / 0.17 m/s): idênticos.
3. **Proxy rejeitado.** Um casco convexo único (união dos 87 cascos do chassi, bit 8, piso
   `conaffinity="12"`) custa o mesmo (765 / 136.6 steps/s) e se comporta igual (89°, CoM a 81 mm), mas
   é geometria duplicada que o `07` teria que manter em sincronia. Com a colisão completa, o conjunto
   de cascos que o `07` decidir manter vale para robô↔robô e para o piso ao mesmo tempo.

Para o `07`: a máscara de geoms não precisa reservar um bit para o piso. Podar um casco agora também o
remove do contato com o piso, o que só importa para cascos da face de baixo e das bordas.
Tier 0: inalterado (só a bola toca o piso, antes e depois; `tier0_*.txt`).
