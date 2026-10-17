# Decisão: o chassi deve colidir com o piso?

Type: grilling
Status: open
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
