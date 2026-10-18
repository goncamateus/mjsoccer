# Decisão: alvos de calibração do chute e do chip

Type: grilling
Status: resolved
Blocked by: 02, 05

## Pergunta

Que distribuição de velocidade de saída o chutador deve produzir, e os parâmetros do mecanismo
precisam ser reajustados para atingi-la?

Hoje o chute é um mecanismo de verdade: `KICK_FORCE = 200` N aplicados por exatamente um step de
controle (16 ms = 8 substeps) a um êmbolo de 0.1413 kg em um trilho `range="0 0.03"` contra
`stiffness="2000"` / `damping="2.0"`, com um bloqueio por software de 250 ms por êmbolo
(`base.py:84-90,160-163`, `models/robot_rollers.xml:270-296`). A velocidade de saída é *emergente* —
`base.py:35` documenta 6-7 m/s para um chute rasteiro. O chipper é o mesmo mecanismo de novo.

Pontos em aberto:
1. **6-7 m/s é legal e realista?** O `02` fornece o teto do livro de regras. Se o livro de regras diz
   6.5 m/s, um mecanismo que emergentemente o ultrapassa é um defeito.
2. **A velocidade de saída sobrevive ao `05`?** Massa da bola, atrito bola↔placa e complacência de
   contato da placa alimentam o impulso. Medir de novo depois do `05`, não calibrar antes.
3. **Uma força fixa de 200 N é a interface certa?** Hoje a ação é um gatilho booleano (`a > 0`,
   depois uma força fixa). O firmware real da SSL comanda uma *velocidade* de chute. Decidir se a
   ação deve escalar a força e, se sim, como o pedido de uma política se mapeia em uma carga de
   capacitor.
4. **O bloqueio de 250 ms é físico?** Ele é estrutural (`base.py:84-87`: sem ele, uma ação de chute
   mantida prende o êmbolo contra o batente e a bola nunca é liberada), mas o número deveria vir de
   um tempo real de recarga do capacitor, não do que fez o êmbolo se comportar.
5. **Recuo.** O chassi absorve a reação (`L4`), algo que o rSim não tem equivalente. Confirmar que
   tem a magnitude certa em vez de ser um artefato dos 0.1413 kg do êmbolo — os êmbolos são 0.2785 kg
   dos 2.692 kg do robô e deslizam à frente do centro (`M3`).

Medido contra o cenário 5 da §13.2 do documento de pesquisa (recuo do chute com o chassi a 0/2/4 m/s)
e sua métrica de velocidade de saída.

## Answer

Resolvido em 2026-10-18. Números e scripts em [`candidate-09/`](../candidate-09/).

**O que a medição mostrou (pós-`05`).** Bola assentada, 200 N: chute 7.06 m/s no tier 0 (acima
do teto de 6.5 do `02` — defeito), 5.87 no tier 1; chip 5.49 / 4.42. Mas esses números são
**artefato de discretização**, não física: a placa fecha ~9 mm por substep de 2 ms, então o
contato é detectado a uma profundidade arbitrária e a velocidade de saída oscila ±35 % com a
força, não monotônica (`fine_current.txt`; a 450 N o chute do tier 0 sai a 3.53,
`firing_substep.txt`). Não é a
geometria (trocar os cascos da placa por uma caixa não muda nada) nem a rigidez (`solref` mais
mole piora). A dt = 0.25 ms a curva é lisa, monotônica, e os tiers concordam: 200 N → chute
5.2 m/s, chip 5.1 / 4.8 (`timestep.txt`, `solref.txt`). Ou seja, 200 N nunca chegou a 6.5.

**Decisões, todas com a equipe:**
1. **Interface de velocidade, meia faixa** (como o `08`): `a ≤ 0` não chuta; `a ∈ (0, 1]` pede
   `6.5·a` m/s. Uma tabela por (tier, êmbolo) — `scene.KICK_TABLE_MS` sobre `KICK_TABLE_N` —
   converte a velocidade pedida em força de um step via `np.interp`, o papel da tabela de
   calibração do firmware (velocidade → largura de pulso). `ctrlrange` 0..200 → **0..500 N**; o
   teto pede ~290-345 N. Calibração por `candidate-09/calibrate.py`: bola assentada e **assentada
   de vez** (30 steps parado — o `seat_ball` entrega a bola ainda acomodando, e um chip do tier 1
   disparado logo saía 9 % mais lento), robô e dribbler parados.
2. **Teto 6.5 m/s** (§8.4.2), chute e chip, nos dois tiers, **relativo ao robô parado**. Robô em
   movimento soma a própria velocidade, como no real (8.1 m/s absolutos com o chassi a 1.5 m/s no
   tier 0).
3. **Discretização: o step de controle em que um êmbolo dispara roda a `dt / 8`**
   (`scene.FIRING_SUBSTEPS`), depois volta. Bate com dt global de 0.25 ms a ~3-7 % (chip do
   tier 1 até ~11 %, absorvido pela tabela, que é medida nesse modo). Custo: **1.64-1.72×** mais lento com ação
   aleatória (que dispara a cada rearme — pior caso; `throughput.txt`), ~nada para uma política que
   chuta pouco. Muta o `MjModel` compartilhado pela duração do laço — seguro com envs que dão step
   um de cada vez. Se o `13` baixar o dt global para ≤ 0.25 ms, apagar.
4. **Bloqueio de 250 ms: aproximação deliberada.** A equipe não tem tempo de recarga do capacitor;
   o número continua estrutural (`base.py`).
5. **Recuo: físico, sem ajuste.** Cenário 5 reduzido a 0 / ~1.5 m/s (o chassi não chega a 4). No
   tier 1 parado, o CoM do robô inteiro recua 0.085 m/s; momento do robô −0.23 N·s contra +0.31 da
   bola, o resto vai para o piso pelas rodas freadas (teto `p_bola / m_robô` = 0.11 m/s). No tier 0
   o servo do chassi absorve quase tudo (−0.03), como se espera do tier ideal. A massa do êmbolo é
   a do CAD. Com o chassi andando, o tier 1 não passou de 0.68 m/s (o `18`).
6. **Chute "rasteiro" que levanta a bola 4-6 cm: aproximação deliberada.** Vira ticket só se o
   `14` mostrar efeito.

**Resultado.** `a = 1` com a bola assentada: 6.25-6.69 m/s nos quatro (±4 % entre assentamentos,
`seating.txt`). Novos testes `test_kick_exit_speed` (a = 0.5 e 1, ±0.3 m/s, chute/chip × tiers)
e `test_kick_neutral_never_fires`. Os testes de rearme continuam passando.

**Herança.** O `11`: um chute de robô em movimento passa de `MAX_BALL_V = 8` (até ~8.5 m/s), então
a observação da velocidade da bola sai de [−1, 1]. O `13`: `FIRING_SUBSTEPS` é a prova de que 2 ms
não resolve o impacto da placa; a varredura de resolução do `13` deve excluir o step de disparo ou
contar com ele.
