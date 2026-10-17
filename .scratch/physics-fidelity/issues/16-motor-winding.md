# Qual enrolamento do EC 45 flat 50 W este robô usa?

Type: task
Status: open
Blocked by: —

## Pergunta

Nada a decidir até chegar um fato, e o fato não está neste repositório — está na documentação de
hardware da equipe ou no próprio motor. HITL: alguém precisa olhar.

A maxon cataloga cinco peças como "EC 45 flat … 50 W" em três enrolamentos e três diâmetros de
carcaça. O `01` assumiu **251601** (Ø42.8 mm, 24 V, `k_M = 33.5` mN·m/A, stall 780 mN·m). A variação
entre as cinco não é acadêmica:

| peça | `k_M` (mN·m/A) | stall (mN·m) | torque de projeto a 10 % do stall | `a_max` frontal resultante |
|---|---|---|---|---|
| 339286 / 400107 (24 V, baixa velocidade) | 47.5 | 402 | 40.2 | **1.54 m/s²** |
| 251601 (24 V) — *assumido* | 33.5 | 780 | 78.0 | **2.98 m/s²** |
| 339287 (36 V) | 101 | 484 | 48.4 | ~1.85 m/s² |
| 651607 (Ø43.5, 24 V) | 36 | 918 | 91.8 | **3.51 m/s²** |

Um fator de 2.3 entre os extremos, e ele se propaga direto para o `forcerange`, o `a_max`, o
`forcerange` dos trilhos do tier 0 (`12`), e todo critério de aceitação que menciona aceleração
(`14`).

O que resolve, do mais barato primeiro:
1. O número da peça impresso no motor, ou a BOM / registros de compra da equipe.
2. A tensão de alimentação em que o robô opera — 36 V seleciona imediatamente o 339287 e descarta
   três.
3. Um ETDP da RobôCIn ou documento interno de hardware que nomeie a peça. O ETDP de 2024 nomeia a
   família do motor e o ponto de projeto do acionamento direto, mas não o enrolamento.
4. Se nada disso der certo: medir. A corrente de stall a uma tensão conhecida dá `k_M` diretamente.

Se não puder ser resolvido, o `04` segue com o 251601 e o registra como suposição declarada, com a
faixa de 1.54–3.51 m/s² escrita na documentação — mas uma linha da equipe remove a maior incerteza
deste mapa inteiro, então pergunte antes de assumir.
