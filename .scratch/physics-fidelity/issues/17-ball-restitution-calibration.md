# Tarefa: encontrar o `solref` que dá à bola o seu COR medido

Type: task
Status: open
Blocked by: —

## Pergunta

O `05` definiu o alvo: coeficiente de restituição bola↔piso e bola↔parede ≈ 0.50 (faixa medida
0.47–0.56, ER-Force/TIGERs). O MuJoCo não tem um coeficiente de restituição literal — o quique emerge
da razão de amortecimento do `solref` — então nenhuma fórmula fechada entrega o valor; ele precisa ser
encontrado por medição, no mesmo formato do diagnóstico do `03`.

Hoje as duas superfícies são criticamente amortecidas (`solref="0.006 1"` em
`models/scene_rollers.xml:23`, `solref="0.004 1"` em `models/field.xml:44`), ou seja, COR ≈ 0.

Entregar:
1. Um harness de teste de queda: soltar a bola de uma altura conhecida acima do piso (e,
   separadamente, em direção à parede a uma velocidade normal conhecida), medir a velocidade de
   rebote, calcular `COR = v_rebound / v_impact`.
2. Varrer a razão de amortecimento do `solref` (mantendo a constante de tempo de referência, ou
   varrendo as duas, o que convergir mais rápido) até o COR medido cair em 0.47–0.56, alvo 0.50.
3. Reportar o par `solref` de cada superfície (não precisam ficar idênticos, já que piso e parede são
   geoms separadas), e se um único valor serve às duas dentro da tolerância exigida ou se precisam
   divergir.
4. Registrar qualquer dependência de velocidade observada (as medições de origem mostram o COR caindo
   com a velocidade de impacto — `docs/research/02-physical-constants.md` §2.1) e se isso importa nas
   velocidades de impacto que este simulador de fato produz (velocidade de saída do chute ≤ 6.5 m/s,
   segundo o `02`).

Este é um ticket `task`: mexe em `models/` só como diagnóstico revertido. Reverter depois de medir; a
edição real entra quando o mapa fechar, conforme a nota de Mode do `map.md`.

## Answer

<!-- preencher quando resolvido -->

## Comments

- 2026-10-18, do `13`: o passo global agora é **1/960 s** (~1.04 ms), não 2 ms. Calibrar o `solref`
  nesse passo. O MuJoCo prende a constante de tempo em ≥ 2·dt, e o quique depende do passo: com o
  `solref` atual o COR de queda de 0.3 m lia 0.136 / 0.107 / 0.144 / 0.151 a 0.5 / 1 / 2 / 4 ms
  (`../candidate-13/sweep_t*.txt`). Conferir o resultado a 0.5 ms antes de fechar; se divergir mais
  de 5 %, a constante de tempo é curta demais para o passo.
