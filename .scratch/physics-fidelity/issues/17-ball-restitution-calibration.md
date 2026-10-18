# Tarefa: encontrar o `solref` que dá à bola o seu COR medido

Type: task
Status: resolved
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

**Mesma constante de tempo, amortecimentos diferentes: piso `solref="0.012 0.225"`, parede
`solref="0.012 0.25"`.** Medido a 1/960 s, de 0.5 a 6.5 m/s de impacto normal: piso **0.476–0.547**
(média 0.513), parede **0.485–0.529** (média 0.509). A 0.5 ms: piso 0.508–0.530, parede 0.499–0.508,
tudo dentro de 0.47–0.56 nos dois passos. Hoje o piso lê 0.00–0.11 e a parede 0.11–0.14.

Harness: [`candidate-17/drop.py`](../candidate-17/drop.py) → [`drop.txt`](../candidate-17/drop.txt).
Lança a bola a 1.0–1.9 mm da superfície com velocidade normal fixa e lê `COR = v_out / v_in`, com
`v_in` no último passo antes do contato e `v_out` no primeiro passo sem contato. O COR é a média sobre
quatro folgas iniciais, porque a fase do passo no início do contato mexe ±2 % nele. O `solref` é
trocado no `MjModel` carregado (`pair_solref` do par bola↔piso; `geom_solref` da bola e das
`col_f_walls`, para a média do `solmix` dar o valor testado). **Nada em `models/` foi editado**, então
não há o que reverter.

| `solref` | piso 1/960 s (0.5 → 6.5 m/s) | parede 1/960 s | desvio 0.5 ms (v ≥ 1) piso / parede | penetração @6.5 piso / parede | contato @6.5 |
|---|---|---|---|---|---|
| atual: piso `0.006 1`, parede mix `0.005 1` | 0.00 → 0.11 | 0.11 → 0.14 | — | 14.1 / 11.7 mm | 23 / 18 ms |
| **piso `0.012 0.225`, parede `0.012 0.25`** | 0.476 → 0.547 | 0.485 → 0.529 | 3.2 % / 4.7 % | 12.9 / 14.0 mm | 8.3 / 9.4 ms |
| valor único `0.02 0.22` | 0.478 → 0.551 | 0.532 → 0.558 | 2.7 % / 2.6 % | 21.3 / 21.1 mm | 13.5 ms |

**Por que a constante de tempo não pode ser curta.** COR 0.50 pede razão de amortecimento ≈ 0.22–0.25,
e no MuJoCo a rigidez escala com `1/(timeconst·dampratio)²`: baixar o amortecimento endurece a mola.
Com `timeconst` 0.004–0.006 o contato dura 2–4 passos, e o COR diverge 10–50 % do de 0.5 ms. O
comentário do `13` pedia ≤ 5 %: 0.010 passa na parede e falha no piso (nenhum `dampratio` o põe na faixa, e `0.22` diverge 6.3 % a v ≥ 1), e
**0.012 é a menor constante de tempo que passa nas duas**. Daí a penetração: ~13–14 mm a 6.5 m/s e
contato de ~8–9 ms, da ordem de hoje (14.1 / 11.7 mm). A bola nunca atravessa a parede de 44 mm
(`field side True` em todas as linhas).

**Ressalva na regra de 5 %.** Tomada sobre *todas* as velocidades, nenhuma linha do piso passa: a
0.5 m/s o piso diverge 6–9 % entre os passos em qualquer `solref` (6.3 % no escolhido). Isso é um quique
de 1.3 cm de altura, com contato de ~8 ms em que a gravidade tira ~15 % de `v_in`. A causa da diferença
entre passos não foi isolada. Li a regra para v ≥ 1 m/s; se ela precisar valer a 0.5 m/s, nenhum
`solref` desta varredura a satisfaz no piso.

**Piso e parede precisam divergir no amortecimento.** No mesmo `solref` a parede lê ~0.02–0.03 acima do
piso (sem gravidade durante o contato, e é uma malha, não um plano). Com `timeconst` ≤ 0.015 nenhum
`dampratio` põe as duas superfícies na faixa ao mesmo tempo: `0.012 0.225` deixa a parede em 0.556, e
`0.012 0.24` deixa o piso em 0.451. Um valor único só existe em `0.02 0.22`, ao custo de 21 mm de
penetração (o raio da bola é 21.5 mm) e 13.5 ms de contato. Como piso e parede vão por `<pair>`
separados de qualquer jeito (abaixo), divergir não custa nada.

**Dependência da velocidade.** No MuJoCo o COR **sobe** com a velocidade de impacto (piso 0.48 → 0.55
de 0.5 a 6.5 m/s), o contrário da bola real (cai, `02` §2.1: −1 % em +14 % de velocidade). A variação
total é de ~0.07, e a faixa inteira até 6.5 m/s cabe em 0.47–0.56. Então ela **não importa** nas
velocidades que este simulador produz, cujo teto é o chute de 6.5 m/s.

**Para a edição real, quando o mapa fechar:**
- Piso: `solref="0.012 0.225"` nos dois `<pair geom1="ball" geom2="floor">` (`scene.xml`,
  `scene_rollers.xml`).
- Parede: **não** pela geom da bola. O `solref` da geom da bola entra na média de todo contato da bola
  sem par explícito, inclusive o casco e o chutador do robô, que o `09` calibrou. Usar `<pair>` bola↔cada
  `col_f_walls_*` (12) com `solref="0.012 0.25"`. Um `<contact>` em `ball.xml` cobre os dois tiers. O
  `<pair>` **não herda** atrito nem `condim` das geoms (cai em μ = 1, `condim` 3, verificado), então
  repetir o que o contato misturado usa hoje: `condim="4" friction="0.6 0.6 0.005 0.0002 0.0002"`.
- Fora do escopo, mas visto: os `col_f_goalpost_*` continuam críticos (`0.004 1`); e o casco convexo
  `col_f_walls_1_c05` está enviesado: a face interna vai de y 4.776 (x 6.3) a ~4.80 (x −6.1), então a
  parede "se move" 2.4 cm ao longo do comprimento.

## Comments

- 2026-10-18, do `13`: o passo global agora é **1/960 s** (~1.04 ms), não 2 ms. Calibrar o `solref`
  nesse passo. O MuJoCo prende a constante de tempo em ≥ 2·dt, e o quique depende do passo: com o
  `solref` atual o COR de queda de 0.3 m lia 0.136 / 0.107 / 0.144 / 0.151 a 0.5 / 1 / 2 / 4 ms
  (`../candidate-13/sweep_t*.txt`). Conferir o resultado a 0.5 ms antes de fechar; se divergir mais
  de 5 %, a constante de tempo é curta demais para o passo.
