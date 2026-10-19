# Aplicar a calibração de restituição da bola do `17` em `models/`

Type: task
Status: resolved
Blocked by: —

## Pergunta

O [`17`](17-ball-restitution-calibration.md) achou os `solref` que dão à bola o COR medido (alvo
0.50, faixa 0.47–0.56, do `05`) e deixou a edição para quando o mapa fechasse. O mapa fechou (todos os
tickets `resolved`), e `models/` continua crítico: a bola quica com COR 0.00–0.11 no piso e
0.11–0.14 na parede.

Aplicar, como o `17` especifica:
1. Piso: `solref="0.012 0.225"` nos dois `<pair geom1="ball" geom2="floor">` (`scene.xml`,
   `scene_rollers.xml`).
2. Parede: um `<pair>` bola↔cada `col_f_walls_*` (12) com `solref="0.012 0.25"`, num `<contact>` em
   `ball.xml`, que cobre os dois tiers. Não pela geom da bola, cujo `solref` entra em todo contato sem
   par (casco e chutador, calibrados pelo `09`). O `<pair>` não herda atrito nem `condim`, então
   repetir o contato misturado de hoje: `condim="4" friction="0.6 0.6 0.005 0.0002 0.0002"`.
3. `scene.build` replica por robô todo par da bola que não seja o do piso. Os pares de parede não são
   do robô e não podem ser replicados.

Fazer:
- Medir o COR no modelo real (sem trocar `solref` no `MjModel` carregado), piso e parede, nos dois
  tiers e a 1/960 s, de 0.5 a 6.5 m/s.
- Medir de novo o que a bola toca no piso: a tabela de chute e o chip do `09`, a retenção do
  `ContestedPossession`/`Dribbling` (o `13` tem os números), e a altura da bola parada (`11`).
- Rodar a suíte, a suíte `validation` e o sweep do `14` (seeds 0-4).

## Answer

Resolvido em 2026-10-18. As edições do `17` foram aplicadas:
- **Piso:** `solref="0.012 0.225"` no par bola↔piso. O par saiu de `scene.xml` e de
  `scene_rollers.xml` e foi para `models/ball.xml`, que os dois tiers incluem; fica uma cópia só.
- **Parede:** 12 `<pair>` bola↔`col_f_walls_1_c00..c11` em `models/ball.xml`, pela classe
  `ball_wall` (`condim="4" friction="0.6 0.6 0.005 0.0002 0.0002" solref="0.012 0.25"`), com os
  números escritos uma vez.
- **`scene.build`:** replicava todo par da bola que não fosse o do piso, então mandaria as paredes
  com prefixo de robô e o compile quebraria. Agora ele replica só os pares cuja geom é do robô
  (`find_all` no corpo `robot`); os do campo ficam, uma vez.

Medições em [`../candidate-21/`](../candidate-21/). O "antes" usa uma cópia de `models/` em `0cced59`,
via `MJSOCCER_MODELS`.

**COR no modelo como sai** ([`cor.txt`](../candidate-21/cor.txt), o harness do `17` sem trocar
nada no `MjModel`): o piso dá 0.476–0.547 e a parede 0.485–0.529, de 0.5 a 6.5 m/s a 1/960 s. A
0.5 ms, 0.508–0.530 e 0.499–0.508, também na faixa; as linhas são idênticas às do `17`. A regra de
5 % entre os passos vale a partir de 1 m/s: o piso a 0.5 m/s fica 6.3 % fora, como o `17` registrou. O par explícito **substitui** o contato
dinâmico em vez de somar: na batida na parede há um contato só, com `solref` `0.012 0.25`. Isso vale
também para o par do piso, que já existia.

**O que a bola toca no piso** ([`ball_before.txt`](../candidate-21/ball_before.txt) →
[`ball_after.txt`](../candidate-21/ball_after.txt), tier 0 / tier 1):

| | antes | depois |
|---|---|---|
| penetração em repouso | 0.04 / 0.04 mm | 0.01 / 0.01 mm |
| rolagem 3 m/s, 2 s | 5.467 / 5.467 m | igual |
| COR de queda de 0.3 m | 0.101 / 0.101 | **0.497 / 0.497** |
| chute 50 % / 100 % | 3.30 / 6.70, 3.41 / 6.53 m/s | 3.31 / 6.71, 3.45 / 6.55 |
| chip 100 % | 6.17 / 6.57 m/s | 6.15 / 6.49 |
| chip: ápice, 1º pouso | 0.418 m, 3.15 m / 0.444, 3.48 | 0.423, 3.15 / 0.466, 3.50 |
| retenção `ContestedPossession` (120) | 116.9 (114) / 111.5 (102) | **120 (120) / 118 (118)** |
| retenção `Dribbling` (120) | 119 / 119 | 117 / 119 |

O chute muda até 1.2 % (chip do tier 1), dentro do `test_kick_exit_speed`. A tabela do `09` fica.
O COR de queda aqui é a métrica do `13` (pico de velocidade antes e depois), que lê 0.101 / 0.497; o
`oblique.py` abaixo usa a do `17` (último passo antes do contato), que lê 0.048 / 0.511 no mesmo
quique. A retenção agora é igual em todas as seeds, e os limites do `test_seated_ball_is_possessed`
seguem a regra dele (5 % abaixo da pior seed do pior tier): 106 → 112 e 109 → 111.

**O chip já quicava antes.** A razão entre o segundo e o primeiro ápice dá um COR vertical efetivo
do primeiro pouso (√(h₂/h₁), `ball_*.txt`) de 0.53 / 0.53 antes e 0.59 / 0.60 depois. A diferença do chip
para a queda é a velocidade horizontal com backspin. [`oblique.py`](../candidate-21/oblique.py)
mede queda de 0.3 m com velocidade horizontal:

| vx (m/s) | 0 | 1 | 2 | 4 | 6 |
|---|---|---|---|---|---|
| antes, sem spin / rolando | 0.048 / 0.048 | 0.065 / 0.073 | 0.083 / 0.092 | 0.196 / 0.127 | 0.345 / 0.157 |
| depois, sem spin / rolando | 0.511 / 0.511 | 0.513 / 0.513 | 0.517 / 0.515 | 0.526 / 0.519 | 0.545 / 0.523 |

O atrito com deslizamento devolve energia na normal, um artefato do contato macio. Antes ele
dominava o quique oblíquo (0.05 → 0.35); agora fica dentro da faixa até 6 m/s sem spin ou rolando.
Com backspin de chip passa um pouco (0.59–0.60). É o item "restituição oblíqua" do mapa, que segue sem
fonte: registrado, não calibrado.

**Sweep, seeds 0-4** ([`sweep/`](../candidate-21/sweep/)):
- 0 tombamentos, nada acima de `MAX_W`.
- O `head_on` a 4 m/s no tier 1 baixa o tilt p99 de 6.2° para 3.0°.
- O `dribble_turn` (tier 1) perde a bola mais cedo: a ≥ 20 rad/s a posse dura 2.5–3 passos, era 8–9.
  Ela já se perdia a ≥ 10 rad/s (`14`); parado, segue 0.99.
- O `competent` no `Dribbling` segura a bola em 0.99 dos passos antes e depois. No tier 1, a fração
  vai de 0.996–0.998 para 0.946–0.995 (seed 3), e a posse se quebra em mais trechos curtos (pela
  tolerância de 50 ms). Por isso a média por trecho cai (390 → 257) e a mediana despenca; no tier 0
  os trechos se juntam (365 → 799).
- Vazão dentro de ±3 %, exceto o `Dribbling` tier 0, que cai 6 % (1227 → 1153), dentro dos 10 %
  do critério 7.

**Testes:**
- `test_ball_bounces_at_its_measured_cor`, novo: piso e parede, a 1 e 6.5 m/s, nos dois tiers, faixa
  0.47–0.56.
- `test_ball_pairs_replicated` agora exige os pares do campo (piso e 12 paredes) uma vez cada.
- 100 testes na suíte normal e 52 na `validation` passam.

Documentação: a linha de restituição da tabela de contato e o parágrafo Multi-robot de
`docs/index.md`, a linha `C2` da tabela de status da auditoria e a docstring de `scene.py`. Fora do
escopo e sem mudança: as traves (`col_f_goalpost_*`), que continuam criticamente amortecidas.
