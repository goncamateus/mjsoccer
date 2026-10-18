# Decisão: tornar honesto o contrato de observação

Type: grilling
Status: resolved
Blocked by: 04

## Pergunta

Três defeitos, e todos precisam do envelope final do `04` antes de poderem ser corrigidos:

1. **`MAX_W = 30.0` não é o máximo do tier 1.** Está documentado como "the tier-0 chassis servo
   `ctrlrange`" (`base.py:34`, `docs/index.md:133`), mas o tier 1 chega a 59.4 rad/s medidos, então o
   canal `omega` rotineiramente lê ≈2.0 sob o próprio acionamento do robô. `docs/index.md:135-137`
   afirma que os valores "sit in roughly [−1, 1]" e culpa as excursões em "a tier-1 chassis pushed by
   another robot" — falso hoje (`L9`). Depois do `04` isso pode se resolver sozinho; confirmar, e
   decidir se `MAX_V`/`MAX_W` devem ser *derivados do modelo* em vez de declarados como constantes.
2. **O canal de altura da bola tem um erro de referência.** `base.py:220` divide o `bp[2]` bruto do
   mundo por `GOAL_HEIGHT = 0.155`, mas o plano do piso está em z = 0.02 (`models/field.xml:42`),
   então o canal carrega um deslocamento sistemático de `+0.02/0.155 = +0.129` e mede a altura acima
   da origem do mundo, não acima do carpete (`U4`). Corrigir a observação, ou mover o piso para
   z = 0? Mover o piso mexe em todo z de spawn do modelo; corrigir a observação é uma subtração.
   Decidir qual, e se mais alguma coisa lê um z bruto.
3. **O `Box` é `(−inf, inf)` e nada faz clip** (`base.py:79`). O rSoccer faz clip de todo canal em
   ±1.2 (`ssl_gym_base.py:27,213-220`). Sob fidelidade, um espaço de observação ilimitado é
   correto-mas-inconveniente, ou o `Box` declarado deveria ao menos ser *verdadeiro*?

Resolvidos juntos porque os três são a mesma pergunta: o contrato de observação descreve o simulador,
ou aspira a ele.

## Comments
- 2026-10-18, do `09`: chutes são limitados a 6.5 m/s **relativos ao robô parado**; um robô em
  movimento soma a própria velocidade e a bola sai a até ~8.5 m/s, acima de `MAX_BALL_V = 8`.

## Answer

Resolvido em 2026-10-18, por grilling, no `main` com `04`-`10` aplicados. Medições em
[`candidate-11/`](../candidate-11/): `extremes.py` (máximo de |obs| por grupo de canais, ação aleatória,
seeds 0-19, seis envs, dois tiers → `extremes_t{0,1}.csv`) e `bounds.py` (o `Box` novo contra episódios
inteiros, seeds 0-9 → `bounds_t{0,1}.csv`).

**O contrato descreve o simulador.** Nada faz clip; o `Box` declara limite finito só onde a física o
garante.

1. **`MAX_V`/`MAX_W` ficam constantes (4.0, 30.0), presos ao `ctrlrange` do servo do tier 0.** O
   item se resolveu com o `04`: sob o próprio acionamento, o tier 1 gira a 29.2 rad/s no comando
   cheio (< 30, já fixado em `test_full_spin_stays_under_max_w`) e anda a ~2 m/s. Ação aleatória:
   |v|/`MAX_V` e |ω|/`MAX_W` chegam a 0.38/0.14 no tier 1 e 0.51/0.88 no tier 0. Não derivados do
   modelo: o tier 1 não carrega o modelo do tier 0, e o mesmo divisor nos dois tiers é o que faz uma
   observação significar uma velocidade. Novo `test_velocity_normalisers_are_the_tier0_servo_range`
   falha se o `12` mexer no servo sem revisitar os divisores.
2. **Altura da bola: subtrair o piso.** `scene.FLOOR_Z = 0.02`, conferido contra o geom `floor` em
   `_check_field`. Bola parada lia **0.267**; agora lê 0.139 (= raio/`GOAL_HEIGHT`)
   (`test_ball_height_is_above_the_carpet`). **Outro leitor de z bruto:** `scene.in_goal` comparava
   o z do mundo com `GOAL_HEIGHT`, então o gol tinha 0.135 m de boca acima do carpete, não 0.155.
   Corrigido igual (`test_crossbar_is_measured_from_the_carpet`): **muda a regra de gol**, um chute
   com o centro da bola entre 0.135 e 0.155 m acima do carpete agora marca. Mover o piso para z = 0
   foi descartado (mexe em todo z de spawn e malha).
3. **`Box` com limite por canal, verdadeiro** (`test_observation_bounds_are_true`):

   | canal | limite | por quê |
   |---|---|---|
   | x, y do robô | ±1.05, ±1.067 | faces internas da parede, `scene.WALL_HALF_X = 6.3`, `WALL_HALF_Y = 4.8` m (conferidas em `_check_field`) |
   | sin, cos, `has_ball` | ±1; 0, 1 | por construção |
   | z da bola | 0 a inf | a bola apoia no carpete; chip não tem teto (lido até ~3) |
   | x, y da bola | inf | a parede tem 0.2 m e um chip passa por cima (ápice medido 0.47 m) |
   | toda velocidade | inf | colisão não tem teto |
   | waypoint do Dribbling | ±2.57, ±3.427 | `hypot(6 + 6.3, 4.5 + 4.8)`: waypoint no campo visto de um robô dentro das paredes |

   `bounds.py`: **0 observações fora do `Box` em ~152 mil steps**; a folga mínima é a do z da bola
   (18-21 mm, o raio menos o esmagamento do contato). O centro do robô fica ≥ ~80 mm dentro das
   paredes. O `contains` do `test_steps_finite` passa a conferir limites de verdade. Ressalva: o
   Dribbling, único env com waypoint e o que mais chipa, só soma ~900 steps por tier (episódios
   curtos); o limite do waypoint vale por construção (rotação preserva norma), não pela amostra.
   À parte: `GOAL_HEIGHT = 0.155` (regra) contra 0.150 m de parede acima do carpete no CAD — o
   `_check_field` não confere isso; não mexido aqui.
4. **`MAX_BALL_V` fica em 8** (comentário do `09`): o chute de um robô em movimento lê até ~1.06.
   Documentado; com velocidade ilimitada no `Box`, não há mentira.

Docs: `docs/index.md` (seção de observação: tabela de limites, números medidos, `FLOOR_Z`/`WALL_*` na
tabela de constantes; a frase falsa "roughly [-1, 1]" / "pushed by another robot" — item 3 do `15` —
saiu) e as tabelas Min/Max das seis páginas de ambiente, mais a regra de gol em
`go_to_ball_shoot.md` e `static_defenders.md`.
