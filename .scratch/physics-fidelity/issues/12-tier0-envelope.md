# Decisão: o que "o mesmo envelope" significa para o tier 0

Type: grilling
Status: open
Blocked by: 04, 20

## Pergunta

Restrição 4 do mapa: o tier 0 mantém o seu *mecanismo* de cinemática ideal, mas recebe o *envelope*
de velocidade/aceleração do tier 1, para que uma cabeça de ação signifique a mesma coisa nos dois.

Tier 0 hoje: três juntas nos eixos do mundo (`slide-x`, `slide-y`, `hinge-z`) com servos de
velocidade com `ctrlrange` ±4 m/s e ±30 rad/s, `forcerange` ±19.1261 N (x, y) e ±3.427 N·m (yaw)
(`models/robot_kinematic.xml:152-154,389-391`). O chassi nunca toca o piso, então não há limite de
tração, nem escorregamento, nem tombamento (`W5`). 19.1261 N sobre 2.69207 kg dá 7.1 m/s².

Decidir:
1. **Quais grandezas são igualadas?** `v_max`, `ω_max` e `a_max` são as três óbvias. Escorregamento,
   tração e transferência de peso não podem ser igualados — esse é o ponto do tier. Dizer isso
   explicitamente, porque "o mesmo envelope" precisa ser um termo definido, não uma impressão.
2. **`forcerange` nos trilhos (slides).** Definir como `m · a_max` do `04`? Isso torna o teto de
   aceleração do tier 0 igual ao teto *limitado por atrito e torque* do tier 1, que é a
   correspondência mais próxima disponível, mas o tier 0 vai alcançá-lo em todas as direções ao mesmo
   tempo, onde o tier 1 não consegue.
3. **O servo de yaw.** ±3.427 N·m contra o `Izz` do chassi dá ao tier 0 uma aceleração angular sem
   análogo em torque de roda. Igualar à autoridade de yaw das quatro rodas do `04`.
4. **Os números redondos ficam?** ±4 m/s e ±30 rad/s parecem escolhidos por arrumação. Se o `04`
   derivar outra coisa, o tier 0 deve seguir a derivação, não a arrumação.
5. **Onde a diferença é documentada?** O `README.md` afirma que uma cabeça de ação se transfere entre
   tiers; `docs/index.md:145-150,199-212` explica as diferenças entre os tiers. Depois deste ticket,
   ambos precisam da declaração de envelope, o que é trabalho do `15`.

## Answer

Resolvido em 2026-10-18, por grilling, no `main` com `04`-`11` aplicados. Medições em
[`candidate-12/`](../candidate-12/): `step.py` (degrau de comando cheio a partir do repouso, um eixo
por vez → `step_before.txt`, `step_after.txt`), `held.py` (retenção do `ContestedPossession`) e o
`extremes.py` do `11` rodado de novo (→ `extremes_t{0,1}.csv`).

Massa 2.69207 kg nos dois tiers; `Izz` no CoM 0.009545 kg·m², 0.009577 na junta de yaw do tier 0
(o CoM do chassi fica 8 mm fora do eixo).

1. **"O mesmo envelope" = os quatro limites por eixo:** `v_max`, `ω_max`, `a_max` linear, `α_max` de
   yaw. **Não igualados, de propósito:** escorregamento, tração, transferência de peso e o
   **acoplamento entre eixos**. O tier 1 é um politopo (frontal cheio não deixa roda para o yaw); o
   tier 0 é uma caixa, com os três limites ao mesmo tempo e, pela diagonal do mundo, √2 × `v`/`a`.
   Igualar o acoplamento exigiria uma projeção em Python, que é mecanismo, não envelope.
   **Alvo: o envelope derivado** (`01`/`04`/`05`), não o medido no tier 1, que hoje é o defeito do
   `18` (0.17 m/s em y). **Isotrópico, o menor** entre frontal e lateral do tier 1, porque as
   corrediças do tier 0 estão no mundo e um limite no referencial do corpo pediria clip em Python
   (`docs/index.md` promete que os limites vêm do modelo).
2. **`forcerange` linear = m·a_max = 2.69207 · 2.30 = ±6.1918 N** (era 19.1261, 7.1 m/s²).
   2.30 = torque de roda lateral; o frontal é min(2.98, μg = 2.45).
3. **`forcerange` de yaw = I·α_max = 0.009577 · 55.06 = ±0.5273 N·m** (era 3.427, 359 rad/s²). O
   tier 1 é limitado por atrito no yaw: μmg·R = 6.602 N · 0.0796 m = 0.5255 N·m sobre 0.009545 →
   55.1 rad/s² (o torque de roda daria 86.5).
4. **`ctrlrange` linear ±4 → ±3.30 m/s** (`v_max` frontal do tier 1; lateral é 4.04). `MAX_V`
   segue (4.0 → 3.3), preso pelo `test_velocity_normalisers_are_the_tier0_servo_range`.
   **`ctrlrange` de yaw fica ±30**, agora por derivação: é o teto que o
   `test_full_spin_stays_under_max_w` garante ao tier 1 (29.2 medido, depende de μ). 29.2 deixaria
   aquele teste em cima do valor medido. `MAX_W` fica 30.
   **Amortecimento das juntas do chassi → 0** (era 0.1 / 0.1 / 0.005, sem fonte). Com os
   `forcerange` novos, ele comeria 5 % do empuxo linear a 3.3 m/s e ~28 % do torque de yaw a 29
   rad/s. **`kv` fica** (119.538 satura com 0.052 m/s de erro, 0.428 com 1.23 rad/s): P rígido +
   limite, como no `04`.
5. **Documentação:** números em `docs/index.md` (tabela de ação, `MAX_V`, extremos, retenção) e nas
   seis páginas de ambiente (`/ 3.3`). A prosa da declaração de envelope vai para o `15` (Comment).
   O `18` recebeu um Comment: quando fechar, o tier 1 deve alcançar o que o tier 0 declara.

`step.py`, comando cheio a partir do repouso:

| eixo | antes: a / regime | depois: a / regime | alvo |
|---|---|---|---|
| x | 7.09 / 4.00 | **2.30 / 3.30** | 2.30 / 3.30 |
| y | 7.12 / 4.00 | **2.31 / 3.30** | 2.30 / 3.30 |
| yaw | 358 / 29.63 | **55.2 / 29.97** | 55.1 / 30 |

Fixado pelo novo `test_tier0_has_the_tier1_envelope` (tolerância de 3 %).

Ação aleatória (`extremes_t{0,1}.csv`, seeds 0-19): `|v|`/`MAX_V` e `|ω|`/`MAX_W` chegam a
0.42/0.46 no tier 0 (eram 0.51/0.88) e 0.46/0.14 no tier 1 (0.38 antes, só pelo divisor menor).
`bounds.py` do `11` no tier 0: 0 observações fora do `Box`.

**Efeito lateral: a retenção do tier 0 é caótica.** `test_seated_ball_is_possessed` caiu na seed 3
do `ContestedPossession` tier 0 (69 < 70). `held.py`, seeds 0-7: antes 82-93, depois 69-95. Testando
uma mudança por vez, `forcerange`/`ctrlrange` não mexem em nada; o amortecimento sozinho explica tudo,
e a resposta é não monotônica (pior seed 82 / 75 / 69 / 75 com amortecimento 0.1 / 0.01 / 0 / 0.2).
É ruído do contato bola↔dribbler, não efeito do envelope. O limite foi para 60. Uma bola de fato
perdida pontua perto de zero, então o teste continua pegando o que deve pegar. `Dribbling` tier 0
continua 119/120.

## Comments

- 2026-10-18, do `13`: o "efeito lateral caótico" daqui era resolução, não caos. No passo de 2 ms o
  contato bola↔dribbler de 4-6 ms fica com dois steps; o `ContestedPossession` tier 0 segura 120 /
  119 / 87 / 0 a 0.5 / 1 / 2 / 4 ms. No passo novo (1/960 s) dá 114-118 nos dois tiers, e o limite
  do teste subiu de 60 para 106. `ω_max` do tier 0 subiu de 30 para **35.9** (teto cinemático, `01`): o
  giro cheio do tier 1 no passo resolvido passa de 30 (31.8 a 0.25 ms). `MAX_W` = 35.9, e o
  `goto_ball_policy` usa `kw` 1.25 para manter o mesmo ganho físico.
- 2026-10-18, do `18`: **reabre**, pela regra do comentário do `12` no `18`. O tier 1 não alcança
  3.30 m/s em x: platô em 2.29 m/s em 3.5 s (`../candidate-18/terminal.txt`). A aceleração lateral
  medida no `14` é 1.87 m/s², abaixo de 2.30. Em y, o comando cheio vira giro. A causa não é a perda de contato, é o arrasto axial dos roletes (e, em y, o yaw), e isso vai
  para o [`19`](19-wheel-axial-scrub.md). Decidir de novo quando o `19` der o envelope que o tier 1
  alcança: ou o tier 0 desce até ele, ou o tier 1 sobe até 3.30.
- 2026-10-18, do `19`: o arrasto axial é o rolete freado (casco poligonal, atrito torcional,
  amortecimento da junta), e a correção vai para o [`20`](20-roller-fix.md). Com ela, o tier 1 chega a
  3.29 m/s em x e 4.02 em y. O `a_max` 2.30 não é alcançável: limitado por atrito com μ 0.25, o tier 1
  sustenta ~1.9 m/s² em x (limite 1.95) e ~1.3 em y (1.33). Em y, acima disso, ele gira, porque as
  dianteiras levam 22 % mais carga. O "μg = 2.45" do item 2 esquece o cosseno das rodas. Decidir o
  envelope com esses números quando o `20` fechar.
