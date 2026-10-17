# 02 — Constantes físicas: a bola, o carpete e o livro de regras

Ticket: `.scratch/physics-fidelity/issues/02-physical-constants.md`. Resolvido em 2026-09-09.
Livro de regras citado a partir das fontes de capítulos do `ssl-rules` na `master` e do livro de
regras renderizado (ambos consultados em 2026-09-09); limites de conformidade da bola das Equipment
Rules da R&A/USGA; todo o resto de medições revisadas por pares, artigos publicados de equipes da SSL,
ou do código-fonte dos próprios simuladores da liga. Valores do repositório citados como `path:LINE`;
upstream em `/tmp/rsim-upstream` (`b413932`).

Pela restrição permanente 1 do mapa, os números do rSim são tratados apenas como referência
diferencial. Dois deles acabam tendo uma proveniência rastreável, e a §3 a nomeia.

## Resumo

| grandeza | modelo atual | recomendado | classe da fonte | citação |
|---|---|---|---|---|
| massa da bola | 0.046 kg (`models/ball.xml:7`) | **0.046 kg — manter** | RULEBOOK | SSL rules §2.2; R&A Regra 4.2 máx. 45.93 g |
| raio da bola | 0.0215 m (`models/ball.xml:7`) | **0.0215 m — manter** | RULEBOOK | SSL rules §2.2 (0.043 m diâm.); R&A Regra 4.3 mín. 42.67 mm |
| COR da bola, piso duro | n/a (`solref="0.006 1"`, criticamente amortecido) | **0.89 em aço, 0.68 em madeira** (só informativo; o MuJoCo não tem COR) | MEASURED | Ismail & Stronge, IOP Conf. Ser. MSE 36 012038 |
| COR da bola, carpete da SSL | **0** (criticamente amortecido) | **0.47–0.56** | MEASURED | ER-Force `simulator.cpp:46`; TIGERs `BallParameters.java:91` |
| desaceleração de rolamento da bola | ≈0.293 m/s² (`μ_roll=0.0009`, `scene_rollers.xml:23`) | **0.26–0.31 m/s²; manter 0.0009** | MEASURED | Rojas & Simon 0.305; ER-Force ETDP 2016 0.30; TIGERs 0.26 |
| atrito de deslizamento da bola (bola↔carpete) | 0.35 (`scene_rollers.xml:23`) | **0.26–0.35; manter 0.35 ou baixar para 0.31** | MEASURED | ER-Force `simulator.cpp:47` (μ_k = 0.35); a_slide 2.5–3.47 m/s² |
| inércia da bola β = I/mr² | 0.4 (esfera uniforme, implícita por `mass=`) | **0.5** (medido; irrepresentável sem um `diaginertia` explícito) | MEASURED | TIGERs `BallParameters.java:49`; Rojas & Simon k=0.5805 |
| velocidade máx. da bola ao sair de um robô | sem teto | **6.5 m/s, rasteiro e chip igualmente** | RULEBOOK | SSL rules §8.4.2 *Ball Speed* |
| velocidade máx. de chip | sem teto | **6.5 m/s — não existe regra separada para chip** | RULEBOOK | SSL rules §8.4.2; "chip" ausente do livro de regras |
| envelope máx. do robô | — | **cilindro Ø0.18 m × 0.15 m** | RULEBOOK | SSL rules §3.2.2 *Shape* |
| cobertura da bola pelo dribbler | — | **≤20 % da área em vista superior ⇒ ≤10.9 mm de penetração no casco** | RULEBOOK | SSL rules §3.2.4 *Dribbling Device* |
| teto de velocidade do robô | — | **nenhum** (1.5 m/s vale só durante Stop e para velocidade de aproximação em colisão) | RULEBOOK | SSL rules §8.4.2 *Crashing*; estabelecido pelo `01` |
| atrito de deslizamento rolete↔piso | 1.0 efetivo (`robot_rollers.xml:311` geom 1.0 vs piso 0.8, o MuJoCo toma o máximo) | **0.25** | MEASURED | Williams, Carter, Gallina & Rosati, IEEE T-RA 2002, Tabela I |
| atrito de deslizamento do piso (`field.xml:42`) | 0.8 | sem base independente; ver linha do rolete | MEASURED | idem |

Veredito por entregável: **1 RULEBOOK · 2 MEASURED · 3 MEASURED · 4 RULEBOOK · 5 MEASURED**.
Nenhum entregável voltou vazio. Dois carregam subquestões que voltaram — o COR tangencial
*específico do carpete* da §2 e o número de *composto de borracha em carpete da SSL* da §5 — e ambos
estão sinalizados no lugar.

---

## 1. A bola regulamentar

**O livro de regras especifica uma bola de golfe, pelo nome, e dá os números.** SSL rules §2.2 *Ball*:

> The ball is a standard orange golf ball. It weights approximately 0.046 kilograms and its
> diameter measures 0.043 meters.
> — [`chapters/playingenvironment.adoc:80-81`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc)

Então a premissa do ticket se confirma: uma bola de golfe é especificada, e as Equipment Rules da
R&A/USGA são donas da sua conformidade. Essas regras dão **limites, não um valor**:

| regra R&A/USGA | texto | tipo |
|---|---|---|
| Regra 4.2 *Weight* | "The weight of the ball must not be greater than 1.620 ounces avoirdupois (45.93 g)." | **máximo**; sem mínimo |
| Regra 4.3 *Size* | "The diameter of the ball must not be less than 1.680 inches (42.67 mm)." | **mínimo**; sem máximo |

([R&A, Rules of Equipment, Part 4 — Conformance of Balls](https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls);
a USGA co-publica texto idêntico como Part 4 Rules 2 e 3 das
[Equipment Rules](https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/equipment-rules.html).)

### 1.1 Então 0.046 ou 0.043 kg está certo? 0.046 — e o 0.043 do rSim é um erro de transcrição

A aritmética resolve de três formas:

```
máximo de conformidade da R&A   45.93 g   = 0.04593 kg
livro de regras da SSL §2.2     0.046 kg              (= o máximo da R&A, arredondado a 2 algarismos)
uma bola real, pesada           46 g                  (estudo de queda do IOP, Tabela 1)
```

O artigo de queda do IOP da §2 pesou e mediu de forma independente a sua bola de golfe em **46 g e
42.80 mm** (Tabela 1) — uma bola real colada nos dois limites de conformidade, como são as bolas de
competição, porque uma bola de massa máxima e diâmetro mínimo voa mais longe. O `mass="0.046"` de
`models/ball.xml:7` está, portanto, correto a 0.15 % do limite de conformidade e exato frente ao
livro de regras. **Manter.**

O `ballMass = 0.043` do rSim (`/tmp/rsim-upstream/src/robosim/sslconfig.h:123`) não é uma bola
diferente. É **o diâmetro da §2.2, em metros, colado no campo de massa**: a linha seguinte,
`ballRadius = 0.0215` (`sslconfig.h:121`), é 0.043/2, então a mesma frase do livro de regras forneceu
os dois, e um dos dois foi lido do substantivo errado. Não existe bola de nenhuma especificação
pesando 43 g e medindo 43 mm; essa combinação só existe como erro de digitação. O "ball mass differs"
do `B9` deve ser registrado como **o rSim está errado**, não como uma discordância — e os 7 % são um
erro de 6.5 % no rSim, não uma escolha de modelagem.

### 1.2 Raio

`size="0.0215"` (43.0 mm de diâmetro) é exatamente o valor do livro de regras. Fica 0.77 % acima do
mínimo da R&A de 42.67 mm e 0.47 % acima da bola medida de 42.80 mm. Um simulador deve escolher os
0.0215 m do livro de regras: é o que a liga declara, é o que todo outro simulador da SSL usa, e a
dispersão abaixo de 1 % está muito abaixo da fidelidade de qualquer outra coisa no modelo de contato.
**Manter.**

**Classe da fonte: RULEBOOK.**

---

## 2. Coeficiente de restituição

O MuJoCo não tem coeficiente de restituição — o quique sai da razão de amortecimento do `solref` —
então esta seção reporta apenas o COR físico e não toca no mapeamento.

### 2.1 Em piso duro

Teste normal de queda revisado por pares, vídeo de alta velocidade a 250 fps, quedas de 1.3–1.7 m
(impacto a 5.05–5.77 m/s), três repetições por configuração, bola pesada em 46 g / 42.80 mm:

| alvo | 1.3 m | 1.4 m | 1.5 m | 1.6 m | 1.7 m |
|---|---|---|---|---|---|
| placa de aço | 0.897 | 0.896 | 0.894 | 0.892 | 0.889 |
| placa de madeira | 0.681 | 0.677 | 0.676 | 0.675 | 0.666 |

(M. R. M. Ismail & W. J. Stronge et al., "Coefficient of restitution of sports balls: A normal
drop test", *IOP Conf. Ser.: Mater. Sci. Eng.* **36** 012038 (2012),
[doi:10.1088/1757-899X/36/1/012038](https://iopscience.iop.org/article/10.1088/1757-899X/36/1/012038/pdf),
Tabelas 1, 3 e 4.)

Duas propriedades a levar adiante. **O COR cai conforme a velocidade de impacto sobe** — 0.897 → 0.889
com um aumento de 14 % na velocidade, e a conclusão do próprio artigo declara a tendência. E **o COR
cai bruscamente conforme o alvo amolece** — 0.90 em aço vira 0.68 em madeira, uma queda de 24 % para
um alvo que ainda é rígido pelos padrões de carpete. Nenhum dos dois é um piso duro sob 6 mm de feltro.

Note que esse 0.89 fica acima da faixa de 0.70–0.85 que a indústria de golfe cita na sua própria
velocidade de entrada padrão de 125 ft/s (38.1 m/s) — consistente com a dependência da velocidade,
já que os 5–6 m/s do teste de queda são seis a sete vezes mais lentos.

### 2.2 No carpete em que a SSL joga

Dois valores derivados da SSL, dos dois simuladores que a liga de fato usa, ambos para a direção
vertical (normal):

| fonte | valor | o que é |
|---|---|---|
| ER-Force `simulator.cpp:40-47`, comentário intitulado "(empirical measurments)" | **0.56** | `Ball vs. Floor: Restitution: sqrt(h'/h) = sqrt(0.314) = 0.56` — um teste de queda no piso da SSL: a bola devolveu 31.4 % da altura de queda |
| TIGERs Sumatra `BallParameters.java:86-91` | **0.47** | `chipDampingZ`, "Chip kick velocity damping factor in Z direction" — a razão de velocidade vertical por salto de uma bola chipada, ou seja, um COR normal por construção |

([ER-Force framework `src/amun/simulator/simulator.cpp`](https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simulator.cpp);
[TIGERs Sumatra `BallParameters.java`](https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java).)

`chipDampingZ` não é um botão de ajuste arbitrário: é um dos três campos de
`SSL_BallModelChipFixedLoss` no protocolo de geometria da própria liga
([`messages_robocup_ssl_geometry.proto:118-127`](https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto)),
que o SSL-Vision transmite a todas as equipes para que possam prever os saltos de uma bola chipada.
Uma equipe que errasse isso perderia recepções de chip o torneio inteiro.

**Um simulador deve escolher 0.50, e dizer por quê.** As duas medições da SSL o cercam (0.47 e
0.56); é o ponto médio com 2 % de erro; e é o valor em que um chip que pousa a 3 m/s rebate a
1.5 m/s e morre em três saltos, que é o que os replays da SSL mostram. Qualquer coisa em 0.47–0.56 é
defensável; fora disso não é.

Os dois valores diferem porque medem coisas diferentes — a ER-Force soltou uma bola na vertical em
baixa velocidade, a TIGERs ajustou uma trajetória de chip inteira cujo primeiro pouso é mais rápido e
oblíquo — e a dependência da velocidade da §2.1 prevê exatamente essa ordem (impacto mais rápido, COR
menor).

### 2.3 O 0.5 do rSim é defensável? Sim — singularmente entre os seus números

`ballBounce = 0.5` (`sslconfig.h:124`) cai dentro da faixa medida de 0.47–0.56, 6 % abaixo do teste
de queda da ER-Force e 6 % acima do ajuste de chip da TIGERs. É a única constante de contato do rSim
que sobrevive ao contato com uma fonte, e a leitura do `C2` deve ser revista: **o quique não é o
contorno do ODE, a ausência de quique é a lacuna de fidelidade.** O `solref="0.006 1"` criticamente
amortecido de `models/scene_rollers.xml:23` dá à bola um COR de 0, ou seja, um chip que pousa morre e
um chute na parede do fundo para — comportamento sem respaldo em nenhuma fonte encontrada aqui.

Duas ressalvas à versão do rSim mesmo assim. `ballBounceVel = 0.1` é um limiar abaixo do qual não há
quique algum, o que nada medido aqui sustenta. E o rSim aplica o *mesmo* 0.5 às paredes
(`sslworld.cpp:151-156,168-169`), onde a superfície é uma tábua dura e não carpete; o contraste
aço-vs-madeira da §2.1 diz que o COR de uma parede deveria ser materialmente maior que o de um
carpete.

### 2.4 O que não tem fonte aqui

O COR **tangencial** — a velocidade horizontal que a bola mantém em um quique oblíquo — tem um número
da SSL (`chipDampingXYFirstHop = 0.8`, `chipDampingXYOtherHops = 0.85`, mesmo arquivo) mas nenhuma
medição por trás que se tenha achado, e o mesmo vale para a restituição bola-saindo-do-robô
(`redirectRestitutionCoefficient = 0.2` da TIGERs; "Ball vs. Robot: Restitution: about 0.60" da
ER-Force). Esses dois discordam em 3×, o que já é evidência de que nenhum foi medido. **Se o projeto
precisar de um modelo de quique oblíquo ou de redirecionamento, essa subquestão está SEM FONTE** e
deve ser registrada como aproximação deliberada.

**Classe da fonte: MEASURED** (para o COR normal, nas duas superfícies).

---

## 3. Desaceleração de rolamento no carpete da SSL

O número mais bem sustentado deste ticket. Três medições independentes, em carpete, por três grupos
diferentes, ao longo de duas décadas:

| fonte | superfície | a_roll (m/s²) | a_slide (m/s²) | k_switch | método |
|---|---|---|---|---|---|
| Rojas & Simon (FU-Fighters) | "our carpet" | **−0.305** | −3.47 | 0.5805 ± 0.042 | 26 chutes capturados pelo sistema de visão computacional deles; ajuste de mínimos quadrados de duas retas com o ponto de quebra escolhido por custo mínimo |
| ER-Force ETDP 2016 §5.2 | o campo de jogo | **−0.3** | −2.5 | 0.6 (teoria: 5/7) | "Using our plotter software, we measured…" a partir de um traço v-t de uma bola chutada a 3.3 m/s |
| TIGERs Sumatra, especialização `ROBOCUP` | carpete de competição da RoboCup | **−0.26** | −3.0 | 0.64 | parâmetros de previsão de bola em produção, por local |

([Rojas, R. & Simon, M., *Like a Rolling Ball*, Freie Universität Berlin](https://robocup.mi.fu-berlin.de/buch/rolling.pdf), §3;
[ER-Force Extended Team Description Paper 2016](https://ssl.robocup.org/wp-content/uploads/2019/01/2016_ETDP_ER-Force.pdf) §5.1–5.2;
[TIGERs `BallParameters.java:24-41`](https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java).)

Estas não são três cópias de um número: Rojas & Simon antecedem os outros dois, a ER-Force os cita
como `[6]` mas mediu de forma independente e obteve um `k_switch` diferente, e o valor da TIGERs é uma
calibração separada por local. O protocolo de geometria da própria liga padroniza exatamente este
modelo — `SSL_BallModelStraightTwoPhase{acc_slide, acc_roll, k_switch}` — e o seu comentário nomeia o
ETDP 2016 da ER-Force como a definição
([`messages_robocup_ssl_geometry.proto:100-116`](https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto)).

**A faixa é 0.26–0.31 m/s² em carpete.** A ER-Force declara a ressalva necessária: "The exact time
points and decelerations are contingent on the playing field" (§5.2), e a TIGERs carrega seis
especializações de local para o mesmo parâmetro. Então um simulador deve escolher o centro da faixa e
tratá-lo como dependente do local, não como constante da natureza.

### 3.1 O 0.29 do modelo é defensável? Sim. O 0.49 do rSim? Não.

Rederivando o valor do modelo a partir das declarações. O coeficiente de atrito de rolamento do
MuJoCo tem dimensão de comprimento e limita o **torque** de resistência ao rolamento em
`μ_roll · f_n`. Com `friction="0.35 0.35 0.005 0.0009 0.0009"` (`models/scene_rollers.xml:23`) e
`mass="0.046"` (`models/ball.xml:7`):

```
f_n   = m g            = 0.046 · 9.81                  = 0.451260 N
τ_max = μ_roll · f_n   = 0.0009 · 0.451260             = 4.06134e-4 N·m
a     = τ_max / (m r (1 + β)),  β = I/mr² = 0.4        (esfera uniforme; padrão do MuJoCo a partir de `mass=`)
      = 4.06134e-4 / (0.046 · 0.0215 · 1.4)
      = 4.06134e-4 / 1.38460e-3                        = 0.2933 m/s²
```

O 0.29 do `P4` se reproduz. **0.293 m/s² fica no meio da faixa medida de 0.26–0.31** — 4 % abaixo de
Rojas & Simon, 2 % abaixo da ER-Force, 13 % acima da TIGERs. É defensável como está e
`μ_roll = 0.0009` não precisa de mudança; para referência, 0.00080 daria 0.26 e 0.00095 daria 0.31,
então toda a faixa medida cabe numa janela de 0.0008–0.00095.

O 0.49 do rSim é **1.6× a 1.9× a faixa medida** e não é defensável. Sua proveniência é rastreável, e
não é uma medição:

```
rSim:      fk = ballFriction · m · g,  ballFriction = 0.05     (sslconfig.h:125, sslworld.cpp:286)
           ⇒ a = 0.05 · 9.81 = 0.4905 m/s²

ER-Force:  const btScalar hackFactor = 1.4;
           const btScalar rollingDeceleration = hackFactor * 0.35;   ⇒ 0.49 m/s²
           precedido por:  // \mu_r = -a / g = 0.0357 (while rolling)   ⇒ 0.350 m/s²
           e:              // rollingFriction in bullet is too unstable to be useful
           e:              // this is quite a hack as it's always applied
                                        (framework/src/amun/simulator/simball.cpp:53-88)
```

O 0.49 do rSim é o 0.35 físico da ER-Force multiplicado por um **`hackFactor` de 1.4 que o próprio
comentário da ER-Force chama de "quite a hack"** — uma compensação de estabilidade do Bullet,
carregada para um simulador ODE onde a instabilidade que ela compensa não existe. A estrutura do rSim
é copiada por atacado: o mesmo ramo "matar a bola abaixo de 0.01 m/s" aparece nos dois
(`sslworld.cpp:295-298` contra `simball.cpp:73-77`), e a versão da ER-Force traz o comentário "the
real ball snaps to a dimple". Mesmo o 0.35 sem hack da ER-Force fica 13 % acima do topo da faixa
medida; o 0.49 é um artefato de um artefato. Esta é a restrição 1 do mapa em sua forma mais pura.

O limiar de morte em si, incomumente, *tem* respaldo. Rojas & Simon observaram que a bola "suddenly
dies" abaixo de "around 2 cm/s", porque "if the ball rolls too slowly, it cannot 'climb' the carpet"
(§2). 0.01 m/s é a ordem de grandeza certa para um efeito real, embora a queixa do `P4` de que a bola
do mjsoccer se arrasta em vez de parar continue justa.

### 3.2 Um subproduto: atrito de deslizamento bola↔carpete

As mesmas três medições fixam o coeficiente de deslizamento, já que `a_slide = μ_k g`:

```
Rojas & Simon    3.47 / 9.81 = 0.354      (o artigo declara 35.4 % de g diretamente)
TIGERs           3.00 / 9.81 = 0.306
ER-Force ETDP    2.50 / 9.81 = 0.255
```

e o simulador da ER-Force o declara como coeficiente medido de forma direta:
`Ball vs. Floor: Friction: \mu_k = -a / g (while slipping) = 0.35` (`simulator.cpp:47`).

**Faixa 0.26–0.35.** O `0.35` de `models/scene_rollers.xml:23` fica no topo dela e bate exatamente com
o valor declarado pela ER-Force. O `1.0` do rSim (`sslworld.cpp:154`) é **3–4× o valor medido** e não
tem respaldo. O `C3` deve ser relido como o `C2`: o mjsoccer está certo e o rSim é o ponto fora da
curva, e não "a backspun ball grips almost three times harder upstream" como uma diferença neutra.
Um simulador poderia igualmente escolher 0.31, o centro da faixa; 0.35 é defensável.

### 3.3 Uma discrepância sinalizada: o momento de inércia da bola

`k_switch` é uma medição direta da distribuição de inércia da bola: para `I = β m r²`, o deslizamento
termina em `v_switch = v₀/(1 + β)`, que é 5/7 = 0.714 para uma esfera uniforme (β = 0.4).

```
Rojas & Simon   k = 0.5805  ⇒ β = 0.723      "the golf balls used in our experiments do not have
ER-Force        k = 0.6     ⇒ β = 0.667       a constant density and their moment of inertia
TIGERs          k = 0.64    ⇒ β = 0.563       differs from (2/5) m R²"  — Rojas & Simon §3
TIGERs, declarado diretamente:  inertiaDistribution = 0.5, "between 0.4 (massive sphere) and 0.66 (hollow sphere)"
```

Toda medição cai **acima** de 0.4. `models/ball.xml:7` declara só `mass="0.046"` em uma geom esférica,
então o MuJoCo calcula β = 0.4 e a bola do modelo é 20–45 % fácil demais de acelerar em giro. Isso
muda por quanto tempo uma bola chutada desliza antes de rolar, e é medido, não chutado. Corrigir
exige uma declaração de inércia explícita; isso é uma edição de modelo e pertence ao ticket que for
dono de `models/ball.xml`, não a este.

### 3.4 Atrito de rolamento ou uma força aplicada?

O MuJoCo pode expressar a desaceleração de rolamento ou pelo termo de atrito de rolamento do par de
contato (o que `scene_rollers.xml:23` já faz) ou como uma força aplicada explícita (o que o rSim e a
ER-Force fazem). **O termo do par de contato é o natural**: é um limite de restrição proporcional à
força normal real, então só se aplica enquanto a bola está no chão e some automaticamente em voo,
enquanto uma força aplicada precisa do teste "estou tocando o piso?" que os dois outros simuladores
tiveram de escrever à mão — e que o rSim erra ao aplicar o arrasto à velocidade 3-D completa da bola,
componente `z` incluída (`sslworld.cpp:288-291`). A escolha da correção é de outro ticket.

**Classe da fonte: MEASURED.**

---

## 4. Limites do livro de regras que restringem o simulador

Tudo do livro de regras da SSL, citado a partir das fontes de capítulo na `master` e conferido contra
a edição renderizada para os números de seção.

| § | regra | limite |
|---|---|---|
| 8.4.2 | *Ball Speed* (falta sem parada) | "A robot must not accelerate the ball faster than 6.5 meters per second in 3D space." |
| 3.2.2 | *Shape* | "A robot must fit inside a 0.18 meters wide and 0.15 meters high cylinder at any point in time." Dispersão de altura da equipe ≤ 0.02 m. |
| 3.2.4 | *Dribbling Device* | "80% of the area of the ball when viewed from above has to be outside the convex hull around the robot. This limitation applies as well to all kicking devices, even if such infringement is momentary." Além disso: não pode elevar a bola; outro robô deve conseguir removê-la; não pode remover todos os seus graus de liberdade. |
| 8.4.1 | *Excessive Dribbling* | não mais que 1 m de onde o drible começou |
| 8.4.2 | *Crashing* | componente da velocidade de aproximação ao longo da linha que une dois robôs adversários > 1.5 m/s é falta |
| 2.1.3 | *Field Markings* | "Distances from a robot are measured from its nearest side to the respective measurement point, with an assumed radius of 0.09 m." |
| 2.1.2 | *Field Surface* | "The playing surface is a green felt mat or carpet. The floor under the carpet is level, flat, and hard." |

([`chapters/offenses.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc),
[`chapters/robots.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc),
[`chapters/playingenvironment.adoc`](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc),
renderizado em [robocup-ssl.github.io/ssl-rules](https://robocup-ssl.github.io/ssl-rules/sslrules.html).)

### 4.1 Velocidade de chute: um teto, e ele cobre o chip

**Não existe limite separado de velocidade de chute de chip, porque não existe regra de chute de chip
nenhuma.** A string `chip` não ocorre em lugar nenhum das fontes de capítulo do livro de regras. A
§8.4.2 diz "6.5 meters per second **in 3D space**" — o módulo do vetor de velocidade 3-D completo,
que é precisamente a formulação necessária para que um número só limite um chute rasteiro e um chip de
forma idêntica. Uma bola chipada lançada a 45° com 6.5 m/s no total é legal; o mesmo chip com
6.5 m/s *horizontais* não é.

Vale dizer isso porque o protocolo de simulação da liga *separa* os dois —
`RobotSpecs.max_linear_kick_speed` e `max_chip_kick_speed`
([`ssl_simulation_config.proto:48-51`](https://github.com/RoboCup-SSL/ssl-simulation-protocol/blob/master/proto/ssl_simulation_config.proto))
— mas essas são declarações de *hardware* por robô, não limites de regra. Um simulador deve escolher
6.5 m/s como um único teto de módulo 3-D para a velocidade da bola após o chute e aplicá-lo aos dois
dispositivos.

Note também que 6.5 m/s é um **limiar de falta, não um teto físico**. A regra existe porque os robôs
podem e de fato a excedem; um simulador de fidelidade ao hardware deve modelar o que o chutador
consegue entregar e deixar os 6.5 m/s serem um evento de árbitro, não limitar a física neles.

### 4.2 Dimensões do robô

Ø0.18 m × 0.15 m é um cilindro envolvente dentro do qual o robô deve caber "at any point in time" —
um envelope, não uma especificação, então qualquer geometria de robô dentro dele está em conformidade.
As regras acrescentam duas grandezas derivadas que um simulador pode usar diretamente: um **raio
assumido de 0.09 m** para toda medição de distância de regra (§2.1.3), e um topo plano do padrão de
visão garantido a caber num círculo de raio 0.085 m recortado 0.055 m à frente do centro (§3.2.3).

### 4.3 Cobertura do dribbler, em metros

Os 80 % da §3.2.4 são uma fração de área do disco da bola em vista superior. Convertendo para a
penetração linear que um simulador pode de fato checar: para um segmento circular de fração de área
0.2, o ângulo central resolve `θ − sin θ = 0.4π`, dando `θ = 2.1131` rad, e a profundidade do
segmento é

```
h = r (1 − cos(θ/2)) = 0.0215 · (1 − cos 1.0566) = 0.0215 · 0.50814 = 0.010925 m
```

**A bola pode estar no máximo 10.9 mm dentro do casco convexo do robô**, e "even if such infringement
is momentary" significa que este é um limite instantâneo, não uma média no tempo — o que é uma
restrição sobre quão profundo pode ser o bolsão de um dribbler, não só sobre onde a bola repousa.

### 4.4 Velocidade do robô

Não rederivado: o `01` estabeleceu que o livro de regras não limita velocidade, aceleração nem
potência do motor do robô, e que os valores de 1.5 m/s são um limite do estado Stop e um limite de
velocidade de aproximação em colisão, e não um teto de velocidade máxima. Nada encontrado aqui muda
isso, e nenhuma regra restringe o drivetrain exceto o cilindro da §3.2.2, que limita o diâmetro da
roda.

**Classe da fonte: RULEBOOK.**

---

## 5. Carpete contra um rolete de borracha de omniwheel

Existe uma medição, e é revisada por pares. Williams, Carter, Gallina & Rosati construíram um robô
omni de três rodas para a equipe de RoboCup da Ohio University, esbarraram em escorregamento nos
testes, e mediram os coeficientes em uma mesa inclinável: as rodas eram fixadas em um ângulo tal que
**só os roletes** tocavam a superfície, uma tábua coberta com o material de teste era girada para
cima até o veículo deslizar, e o coeficiente era tomado como a tangente do ângulo de deslizamento,
com média sobre várias tentativas.

| superfície | μ′_W máx (rolete, direção de acionamento) | μ′_T máx (rolete, transversal) | μ″_W máx / μ″_T máx (cubo entre roletes) |
|---|---|---|---|
| papel | 0.26 | 0.09 | 0.47 |
| **carpete** | **0.25** | **0.15** | **0.56** |

(R. L. Williams II, B. E. Carter, P. Gallina & G. Rosati, "Dynamic Model with Slip for Wheeled
Omni-Directional Robots", *IEEE Transactions on Robotics and Automation* **18**(3), 285–293, junho de
2002; manuscrito final em
[people.ohio.edu/williams/html/PDF/IEEETRA02.pdf](https://people.ohio.edu/williams/html/PDF/IEEETRA02.pdf),
§III.A.2 e Tabela I.)

**μ′_W = 0.25 é o número que este entregável pede**: o rolete deslizando no carpete ao longo da
direção de acionamento da roda, que é o orçamento de tração. μ′_T = 0.15 é a direção transversal, em
que o rolete rola em vez de deslizar. μ″ = 0.56 é o material rígido do cubo nas folgas entre roletes,
e o achado principal do artigo é que essas folgas — 10 % de cada setor angular na roda deles —
dominam o comportamento de escorregamento "even at low rates and accelerations", porque o coeficiente
de contato mais que dobra conforme cada folga passa pela área de contato.

**O que um simulador deve escolher, e a ressalva honesta.** 0.25 para o contato do rolete. Mas duas
incompatibilidades precisam ser registradas em vez de encobertas. O **composto do rolete não é
declarado** — as rodas eram unidades de prateleira "developed for material handling applications", não
os roletes de borracha moldada que um robô SSL moderno carrega — e o **carpete não é identificado**,
certamente não o tapete de feltro verde que a §2.1.2 especifica. Então, embora *um rolete de omniwheel
sobre carpete* esteja medido, a pergunta mais estreita — *um rolete de borracha da SSL sobre feltro da
SSL* — **não tem fonte**: nenhuma equipe da SSL publica um coeficiente roda/carpete, e uma busca no
arquivo de ETDPs achou apenas a observação qualitativa de que o atrito do carpete muda ao longo de um
torneio conforme os robôs o desgastam, forçando reajuste de ganhos. Essa subquestão deve ser
registrada como aproximação deliberada.

### 5.1 O que isso diz sobre o modelo

`models/robot_rollers.xml:311` dá a toda geom de colisão de rolete `friction="1.0 …"` contra o piso de
`models/field.xml:42` em `0.8`; o MuJoCo mistura geoms sem par tomando o máximo elemento a elemento,
então o **coeficiente de deslizamento efetivo rolete↔piso é 1.0** — quatro vezes o único valor
medido. Nem 1.0 nem 0.8 têm base; 0.8 em particular não aparece em nenhuma fonte consultada aqui.

Isso importa além deste ticket. O `01` dimensionou o drivetrain contra um orçamento de atrito de
`μ·m·g = 26.4` N calculado com μ = 1.0. Com μ = 0.25 esse orçamento é
`0.25 · 2.69207 · 9.81 = 6.60` N contra os 8.16 N de força frontal no chão que o `01` recomenda — por
roda, `0.25 · 6.602 = 1.65` N de capacidade contra 2.59 N exigidos. **O robô escorregaria em vez de
ser limitado por torque**, o que inverte a conclusão central do `01`. A recomendação de `forcerange`
do `01` não muda (vem do motor, não do piso), mas o seu achado "31 % do orçamento de atrito, então o
escorregamento é um evento raro" depende de um coeficiente que não tem fonte. Esta é uma sinalização
entre tickets, não uma rederivação: o ticket que definir o método para justificar um coeficiente de
contato deve carregá-la, e o mapa já lista atrito robô↔robô e robô↔parede como pendentes da mesma
questão.

**Classe da fonte: MEASURED** (para um rolete de omniwheel sobre carpete; o valor específico da SSL,
borracha sobre feltro, está SEM FONTE, §5 acima).

---

## Não resolvido, e suposições declaradas

1. **A faixa de COR 0.47–0.56 é só na direção normal.** A restituição tangencial e a de
   bola-saindo-do-robô não têm fonte (§2.4), e os dois valores publicados da SSL para a segunda
   diferem em 3×.
2. **O 0.56 e o μ_k = 0.35 da ER-Force são rotulados "(empirical measurments)" em um comentário de
   código-fonte,** não em um artigo. O método é declarado (`sqrt(h'/h)`, `−a/g`) mas os dados brutos
   não são publicados. São tratados como medições aqui porque são a calibração do simulador de
   referência da própria liga, que o ticket nomeia como aceitável, e porque ambos são corroborados de
   forma independente — o 0.56 pelo 0.47 da TIGERs, e o 0.35 pelo 0.354 de Rojas & Simon.
3. **As três medições de desaceleração de rolamento são específicas de local.** A ER-Force diz isso
   explicitamente e a TIGERs carrega seis especializações de local. 0.26–0.31 é uma faixa, não uma
   constante, e um simulador que queira um número só está escolhendo.
4. **Qual carpete.** Nenhuma fonte consultada identifica o carpete por produto. A §2.1.2 diz apenas
   "green felt mat or carpet". Todo número dependente de carpete aqui herda, portanto, uma dispersão
   de local não quantificada.
5. **Williams et al. é uma medição de 2002 em roletes e carpete não especificados** (§5). É a única
   medição encontrada; não é uma medição em carpete da SSL.
6. **O momento de inércia da bola** (§3.3) é medido em β ≈ 0.5–0.7 contra os 0.4 implícitos no
   modelo, e não é corrigível sem uma declaração de inércia explícita. Registrado, não corrigido.
7. **`ballBounceVel = 0.1`** no rSim — uma velocidade abaixo da qual não há quique — não tem respaldo
   em nada do que foi encontrado, ao contrário do limiar de morte de rolamento de 0.01 m/s, que tem
   (§3.1).
8. **O teto de 6.5 m/s da bola é um limiar de falta, não um limite físico** (§4.1). Tratá-lo como
   limitador do chutador seria um modelo de regras, não um modelo de hardware.

## Fontes

Liga (RoboCup Small Size League), consultadas em 2026-09-09:
- Rules of the RoboCup Small Size League, renderizado — https://robocup-ssl.github.io/ssl-rules/sslrules.html (§2.1.2 Field Surface, §2.1.3 Field Markings, §2.2 Ball, §3.2.2 Shape, §3.2.4 Dribbling Device, §8.4.1 Excessive Dribbling, §8.4.2 Ball Speed / Crashing)
- Fontes de capítulos na `master` — https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/playingenvironment.adoc, .../robots.adoc, .../offenses.adoc
- Protocolo de geometria do SSL-Vision, `SSL_BallModelStraightTwoPhase` e `SSL_BallModelChipFixedLoss` — https://github.com/RoboCup-SSL/ssl-vision/blob/master/src/shared/proto/messages_robocup_ssl_geometry.proto
- Protocolo de simulação da SSL, `RobotSpecs` — https://github.com/RoboCup-SSL/ssl-simulation-protocol/blob/master/proto/ssl_simulation_config.proto
- ER-Force Extended Team Description Paper 2016, §5 "Ball model" — https://ssl.robocup.org/wp-content/uploads/2019/01/2016_ETDP_ER-Force.pdf
- RobôCIn Extended Team Description Paper 2024 — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf

Conformidade da bola de golfe (R&A / USGA, co-proprietárias das Equipment Rules):
- R&A, Rules of Equipment, Part 4 — Conformance of Balls, Regra 4.2 (Weight) e Regra 4.3 (Size) — https://www.randa.org/en/roe/the-rules-of-equipment/part-4-conformance-of-balls
- USGA Equipment Rules, Part 4 (texto idêntico, Regras 2 e 3) — https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/equipment-rules.html

Trabalho de medição:
- Rojas, R. & Simon, M., *Like a Rolling Ball*, Freie Universität Berlin — https://robocup.mi.fu-berlin.de/buch/rolling.pdf (26 chutes rastreados por visão em carpete; a_slide −3.47 m/s², a_roll −0.305 m/s², k = 0.5805 ± 0.042)
- Ismail, M. R. M. & Stronge, W. J. et al., "Coefficient of restitution of sports balls: A normal drop test", *IOP Conf. Ser.: Mater. Sci. Eng.* **36** 012038 (2012), doi:10.1088/1757-899X/36/1/012038 — https://iopscience.iop.org/article/10.1088/1757-899X/36/1/012038/pdf (bola de golfe 46 g / 42.80 mm; COR 0.889–0.897 em aço, 0.666–0.681 em madeira, quedas de 1.3–1.7 m)
- Williams II, R. L., Carter, B. E., Gallina, P. & Rosati, G., "Dynamic Model with Slip for Wheeled Omni-Directional Robots", *IEEE Transactions on Robotics and Automation* **18**(3), 285–293, junho de 2002 — https://people.ohio.edu/williams/html/PDF/IEEETRA02.pdf (atrito de rolete em carpete na mesa inclinável: μ′_W 0.25, μ′_T 0.15, cubo 0.56)

Calibração de simuladores (os dois simuladores de referência da liga):
- ER-Force `framework`, `src/amun/simulator/simulator.cpp:40-65` — o bloco "(empirical measurments)": restituição bola/piso 0.56 a partir de `sqrt(h'/h) = sqrt(0.314)`, atrito bola/piso μ_k = 0.35, restituição bola/robô 0.60 e atrito 0.18 — https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simulator.cpp
- ER-Force `framework`, `src/amun/simulator/simball.cpp:36-88` — `\mu_r = -a/g = 0.0357`, `hackFactor = 1.4`, `rollingDeceleration = 0.49`, o ramo snap-to-dimple de 0.01 m/s, e a citação de Rojas & Simon — https://github.com/robotics-erlangen/framework/blob/master/src/amun/simulator/simball.cpp
- TIGERs Mannheim `Sumatra`, `modules/moduli-geometry/.../BallParameters.java` — accSlide −3000 mm/s², accRoll −260 mm/s², kSwitch 0.64, inertiaDistribution 0.5, chipDampingZ 0.47, chipDampingXY 0.8/0.85, redirectRestitutionCoefficient 0.2, cada um com especializações por local incluindo `ROBOCUP` — https://github.com/TIGERs-Mannheim/Sumatra/blob/master/modules/moduli-geometry/src/main/java/edu/tigers/sumatra/geometry/BallParameters.java

Código-fonte (este repositório, revisão local):
- `models/ball.xml:7` (geom da bola: massa, raio, atrito, `solref`)
- `models/scene_rollers.xml:21-23` (pares de contato bola/dribbler e bola/piso)
- `models/field.xml:42` (plano do piso, `friction="0.8 …"`), `:44` (`solref` da parede)
- `models/robot_rollers.xml:311` (geom de colisão do rolete, `friction="1.0 …"`)
- `docs/research/rsoccer-vs-mjsoccer.md` §1 (`P4`, `P5`), §3 (`B9`), §5 (`C2`, `C3`)
- `.scratch/physics-fidelity/research/01-drivetrain-envelope.md` §3.1, §5

Upstream (`/tmp/rsim-upstream`, `b413932`):
- `src/robosim/sslconfig.h:119-128` (`ballRadius` 0.0215, `ballMass` 0.043, `ballFriction` 0.05, `ballBounce` 0.5, `ballBounceVel` 0.1, mais os campos mortos `ballSlip`/damping do `P5`)
- `src/robosim/sslworld.cpp:151-182` (superfícies bola/parede e bola/chão, `mu = 1`, bounce 0.5)
- `src/robosim/sslworld.cpp:279-300` (o arrasto de Coulomb constante aplicado à velocidade 3-D completa, e o corte em 0.01 m/s)
