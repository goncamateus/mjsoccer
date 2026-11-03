# 03 — Material do dribbler: o que segura a bola num giro, o que as equipes medem e o que o MuJoCo consegue representar

Pesquisa de 2026-11-02, motivada pela perda da bola com o robô girando no lugar a ≥ ~1 rad/s
(26–85 de 120 passos retidos a 1 rad/s, 12–14 a 2 rad/s, ticket `22` (sondas descartadas; números no §Diagnóstico),
ticket `22` §Diagnóstico 5). Fontes consultadas em 2026-11-02: os arquivos de TDP/ETDP da liga em
https://ssl.robocup.org/team-description-papers/ (ZIPs 2013–2025, lidos por inteiro via `pdftotext`) mais
os ETDPs de 2026 baixados um a um de https://ssl.robocup.org/robocup-2026-teams/; dois artigos revisados
por pares sobre dribbler da SSL (ZJUNlict, IEEE RCAR 2019; RobôCIn, SBR/LARS 2021); o livro de regras
`RoboCup-SSL/ssl-rules` em `2ff4be0` e as regras dos desafios de hardware 2021/2022; o código-fonte de
grSim `fe2bd29`, rSim `b413932`, ER-Force `framework` `38563d1`, TIGERs `Sumatra` `496376d` e Bullet3
`d3afe76`; e a documentação **e o código-fonte** do MuJoCo na tag `3.11.0` (`b85fdca`), que é a versão
travada em `uv.lock:413`. Valores do repositório citados como `path:LINE`; URLs de PDF com página e seção.
Classes: RULEBOOK > MEASURED > VENDOR > DELIBERATE APPROXIMATION. Parágrafos marcados **Inferência**
são raciocínio meu sobre as fontes, não o que elas dizem.

## Resumo

| grandeza | modelo atual | recomendado | classe da fonte | citação |
|---|---|---|---|---|
| material do rolete do robô de referência (RobôCIn v2025) | rígido, sem material (`models/robot_kinematic.xml:337`) | **PU Shore 65A** sobre eixo | VENDOR (a própria equipe) | [RobôCIn ETDP 2026][RC26] p.3 §2.2 |
| diâmetro do rolete | **18 mm** de colisão (`size="0.009 0.036"`, `models/robot_kinematic.xml:337`, `models/robot_rollers.xml:250`) | **13 mm** (r = 6.5 mm) — é o que a malha visual do próprio CAD mede | VENDOR | [RobôCIn ETDP 2025][RC25] p.3 Tab.1; [ETDP 2026][RC26] p.2 Tab.1; `models/assets/visual/Dribblerschnecke.stl` medido: r_ext 6.5 mm, r_int 3.0 mm, 72 mm |
| geometria do rolete | cilindro liso | liso é o que a malha do CAD mostra e o que a TIGERs usa desde 2024 (a forma do rolete PU da RobôCIn v2025 não é declarada); dividido/rosca **mede** melhor em giro | MEASURED | [KIKS 2022][KIKS22] p.3; [ZJUNlict 2019][ZJU19] p.4 Tab.1; [TIGERs 2024][TIG24] p.1 |
| μ rolete↔bola | 2.5 (`models/scene.xml:21-22`) | **sem medição publicada.** O único número é ≈0.8 (silicone, sem método); manter 2.5 como aproximação | DELIBERATE APPROXIMATION | [Parsian 2013][PAR13] p.8; [ZJUNlict 2025][ZJU25] p.2 ("hard to obtain") |
| rigidez normal do contato | ~2×10⁵ N/m efetivos (**Inferência**, de `solref="0.004 0.5"`, `solimp="0.95 0.99 0.001"`) | **1–3.5×10⁴ N/m** (Hertz, PU 65A, 1–6 N); afundamento 0.08–0.4 mm | DELIBERATE APPROXIMATION (correlação Shore→E + Hertz) | §5.5; Gent 1958 / Qi 2003 via [Shore durometer][WIKI]; [ZJUNlict 2025][ZJU25] p.3 Tab.1 (k₁ = 4000 N/m, suposto) |
| COR normal bola↔revestimento | ≈0.16 (**Inferência**: oscilador linear com ζ = 0.5; não medido no MuJoCo) | **≈0.63** em silicone Shore A10 (bloco de 13.5 mm); sobe com a dureza | MEASURED (outro material) | [Luhbots 2026][LUH26] p.6–7 §5 |
| velocidade superficial do rolete | 2.83 m/s alvo (314 rad/s × 9 mm), ~2.0 m/s carregado | TIGERs: comando 3 m/s **com teto de força 5 N**, operando saturado no teto | VENDOR | Sumatra `config/botParamsDatabase.json:79-84`; [TIGERs 2024][TIG24] p.4 §2.1, p.7 §2.2 |
| controle do motor do rolete | velocidade (`kv`, `models/robot_kinematic.xml:370`) | **torque/corrente limitado** | MEASURED | [ZJUNlict 2019][ZJU19RCAR] Tab.2 (fuga em ré 1.63 → 2.5 m/s); [TIGERs 2024][TIG24] p.4 |
| guinada sustentada, barra única | perde ≥ ~1 rad/s | KIKS: barra não dividida "dificilmente" segura > 1 rad/s; ZJUNlict (lisa, 3 pontos): foge a 7.0 rad/s; RobôCIn 2021 (silicone, forma não declarada): controle até ±6.5 rad/s | MEASURED | [KIKS 2022][KIKS22] p.3; [ZJUNlict 2019][ZJU19] p.4; [Araújo et al. 2021][LARS21] |
| guinada com barra macia / dividida / rosca | — | gel Asker C15: >10 s abaixo de 5.2 rad/s; dividida PU A50: ≥15 s a 3 rad/s; rosca: foge a 10.9 rad/s | MEASURED | [KIKS 2023][KIKS23] p.6; [KIKS 2022][KIKS22] p.3; [ZJUNlict 2019][ZJU19] p.4 |
| limites de skill com bola (TIGERs) | — | 3.5 rad/s, 20 rad/s², 1.2 m/s, 1.0 m/s² (proteger bola) | VENDOR (configuração de equipe) | Sumatra `DragBallSkill.java:24-37` |
| regras do dispositivo | — | cobertura ≤ 20 %; não levantar a bola; outro robô tem de conseguir tirá-la; não remover todos os GDL; ≤ 1 m driblando | RULEBOOK | SSL rules §3.2.4; §8.4.1 *Excessive Dribbling*, *Ball Holding* |
| mecanismo nos outros simuladores | contato de atrito real | grSim/rSim: junta hinge bola↔kicker; ER-Force: atrito real Bullet μ 1.5 **ou** restrição ponto-a-ponto; Sumatra: "sticky" cinemático que recentra a bola | — (código) | §4 |
| MuJoCo: contato bola↔cilindro | 1 ponto, normal ⊥ ao eixo | é assim por construção; sem rigidez tangencial; `impratio`/`noslip` não agem fora do cone | — (docs/código 3.11.0) | §5.2–5.4 |
| MuJoCo: o que dá "patch"/aderência | — | `flexcomp` (deformável, `young` em Pa) e `adhesion` (novo na 3.11.0); `friction` anisotrópico em `pair` **não serve** (eixo tangente ancorado no mundo) | — (docs/código 3.11.0) | §5.4 |

Veredito por pergunta: **1 VENDOR/MEASURED · 2 não encontrado (um número sem método) · 3 MEASURED +
RULEBOOK · 4 código · 5 docs/código.** A pergunta 2 voltou vazia de verdade: nenhuma fonte primária
mede o atrito de um material de dribbler contra uma bola de golfe.

Achado colateral, fora do material: o cilindro de colisão do rolete tem **r = 9 mm**, 2.5 mm a mais
que a malha visual do mesmo CAD e que o Ø13 mm declarado pela equipe. Não encontrei justificativa no
repositório (`docs/index.md:388-412` não lista o raio; `docs/research/rsoccer-vs-mjsoccer.md:243` só o
registra).

---

## 1. Materiais e geometrias de rolete na SSL

### 1.1 O robô de referência

Que este CAD seja o robô RobôCIn 2025 é inferência registrada em
`.scratch/physics-fidelity/research/01-drivetrain-envelope.md:355-358` (nomes de malha
`Dribbler_Robocup2025v3`). O que a RobôCIn declara do dribbler:

- **Rolete Ø13 mm nas três últimas gerações.** v2022: Maxon EC-max 22 25 W, engrenagem 1:1:1, cobertura
  máx. 19.55 %; v2024: motor e engrenagem de dribbler "Not used", cobertura 17.91 %; v2025: EC-max 22
  25 W, 1:1, cobertura **20.00 %** ([ETDP 2025][RC25] p.3 Tab.1; [ETDP 2026][RC26] p.2 Tab.1).
- **v2025: "a roller bar coated in Polyurethane (PU) with a Shore 65A hardness"**, com suspensão de
  amortecimento lateral; protótipo funcional, mas por falta de motores usado só em alguns robôs, "primarily
  for pass-damping and ball reception"; absorve passes até 5 m/s ([ETDP 2026][RC26] p.3 §2.2).
- Gerações anteriores: barra de três camadas silicone/PLA/aço, "**silicone has 40 shore hardness**", forma
  baseada nas barras da TIGERs e da ZJUNlict, com baixa durabilidade (rasga) e "small bumps"
  ([TDP 2020][RC20] p.6 §2.4); em 2022 a equipe declara que analisou a barra **lisa** e que estudará
  perfis de centragem "modify[ing] only the silicone bar" ([TDP 2022][RC22] §7.2).

A malha visual do rolete neste repositório confere com o Ø13 mm: medi
`models/assets/visual/Dribblerschnecke.stl` — tubo reto, raio externo 6.50 mm, raio interno 3.00 mm,
comprimento 72 mm. O geom de colisão que de fato toca a bola é um cilindro de **r = 9 mm**, 72 mm
(`models/robot_kinematic.xml:337`, `models/robot_rollers.xml:250`), e há um segundo cilindro coaxial de
r = 5 mm, 94 mm (`models/robot_kinematic.xml:336`) que fica dentro do primeiro e só pode tocar a bola além
das pontas.

### 1.2 Levantamento das equipes

Tudo abaixo é declaração das próprias equipes em TDP/ETDP; "medida" só onde a equipe relata um teste.

| equipe, ano | material do rolete | dureza | Ø | geometria | teste/observação relatada | fonte |
|---|---|---|---|---|---|---|
| TIGERs 2020 (v2020) | PU moldado (Smooth-On VytaFlex 60) | 60A | — | **hélice** | motor = barra (1:1:1); 55 W, 17 000 rpm nominal | [TIGERs 2020][TIG20] p.9 §2.3 |
| TIGERs 2022 | o mesmo | — | — | hélice | "during lateral movement the ball is lost easily"; resolvido trocando o **amortecimento** (2 GDL, TPE 70A), não a barra; material "very soft and abrasive" | [TIGERs 2022][TIG22] p.1–2 §1.1 |
| TIGERs 2024 | TPU impresso (Recreus Filaflex 60A) | 60A | 14 mm | **lisa** (abriu mão da centragem) | dribla devagar, com torque limitado; a bola às vezes para e fica presa | [TIGERs 2024][TIG24] p.1, p.4 §2.1 |
| ZJUNlict 2019 | "3D printing rubber" | — | — | lisa vs **rosca** | fuga em giro: lisa 402 °/s, rosca 622 °/s (§3.1) | [ZJUNlict 2019][ZJU19] p.4 |
| KIKS 2017 | poliéster-uretano | — | 15 mm | lisa | — | [KIKS 2017][KIKS17] Tab. |
| KIKS 2022 | PU | A50 | 15 mm | **dividida**, vão central de 21 mm | giro: não dividida < 1 rad/s, dividida ≥ 15 s a 3 rad/s (§3.1) | [KIKS 2022][KIKS22] p.2–3 §2.1 |
| KIKS 2023 | gel de uretano super macio (EXSEAL "HITOHADA") | **Asker C15** | — | cônica | giro: gel > 10 s abaixo de 300 °/s; atual instável ≥ 200 °/s (§3.1) | [KIKS 2023][KIKS23] p.3–7 §2 |
| RoboDragons 2013–2016 | borracha de silicone | — | 16 mm × 73 mm | lisa | — | [RoboDragons 2013][RD13] |
| RoboDragons 2017–2019 | borracha "non-repulsive" | A70 (2020) | 16 mm × 61 mm | lisa | 2019: "slippery against a ball" | [RoboDragons 2017][RD17]; [2019][RD19] |
| RoboDragons 2020 | poliéster-uretano ("Ester Polyurethane") | A50 | — | lisa | ré a aceleração constante 0.15–0.8 m/s²: PU melhor que a borracha A70 (só em gráfico) | [RoboDragons 2020][RD20] p.2–3 §2 |
| RoboTeam Twente 2020 | tubo de silicone sobre eixo metálico | — | — | reta; testou ampulheta, rosca, polia abaulada | **ampulheta empurra a bola para fora** (v = rω cresce para as pontas) | [RTT 2020][RTT20] p.9–10 §4.4 |
| ER-Force 2014 | mangueira de **neoprene** (avaliou PU e silicone) | A70 | 16 mm | lisa | "best friction"; muito abaixo de 70A a barra deforma e a bola pula; ≥ 90A rebate demais | [ER-Force 2014][ERF14] p.3 §2.2 |
| RoboFEI 2013 | borracha natural, PU, silicone | 20–30A | — | lisa | em um dos carpetes, só PU 30A e silicone 20A sem vibração excessiva (corrente do motor) | [RoboFEI 2013][RF13] p.6 |
| RoboFEI 2018 / 2023 | PU 30A liso → silicone 20A com rosca → silicone | 30A / 20A / 50–60A | — | lisa → rosca | — | [RoboFEI 2018][RF18]; [2023][RF23] |
| Op-AmP 2017 | **selante de silicone** (testou uretano macio/duro, silicone macio) | — | — | autocentrante, centro rebaixado | uretano duro "did not have enough grip"; macios sem resistência | [Op-AmP 2017][OPA17] p.5 §2.3 |
| Parsian 2013 | tubo oco de silicone moldado | — | — | lisa | "friction coefficient is about 0.8" (§2) | [Parsian 2013][PAR13] p.8 |
| MRL 2013–2016 | "carbon silicon tube" (vs PU, silicone) | — | — | lisa | escolhido pela "higher capability in ball handling" | [MRL 2013][MRL13] |
| RFC Cambridge 2014 | silicone (depois de uretano) | Shore **OO-30** | — | côncava/convexa | — | [RFC Cambridge 2014][RFC14] p.2 |
| KgpKubs 2020 | Ecoflex 00-50 | OO-50 | — | hélice | — | [KgpKubs 2020][KGP20] p.3 |
| ITAndroids 2019 | borracha de silicone | 25A | — | estrias espelhadas | — | [ITAndroids 2019][ITA19] |
| RoboJackets 2019 | uretano | 30 | 15 mm (2024–26) | — | rolete anterior desintegrava | [RoboJackets 2019][RJ19] p.5 |
| UBC Thunderbots 2023 / 2025 | PU 1:1 → TPU impresso | 50A → 60A | — | rosca → **lisa** | — | [UBC 2023][UBC23] p.4; [UBC 2025][UBC25] p.4 |
| Project Neon 2025 | silicone | A14 | 12 mm | hélice (à la TIGERs) | — | [Project Neon 2025][PN25] p.11–12 |
| TurtleRabbit 2026 | VytaFlex 60 (uretano) | 60A | — | cônico → — | escolhido "for its high friction coefficient" (sem valor) | [TurtleRabbit 2026][TR26] p.4 |
| Luhbots 2026 | silicone SH45 → **SH10** | A45 → A10 | — | **dois roletes independentes** | drop test de bola de golfe (§1.3); divisão visa "fast turns" | [Luhbots 2026][LUH26] p.3–7 |

Padrão (**Inferência**, de contagem na tabela): os elastômeros dominantes são PU/uretano fundido
(30A–65A), silicone fundido (OO-30 a 60A) e, desde 2024, TPU 60A impresso. A dureza mais comum nas
equipes de topo está em **50A–65A**; o extremo macio (Asker C15, OO-30, A10, A14) aparece justamente
onde a equipe persegue retenção ou absorção, sempre com queixa de desgaste. Diâmetros declarados: 12–16 mm.

### 1.3 O que as medições dizem sobre material vs. geometria

- **Material importa, e o mecanismo declarado é área de contato.** KIKS 2023 compara a barra atual
  (PU, a dividida de 2022) com uma cônica de gel Asker C15 em giro a velocidade angular constante, motor
  ~6300 rpm: a atual "cannot keep the ball stably when the robot's angular velocity is 200 deg/s or
  higher"; a de gel "could keep the ball stably for more than 10 seconds if the angular velocity was less
  than 300 deg/s" — e, contra a previsão deles, mantinha a bola **na ponta** da barra, o que atribuem a
  "the contact area with the ball is larger, which may be the result of sufficient frictional force"
  ([KIKS 2023][KIKS23] p.6–7). No mesmo documento, um cone duro e um cone de gel com **a mesma forma**
  movem a bola lateralmente em **sentidos opostos**; a explicação deles é que no material macio "the ball
  dips into the dribble bar", aumentando a área de contato (p.3–4, Fig.3); a velocidade lateral medida
  ao longo da barra é ~0.1 m/s nos três carpetes (p.4 Tab.1).
- **Geometria/topologia de contato pesa tanto quanto o material.** KIKS 2022: com a barra dividida
  (dois segmentos de PU A50, Ø15 mm, vão de 21 mm), a bola fica ≥ 15 s a 3 rad/s; a barra não dividida
  anterior "can hardly hold a ball" acima de 1 rad/s ([KIKS 2022][KIKS22] p.3). ZJUNlict 2019: mesma
  borracha, rosca vs lisa, 402 → 622 °/s ([ZJU19] p.4). RTT 2020: a ampulheta "drove the ball to the
  sides" porque a velocidade periférica cresce para as pontas ([RTT20] p.10).
- **Amortecimento e controle do motor resolveram perdas laterais na prática.** TIGERs 2022 perdia a
  bola em movimento lateral com o dribbler v2020 e resolveu com a estrutura de amortecimento de 2 GDL
  ([TIG22] p.1–2); TIGERs 2024 passou a driblar devagar com controle de torque (§3.1).
- **Restituição medida com bola de golfe (normal, não atrito).** Luhbots 2026 solta uma bola de golfe de
  0.74 m sobre blocos de silicone de 70 × 50 × 13.5 mm (Shore 00, 10, 25, 33, 45; 10 quedas cada): a altura
  de rebote sobe com a dureza; **SH10 rebate 0.29 m em média, 61.4 % da energia absorvida**
  ([LUH26] p.6–7 §5). **Inferência:** COR = √(0.29/0.74) ≈ **0.63**; os demais valores só existem no
  boxplot.
- **O que não ajuda.** ER-Force 2014: abaixo de 70A a barra deforma e a bola "repetitively jumps away";
  acima de 90A rebate demais ([ERF14] p.3). ZJUNlict 2019 afirma que, na arquitetura de dois pontos com
  controle passivo, "it is also hard to greatly improve its performance by simply changing the material
  of dribbling-bar" ([ZJU19] p.2).

### 1.4 Repositórios de hardware aberto

Não encontrei especificação de material de rolete em repositório de hardware aberto de equipe (busca de
código no GitHub por "dribbler"/"shore" sem resultado útil). O único dado de equipe em código aberto que
usei é o Sumatra da TIGERs (§3.2), que traz parâmetros de operação, não de material.

---

## 2. Coeficiente de atrito do material contra a bola de golfe

**Não encontrado.** Nenhum TDP/ETDP 2013–2026, nenhum dos dois artigos revisados por pares e nenhuma
busca por literatura de bola de golfe contra elastômero trouxe uma medição.

O que existe:

| fonte | valor | o que é |
|---|---|---|
| [Parsian 2013][PAR13] p.8 | "about 0.8" | tubo oco de silicone; sem método, sem contraparte declarada |
| [ZJUNlict 2025][ZJU25] p.2 | — | lista "the friction coefficient between the ball and the dribble-bar" entre os parâmetros "hard to obtain"; por isso modelam o contato como mola-amortecedor 1-D com valores **supostos** (p.3 Tab.1: k₁ = 4000 N/m, c₁ = 5 N·s/m, "we do not know the true values") |
| [ZJUNlict 2019][ZJU19] p.3 | qualitativo | μ dinâmico bola↔carpete > bola↔pá do chip |
| [ER-Force 2014][ERF14] p.3; [TurtleRabbit 2026][TR26] p.4 | qualitativo | neoprene A70 "best friction"; VytaFlex 60 "high friction coefficient" |
| grSim `config` padrão, `src/configwidget.cpp:270` | 0.8 | `KickerFriction`, caixa do kicker que faz papel de dribbler |
| rSim `src/robosim/sslconfig.h:169` | 0.8 | `kickerFriction` |
| ER-Force `simrobot.cpp:109` × `simball.cpp:54` | 1.5 | atrito do corpo do dribbler (1.5) × bola (1.0); o Bullet multiplica (`btManifoldResult.cpp:56-66`) |
| este repositório, `models/scene.xml:21-22` | 2.5 | DELIBERATE APPROXIMATION, "above 1 so the bar can hold the ball" (`docs/index.md:406`) |

Conclusão: o 2.5 continua sem base; os únicos números de outros simuladores (0.8, 1.5) também não têm.
Como μ 5 e μ 10 já foram testados e não resolveram, **μ não é a alavanca** — e não há valor medido para
calibrá-lo.

---

## 3. Retenção medida em giro e aceleração; regras

### 3.1 Medições publicadas

| fonte | configuração | protocolo | resultado |
|---|---|---|---|
| [KIKS 2022][KIKS22] p.3 Fig.5 | barra não dividida vs dividida (PU A50, Ø15 mm, vão 21 mm), motor 4500–6300 rpm, 3 carpetes de polipropileno | gira com a bola, tempo médio de retenção (teto 20 s) | não dividida: "can hardly hold a ball" acima de **1 rad/s**; dividida: **≥ 15 s a 3 rad/s**; menos eficaz no carpete de atrito alto |
| [KIKS 2023][KIKS23] p.6 Fig.7 | barra atual vs gel Asker C15, motor ~6300 rpm | gira a ω constante até perder | atual instável a ≥ **200 °/s (3.5 rad/s)**; gel > 10 s abaixo de **300 °/s (5.2 rad/s)** |
| [ZJUNlict 2019 ETDP][ZJU19] p.4 Tab.1; [RCAR 2019][ZJU19RCAR] p.5 | dribbler de 3 pontos de contato (bola toca carpete, rolete e pá do chip), mesma borracha, lisa vs rosca | rampa de guinada **20 °/s²**, registra ω na fuga, 10 repetições | lisa: média **402 °/s (7.0 rad/s)**, faixa 340–520; rosca: **622 °/s (10.9 rad/s)**, 580–680 |
| [ZJUNlict RCAR 2019][ZJU19RCAR] p.5 Tab.2 | controle passivo (velocidade) vs ativo (torque) | ré a 0.2 m/s², registra velocidade na fuga | passivo **1.63 m/s**, torque **2.5 m/s** |
| [RoboDragons 2020][RD20] p.3 Fig.4 | borracha A70 vs PU A50 | ré a aceleração constante 0.15–0.8 m/s² | PU melhor; valores só no gráfico |
| [Araújo et al. 2021][LARS21] (resumo); [RobôCIn 2022][RC22] p.9 §5 | dribbler RobôCIn 2021, barra de silicone (a análise a supõe lisa, §7.2; a forma real não é declarada), amortecimento EVA | testes "in a competition environment" | "keep the ball control" com vx −0.5…1.5 m/s, vy ±0.5 m/s, **ω −6.5…6.5 rad/s**; nenhum critério de perda declarado |

Para comparar com a sonda deste repositório: 120 passos de controle = **2 s** (16 passos físicos de
1/960 s por passo de controle, `docs/index.md:501-502`, `models/scene.xml:4`). As medições acima usam
retenção de 10–20 s ou a ω de fuga numa rampa; a sonda de 2 s é mais branda que ambas.

**Inferência.** As fontes não convergem num único número para "barra única": a barra não dividida da KIKS
falha perto de 1 rad/s, como a simulação, enquanto a lisa da ZJUNlict aguenta 7 rad/s — mas com terceiro
ponto de contato (e, no mesmo robô, controle de torque) — e a RobôCIn 2021 declara ±6.5 rad/s. A medição mais próxima da
linhagem deste CAD é a da própria RobôCIn, só que com outro rolete (silicone, 2021) que não é o da v2025
(PU 65A). O dado de domínio "não há guinada medida" não vale para a equipe: há um número publicado e
revisado por pares, embora sem critério de perda.

### 3.2 Limites de operação em código (TIGERs Sumatra, `496376d`)

- `DragBallSkill` (proteger a bola contra um oponente): `accMax` 1.0 m/s², `accMaxW` **20 rad/s²**,
  `velMax` 1.2 m/s, `velMaxW` **3.5 rad/s** (`modules/sumatra-skillsystem/.../dribbling/DragBallSkill.java:24-37`).
- `DribbleKickSkill`: `accMax` 1.5 m/s², `accMaxW` 15 rad/s²; no chute, 2.0 m/s² e 25 rad/s²
  (`DribbleKickSkill.java:35-45`).
- Robô v2020: `dribblerSpecs` `defaultSpeed` 3.0, `defaultForce` 5.0 (`config/botParamsDatabase.json:79-84`),
  em **m/s de superfície da barra** e **N de força máxima**, quantizados em 0.125 m/s e 0.25 N
  (`modules/moduli-botmanager/.../KickerDribblerCommands.java:20-65`).
- Estado de tração do dribbler reportado pelo robô: OFF, IDLE, LIGHT, STRONG
  (`modules/common-bot/.../EDribbleTractionState.java`); a `DragBallSkill` só libera `accMax` quando STRONG
  (`DragBallSkill.java:67-79`).

São limites de operação escolhidos pela equipe, não limites de retenção medidos.

### 3.3 Modelos analíticos publicados

- **CMDragons 2015** ([ETDP][CMD15] p.7 §4): girar em torno da bola funciona sem spin, mas "when the robot
  has imparted backspin on the ball, the ball will roll and be lost as the robot turns around it"; girar
  no lugar é o caso particular do arco com R = distância centro→dribbler. Modelo: a força de atrito do
  carpete |f_F| = μmg tem de ser equilibrada pela componente da normal do robô, inclinando a frente para
  dentro de φ = tan⁻¹(μg / ω²R). "A more sophisticated model, accounting for the spin of the ball, is
  ongoing work."
- **TIGERs `DribblingVerifier`** (`modules/common/.../trajectory/DribblingVerifier.java:206-224`): no
  referencial do robô, aceleração da bola = −a_robô + (α·R, ω²·R) + puxada do dribbler
  (`maxDribbleBallAcceleration`); a bola é perdida quando o ângulo desse vetor em relação ao eixo do
  robô passa de `maxRetainingBallAngle`. Os dois parâmetros são "Available in BotParams" segundo o
  javadoc (`:37-38`), mas **não existem** no `BotParams`/`botParamsDatabase.json` publicados — não
  encontrei os valores. O modelo é puramente cinemático: sem spin da bola, sem atrito do carpete.

**Inferência**, com os números deste modelo (bola 93.8 mm à frente, `probe_tier0.json`): girando no lugar
a ω constante, a aceleração centrípeta da bola é ω²R = 0.094 m/s² a 1 rad/s (4.3 mN) e 0.375 m/s² a
2 rad/s — pequena, e na direção *radial*, que o rolete absorve pela normal. O limite do CMDragons para a
força lateral de carpete (μmg com o μ 0.35 deste modelo) é 0.16 N. Nenhuma das duas fontes modela o termo
que **eu** acho que domina aqui (§5.6).

### 3.4 Regras e desafios técnicos

- **§3.2.4 *Dribbling Device*** (`chapters/robots.adoc:44-50`): dribblers que aplicam spin são permitidos
  se: não levantarem a bola; "another robot must be able to remove the ball from a robot with the ball";
  "a robot must not take full control of the ball by removing all of its degrees of freedom"; e 80 % da
  área da bola vista de cima fica fora do fecho convexo do robô (também para kickers, mesmo
  momentaneamente). O `02` traduziu isso para ≤ 10.9 mm de penetração no casco.
- **§8.4.1 *Stopping Fouls* → *Excessive Dribbling*** (`chapters/offenses.adoc:120-123`, título de 5º
  nível, sem número no livro renderizado): não driblar mais de 1 m em linha reta desde o início; o drible
  começa no contato e termina com "observable separation". **→ *Ball Holding*** (`:92-93`): não cercar a
  bola para impedir acesso. Evento `BOT_DRIBBLED_BALL_TOO_FAR` detectado pelo auto-referee
  (`chapters/appendix.adoc:103`).
- **Nenhuma regra fixa guinada, aceleração angular ou aceleração lateral com bola.** O Hardware Challenge
  2021, *Challenge 3: Dribbling* (atualizado 2021-06-16): percurso de portões, o último cruzado três vezes,
  2 min por tentativa, drible > 1 m permitido, perder a bola não falha a tentativa
  ([regras][HWC21]). O *Dribbling Challenge* 2022 (2022-06-15): sequência de portões A–F, sem requisito
  angular ([regras][DC22]). A TIGERs venceu o de 2021 ([TIG22] p.2).

---

## 4. Como os outros simuladores modelam o dribbler

| simulador (commit) | mecanismo | onde |
|---|---|---|
| **grSim** (`fe2bd29`, ODE) | **junta, não atrito.** Com o roller ligado e a bola tocando a caixa do kicker, zera a velocidade da bola e cria `dJointCreateHinge` entre caixa e bola; destrói quando a bola deixa de tocar ou o roller desliga, e antes de todo chute. O comando é binário (`dribbler_speed() > 0 ? 1 : 0`). O contato bola↔kicker continua sendo de atrito (μ `KickerFriction` 0.8, `slip1` 5) | `src/robot.cpp:95-112`, `:181-204`, `:159`; `src/sslworld.cpp:293-296`, `:921-923`, `:634-636` |
| grSim antes de `19f6355` (2019-09-14, "make it dribble") | **força especial.** Impunha a velocidade angular da bola (`dBodySetAngularVel`, `RollerTorqueFactor × 1400` = 84 rad/s com o padrão 0.06) e um torque de centragem proporcional ao deslocamento lateral (`RollerPerpendicularTorqueFactor`). Os dois parâmetros ainda são lidos (`src/configwidget.cpp:268-269`) mas não usados | `src/robot.cpp:108-126` em `19f6355^` |
| **rSim** (`b413932`, ODE; motor do rSoccer `3adf7c3`) | cópia do grSim: hinge caixa↔bola com o dribbler ligado e a bola tocando; μ bola↔kicker 0.8. Detalhe: o passo só faz `setDribbler(true)` e nada o desliga depois; `toggleDribbler` é um no-op (`!=`) | `src/robosim/sslrobot.cpp:110-125`, `:202-229`, `:162-165`; `src/robosim/sslworld.cpp:163-166`, `:320`; `sslconfig.h:169` |
| **ER-Force** (`38563d1`, Bullet) | **dois modos.** (a) físico, padrão: corpo cilíndrico de r = 7 mm × `dribbler_width` (70 mm na `config/robots/generation_2020.txt:13`), atrito 1.5, restituição 0.2, preso ao robô por `btHingeConstraint` com motor angular (alvo ≤ 150 rad/s, impulso máx. 20·fração) — atrito real contra a bola (atrito 1.0); (b) "perfect": `btPoint2PointConstraint` robô↔bola (bola presa por um ponto, rotação livre) mais uma restrição anti-tombamento. Padrão `dribblePerfect = false`; o preset `RC2021` liga o perfeito (`simulate_dribbling: false`), os presets `Realistic`/`Friendly`/`None` usam o físico | `src/amun/simulator/simrobot.cpp:91-121`, `:155-199`; `simulator.cpp:175`, `:858-861`; `config/simulator-realism/RC2021.txt:15`, `Realistic.txt:15` |
| ER-Force, atrito do Bullet (`d3afe76`) | por padrão **uma única direção de atrito**, recalculada a cada passo ao longo da velocidade relativa tangencial, limitada a ±μ × impulso normal (`m_solverMode` padrão = `SOLVER_USE_WARMSTARTING` + `SOLVER_SIMD`); coeficiente combinado = produto | `btSequentialImpulseConstraintSolver.cpp:1080-1105`, `:605-606`; `btContactSolverInfo.h:102`; `btManifoldResult.cpp:56-66` |
| **TIGERs Sumatra** (`496376d`, próprio, 2-D) | **cinemático "sticky".** Com `dribbleRpm > 0` a linha frontal do kicker vira pegajosa: a colisão vira totalmente inelástica nas duas direções, a bola é **teleportada para o centro da linha frontal** (`stick()` = `getPathCenter()`) 1 mm para dentro, recebe a velocidade da superfície (translação + rotação do robô) e um spin `dribblingSpeed` (padrão 0 rpm); a linha aplica aceleração de 10 (unidade não declarada) puxando a bola para o robô; detecção "dentro" com margem de 10 mm enquanto pegajosa | `modules/moduli-vision-simulation/.../BallCollisionHandler.java:34-35`, `:155-166`, `:214-262`; `BotCollisionObject.java:84-110`, `:158`; `KickerFrontLineCollisionObject.java:52-55`, `:93-98` |

**Inferência.** Só o modo físico do ER-Force faz o que este repositório faz (atrito de um cilindro
girando contra a bola) — e o Bullet, com uma única linha de atrito alinhada com a velocidade de
deslizamento total, tem a mesma propriedade de "o atrito aponta contra o deslizamento total" que o cone
elíptico do MuJoCo (§5.3). Os outros três prendem a bola por construção (junta, ponto ou teleporte), e o
Sumatra ainda a recentra a cada passo — reter em giro é trivial neles, e não é evidência de que um
contato de atrito deveria reter.

---

## 5. Material → parâmetros de contato do MuJoCo 3.11.0

### 5.1 Versão

`uv.lock:413` trava `mujoco` 3.11.0 (dependência `>=3.11.0`, `uv.lock:386`); conferido no ambiente.
Toda referência abaixo é a `doc/` e `src/engine/` na tag `3.11.0` (`b85fdca`, 2026-07-27) do repositório
`google-deepmind/mujoco`; a documentação online equivalente é https://mujoco.readthedocs.io/. Dois recursos
relevantes **estreiam exatamente nessa versão**: `geom/surfacevel` e `geom|pair/adhesion`
(`doc/changelog.rst:16-35`).

### 5.2 Semântica dos parâmetros

- **`friction`** do geom = (deslizante, torcional, rolante); o deslizante age nos dois eixos tangentes; o
  torcional e o rolante têm **unidade de comprimento** — o torcional "can be interpreted as the diameter
  of the surface contact patch", o rolante como profundidade da deformação local
  (`doc/XMLreference.rst:2708-2715`; `doc/computation/index.rst:959-976`). O `pair` tem 5 coeficientes; os
  dois primeiros diferentes dão atrito tangencial anisotrópico (`XMLreference.rst:4210-4217`). Um `pair`
  explícito ignora os parâmetros dos geoms; sem `pair`, o atrito é o **máximo elemento a elemento** e
  `solref`/`solimp` são média ponderada por `solmix` (mínimo se algum usar o formato direto)
  (`doc/modeling.rst:418-470`).
- **`condim`**: 1 sem atrito; 3 atrito tangencial "opposing slip"; 4 soma atrito torcional, "useful for
  modeling soft fingers"; 6 soma rolante (`computation/index.rst:946-982`). O par bola↔rolete usa 4 com
  torcional 0.02 m (`models/scene.xml:21-22`), ou seja um "patch" de 20 mm de diâmetro.
- **Cone** (`option/cone`, `XMLreference.rst:398-402`): o elíptico é
  f₁ ≥ 0, f₁² ≥ Σᵢ₌₂ⁿ fᵢ²/μᵢ₋₁² (`computation/index.rst:994-997`): **um único orçamento** compartilhado
  pelos dois eixos tangentes e, com condim 4, pelo torcional. O cenário já usa `cone="elliptic"`, Newton
  (`models/scene.xml:4`).
- **`solref`**: positivo = (timeconst, dampratio); negativo = (−rigidez, −amortecimento), "recommended
  format for system identification"; dampratio 1 ≡ amortecimento = 2√rigidez
  (`modeling.rst:328-398`). Afundamento em repouso, impedância constante d: r = a_u(1−d)·timeconst²·dampratio²
  no formato positivo e r = a_u(1−d)/rigidez no direto (`:363-367`, `:382-386`), com a_u a aceleração
  sem restrição. timeconst ≥ 2·timestep, imposto salvo `refsafe` (`:358-361`).
- **Dimensões de atrito do cone elíptico**: resíduo de posição identicamente zero, portanto **sem mola
  tangencial**: dinâmica de primeira ordem (decaimento da velocidade de deslizamento), impedância sempre
  d₀ = `solimp[0]`, no formato positivo só o timeconst conta, no direto só o amortecimento
  (`modeling.rst:400-416`; `computation/index.rst:1170-1172`, `:1530-1534`). `pair/solreffriction`
  separa esse timeconst do normal (só cone elíptico, `XMLreference.rst:4226-4238`). O MuJoCo também não
  guarda estado de deformação entre passos: "There is a related phenomenon associated with slip in
  friction dimensions which is also ignored" (`computation/index.rst:1205-1223`).
- **`solimp`**: (d₀, d_width, width, midpoint, power) — impedância em função da penetração; é o que modela
  uma "camada macia" (`modeling.rst:233-302`; `computation/index.rst:1562-1567`).
- **`margin`/`gap`**: o contato gera força a distância ≤ `margin` (soma dos dois geoms) e a impedância é
  aplicada a distância − margin; `gap` só detecta (`XMLreference.rst:2756-2772`, `:4240-4251`;
  `computation/index.rst:912-945`).
- **`impratio`**: razão de impedância atrito/normal no cone elíptico, "having the general effect of
  preventing slip, without increasing the actual friction coefficient" (`XMLreference.rst:323-330`).
  **`noslip_iterations`**: pós-processamento PGS com R = 0 nas dimensões de atrito; "ad-hoc correction"
  (`computation/index.rst:1413-1419`; `XMLreference.rst:448-452`).
- **O limite de todos esses botões** está escrito: "*Slip-preventing contact forces are outside the
  friction cone.* This implies that the physics cannot prevent slip, even in principle" — e para isso a
  receita é aumentar a normal, o μ, o condim, ou "improve the geometry of the contacting geoms in order to
  add more contact points, possibly with non-flat geometry (e.g., bumps), so slippage is prevented by the
  normal force and not only frictional components"; `impratio` e `noslip` são a receita para "slow
  slippage", que é outro problema (`modeling.rst:1682-1738`; `overview.rst:833-863`).

### 5.3 Um contato deslizante transmite força lateral?

**O que a fonte diz.** A força de atrito pertence ao cone elíptico acima: o módulo tangencial total está
limitado por μ·f_N, compartilhado entre as direções (`computation/index.rst:994-997`). O condim 3
"generate[s] … a tangential friction force opposing slip" (`:959-962`). O modelo convexo não impõe
complementaridade estrita, então "frictional forces may not be maximally dissipative" (`:52-54`).

**Inferência.** Sim, transmite — mas só a fatia lateral de um único orçamento. Sob máxima dissipação (de
que o MuJoCo se aproxima), um contato saturado deslizando com velocidade v = (v_giro, v_lat) dá
f_lat ≈ μ f_N · v_lat/|v|. Com v_giro da ordem de 1–2 m/s e os 56 mm/s laterais observados, isso é 3–6 %
de μ f_N. Essa é a hipótese de trabalho, e ela é coerente com a documentação. A §5.6 mostra por que vale
medir antes de aceitá-la.

### 5.4 Existe no MuJoCo um "patch" que dê aderência lateral a um rolete macio girando?

- **Não com primitivas.** `mjc_SphereCylinder` resolve o contato lateral projetando o centro da esfera
  no eixo do cilindro e chamando esfera-esfera (`src/engine/engine_collision_primitive.c:345-383`), que
  produz **exatamente um contato** com normal ao longo da linha entre os centros (`:262-293`). A normal é
  sempre ⊥ ao eixo do rolete, qualquer que seja a penetração: amolecer `solref`/`solimp` aprofunda o
  ponto, mas **não cria componente lateral da normal** (o "berço" que um elastômero real forma). Retenção
  lateral num cilindro liso só pode vir do atrito.
- **Mais geoms = mais pontos.** É o caminho da receita "add more contact points" (§5.2). V, ampulheta e
  hélice já foram testados e não resolveram; RTT 2020 explica por que a ampulheta piora (§1.3).
- **Atrito anisotrópico em `pair` não serve aqui.** O esfera-esfera zera a tangente
  (`engine_collision_primitive.c:292`) e o driver completa o referencial com `mju_makeFrame`
  (`src/engine/engine_collision_driver.c:1854`), que escolhe o 1º eixo tangente como **o eixo y do
  mundo** projetado no plano tangente (ou z do mundo se a normal for quase paralela a y)
  (`src/engine/engine_util_spatial.c:508-531`). **Inferência:** com o robô girando, os coeficientes
  "lateral" e "de giro" de um `pair` anisotrópico giram em relação à barra; só ficariam alinhados com o
  robô parado de frente para ±x.
- **Deformáveis: `flex`/`flexcomp`.** Corpos-vértice ligados por elementos que deformam, com colisão e
  contato próprios; um contato com elemento deformável pode envolver vários corpos
  (`modeling.rst:1337-1428`). Pontos "pinned" pertencem ao corpo pai do `flexcomp`, os demais viram
  corpos filhos com 3 juntas deslizantes (`XMLreference.rst:3614-3642`, `:3999-4011`); elasticidade
  Saint Venant–Kirchhoff com **`young` em pressão** e `poisson` < 0.5 (`XMLreference.rst:4426-4450`);
  atributos de contato iguais aos de geom (`:4513-4530`). Parametrizações `trilinear`/`quadratic`
  reduzem os GDL (`modeling.rst:1388-1399`). **Inferência:** uma luva de elastômero com os nós internos
  presos ao corpo `dribbler` seria o único jeito de ter um patch real, e o módulo vem direto da dureza
  (§5.5) — mas `pair` só referencia geoms (`XMLreference.rst:4193-4200`), então o contato bola↔luva usa
  mistura de atributos, e o custo em GDL a 960 Hz é desconhecido.
- **"Elasticity plugins".** A antiga via plugin 2-D/3-D com `composite` foi removida em favor do
  `flexcomp` (3.2.3, `changelog.rst:1337-1338`); o plugin `shell` também (3.3.3, `:975-976`). Resta o
  plugin `cable` (1-D), inútil para um rolete.
- **`adhesion` (novo na 3.11.0).** Translada o cone ao longo da normal: o contato pode puxar até
  `adhesion` e "the friction budget becomes μ(f_N + adhesion). Contacts resist sliding even under zero
  normal force … useful for sticky materials (tape, gecko feet, tacky rubber)"; é por contato, soma dos
  geoms ou o valor do `pair`; com `gap` age a distância (`XMLreference.rst:2801-2826`, `:4253-4258`;
  `computation/index.rst:1541-1560`). Contatos adesivos dentro do `gap` perdem as linhas tangenciais
  (`engine_collision_driver.c:1846-1851`).
- **`surfacevel` (novo na 3.11.0).** Superfície que se move com o geom parado (esteiras, mesas
  giratórias); compõe com o movimento do corpo e, com condim ≥ 4, o termo angular também alimenta o
  atrito torcional (`XMLreference.rst:2774-2799`; `computation/index.rst:1535-1540`). **Inferência:**
  permite um rolete sem a junta `dribbler_spin` e sem a queda de velocidade do atuador, mas não muda o
  cone — não é, por si, uma solução para a retenção lateral.

### 5.5 Números do material de referência

Fórmulas de dureza→módulo tiradas da página [Shore durometer][WIKI], que as atribui a A. N. Gent, *Rubber
Chem. Technol.* **31**(4):896 (1958) — E[MPa] = 0.0981(56 + 7.62336 S)/(0.137505 (254 − 2.54 S)) — e a
Qi, Joyce & Boyce (2003) — log₁₀E = 0.0235 S − 0.6403. **Não consultei os artigos originais**; o resumo de
Gent dá validade para dureza ≳ 40. Hertz com bola rígida, ν = 0.5, raio equivalente √(R′R″) para o
contato esfera(21.5 mm)×cilindro(6.5 mm) — resultado clássico, aplicado por mim (**Inferência** do começo
ao fim desta tabela).

| | PU 65A, Gent (E = 4.4 MPa) | PU 65A, Qi (E = 7.7 MPa) |
|---|---|---|
| afundamento δ a 1 / 3 / 6 N | 0.12 / 0.24 / 0.38 mm | 0.08 / 0.17 / 0.26 mm |
| raio do patch a | 1.1 / 1.6 / 2.0 mm | 0.9 / 1.3 / 1.7 mm |
| rigidez tangente 1.5F/δ | 1.3 / 1.9 / 2.4 ×10⁴ N/m | 1.9 / 2.7 / 3.4 ×10⁴ N/m |

(A 9 mm de raio os números mudam < 5 %.) Para comparação, o ZJUNlict supõe k₁ = 4000 N/m, c₁ = 5 N·s/m
([ZJU25] p.3).

Modelo atual, com a fórmula de repouso da §5.2 e massa efetiva ≤ 0.046 kg: a 3 N, r ≈ 13 µm (d = 0.95)
a 2.6 µm (d = 0.99), ou seja ~2×10⁵ a 10⁶ N/m — **uma ordem de grandeza mais duro** que o PU 65A, e
patch de 20 mm no coeficiente torcional contra 2–4 mm de Hertz. Para reproduzir δ ≈ 0.2 mm a 3 N com
d₀ = 0.95: timeconst ≈ √(δ / (a_u(1−d))) / dampratio ≈ 7–9 ms com dampratio 1 (14–17 ms com 0.5). No
formato direto, a mesma fórmula pede rigidez = (1−d)·k_físico/m, não k_físico/m.

Restituição: num oscilador linear, COR = exp(−ζπ/√(1−ζ²)); o `dampratio` 0.5 atual daria ~0.16, e o
COR 0.63 do silicone A10 da Luhbots pediria ζ ≈ 0.15. No MuJoCo, porém, o COR depende também de
`solimp`, e os tickets 17/21 deste repositório chegaram ao `solref` da bola **medindo** o COR em simulação
(`docs/index.md:401`), não pela fórmula.

### 5.6 O que a física do giro sugere sobre a falha

**Inferência** do começo ao fim, para orientar a sonda:

1. **O deslizamento "macio" de um contato que adere é pequeno demais para explicar 56 mm/s.** Nas
   dimensões de atrito, a velocidade de deslizamento em regime vale ≈ (1−d₀)/(d₀·b)·a_u, com
   b = 2/(d_width·timeconst) ≈ 505 s⁻¹ (`modeling.rst:340-352`, `:400-416`): ≈ 1×10⁻⁴ s × a_u, ou seja
   < 0.5 mm/s para cargas laterais de até 0.2 N. O deslizamento lateral observado exige ou **cone
   saturado** ou **contato ausente**.
2. **O contato está ausente boa parte do tempo.** Mesmo parado, o contato bruto bola↔rolete existe em só
   74 % (tier 0) e 65 % (tier 1) dos passos — o rolete "sticks, slips and re-grips", e `holds()` só
   funciona por causa da janela de 0.05 s (`docs/index.md:308-322`). No traço a 1 rad/s a normal oscila
   entre 0.48 e 6.45 N de uma amostra para a outra (ticket `22` §Diagnóstico 2). Fora do contato, a bola fica
   só sobre o carpete, cujo contato também desliza no sentido do giro da bola e oferece pouco
   lateralmente. Isso explicaria por que μ×4, `impratio` e `noslip` não mudam nada (não agem quando não há
   contato) — e é exatamente o problema de "bouncing" contra o qual ZJUNlict (3 pontos), KIKS (amortecedor,
   gel) e TIGERs (2 GDL) projetaram.
3. **O rolete não satura em média.** O torque do atuador sob carga, kv·(alvo − medido), dá ≈ 0.012 N·m
   tanto em kv×1 (2178 rpm) quanto em kv×10 (2921 rpm) (ticket `22` §Diagnóstico 5, `models/robot_kinematic.xml:370`),
   ~1.3 N tangenciais a 9 mm. O orçamento μ·f_N = 2.5 × (0.5–6.5 N) só fica abaixo disso nas fases de
   normal baixa — de novo o ciclo de quique.
4. **O vetor de spin da bola tem de girar junto com o robô.** Com o rolete rolando sobre a bola a ~2 m/s,
   a bola gira a ~93 rad/s; com β = 0.5 (I = 1.06×10⁻⁵ kg·m²), girar esse momento angular a ω_z exige um
   torque ω_z·I·ω_bola ≈ 1×10⁻³ N·m por rad/s em torno do eixo frontal do robô, ~0.05 N por rad/s de força
   lateral nos contatos (alavanca r_bola). Isso é da mesma ordem do limite μmg = 0.16 N do CMDragons, e
   **cresce com a velocidade do rolete** — coerente com a varredura de rpm, em que a retenção cai com a
   rotação em todo ω (kv×1, ω = 0.5: 118/104/69/20 a 1.5/3/6/9 krpm; ticket `22` §Diagnóstico 5). Essa física
   vale também no robô real; nenhuma das fontes a modela. CMDragons a deixa explicitamente para depois
   (§3.3).
5. **A simulação pode não estar "errada" para uma barra lisa única.** A KIKS mediu a mesma falha perto de
   1 rad/s numa barra não dividida (§3.1). O que separa os robôs que retêm a 3–7 rad/s está documentado:
   contato dividido ou de três pontos, amortecimento/material que mantêm o contato, e torque limitado em
   rotação baixa.

---

## Implicações para o modelo

Nada implementado. Cada opção traz a fonte em que se apoia e a sonda que a verifica. A sonda padrão é
retidos/120 a ω = 0.5/1/2 rad/s com o rolete em comando máximo, seeds 0–4. A ordem sugerida é 0 → 1 → (2, 3, 5)
→ 4 → 6 → 7.

0. **Diagnóstico antes de mudar parâmetros.** A cada passo físico da sonda, registrar: (a) ciclo de
   trabalho do contato bola↔rolete; (b) razão de saturação √(f₂²/μ₁² + f₃²/μ₂² + f₄²/μ_t²)/f₁ via
   `mj_contactForce`; (c) velocidade de deslizamento no contato decomposta em giro/lateral (linhas de
   atrito em `efc_vel`); (d) ângulo entre ω da bola e o eixo do rolete; (e) força lateral no contato bola↔
   carpete. Regra de decisão: deriva lateral acumulada **sem contato** → opções 2–4; **com contato
   saturado** → 5–6; **com contato aderente** → revisar o item 4 da §5.6. Base: cone
   (`computation/index.rst:994-997`), estimativa de deslizamento macio (§5.6.1), ciclo de 65–74 %
   (`docs/index.md:317-322`).
1. **Raio de colisão 9 → 6.5 mm** (e reposicionar o eixo para manter o assento e a cobertura ≤ 20 %,
   critério do `02`). Fonte: VENDOR ([RC25] Tab.1, [RC26] Tab.1) + a malha visual do próprio CAD. Sonda:
   padrão + profundidade de cobertura. Muda a altura do ponto de contato e a velocidade superficial
   (2.04 m/s a 314 rad/s).
2. **Contato normal com o material da v2025 (PU 65A).** `solref` positivo com timeconst 7–9 ms e
   dampratio 1 (ou 14–17 ms / 0.5), mantendo `solimp` d₀ = 0.95; variante com `margin` ≈ 0.1–0.2 mm
   (escala do δ de Hertz) para o contato sobreviver a pequenas separações; torcional 0.002–0.004 m (2a de
   Hertz) em vez de 0.02. Fonte: [RC26] p.3 + Gent/Qi + Hertz (§5.5) — DELIBERATE APPROXIMATION derivada
   de material. Sonda: padrão + ciclo de contato + desvio-padrão da normal (opção 0). Expectativa
   (**Inferência**): pouco efeito direto na lateral (a normal continua ⊥ ao eixo), efeito via quique.
3. **Amortecimento do braço do dribbler.** Hoje `dribbler_pivot` tem rigidez 1, amortecimento 0.0233,
   faixa 0–0.14 rad (`models/robot_kinematic.xml:329`). Varrer o amortecimento ×3/×10 e a rigidez. Fonte:
   TIGERs 2022 resolveu perda lateral trocando o amortecimento ([TIG22] p.1–2); ZJUNlict 2019 (amortecedor
   bidirecional, [ZJU19] p.3–4); KIKS 2023 (dampers, [KIKS23] p.5–6) — MEASURED (qualitativo). Sonda: padrão
   + ciclo de contato.
4. **Terceiro ponto de contato.** Hoje a bola só toca o rolete (e o piso). Testar geometria em que a bola
   assente também na placa do kicker/chip em regime, dentro dos 20 %. Fonte: ZJUNlict "3-touch-point"
   (bola apoiada no carpete, no rolete e na pá do chip, [ZJU19] p.3–4) e TIGERs 2024 (pá do chip como
   batente final, [TIG24] p.4) — MEASURED. Sonda: padrão + cobertura.
5. **Rolete com torque limitado e rotação baixa.** Substituir o regime atual — atuador de velocidade bem
   abaixo do `forcerange` de 0.05 N·m, usando ~0.012 N·m — por um rolete que opere saturado: kv alto e
   `forcerange` = F_lim × r, com F_lim = 5 N da TIGERs v2020 (0.0325 N·m a 6.5 mm), alvo de superfície
   3 m/s; variante de alvo baixo, para a bola quase parar ("firm grip", [TIG24] p.4). Fonte:
   `botParamsDatabase.json:79-84`, [TIG24] §2.1–2.2, ZJUNlict torque 1.63 → 2.5 m/s ([ZJU19RCAR] Tab.2) —
   VENDOR/MEASURED. Sonda: padrão + rotação medida do rolete. Expectativa (**Inferência**): spin menor
   reduz o item 4 da §5.6 e deixa o cone menos ocupado no sentido do giro.
6. **`adhesion` no `pair` bola↔rolete** (0.1–0.5 N). Orçamento μ(f_N + adhesion) mesmo nas fases de normal
   quase nula. Fonte: `XMLreference.rst:2801-2826` ("tacky rubber") — DELIBERATE APPROXIMATION (a
   aderência de silicone/TPU é real, mas nenhum valor foi medido). Sonda: padrão + um teste de roubo (outro
   robô tem de conseguir tirar a bola, §3.2.4) + z da bola (não pode levantar).
7. **Luva deformável (`flexcomp`) presa ao corpo `dribbler`**, com `young` ≈ 4–8 MPa e `poisson` ≈ 0.45–0.49
   para PU 65A, só como protótipo descartável. Fonte: `modeling.rst:1337-1428`, `XMLreference.rst:3614-3642`,
   `:4426-4450`; módulo da §5.5. Sonda: padrão + passos/s (`docs/index.md:267` dá a referência de custo).
   Risco: comportamento de nós presos a um pai girando a ~300 rad/s, não verificado.
8. **Último recurso, como os outros simuladores: restrição de posse.** `equality/connect` bola↔`dribbler`
   ligado/desligado em tempo de execução por `mjData.eq_active` (`XMLreference.rst:4733-4738`), análogo
   ao `btPoint2PointConstraint` do ER-Force (rotação livre, o que respeita "not removing all of its degrees
   of freedom") ou à hinge do grSim/rSim, com regra de soltura (força limite, ou um critério à
   `DribblingVerifier`, cujos parâmetros não são públicos). Fonte: §4 — DELIBERATE APPROXIMATION. Sonda:
   120/120 sai trivialmente, então a sonda útil é o teste de roubo e a fuga em rampa.

**Não perseguir:** `friction` anisotrópico em `pair` (o eixo tangente segue y do mundo, §5.4);
`impratio`/`noslip`/μ mais alto (já testados; a documentação diz que não agem sobre força fora do cone,
§5.2); ampulheta/V (piora por v = rω, [RTT20] p.10).

**Alvo físico e o que pedir à equipe.** Não há um alvo único (§3.1). O mais barato que fecharia a pergunta
é a equipe repetir no robô v2025 o protocolo da ZJUNlict: rolete em regime, rampa de guinada de
20 °/s² (0.349 rad/s²), ω no instante da fuga, 10 repetições ([ZJU19] p.4). A mesma rampa na simulação
substituiria a contagem de 120 passos por uma ω de fuga diretamente comparável a 402/622 °/s
(ZJUNlict), ~57 °/s (KIKS, não dividida) e 372 °/s (RobôCIn 2021, ±6.5 rad/s, sem critério de perda).

---

## Fontes

TDP/ETDP (arquivo da liga; páginas do PDF):
- [RC20]: RobôCIn TDP 2020 — https://ssl.robocup.org/wp-content/uploads/2020/03/2020_TDP_RoboCIn.pdf
- [RC22]: RobôCIn (E)TDP 2022, §5, §7.2 — https://ssl.robocup.org/wp-content/uploads/2022/04/2022_TDP_RoboCIn.pdf
- [RC25]: RobôCIn ETDP 2025, Tab.1 — https://ssl.robocup.org/wp-content/uploads/2025/04/2025_ETDP_RoboCIn.pdf
- [RC26]: RobôCIn ETDP 2026, Tab.1, §2.2 — https://ssl.robocup.org/wp-content/uploads/2026/04/2026_ETDP_RoboCin.pdf
- [TIG20]: TIGERs Mannheim ETDP 2020, §2.3 — https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_TIGERS.pdf
- [TIG22]: TIGERs Mannheim ETDP 2022, §1.1 — https://ssl.robocup.org/wp-content/uploads/2022/04/2022_ETDP_TIGERs-Mannheim.pdf
- [TIG24]: TIGERs Mannheim ETDP 2024, §1, §2.1–2.2 — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf
- [ZJU19]: ZJUNlict ETDP 2019, §2 — https://ssl.robocup.org/wp-content/uploads/2019/03/2019_ETDP_ZJUNlict.pdf (também arXiv:1905.09157)
- [ZJU25]: ZJUNlict ETDP 2025, §2.1 — https://ssl.robocup.org/wp-content/uploads/2025/04/2025_ETDP_ZJUNlict.pdf
- ZJUNlict ETDP 2026, §2.1 "Active Dribbling" (o motor pressiona a bola contra o carpete: f = μ(F_N sin α + mg) − F_N cos α) — https://ssl.robocup.org/wp-content/uploads/2026/03/2026_ETDP_ZJUNlict.pdf
- [KIKS17]: KIKS TDP 2017 — https://ssl.robocup.org/wp-content/uploads/2019/01/2017_TDP_KIKS.pdf
- [KIKS22]: KIKS ETDP 2022, §2.1 — https://ssl.robocup.org/wp-content/uploads/2022/04/2022_ETDP_KIKS.pdf
- [KIKS23]: KIKS ETDP 2023, §2.1–2.2 — https://ssl.robocup.org/wp-content/uploads/2023/02/2023_ETDP_KIKS.pdf
- [RD13]: RoboDragons ETDP 2013 — https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_RoboDragons.pdf
- [RD17]: RoboDragons ETDP 2017 — https://ssl.robocup.org/wp-content/uploads/2019/01/2017_ETDP_RoboDragons.pdf
- [RD19]: RoboDragons ETDP 2019 — https://ssl.robocup.org/wp-content/uploads/2019/03/2019_ETDP_RoboDragons.pdf
- [RD20]: RoboDragons ETDP 2020, §2 — https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_RoboDragons.pdf
- [RTT20]: RoboTeam Twente ETDP 2020, §4.4 — https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_RTT.pdf
- [ERF14]: ER-Force TDP 2014, §2.2 — https://ssl.robocup.org/wp-content/uploads/2019/01/2014_TDP_ER-Force.pdf
- [RF13]: RoboFEI TDP 2013 — https://ssl.robocup.org/wp-content/uploads/2019/01/2013_TDP_RoboFEI.pdf
- [RF18]: RoboFEI TDP 2018 — https://ssl.robocup.org/wp-content/uploads/2019/01/2018_TDP_RoboFEI.pdf
- [RF23]: RoboFEI TDP 2023 — https://ssl.robocup.org/wp-content/uploads/2023/02/2023_TDP_RoboFEI.pdf
- [OPA17]: OP-AmP TDP 2017, §2.3 — https://ssl.robocup.org/wp-content/uploads/2019/01/2017_TDP_Op-Amp.pdf
- [PAR13]: Parsian ETDP 2013 — https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_Parsian.pdf
- [MRL13]: MRL ETDP 2013 — https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_MRL.pdf
- [RFC14]: RFC Cambridge TDP 2014 — https://ssl.robocup.org/wp-content/uploads/2019/01/2014_TDP_RFC_Cambridge.pdf
- [KGP20]: KgpKubs TDP 2020 — https://ssl.robocup.org/wp-content/uploads/2020/03/2020_TDP_KgpKubs.pdf
- [ITA19]: ITAndroids TDP 2019 — https://ssl.robocup.org/wp-content/uploads/2019/03/2019_TDP_ITAndroids_Small_Size.pdf
- [RJ19]: RoboJackets TDP 2019, §1.4 — https://ssl.robocup.org/wp-content/uploads/2019/03/2019_TDP_RoboJackets.pdf
- [UBC23]: UBC Thunderbots TDP 2023 — https://ssl.robocup.org/wp-content/uploads/2023/02/2023_TDP_UBC_Thunderbots.pdf
- [UBC25]: UBC Thunderbots TDP 2025 — https://ssl.robocup.org/wp-content/uploads/2025/04/2025_TDP_UBC-Thunderbots.pdf
- [PN25]: Project Neon TDP 2025 — https://ssl.robocup.org/wp-content/uploads/2025/04/2025_TDP_Project-Neon.pdf
- [TR26]: TurtleRabbit TDP 2026, §2.2.2 — https://ssl.robocup.org/wp-content/uploads/2026/04/2026_TDP_TurtleRabbit.pdf
- [LUH26]: luhbots ETDP 2026, §3, §5 — https://ssl.robocup.org/wp-content/uploads/2026/04/2026_ETDP_luhbots.pdf
- [CMD15]: CMDragons ETDP 2015, §4 — https://ssl.robocup.org/wp-content/uploads/2019/01/2015_ETDP_CMDragons.pdf

Artigos revisados por pares:
- [ZJU19RCAR]: Huang, Wang, Chen, Li, Chen & Xiong, "Mechatronic Design of a Dribbling System for RoboCup Small Size Robot", IEEE RCAR 2019 — https://arxiv.org/abs/1905.09934 (§IV–V, Tab.1–2)
- [LARS21]: Araújo, Silva, Monteiro, Sousa, Silva, Santana & Barros, "Mechanical and Dynamic Analysis for Design and Development of a RoboCup SSL Dribbling Mechanism", SBR/LARS 2021, pp. 312–317 — https://sol.sbc.org.br/index.php/sbrlars/article/view/18358 (só o resumo foi lido)

Regras:
- SSL rules `2ff4be0`: `chapters/robots.adoc:44-50` (§3.2.4), `chapters/offenses.adoc:92-93`, `:120-123` (§8.4.1), `chapters/appendix.adoc:103` — https://github.com/RoboCup-SSL/ssl-rules/tree/2ff4be0c4bdcc203555f7f3baedb2bc22d297e41/chapters
- [HWC21]: SSL Hardware Challenge 2021, *Challenge 3: Dribbling* — https://robocup-ssl.github.io/ssl-hardware-challenge-rules/rules.html#_challenge_3_dribbling
- [DC22]: RoboCup 2022 SSL Dribbling Challenge Rules — https://robocup-ssl.github.io/technical-challenge-rules/2022-ssl-dribble-rules.html

Código de simuladores:
- grSim `fe2bd2915a46f9f11ea6cb48dc426b8047952073` — https://github.com/RoboCup-SSL/grSim/blob/fe2bd2915a46f9f11ea6cb48dc426b8047952073/src/robot.cpp ; commit `19f6355da700de671242436d558ad1abd3ad430e` — https://github.com/RoboCup-SSL/grSim/commit/19f6355da700de671242436d558ad1abd3ad430e
- rSim `b413932b7b4bbc0fc399f6a979e4cfcc4f56264f` — https://github.com/robocin/rSim/blob/b413932b7b4bbc0fc399f6a979e4cfcc4f56264f/src/robosim/sslrobot.cpp ; rSoccer `3adf7c3`
- ER-Force `framework` `38563d11879b4b8dd4ee8009b1d49cc88fae8abf` — https://github.com/robotics-erlangen/framework/blob/38563d11879b4b8dd4ee8009b1d49cc88fae8abf/src/amun/simulator/simrobot.cpp
- Bullet3 `d3afe76132fc76b0f9b2359cbaef47a3766e7fbb` — https://github.com/bulletphysics/bullet3/blob/d3afe76132fc76b0f9b2359cbaef47a3766e7fbb/src/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp
- TIGERs `Sumatra` `496376d77d9e208736257f4410ad1dd5ca9cf72d` — https://github.com/TIGERs-Mannheim/Sumatra/tree/496376d77d9e208736257f4410ad1dd5ca9cf72d

MuJoCo 3.11.0 (`b85fdca54f0e0038b804af146a0b4e94199e00d0`):
- `doc/computation/index.rst`, `doc/modeling.rst`, `doc/XMLreference.rst`, `doc/overview.rst`, `doc/changelog.rst` — https://github.com/google-deepmind/mujoco/tree/3.11.0/doc (renderizado em https://mujoco.readthedocs.io/)
- `src/engine/engine_collision_primitive.c`, `engine_collision_driver.c`, `engine_util_spatial.c` — https://github.com/google-deepmind/mujoco/tree/3.11.0/src/engine

Dureza→módulo (secundária; originais não consultados):
- [WIKI]: "Shore durometer", Wikipedia (fórmulas de Gent 1958 e Qi, Joyce & Boyce 2003) — https://en.wikipedia.org/wiki/Shore_durometer

Este repositório: `uv.lock:386,413`; `models/scene.xml:4,21-22`; `models/robot_kinematic.xml:329,336-337,370`;
`models/robot_rollers.xml:249-250,807`; `models/assets/visual/Dribblerschnecke.stl`; `docs/index.md:308-322,388-412,501-502`;
ticket `22` (sondas descartadas; números no §Diagnóstico).

[RC20]: https://ssl.robocup.org/wp-content/uploads/2020/03/2020_TDP_RoboCIn.pdf
[RC22]: https://ssl.robocup.org/wp-content/uploads/2022/04/2022_TDP_RoboCIn.pdf
[RC25]: https://ssl.robocup.org/wp-content/uploads/2025/04/2025_ETDP_RoboCIn.pdf
[RC26]: https://ssl.robocup.org/wp-content/uploads/2026/04/2026_ETDP_RoboCin.pdf
[TIG20]: https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_TIGERS.pdf
[TIG22]: https://ssl.robocup.org/wp-content/uploads/2022/04/2022_ETDP_TIGERs-Mannheim.pdf
[TIG24]: https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf
[ZJU19]: https://ssl.robocup.org/wp-content/uploads/2019/03/2019_ETDP_ZJUNlict.pdf
[ZJU25]: https://ssl.robocup.org/wp-content/uploads/2025/04/2025_ETDP_ZJUNlict.pdf
[ZJU19RCAR]: https://arxiv.org/abs/1905.09934
[LARS21]: https://sol.sbc.org.br/index.php/sbrlars/article/view/18358
[KIKS17]: https://ssl.robocup.org/wp-content/uploads/2019/01/2017_TDP_KIKS.pdf
[KIKS22]: https://ssl.robocup.org/wp-content/uploads/2022/04/2022_ETDP_KIKS.pdf
[KIKS23]: https://ssl.robocup.org/wp-content/uploads/2023/02/2023_ETDP_KIKS.pdf
[RD13]: https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_RoboDragons.pdf
[RD17]: https://ssl.robocup.org/wp-content/uploads/2019/01/2017_ETDP_RoboDragons.pdf
[RD19]: https://ssl.robocup.org/wp-content/uploads/2019/03/2019_ETDP_RoboDragons.pdf
[RD20]: https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_RoboDragons.pdf
[RTT20]: https://ssl.robocup.org/wp-content/uploads/2020/03/2020_ETDP_RTT.pdf
[ERF14]: https://ssl.robocup.org/wp-content/uploads/2019/01/2014_TDP_ER-Force.pdf
[RF13]: https://ssl.robocup.org/wp-content/uploads/2019/01/2013_TDP_RoboFEI.pdf
[RF18]: https://ssl.robocup.org/wp-content/uploads/2019/01/2018_TDP_RoboFEI.pdf
[RF23]: https://ssl.robocup.org/wp-content/uploads/2023/02/2023_TDP_RoboFEI.pdf
[OPA17]: https://ssl.robocup.org/wp-content/uploads/2019/01/2017_TDP_Op-Amp.pdf
[PAR13]: https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_Parsian.pdf
[MRL13]: https://ssl.robocup.org/wp-content/uploads/2019/01/2013_ETDP_MRL.pdf
[RFC14]: https://ssl.robocup.org/wp-content/uploads/2019/01/2014_TDP_RFC_Cambridge.pdf
[KGP20]: https://ssl.robocup.org/wp-content/uploads/2020/03/2020_TDP_KgpKubs.pdf
[ITA19]: https://ssl.robocup.org/wp-content/uploads/2019/03/2019_TDP_ITAndroids_Small_Size.pdf
[RJ19]: https://ssl.robocup.org/wp-content/uploads/2019/03/2019_TDP_RoboJackets.pdf
[UBC23]: https://ssl.robocup.org/wp-content/uploads/2023/02/2023_TDP_UBC_Thunderbots.pdf
[UBC25]: https://ssl.robocup.org/wp-content/uploads/2025/04/2025_TDP_UBC-Thunderbots.pdf
[PN25]: https://ssl.robocup.org/wp-content/uploads/2025/04/2025_TDP_Project-Neon.pdf
[TR26]: https://ssl.robocup.org/wp-content/uploads/2026/04/2026_TDP_TurtleRabbit.pdf
[LUH26]: https://ssl.robocup.org/wp-content/uploads/2026/04/2026_ETDP_luhbots.pdf
[CMD15]: https://ssl.robocup.org/wp-content/uploads/2019/01/2015_ETDP_CMDragons.pdf
[HWC21]: https://robocup-ssl.github.io/ssl-hardware-challenge-rules/rules.html#_challenge_3_dribbling
[DC22]: https://robocup-ssl.github.io/technical-challenge-rules/2022-ssl-dribble-rules.html
[WIKI]: https://en.wikipedia.org/wiki/Shore_durometer
