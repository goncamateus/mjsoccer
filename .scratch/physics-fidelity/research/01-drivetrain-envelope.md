# 01 — Envelope do drivetrain: o que os motores deste robô realmente conseguem fazer?

Ticket: `.scratch/physics-fidelity/issues/01-drivetrain-envelope.md`. Resolvido em 2026-09-08.
Dados do fabricante das páginas de produto de maxongroup.com/maxongroup.us (consultadas em
2026-09-08); regras do código-fonte do livro de regras da RoboCup SSL na `master` (consultado em
2026-09-08); todo o resto citado como `path:LINE` neste repositório ou em `/tmp/rsim-upstream`
(`b413932`).

Grandezas medidas tomadas como dadas de `docs/research/rsoccer-vs-mjsoccer.md` §4/§6: massa total
`m = 2.69207` kg, `r_eff = 30.089` mm, `R = 0.0796` m, rodas a 60/135/225/300°, CoM 53.99 mm acima do
carpete com deslocamento lateral de 3.459 mm, limiar estático de tombamento frontal 7.86 m/s²,
`ω_tip = 47.7` rad/s, orçamento de atrito `μ·m·g = 26.4` N com `μ = 1.0`. Todas foram reverificadas
aritmeticamente aqui e todas se reproduzem (7.8585 m/s², 47.665 rad/s, 26.409 N).

## Valores recomendados

| declaração | hoje (`models/robot_rollers.xml:827-830`) | recomendado | base |
|---|---|---|---|
| `forcerange` (N·m por roda) | `-3 3` | **`-0.078 0.078`** | ponto de operação de projeto do EC 45 flat 50 W, acionamento direto (§2, §3) |
| `ctrlrange` (rad/s por roda) | `-200 200` | **`-95 95`** | `ω_tip` com margem de projeto declarada de 25 % (§3) |
| `kv` | `0.103696` | inalterado | satura acima de 0.75 rad/s de erro, ou seja, limitado por torque em todo transitório (§3) |

| grandeza resultante | hoje | recomendado | limiar | veredito |
|---|---|---|---|---|
| `a_max` frontal (+x) | 9.81 (cortado pelo atrito) | **2.98 m/s²** | 7.86 m/s² tombamento estático | passa, 62 % abaixo |
| `a_max` lateral (+y) | 9.81 (cortado pelo atrito) | 2.30 m/s² | — | — |
| força no chão exigida, frontal | 1255 N contra orçamento de 26.4 N | 8.16 N = 31 % do orçamento | 26.4 N | limitado por torque, não por atrito |
| velocidade de superfície da roda | 6.02 m/s | **2.86 m/s** | — | — |
| `v_max` frontal / lateral | 6.95 / 8.51 m/s | **3.30 / 4.04 m/s** | — | dentro da faixa real da SSL (§5) |
| `ω_max` (comando uniforme nas rodas) | 75.6 rad/s (59.4 medido) | **35.9 rad/s** | 47.7 rad/s tombamento dinâmico | passa, 25 % abaixo |

A mudança principal é qualitativa: `forcerange="-3 3"` faz da roda um servo de torque infinito
limitado por atrito, `forcerange="-0.078 0.078"` faz dela um motor.

## 1. Os valores do datasheet do Maxon EC 45 flat 50 W

A maxon publica **cinco** números de catálogo sob o nome "EC 45 flat … 50 W", em três enrolamentos
distintos e três diâmetros de carcaça. Todos os valores vêm das páginas de produto da própria maxon.

| peça | carcaça | U_nom | velocidade em vazio | velocidade nominal | torque nominal (contínuo máx.) | corrente nominal (contínua máx.) | torque de stall | corrente de stall / partida | constante de torque | fonte |
|---|---|---|---|---|---|---|---|---|---|---|
| 251601 | Ø42.8 mm | 24 V | 6710 rpm | 5240 rpm | 83.4 mN·m | 2.33 A | 780 mN·m | 23.3 A | 33.5 mN·m/A | [maxon 251601](https://www.maxongroup.com/maxon/view/product/251601) |
| 651607 | Ø43.5 mm | 24 V | 6250 rpm | 5170 rpm | 91.1 mN·m | 2.52 A | 918 mN·m | 25.5 A | 36 mN·m/A | [maxon 651607](https://www.maxongroup.com/maxon/view/product/651607) |
| 339286 | Ø42.8 mm | 24 V | 4730 rpm | 3480 rpm | 69.6 mN·m | 1.41 A | 402 mN·m | 8.47 A | 47.5 mN·m/A | [maxon 339286](https://www.maxongroup.com/maxon/view/product/339286) |
| 400107 | Ø45 mm, com cabo | 24 V | 4730 rpm | 3480 rpm | 69.6 mN·m | 1.41 A | 402 mN·m | 8.47 A | 47.5 mN·m/A | [maxon 400107](https://www.maxongroup.com/maxon/view/product/400107) |
| 339287 | Ø42.8 mm | 36 V | 3360 rpm | 2360 rpm | 90.5 mN·m | 0.828 A | 484 mN·m | 4.81 A | 101 mN·m/A | [maxon 339287](https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/339287) |

Características de apoio para o 251601, a variante usada abaixo: corrente em vazio 185 mA, resistência
de terminal 1.03 Ω, constante de velocidade 285 rpm/V, gradiente velocidade/torque 8.77 rpm/mN·m,
eficiência máx. 83 %, inércia do rotor 135 g·cm²
([maxon 251601](https://www.maxongroup.com/maxon/view/product/251601)).

**O limite de corrente que "50 W" implica.** "50 W" é a classe de potência *de saída* nominal, não um
limite separado; o limite é a corrente nominal. Para o 251601 o limite elétrico contínuo é 2.33 A, o
que, pela constante de torque, é

```
τ_cont = I_nom · k_M = 2.33 A · 33.5 mN·m/A = 78.1 mN·m
```

e a saída mecânica no ponto nominal é
`0.0834 N·m · 5240 rpm · 2π/60 = 0.0834 · 548.7 = 45.8 W`; para o 651607,
`0.0911 · 541.4 = 49.3 W`. Então o rótulo da classe confere, e o "torque nominal 83.4 mN·m" do próprio
datasheet fica cerca de 7 % acima de `I_nom · k_M`. **78.1 mN·m é o valor sobre o qual construir**,
porque é o que o limite de corrente do driver de fato impõe.

Dois fatos a levar adiante: `I_nom · k_M = 78.1` mN·m é também **exatamente 10.0 %** dos 780 mN·m de
torque de stall, e 10 % do stall é onde o próprio artigo de hardware da equipe diz que esses motores
são operados (§2). Três caminhos independentes chegam a ~78 mN·m.

**Suposição (rotulada).** Qual das cinco variantes este robô usa não está registrado em lugar nenhum
do repositório. Assume-se o 251601. A faixa de torque entre as cinco é 40.2–91.8 mN·m a 10 % do stall
e 69.6–91.1 mN·m no nominal, o que move `a_max` entre 1.54 e 3.51 m/s² (§3) — um fator de 2.3. A
escolha do enrolamento é o maior input isolado ainda não resolvido deste ticket.

## 2. Pela 18:60 até a roda, e o que são os 0.070 N·m do rSim

18 dentes no pinhão do motor acionando 60 na roda é uma redução de `60/18 = 3.3333:1`, então um
estágio ideal (100 % de eficiência, sem folga) dá

```
τ_roda = τ_motor · 3.3333          ω_roda = ω_motor / 3.3333
```

Os dois sentidos aplicados ao 251601:

| grandeza no motor | valor | na roda pela 18:60 |
|---|---|---|
| `I_nom · k_M` (contínuo) | 78.1 mN·m | 260.3 mN·m |
| torque nominal | 83.4 mN·m | 278.0 mN·m |
| torque de stall | 780 mN·m | 2600 mN·m |
| velocidade nominal | 5240 rpm | 1572 rpm = 164.6 rad/s |
| velocidade em vazio | 6710 rpm | 2013 rpm = 210.8 rad/s |

### 2.1 O `wheelMotorMaxRPM = 1557` do rSim é a velocidade nominal do motor

`1557 rpm · 3.3333 = 5190 rpm` no motor. Isso cai dentro da faixa de velocidade nominal dos dois
enrolamentos de alta velocidade de 24 V e 50 W — 5170 rpm (651607) e 5240 rpm (251601) — com
diferença menor que 1 %, e é 77 % dos 6710 rpm de velocidade em vazio do 251601. Então
`wheelMotorMaxRPM` (`/tmp/rsim-upstream/src/robosim/sslconfig.h:160`) é a **velocidade nominal (com
carga nominal)** do motor reduzida pela engrenagem, não a velocidade em vazio. O 1 % residual é
consistente com uma edição diferente do catálogo; não é resolvido aqui e não importa.

### 2.2 O `wheelMotorMaxTorque = 0.070` do rSim é uma redução deliberada de ~3.7×

Voltando pelo mesmo estágio:

```
τ_motor = 0.070 N·m / 3.3333 = 0.0210 N·m = 21.0 mN·m
        = 26.9 % de I_nom·k_M (78.1)   = 25.2 % do torque nominal (83.4)   = 2.7 % do stall (780)
corrente implícita = 21.0 / 33.5 = 0.627 A, ou seja, 27 % do limite contínuo de 2.33 A
```

Não é nem nominal, nem stall, nem um valor térmico. É um corte deliberado para cerca de um quarto da
capacidade contínua, que é exatamente o que o comentário "reduced to 3m/s2 accel"
(`sslconfig.h:159`) afirma. Aquilo *a partir de que* foi reduzido são 260 mN·m na roda — uma redução
de 3.72×.

### 2.3 O comentário "3 m/s²" confere? Sim — como valor do eixo fraco

As quatro rodas ficam a 60/135/225/300°, então suas direções de acionamento no chão são tangenciais e
nenhuma direção recebe as quatro com projeção unitária. Com `α_i` os ângulos de montagem, a força no
chão alcançável a partir de uma força por roda `F` é `Σ|sin α_i| = 3.14626` ao longo de +x e
`Σ|cos α_i| = 2.41421` ao longo de +y (ambos com momento líquido de yaw zero: o padrão de sinais que
maximiza a força também cancela `Σ f_i`). Com os números do próprio rSim (`m = 2.42` kg,
`wheelRadius = 0.02475`, `sslconfig.h:153,155-157`, `sslrobot.cpp:50-56,63-68`):

```
F_roda = 0.070 / 0.02475            = 2.828 N
frontal:  3.14626 · 2.828 / 2.42    = 3.68 m/s²   (3.63 incluindo a inércia de aceleração da roda)
lateral:  2.41421 · 2.828 / 2.42    = 2.82 m/s²   (2.79 incluindo a inércia de aceleração da roda)
"quatro rodas" ingênuo:  4 · 2.828 / 2.42 = 4.67 m/s²
```

Então: o "3 m/s²" do comentário está certo com 6 % de erro **na direção fraca (lateral)** e subestima o
valor frontal em 19 %. Os 4.67 m/s² citados no `W3` de `docs/research/rsoccer-vs-mjsoccer.md`
**não são alcançáveis em nenhuma direção** — supõem um fator de projeção de 4 onde a geometria o
limita a 3.146 — e superestimam o teto frontal real do rSim em 27 %. A conclusão do `W3` sobrevive
(o rSim é limitado por torque: 8.90 N de força frontal contra o seu próprio orçamento de atrito
tangencial `0.8 · 2.42 · 9.81 = 19.0` N), mas o seu número deve ser corrigido para 3.68 m/s² frontal /
2.82 m/s² lateral.

### 2.4 A suposição 18:60 não vale para este CAD

Duas evidências dizem que este robô **não tem estágio de redução na roda**.

1. **Os nomes das malhas do próprio modelo.** As malhas visual e de colisão de cada cubo de roda são
   `Direct_Transmission_new_Simplify_2` (`models/robot_rollers.xml:300-301`, e a mesma malha em
   `wheel2`/`wheel3`/`wheel4`). As únicas malhas de engrenagem no arquivo — `gear_z18_21101800_di3`
   (uma engrenagem de 18 dentes) e `Middle_Gear_Simplify_1` — estão presas ao **chassi** em
   `x = 0.0634, y = -0.04325`, ao lado do dribbler, e se chamam
   `r_vis_gearMotorDribbler` / `r_vis_gearMiddleDribbler` (`models/robot_rollers.xml:164-165`,
   `250-257`). O pinhão de 18 dentes neste CAD aciona o dribbler, não uma roda.
2. **O artigo de hardware da própria equipe.** O ETDP 2024 da RobôCIn afirma que as rodas da v2022
   tinham 45 mm de diâmetro com um "embedded gearing system (3.3:1)", que os robôs v2024 passaram para
   **acionamento direto sem engrenagens**, e que para isso a roda cresceu para "a total diameter of
   62 mm" ([2024 ETDP RoboCIn](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf),
   §1.2 *Direct Drive*). O `r_eff = 30.089` mm medido neste modelo é uma roda de 60.18 mm — a geração
   de acionamento direto, não a de 45 mm com redução. A mesma seção declara o ponto de operação de
   projeto: "The 50W Maxon motors through direct drive work at a maximum operating point around 10
   percent of the stall torque, near maximum efficiency." A malha do dribbler no CAD se chama
   `Dribbler_Robocup2025v3` (`models/robot_rollers.xml:159`), ou seja, um robô da geração 2025.
3. **A aritmética concorda.** Se a 18:60 *estivesse* presente nesta roda, o torque contínuo na roda
   seria 260 mN·m = 8.65 N por roda = **27.2 N de força frontal no chão contra um orçamento de atrito
   de 26.4 N**. O robô ainda seria limitado por atrito em 9.81 m/s², ainda acima do seu próprio
   limiar de tombamento de 7.86 m/s², e o teto de torque ainda seria comportamentalmente inerte. Um
   estágio de 3.33:1 e uma roda de 30 mm não são uma combinação que alguém construiria.

Isso substitui a restrição permanente 2 de `.scratch/physics-fidelity/map.md` ("Motor envelope
default: Maxon EC45 flat 50 W, 18:60 gearing. No real drivetrain spec was available"): uma
especificação de drivetrain **está** disponível, é o ETDP da própria equipe, e diz acionamento direto.
Os 0.070 N·m e 1557 rpm do rSim descrevem a geração *anterior, com redução*, do robô, e é por isso que
não se transferem como números.

## 3. `a_max`, `v_max`, `ω_max` e o par `forcerange` / `ctrlrange`

Note primeiro que o `forcerange` de um atuador `velocity` do MuJoCo ligado à junta do cubo é um
torque **de roda**. A questão da engrenagem da §2 só decide, portanto, *qual* torque de roda o motor
consegue sustentar; ela nunca aparece na declaração em si.

### 3.1 Torque → força no chão → aceleração

Acionamento direto, então `τ_roda = τ_motor = 78 mN·m` (§1: `I_nom · k_M = 78.1` mN·m = 10.0 % do
stall, o ponto de operação que o ETDP nomeia). Arredondando para baixo para 0.078 N·m:

```
F_roda = τ / r_eff         = 0.078 / 0.030089            = 2.5923 N
F_x,total = 3.14626 · F    = 8.1561 N     (Σ|sin α_i|, yaw líquido zero)
F_y,total = 2.41421 · F    = 6.2584 N     (Σ|cos α_i|, yaw líquido zero)
a_x = F_x / m              = 8.1561 / 2.69207            = 3.029 m/s²
a_y = F_y / m              = 6.2584 / 2.69207            = 2.325 m/s²
```

Incluindo a aceleração da própria roda (hub `Iyy = 5.93133e-6` mais `armature="1e-5"`,
`models/robot_rollers.xml:298-299`, então `I = 1.5931e-5` kg·m²; a massa efetiva adicionada é
`I·Σsin²α_i/r_eff² = 1.5931e-5 · 2.5 / 0.030089² = 0.0440` kg, +1.63 %):

```
a_x = 8.1561 / 2.73606 = 2.981 m/s²        a_y = 6.2584 / 2.71847 = 2.302 m/s²
```

Verificações de sanidade da recomendação: 8.16 N de força frontal no chão são **31 % do orçamento de
atrito de 26.4 N**, então a roda é limitada por torque e o escorregamento é um evento raro em vez do
regime de operação; e 0.078 N·m é 100 % do limite de corrente contínua e 10 % do stall, então nada
térmico está sendo violado. É um teto rígido, não um valor de pico com folga — acionamentos brushless
reais podem puxar várias vezes `I_nom` por pouco tempo, o que este modelo não representa (registrado
como aproximação deliberada).

### 3.2 Velocidade → velocidade de superfície → velocidade do corpo e yaw

Para um comando uniforme `w` nas quatro rodas, a matriz de rodas medida
(`docs/research/rsoccer-vs-mjsoccer.md` §6) se reduz a `ω = w · r_eff / R` (sinal invertido, `W2`),
e para translação pura a roda dominante é aquela cuja direção de acionamento está mais próxima do
movimento: `w = v_x · sin 60° / r_eff` frontal, `w = v_y · cos 45° / r_eff` lateral. Então

```
velocidade de superfície = w · r_eff
v_x,max       = w · r_eff / 0.866025
v_y,max       = w · r_eff / 0.707107
ω_max         = w · r_eff / R = w · 0.030089 / 0.0796 = 0.37800 · w
```

Acionamento direto significa que o motor não impõe praticamente nenhum teto de velocidade nesta roda:
os 5240 rpm nominais do 251601 **são** a velocidade da roda, ou seja, 548.7 rad/s = 16.5 m/s de
velocidade de superfície. O teto de velocidade é, portanto, uma escolha de projeto, e a única âncora
física disponível é `ω_tip = 47.7` rad/s, que `ω = 0.378 w` atinge em `w = 126.2` rad/s:

| `ctrlrange` (rad/s) | `ω_max` (rad/s) | margem abaixo de 47.7 | superfície (m/s) | `v_x` (m/s) | `v_y` (m/s) | rpm do motor |
|---|---|---|---|---|---|---|
| 86 | 32.5 | 32 % | 2.59 | 2.99 | 3.66 | 821 |
| **95 (recomendado)** | **35.9** | **25 %** | **2.86** | **3.30** | **4.04** | **907** |
| 110 | 41.6 | 13 % | 3.31 | 3.82 | 4.68 | 1050 |
| 126.2 | 47.7 | 0 % | 3.80 | 4.39 | 5.37 | 1205 |
| 164.6 (nominal 18:60, hipotético) | 62.2 | −30 % | 4.95 | 5.72 | 7.00 | — |
| 200 (hoje) | 75.6 | −59 % | 6.02 | 6.95 | 8.51 | 1910 |

**Uma margem de 25 % abaixo de `ω_tip` é uma escolha de projeto declarada, não um número de
hardware.** É o único lugar deste ticket em que um número é escolhido em vez de derivado. A
justificativa é que a previsão de tombamento é um balanço estático de momentos e o início medido a
cerca em vez de coincidir com ela (42.1 rad/s de yaw já inclinam o chassi 6.0° com um comando de
120 rad/s, `docs/research/rsoccer-vs-mjsoccer.md` §4), então o envelope não deve ser posto no
limiar.

`±95` rad/s usa só 17 % da velocidade nominal do motor. A leitura honesta é que uma roda de 30 mm em
acionamento direto é rica em velocidade e pobre em torque: a aceleração é limitada pelo hardware, a
velocidade máxima não, e o yaw é limitado pelo chassi e não pelo drivetrain. Se a capacidade plena em
linha reta algum dia for desejada, a alternativa é deixar o `ctrlrange` no limite do motor e limitar
o **yaw** na camada de ação — que é o que os ambientes do rSoccer fazem (`max_w = 10` rad/s,
`docs/research/rsoccer-vs-mjsoccer.md` §7). Essa é uma decisão de espaço de ações, não de atuador, e
fica para os tickets que são donos do espaço de ações.

### 3.3 O par

```xml
<velocity name="wheelN_vel" joint="wheelN_spin" kv="0.103696"
          ctrlrange="-95 95" forcerange="-0.078 0.078" />
```

com `kv` inalterado: o servo satura acima de `0.078 / 0.103696 = 0.75` rad/s de erro de rastreamento,
então se comporta como um motor limitado por torque em todo transitório e só entra na região
proporcional a menos de 0.75 rad/s do alvo. (Hoje ele satura acima de 28.9 rad/s, e é por isso que
parece um servo ideal.) Se `kv` deveria ser rederivado a partir do gradiente velocidade/torque do
motor é uma pergunta separada e não é respondida aqui.

Tempo para atingir `v_x,max` com `a_x`: `3.30 / 2.98 = 1.11` s. Números do envelope completo:

| | frontal (+x) | lateral (+y) | yaw |
|---|---|---|---|
| `a_max` | 2.98 m/s² | 2.30 m/s² | — |
| `v_max` | 3.30 m/s | 4.04 m/s | 35.9 rad/s |

Pela restrição 4 do mapa, o tier 0 deve receber o mesmo envelope: `2.69207 · 2.98 = 8.02` N nos
atuadores de slide x/y (contra os `±19.1261` de hoje, `models/robot_kinematic.xml:389-390`) e um
`ctrlrange` de yaw de 35.9 rad/s (contra os 30 de hoje). Derivar a faixa de *força* de yaw do tier 0
exige o `Izz` do robô e está fora do escopo deste ticket.

## 4. Passa pelos dois limiares?

| limiar | valor | envelope recomendado | veredito |
|---|---|---|---|
| tombamento estático frontal | 7.86 m/s² | 2.98 m/s² | **passa — 4.88 m/s² de margem, 62 % abaixo** |
| tombamento dinâmico por yaw `ω_tip` | 47.7 rad/s | 35.9 rad/s | **passa — 11.8 rad/s de margem, 25 % abaixo** |
| orçamento de atrito `μ·m·g` | 26.4 N | 8.16 N exigidos | passa — 31 % do orçamento usado |

Os dois passam, e a razão em relação ao limiar de tombamento (0.379 frontal) é *mais* conservadora do
que a do próprio rSim (3.68 / 6.28 = 0.586, ou 2.82 / 6.28 = 0.449 no eixo fraco). O robô não consegue
mais tombar sozinho acelerando em linha reta, e não consegue mais comandar uma taxa de yaw que o
tombe. Ele também não consegue atingir 47.7 rad/s de yaw *transitoriamente* a partir de um comando
saturado, porque a taxa de yaw agora é limitada cinematicamente em vez de pelo torque disponível.

Duas ressalvas do lado do yaw. Primeiro, o `ω_max` acima é a previsão cinemática; o valor medido fica
um pouco abaixo (um comando de 120 rad/s prevê 45.4 rad/s e mede 42.1), então a margem real é um
pouco maior. Segundo, `ω_tip = 47.7` rad/s depende do deslocamento do CoM de 3.459 mm, que é um
artefato do CAD de fidelidade desconhecida; se esse deslocamento for corrigido ou se descobrir que é
uma assimetria que o robô real não tem, o `ω_tip` se move e a recomendação de 95 rad/s deve ser
revisitada. A recomendação de torque não depende disso.

## 5. As regras da SSL ou a prática da Divisão A limitam velocidade, aceleração ou potência do motor?

**O livro de regras não limita nenhuma das três.** O capítulo 3 *Robots* → *Hardware And Software
Constraints* cobre segurança, forma, cor, dispositivo de drible, padrão de visão, rádio e autonomia, e
**não** contém nenhum limite de velocidade, aceleração, velocidade de roda, potência do motor,
capacidade de bateria ou drivetrain de qualquer tipo
([chapters/robots.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc)).
A única restrição dimensional que toca o drivetrain indiretamente é *Shape*: "A robot must fit inside
a 0.18 meters wide and 0.15 meters high cylinder at any point in time" — que limita o diâmetro da roda
e, portanto, a relação torque/velocidade, e nada mais.

O que o livro de regras de fato limita:

| regra | limite | onde |
|---|---|---|
| Ball Speed (falta sem parada) | "A robot must not accelerate the ball faster than 6.5 meters per second in 3D space" | [chapters/offenses.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc) |
| Robot Stop Speed | "A robot must not move faster than 1.5 meters per second during stop", 2 s de tolerância, não se aplica durante posicionamento da bola | idem |
| Crashing | no contato entre robôs adversários, a componente da velocidade de aproximação ao longo da linha que os une deve ser menor que 1.5 m/s; ambos são culpados se as velocidades dos dois robôs diferem em menos de 0.3 m/s | idem |
| Stop command | "all robots have to slow down to less than 1.5 m/s" mais 0.5 m da bola | [chapters/refereecommands.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/refereecommands.adoc) |
| Safety | só qualitativo: "high power or high velocity uncontrolled robot operation" é listado como falha insegura pela qual um árbitro pode remover um robô | [chapters/offenses.adoc](https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/offenses.adoc) |

Então os valores de 1.5 m/s são limites de estado de jogo e de colisão, não um limite de velocidade
máxima; o jogo livre não tem teto, e não existe regra que o envelope recomendado de 3.30 m/s /
2.98 m/s² pudesse violar. O limite de 6.5 m/s da bola pertence ao ticket do chutador, não a este.

**A prática da Divisão A** não está codificada em lugar nenhum e nenhum valor isolado pode ser citado
como "a" prática. O que é citável, dos artigos publicados pelas próprias equipes: a RobôCIn afirma que
"most of the time, an SSL robot is accelerating, rarely moving at a constant speed", e que dimensiona
a roda para colocar o ponto de operação do motor perto da eficiência máxima em ~10 % do torque de
stall ([2024 ETDP RoboCIn](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf)
§1.2) — ou seja, o alvo de projeto é aceleração, não velocidade máxima, o que é consistente com um
envelope limitado por torque como o recomendado. O ETDP 2024 da TIGERs Mannheim nomeia o "Maxon EC-45
flat 70 W" como opção de acionamento e menciona "constraints, like maximum velocity and acceleration"
sendo repassadas ao seu planejador de trajetória, mas as publica como entradas do planejador e não
como valores de hardware
([2024 ETDP TIGERs Mannheim](https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf)).
Os 3.30 m/s recomendados estão na faixa certa em relação ao único número firme que a própria liga usa
para a velocidade dos robôs — o limite de parada de 1.5 m/s, que existe justamente porque os robôs
normalmente viajam bem mais rápido.

## Não resolvido, e suposições declaradas

1. **Qual enrolamento do EC 45 flat 50 W.** Assumido o 251601 (24 V, `k_M = 33.5` mN·m/A, 780 mN·m de
   stall). Não registrado no repositório. Entre as cinco variantes do catálogo o torque de projeto
   varia de 40.2 a 91.8 mN·m a 10 % do stall, isto é, `a_x` entre 1.54 e 3.51 m/s². Se o acionamento
   for o enrolamento de baixa velocidade de 24 V (339286/400107), a recomendação cai para
   `forcerange="-0.040 0.040"` e 1.54 m/s²; se for o 651607, `-0.092 0.092` e 3.51 m/s². **Este é o
   input que mais mudaria a resposta, e uma linha de documentação da equipe o resolveria.**
2. **O limite de corrente do driver.** Assumido igual à corrente nominal do motor (2.33 A), isto é, o
   ESC não permite sobrecorrente transitória. Acionamentos reais permitem várias vezes `I_nom` por
   dezenas de milissegundos; representar isso exigiria um envelope de torque, não um `forcerange`
   constante. Aproximação deliberada, citada como tal.
3. **Eficiência do drivetrain.** A aritmética 18:60 da §2 supõe um estágio ideal. Irrelevante para a
   recomendação porque a §2.4 conclui que não há estágio.
4. **A margem de 25 % de yaw** da §3.2 é uma escolha de projeto, não uma grandeza derivada.
5. **`ω_tip` depende do deslocamento de CoM de 3.459 mm**, cuja fidelidade ao robô físico não está
   verificada (`M1`). Ele condiciona o `ctrlrange`, não o `forcerange`.
6. **Se este CAD é o robô RobôCIn 2025** é inferido dos nomes das malhas (`Dribbler_Robocup2025v3`,
   `Direct_Transmission_new_Simplify_2`) e da roda de 60.18 mm batendo com a roda de acionamento
   direto de 62 mm do ETDP. Não está declarado no repositório. A conclusão de acionamento direto da
   §2.4 não se apoia só na identificação — o item 3 da §2.4 descarta a 18:60 pela aritmética,
   independentemente de quem seja este robô.
7. **`kv`** não foi rederivado a partir do gradiente velocidade/torque do motor (8.77 rpm/mN·m para o
   251601). Deixado inalterado; sinalizado para o ticket que for dono do modelo do servo.

## Fontes

Fabricante (maxon group, dono dos dados), todas consultadas em 2026-09-08:
- EC 45 flat Ø42.8 mm, 50 W, 24 V, peça 251601 — https://www.maxongroup.com/maxon/view/product/251601 e https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/251601
- EC 45 flat Ø43.5 mm, 50 W, 24 V, peça 651607 — https://www.maxongroup.com/maxon/view/product/651607
- EC 45 flat Ø42.8 mm, 50 W, 24 V, peça 339286 — https://www.maxongroup.com/maxon/view/product/339286
- EC 45 flat Ø45 mm, 50 W, 24 V, peça 400107 — https://www.maxongroup.com/maxon/view/product/400107
- EC 45 flat Ø42.8 mm, 50 W, 36 V, peça 339287 — https://www.maxongroup.us/maxon/view/product/motor/ecmotor/ecflat/ecflat45/339287

Liga:
- Rules of the RoboCup Small Size League — https://robocup-ssl.github.io/ssl-rules/sslrules.html; fontes dos capítulos na `master`: https://github.com/RoboCup-SSL/ssl-rules/blob/master/chapters/robots.adoc, .../offenses.adoc, .../refereecommands.adoc
- RobôCIn SSL Extended Team Description Paper for RoboCup 2024, §1.2 Direct Drive — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_RoboCIn.pdf
- TIGERs Mannheim Extended Team Description Paper 2024 — https://ssl.robocup.org/wp-content/uploads/2024/04/2024_ETDP_TIGERsMannheim.pdf

Código-fonte (este repositório, revisão local):
- `models/robot_rollers.xml:153` (massa/inércia do chassi), `:159` `:164-165` `:250-257` (malhas de engrenagem do dribbler), `:297-301` (corpo wheel1, junta, inércia, malhas Direct_Transmission), `:827-830` (os quatro atuadores de roda)
- `models/robot_kinematic.xml:389-391` (atuadores do tier 0)
- `docs/research/rsoccer-vs-mjsoccer.md` §4 (`M1`–`M6`), §6 (`W1`–`W7`, matriz de rodas medida), §7 (tabela de limites)
- `docs/analysis/ssl-simulator-rsoccer-comparison.md` (linha de limites de acionamento)

Upstream (`/tmp/rsim-upstream`, `b413932`):
- `src/robosim/sslconfig.h:146-160` (ângulos das rodas, raio, massas, `wheelMotorMaxTorque = 0.070`, `wheelMotorMaxRPM = 1557.0`)
- `src/robosim/sslrobot.cpp:23-68` (hinge + AMotor por roda, `dParamFMax` = o torque, `dParamVel` = o alvo, limitação de RPM→rad/s)
