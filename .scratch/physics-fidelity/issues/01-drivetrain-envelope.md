# Envelope do drivetrain: o que os motores deste robô realmente conseguem fazer?

Type: research
Status: resolved
Blocked by: —

## Pergunta

Quais são os limites fisicamente defensáveis para um atuador de roda tier-1, derivados de um Maxon
EC45 flat 50 W passando por uma redução 18:60 até a roda medida deste modelo?

Hoje `models/robot_rollers.xml:827-830` declara `forcerange="-3 3"` N·m e
`ctrlrange="-200 200"` rad/s por roda. Com o `r_eff = 30.089 mm` **medido**, isso dá 399 N de
força no chão por roda contra um orçamento total de atrito de `μ·m·g = 1.0 · 2.69207 · 9.81 = 26.4 N`
(`W3`, `L1`). Isso não é um atuador; é "torque infinito, limitado por atrito". O equivalente no
rSim é `wheelMotorMaxTorque = 0.070` N·m com `wheelMotorMaxRPM = 1557`
(`/tmp/rsim-upstream/src/robosim/sslconfig.h:159-160`), comentado "Maxon EC45 flat 50w with
18:60 gear ratio, reduced to 3m/s2 accel" — o único número do rSim com origem no hardware.

Entregar:
1. Os valores do EC45 flat 50 W tirados do próprio datasheet da Maxon: torque nominal e de bloqueio
   (stall), velocidade nominal e em vazio, e o limite de corrente que "50 W" implica.
2. Passando pela 18:60 (redução 3.333:1), o torque e a velocidade **na roda**, e se os 0.070 N·m do
   rSim correspondem ao valor nominal, contínuo ou a um valor deliberadamente reduzido — o comentário
   diz "reduced to 3m/s2 accel", então estabelecer *a partir de quê* foi reduzido.
3. Com `r_eff = 30.089 mm` e `R = 0.0796 m` (ambos medidos, §6 do documento de pesquisa), os
   `a_max`, `v_max` e `ω_max` resultantes para um robô de 2.69207 kg, e o par `forcerange` /
   `ctrlrange` que os expressa.
4. Se o resultado passa pelos dois limiares que importam: o limiar estático de tombamento frontal
   derivado `g · 0.04325 / 0.05399 = 7.86` m/s², e `ω_tip = 47.7` rad/s (`M1`). Um envelope
   defensável deve ficar confortavelmente abaixo de ambos, assim como os 4.67 m/s² do rSim ficam
   abaixo dos seus próprios 6.28.
5. Se as regras da SSL ou a prática comum da Divisão A limitam a velocidade ou a aceleração da roda
   de forma independente.

Somente fontes primárias: o datasheet publicado pela Maxon para o EC45 flat 50 W, o livro de regras
da SSL e os dois clones. Citar cada número no documento ao qual ele pertence. Registrar
explicitamente tudo o que não pode ser resolvido sem a especificação do robô real — a equipe não tem
documento do drivetrain, então uma suposição declarada é uma resposta aceitável, um número inventado
não é.

## Answer

Resolvido em 2026-09-08. Desenvolvimento completo: [`research/01-drivetrain-envelope.md`](../research/01-drivetrain-envelope.md).

**Envelope recomendado** — `forcerange="-0.078 0.078"` N·m e `ctrlrange="-95 95"` rad/s por
roda, `kv` inalterado. Resulta em `a_max` 2.98 m/s² frontal / 2.30 lateral, `v_max` 3.30 / 4.04 m/s,
`ω_max` 35.9 rad/s. Passa pelo limiar estático de tombamento frontal (7.86 m/s²) com 62 % de folga e
por `ω_tip` (47.7 rad/s) com 25 %, e exige 31 % do orçamento de atrito em vez de 4750 %.

As duas metades têm status diferentes: **`forcerange` é um fato de hardware** (`I_nom · k_M` do
motor, acionamento direto), **`ctrlrange` é uma escolha de projeto** — a velocidade em vazio do motor
é ~702 rad/s na roda, então 95 rad/s é escolhido para limitar o yaw com 25 % de margem abaixo de
`ω_tip`, não porque o acionamento não consiga girar mais rápido. Essa escolha pertence ao `04`.

**A descoberta que importa: este robô é de acionamento direto (direct drive), não 18:60.** Três
caminhos independentes. (a) A malha de todo cubo de roda é `Direct_Transmission_new_Simplify_2`
(`models/robot_rollers.xml:300-301`); a única engrenagem de 18 dentes no CAD é
`r_vis_gearMotorDribbler` no chassi, que aciona o dribbler
(`models/robot_rollers.xml:164-165,250-257`). (b) O ETDP 2024 da RobôCIn, §1.2 *Direct Drive*,
registra a mudança da roda de 45 mm com redução (3.3:1) para uma roda de 62 mm com acionamento
direto, e declara o ponto de operação de projeto como "around 10 percent of the stall torque, near
maximum efficiency" — o `r_eff = 30.089` mm medido neste modelo corresponde a uma roda de 60.18 mm.
(c) Aritmética: com 18:60 a roda geraria 27.2 N de força no chão contra um orçamento de atrito de
26.4 N, deixando o robô limitado por atrito em 9.81 m/s² e o teto de torque comportamentalmente
inerte. Três caminhos também convergem no torque: `I_nom · k_M = 78.1` mN·m, torque nominal
83.4 mN·m, e 10 % dos 780 mN·m de stall = 78.0 mN·m.

Portanto, os `0.070` N·m e `1557` rpm do rSim descrevem a **geração anterior do robô, com
redução**. Nunca foram transferíveis como números, e por isso o ticket rederivou em vez de copiar.

**Correções que isso impõe às auditorias.** O `W3` de `docs/research/rsoccer-vs-mjsoccer.md` dá o
teto do rSim como 4.67 m/s²; isso supõe que as quatro rodas projetam força unitária. A geometria
limita a projeção a `Σ|sin α| = 3.14626` frontal e `Σ|cos α| = 2.41421` lateral (verificado de forma
independente: ambos se reproduzem, e o padrão de sinais que maximiza a força dá `Σf_i = 0`, então não
custa yaw). O teto real do rSim é **3.68 m/s² frontal / 2.82 lateral**, o que confirma o próprio
comentário "reduced to 3m/s2 accel" no eixo fraco. A conclusão do `W3` (rSim limitado por torque,
mjsoccer limitado por atrito) não é afetada.

**Não resolvido, levado para o `16`.** Qual dos cinco enrolamentos do EC 45 flat 50 W este robô usa
não está registrado em lugar nenhum do repositório. Assume-se o 251601. Ao longo do catálogo, o torque
de projeto varia de 40.2 a 91.8 mN·m, movendo `a_max` entre 1.54 e 3.51 m/s² — um fator de 2.3, e o
maior input não resolvido do `04`. Também não resolvidos e registrados como aproximações
deliberadas: a tolerância a sobrecorrente transitória do ESC (um `forcerange` constante não consegue
expressá-la), e se `kv` deveria ser rederivado a partir do gradiente velocidade/torque do motor,
8.77 rpm/mN·m.

**Livro de regras:** não limita velocidade, aceleração, velocidade da roda nem potência do motor. O
capítulo 3 *Robots* → *Hardware And Software Constraints* restringe apenas forma, cor, dribbler,
padrão de visão, rádio e autonomia.

## Adendo (2026-09-09, vindo do `02`)

O `02` estabelece que o único coeficiente medido de rolete de omniwheel sobre carpete é **0.25**,
não o 1.0 que este ticket supôs (o modelo declara `friction="1.0"` no rolete, 0.8 no piso, e o MuJoCo
usa o máximo elemento a elemento — então 1.0 é o que vale hoje, mas não tem fonte). Com μ = 0.25 o
orçamento é 6.602 N contra os 8.16 N que este ticket recomenda exigir.

`forcerange="-0.078 0.078"` não é afetado — é derivado do motor. O que muda é a *restrição
dominante*: o robô seria limitado por atrito em `μ·g = 2.45` m/s², não limitado por torque em 2.98.
A conclusão "passa pelo limiar de tombamento" se mantém com mais margem, não menos. O `04` e o `05`
agora precisam ser decididos juntos; o `05` foi adicionado aos bloqueadores do `04`.
