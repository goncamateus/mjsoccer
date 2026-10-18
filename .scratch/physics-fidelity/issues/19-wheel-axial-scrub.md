# Por que o tier 1 não alcança `v_max` cinemático: arrasto axial dos roletes e colapso lateral em yaw

Type: task
Status: resolved
Blocked by: —

## Pergunta

O `18` mostrou que a perda de contato não é o que trava a velocidade: a força normal média continua
mg, e curar o contato rende +0.4 m/s
([`../candidate-18/`](../candidate-18/), seção *Velocidade* do [`18`](18-wheel-contact-loss.md)).
Sobram dois mecanismos.

1. **x: platô em ~2.3 de 3.30 m/s** (`terminal.txt`). Mesmo a 0.66 m/s o regime fica 15 % abaixo,
   com o arrasto axial cancelando o empurrão das rodas. Cada roda leva 0.8–0.9 N do piso **ao longo do eixo**, o que um rolete livre
   deveria zerar. Cada rolete novo entra no contato a 4–8 rad/s quando precisa de ~40, e leva ~8 ms
   para chegar (`spinup.txt`). Não muda com `armature`, atrito torcional, amortecimento, `impratio` nem
   cone. O casco de rolete em resolução cheia (`hull`) e o rolete elipsoide (`ellip`) levam o terminal
   a 2.76–2.80, mas o `ellip` ainda tem 0.7 N axiais.
2. **y: colapso em yaw** a k ≥ 0.5 (`v` ~0, yaw 4–9 rad/s) em todas as variantes do `18`, inclusive
   com contato a 0.7–0.9. A hipótese não medida é a assimetria frente/trás (rodas a 60° e a 45°): com
   atrito saturado desigual, o momento não se cancela.

Fazer:
1. Fechar o mecanismo do arrasto axial. Testar: por que um rolete liso ainda arrasta (transição entre
   roletes vizinhos sobrepostos? rolete que já sai do contato ainda girando?), e se o torque de `Fn`
   fora do eixo do casco de 32 vértices é o que segura o rolete na base.
2. Medir o momento de yaw por roda no comando lateral e confirmar ou derrubar a hipótese
   frente/trás.
3. Dizer se a correção é de malha (`07`: `maxhullvert` do rolete, forma do rolete), de contato ou de
   atuação, e qual `v_max` e `a_max` lateral o tier 1 alcança depois dela. O `12` espera esse número.

## Answer

Resolvido em 2026-10-18. Commit medido: `64a8b8a` (passo 1/960 s, atuador do `04`, μ 0.25).
Scripts e saídas brutas em [`../candidate-19/`](../candidate-19/); `run.sh <dir>` refaz tudo (~1.5 min)
em cima das variantes do `18`, menos o que vai marcado como "checado à parte". As variantes daqui são cópias editadas por
[`variants.py`](../candidate-19/variants.py), todas com a colisão das paredes desligada, só para que
uma corrida de 4.5 s que deriva no heading não termine numa parede. **Nada em `models/` foi editado.**

**Resposta curta.** O arrasto axial vem de um rolete que **não gira livre**. Três coisas freiam o giro
dele, e um rolete freado escorrega ao longo do eixo da roda:

- o casco de 32 vértices, em que a normal age 0.4–0.9 mm fora do eixo do rolete e dá um torque do
  tamanho do torque do atrito;
- o atrito torcional (`condim 4`, 0.005 m), que entra no giro do rolete quando o eixo dele inclina;
- o amortecimento da junta do rolete, que faz o rolete perder o giro na volta fora do piso.

Tirar as três (rolete elipsoide, `condim 3`, amortecimento 0) leva o arrasto de **3.41 N para 0.009 N**
(soma das quatro rodas). O tier 1 chega a **3.29 de 3.30 m/s em x e 4.02 de 4.04 em y**.

O colapso em y **não é defeito do modelo**. É o limite de atrito com cargas desiguais: as rodas
dianteiras levam 22 % mais carga que as traseiras, então, saturadas, empurram mais, e o momento sobra.
A correção é de atuação. Com μ 0.25, o tier 1 sustenta **~1.9 m/s² em x e ~1.3 em y**, não 2.30.

### 1. Arrasto axial: o que segura o rolete

[`torque.py`](../candidate-19/torque.py) → [`torque.txt`](../candidate-19/torque.txt). Avanço a
k = 0.2, segundo 1–2, todo contato rolete↔piso. O arrasto é a força axial do piso contra o
movimento axial do cubo, somada nas quatro rodas. A base dá 3.41 N, os 0.85 N por roda do `18`. Os
torques são em torno do eixo do rolete, média de |·| por contato. `tors` zera o atrito torcional no
modelo compilado, como o `axial.py` do `18`.

| variante | arrasto (N) | braço da normal | torque tangencial / normal / torcional (mN·m) |
|---|---|---|---|
| base | 3.41 | 0.80 mm | 4.14 / 3.87 / 1.01 |
| base `tors` | 3.20 | 0.87 mm | 3.61 / 3.73 / 0 |
| `hull` | 2.59 | 0.43 mm | 3.49 / 2.36 / 1.86 |
| `hull` `tors` | 1.72 | 0.53 mm | 2.39 / 2.40 / 0 |
| `ellip` | 2.92 | 0 | 2.87 / 0 / 2.65 |
| `ellip` `tors` | 0.42 | 0 | 0.20 / 0 / 0 |
| `base_c3_d0` | 2.86 | 0.83 mm | 3.32 / 3.48 / 0 |
| `hull_c3_d0` | 1.45 | 0.52 mm | 2.29 / 2.34 / 0 |
| **`ellip_c3_d0`** | **0.009** | 0 | 0.06 / 0 / 0 |

- **Transição entre roletes vizinhos: não é.** Roletes da mesma roda têm `contype 4` e nenhum
  `<exclude>`, mas não se tocam nem parados nem andando (0 contatos,
  [`probe.txt`](../candidate-19/probe.txt)).
- **Rolete que sai ainda girando: não medido aqui.** O que o `spinup.txt` do `18` mostra é o rolete que
  **entra**: chega lento (4–8 rad/s) e leva ~8 ms para pegar giro, com 1–3 N axiais. Os três freios
  abaixo explicam essa entrada, e tirá-los zera o arrasto, então a saída não tem papel visível.
- **Normal fora do eixo: é, no casco.** No casco de 32 vértices, a normal age 0.8 mm fora do eixo, e
  o torque dela é do tamanho do torque do atrito (o `torque.txt` guarda |·|, não o sinal). Que ele se
  opõe ao atrito vem do `18`: o torque líquido no rolete é ~12× menor que `F_ax · r`. O rolete
  "engata" de face em face. O casco cheio corta o braço pela metade (0.43–0.53 mm), mas não
  zera. O elipsoide zera, porque é superfície de revolução: a normal cruza o eixo.
- **Atrito torcional: é, no elipsoide.** Sem o braço da normal, o torcional (2.65 mN·m) é do tamanho
  do tangencial. O `18` testou `tors` só na base, onde o casco domina (3.41 → 3.20), por isso pareceu
  inerte. No elipsoide, `tors` sozinho leva 2.92 → 0.42 N.
- **Amortecimento do rolete: é, a velocidade.** Com `armature` 1e-7 e `damping` 1e-6, o giro decai
  com τ = (8.07e-9 + 1e-7)/1e-6 = 0.11 s (inércia axial e `armature` de `robot_rollers.xml`), e cada
  rolete passa ~19/20 da volta fora do piso (1 de 20 na base). Pesa quando a
  roda gira devagar para o tanto que o cubo anda no eixo, como nas dianteiras em y (0.866 axial,
  0.5 rolando). `ellip_c3` chega a 3.27 m/s em x, mas em y emperra em 2.07 m/s. Com `damping` 0, chega
  a 4.02 m/s. A `armature` não pesa: 1e-8 com `damping` 1e-6 dá o mesmo que `damping` 0 (checado à
  parte, não salvo).

As três correções são necessárias. [`drive.py`](../candidate-19/drive.py) →
[`drive.txt`](../candidate-19/drive.txt): `v` aos 4.5 s com comando cheio em x, e com rampa de
1 m/s² em y (k = 1 em y colapsa em todas, ver §2). `c3` = rolete `condim 3`; `d0` = `damping` 0.

| variante | x, de 3.30 | y rampa 1, de 4.04 |
|---|---|---|
| base | 2.40 | colapsa (2.5 s) |
| `base_c3_d0` | 2.66 | colapsa (1.9 s) |
| `hull` | 2.74 | colapsa (3.3 s) |
| `hull_c3_d0` | 3.04 | 2.23 |
| `ellip` | 2.85 | colapsa (3.8 s) |
| `ellip_c3` | 3.27 | 1.76 (pico 2.07) |
| `ellip_d0` | 2.93 | colapsa (3.9 s) |
| **`ellip_c3_d0`** | **3.29** | **4.02** |

### 2. y: o momento de yaw por roda

[`yaw.py`](../candidate-19/yaw.py) → [`yaw.txt`](../candidate-19/yaw.txt). As quatro rodas ficam a
79.6 mm do centro, com eixo radial, então cada uma empurra na tangente e dá `Mz = ±R·F`. As
dianteiras (1, 4) ficam em x = +39.8 mm, a 60°. As traseiras (2, 3) ficam em x = −56.3 mm, a 45°.

- **A hipótese frente/trás se confirma, mas pela carga, não pelos ângulos.** No y puro, `Fx = 0` e
  `Mz = R·2(F_f − F_r)`, então o momento zera só com força igual nas quatro rodas, sejam os ângulos
  60°/45° ou quaisquer outros. O que desiguala as forças é a carga. Com o CoM a −3.45 mm (`03`), as
  dianteiras ficam mais perto dele, e a normal estática fica em 7.53 / 5.65 / 6.24 / 6.99 N:
  frente/trás **1.22**, igual à previsão geométrica (56.3 − 3.45)/(39.8 + 3.45).
- Em degrau, as quatro rodas empurram perto do μN delas (forças ao longo do rolamento de 0.2 a 2.5 N,
  quase todas entre 1 e 2). Nos primeiros 0.35 s, as dianteiras dão +85 a +185 mN·m cada e as
  traseiras −51 a −170. A soma em janelas de 50 ms é **positiva nas quatro corridas**, de +4 a
  +112 mN·m, com média 54 na base e 65 no `ellip_c3_d0` a k = 1. A conta R·2μ(N_f − N_r) dá 52.
  Depois disso o yaw já cresceu, e a k = 0.3 uma roda dianteira desatura, e a soma troca de sinal. É a
  sobra inicial que dispara o giro. Em x, as rodas 1↔4 e 2↔3
  cancelam por simetria esquerda/direita com qualquer carga, e por isso x não gira.
- **Não é defeito.** O motor dá 0.078 N·m (`forcerange` do atuador) / 30 mm = 2.6 N por roda, acima do μN, então o robô real
  também é limitado por tração com μ 0.25 e giraria com o mesmo degrau sem realimentar o heading.

**Limites de aceleração**, com N medido e m = 2.692 kg:

- x: μ(0.866·ΣN_f + 0.707·ΣN_r)/m = 0.25 · (0.866 · 14.52 + 0.707 · 11.89)/2.692 = **1.95 m/s²**.
- y sem momento: F_f = F_r ≤ μN_r, logo μ · (2·0.5 + 2·0.707) · N̄_r/m = 0.25 · 2.414 · 5.945/2.692 =
  **1.33 m/s²**.

O "μg = 2.45" do `12` esquece que cada roda empurra na sua tangente, 30° ou 45° fora do movimento.
Medido em [`ramp.txt`](../candidate-19/ramp.txt), com `ellip_c3_d0` e o comando subindo a `a` m/s²:

| eixo | acompanha | não acompanha |
|---|---|---|
| x | 1.6 (`v` 1.57 a 1 s) | 2.0 (1.83 a 1 s), 2.3 (1.81). O degrau dá 1.84 de média no 1º segundo, e o pico de 2.56 em 0.1 s é transiente da partida. |
| y | 1.2 (4.02 m/s a 4 s, yaw ≤ 0.5) | 1.4: colapsa aos 2.7 s, yaw 19 rad/s |

Mesmo abaixo do limite, a sobra do momento deriva o heading devagar (+0.2 a +0.9 rad em 4.5 s, yaw
≤ 0.5 rad/s): o `wheel_command` comanda no referencial do corpo, sem realimentar o heading.

### 3. Onde vai a correção, e o envelope

- **Malha (`07`): o rolete de colisão vira superfície de revolução lisa.** Usar o elipsoide do
  `ellip` do `18`: semi-eixos 6.3 / 13.77 / 6.3 mm, envelope circular, `contype 0` /
  `conaffinity 4`, porque os elipsoides vizinhos se sobrepõem e não podem colidir entre si. Ele
  continua colidindo com o piso (`contype 5`) e com a bola (`7`). Primitiva em vez de malha, presumivelmente
  mais barata que o casco de 32 vértices (não medido; é do `20`). O casco cheio não basta: braço 0.43 mm, e 3.04 / 2.23 m/s com as outras correções.
- **Contato:** `condim="3"` nos roletes.
- **Junta:** `damping` do rolete 1e-6 → 0. Sem fonte; a `armature` 1e-7 pode ficar.
- **Atuação: o colapso em y.** Nada muda no modelo. Quem comanda o tier 1 sem realimentar o heading
  não deve pedir mais que ~1.3 m/s² de lateral. Isso é do envelope (`12`), não deste ticket.

**Envelope do tier 1 depois das três:** `v_max` **medido** 3.29 m/s em x e 4.02 em y (cinemáticos 3.30
/ 4.04). `a_max` sustentado: os **limites analíticos** são 1.95 m/s² em x e 1.33 em y, e as rampas
medidas os cercam (x: 1.6 acompanha, 2.0 não; y: 1.2 acompanha, 1.4 colapsa). Acima disso, em y, o
robô gira.

### Consequências

- Novo [`20`](20-roller-fix.md): aplicar as três em `models/` e medir de novo o que o `14` mede.
- `12`: `v_max` 3.30 fica alcançável depois do `20`. `a_max` 2.30 não é alcançável em eixo nenhum
  com μ 0.25 (1.95 / 1.33). Passa a esperar o `20`.
- `14`: nada muda agora. A janela de 0.1 s do critério 5 pega o transiente da partida (2.5 m/s² em x),
  acima do limite sustentado. Medir de novo no `20`.
