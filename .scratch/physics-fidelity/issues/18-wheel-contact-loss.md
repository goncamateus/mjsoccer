# Por que as rodas do tier 1 perdem contato com o piso ao dirigir?

Type: task
Status: resolved
Blocked by: —

## Pergunta

Nada a decidir até ter o mecanismo. Medição feita no `04`
([`../candidate-04/contact.py`](../candidate-04/contact.py)): com o comando cinemático de avanço
cheio, a fração de steps em que cada roda toca o piso cai de 0.99 (parado) para 0.27-0.42 com o
atuador novo e μ 0.25, e para **0.06-0.11** no modelo atual. O chassi chega a ~1.7-2 m/s em x contra 3.30
cinemáticos, e a **0.17 m/s em y** contra 4.04 (`line_c95.txt`: aceleração ~0, pico de yaw 6.6
rad/s) — medir os dois eixos.

Fazer:
1. Separar as hipóteses: (a) poligonalidade da roda — vãos entre os 20 roletes de colisão
   (`col_wheelN_roller_*`), cada roda batendo de rolete em rolete; (b) rigidez/amortecimento do
   contato do rolete (`solref="0.004 1"`, `condim="4"`); (c) indeterminação estática de quatro rodas
   num chassi rígido (sempre uma roda "sobrando"); (d) `timestep` 0.002 grande demais para o contato
   dos roletes (cruzar com o `13`).
2. Para cada uma, uma variante no scratchpad via `MJSOCCER_MODELS`, como no `03`/`04`, medindo fração
   de contato por roda e `|v|` final.
3. Registrar qual mecanismo domina, e se a correção é de `07` (malha de colisão), `13` (timestep) ou
   do próprio contato.

Até isto fechar, o `14` não deve usar `v_max`/`a_max` cinemáticos como critério.

## Answer

Resolvido em 2026-10-18. Commit medido: `40d625d` (passo 1/960 s, atuador do `04`, μ 0.25).
Scripts e saídas brutas em [`../candidate-18/`](../candidate-18/); `run.sh <dir>` refaz tudo (~1.5 min).
As variantes são cópias de `models/` editadas por [`variants.py`](../candidate-18/variants.py) e
rodadas com `MJSOCCER_MODELS`. **Nada em `models/` foi editado.**

**Resposta curta.** A perda de contato é o robô inteiro **saltando**, não uma roda "sobrando". Nenhuma
das hipóteses (a), (b), (d) a explica, e só uma suspensão a remove. Mas ela **não custa tração**: a força
normal média continua igual a mg, e curar o contato rende só +0.4 m/s. O que trava o `v` são outros dois
mecanismos, que vão para o novo [`19`](19-wheel-axial-scrub.md): arrasto **axial** dos roletes (que
cancela o empurrão das rodas) e, em y, um **colapso em yaw**. O tier 1 **não alcança** 3.30 m/s em x
(platô ~2.3), e a aceleração lateral medida no `14` (1.87 m/s²) fica abaixo de 2.30. Então o `12`
reabre, como o comentário dele pede.

### Contato: as quatro hipóteses

[`contact.py`](../candidate-18/contact.py) → [`contact.txt`](../candidate-18/contact.txt). Comando
cheio em x (cinemático 3.30 m/s), 2 s depois de assentar, com `k` = fração do comando. `n=0` é a
fração de steps com **nenhuma** roda no piso; `n=3`, com uma roda sobrando.

| variante | hipótese | contato por roda, k = 1 | `n=0` / `n=3` | `v` 2 s, k = 1 |
|---|---|---|---|---|
| base | — | 0.33–0.40 | 0.23 / 0.11 | 2.07 |
| `dt05` (1/1920 s) | (d) | 0.32–0.38 | 0.21 / 0.11 | 2.16 |
| `soft8/16/32` (`solref` 0.008/0.016/0.032) | (b) | 0.35–0.43 / 0.40–0.48 / 0.50–0.57 | 0.19 / 0.13, 0.13 / 0.17, 0.08 / 0.21 | 1.98 / 2.10 / 2.09 |
| `damp3` (`0.004 3`) | (b) | 0.33–0.38 | 0.23 / 0.11 | 2.06 |
| `hull` (casco do rolete em resolução cheia) | (a) rolete | 0.37–0.42 | 0.17 / 0.12 | 2.47 |
| `ellip` (rolete elipsoide, envelope circular) | (a) roda | 0.32–0.37 | 0.26 / 0.07 | 2.43 |
| `susp` (mola vertical por roda, 1 mm de assentamento) | (c) | **0.79–0.85** | **0.00** / 0.39 | 2.35 |

- **(a) poligonalidade: não é.** O raio efetivo da roda ondula **0.237 mm** a cada 18°
  ([`diag.txt`](../candidate-18/diag.txt)). A conta `A·Ω² > g` previa a perda a partir de ~14 rad/s
  de roda. Mas o `ellip` derruba a ondulação para **0.007 mm** e o contato a k = 1 fica igual
  (0.35–0.39 no `diag`). O liga/desliga do contato oscila a **40–45 Hz** na base e a 69–88 Hz no
  `ellip`, nunca na passagem de rolete (240–295 Hz). Só a `susp` trava nela.
- **(b) rigidez/amortecimento: ajuda pouco.** Amolecer 8× leva o contato de 0.33–0.40 a 0.50–0.57 e
  não muda `v`; amortecer 3× não muda nada. A frequência do liga/desliga quase não se mexe com o `solref`
  (40–45 → 35–39 → 29–35 Hz em 0.004 / 0.016 / 0.032), quando o modo de quique do chassi na mola do
  contato cairia ~8× (a frequência dele escala com 1/`timeconst`). Então não é esse modo.
- **(c) indeterminação: não na forma do ticket.** Os steps com zero rodas (23 %) são o dobro dos
  steps com três (11 %): o robô sai do chão inteiro, ele não fica apoiado em três rodas. Mas compliance
  entre chassi e roda resolve: com a `susp`, `n=0` cai a zero e o contato vai a 0.8.
- **(d) passo: não é.** O `dt05` acompanha a base: contato dentro de ±0.04 em toda linha, `v` x a
  k = 1 2.16 contra 2.07. O passo testado é 0.52 ms, um fio acima dos ≤ 0.5 ms do comentário do `13`;
  como nada mudou de 1.04 para 0.52 ms, não rodei de novo a 0.5 ms.

A perda escala com a velocidade de roda em todas as variantes: a k = 0.1 o contato fica em
0.97–0.99. A fonte da excitação a 30–45 Hz **não foi isolada**.

**Por que isso não importa para a tração.** A força normal média somada fica em **0.996–1.001 mg**
nas sete variantes do `diag.txt` (x, k = 0.2 e 1). O atrito integrado no tempo é limitado por μ × o
impulso normal, então o orçamento médio de tração é μmg com ou sem contato contínuo. A `susp` mostra
isso na prática: contato 0.8, e o terminal vai de 2.29 a 2.69 m/s, 40 % do déficit. É o mesmo ganho
de `hull`/`ellip`, que não mexem no contato, então o ganho da `susp` provavelmente também não vem do
contato.

**Onde vai a correção:** em nenhum lugar por enquanto. Não é do `07` (geometria), nem do `13`
(passo), nem do `solref`. A suspensão é a única que cura, mas ela inventa uma mola que o robô real não
tem com essa rigidez. Não fazer nada até um consumidor precisar de forças normais por roda (*Fidelidade
de escorregamento por roda* em `map.md`).

### Velocidade: o que de fato trava o `v`

**x: é platô, não é falta de tempo.** Na [`terminal.py`](../candidate-18/terminal.py) →
[`terminal.txt`](../candidate-18/terminal.txt) (comando cheio na diagonal, 3.5 s), o `v` em
1 / 2 / 3 / 3.5 s fica assim:

| variante | `v` (m/s) | de 3.30 |
|---|---|---|
| base | 1.47 / 2.10 / 2.31 / **2.29** | 69 % |
| `hull` | 1.66 / 2.49 / 2.70 / 2.76 | 84 % |
| `ellip` | 1.62 / 2.47 / 2.74 / 2.80 | 85 % |
| `susp` | 1.57 / 2.36 / 2.64 / 2.69 | 82 % |
| `soft32` | 1.40 / 2.08 / 2.36 / 2.42 | 73 % |

**O arrasto axial cancela o empurrão.** A k = 0.2 (cinemático 0.66 m/s), com o contato em 0.81–0.90, o `v` assenta em
0.52–0.55: um déficit de 15–17 % em regime, a baixa velocidade. A força do piso em cada roda
([`axial.py`](../candidate-18/axial.py) → [`axial.txt`](../candidate-18/axial.txt)) mostra as quatro
rodas empurrando ao longo do rolamento (0.48–0.82 N, no sentido do movimento). Mas a componente **ao
longo do eixo da roda** é de **0.79–0.93 N por roda**, contra o movimento axial do cubo, com μN ≈
1.65 N. É o que um rolete livre deveria zerar. Nas traseiras (a 45°), com mais movimento axial, esse
arrasto vence o empurrão em x, e a soma assenta em ~0 abaixo do cinemático. As rodas giram no comando
(18.7 de 19 rad/s); o que escorrega é o eixo.

- Isso não é geometria do eixo: os eixos dos roletes estão tangentes (produto escalar 0.000 com o eixo
  da roda, checado à parte, não salvo), e a cambagem é ≤ 0.0005 (`axial.txt`). Também não é `armature` do rolete (1e-7 → 1e-9), atrito torcional (0.005 → 0),
  amortecimento do rolete, `impratio` 10 nem cone piramidal: o arrasto axial fica em 0.6–0.9 N em todos.
- Passo a passo ([`spinup.py`](../candidate-18/spinup.py) → [`spinup.txt`](../candidate-18/spinup.txt)),
  cada rolete **novo** entra no contato a 4–8 rad/s quando precisa de ~40, e leva ~8 ms para chegar lá,
  com 1–3 N axiais nesse intervalo. A k = 0.2 isso acontece a 50–60 Hz por roda. O torque líquido no
  rolete é ~12× menor que `F_ax · r`. A leitura provável é que a normal age fora do eixo do rolete, o
  cogging do casco de 32 vértices, mas o `ellip`, que é liso, ainda tem 0.7 N axiais. Então o mecanismo
  **não está fechado**.
- O `hull` e o `ellip` são as únicas variantes que sobem o terminal de verdade (+0.5 m/s).

**y: colapso em yaw, em todas as variantes.** Com k ≥ 0.5 o comando lateral vira giro: `v` ~0 e yaw de
3.6–9.3 rad/s, inclusive com a `susp` (contato 0.7–0.9). A k = 0.3 o y anda (0.52–0.91 de 1.21). A
leitura provável é que as rodas dianteiras (a 60°) e as traseiras (a 45°) pedem velocidades diferentes
em y. Quando o atrito satura de forma desigual, o momento da frente e o de trás deixam de se cancelar,
ao contrário de x, que é simétrico esquerda/direita. Isso **não foi medido**, e vai para o `19`.

### Consequências

- `12`: o tier 1 não alcança 3.30 m/s em x (2.29 base), nem 2.30 m/s² lateral (1.87, `14`). O `12`
  **reabre**, bloqueado pelo `19`.
- `14`: nada muda. O critério 5 (aceleração) não depende de `v_max` e continua passando, e nenhum
  critério deve usar `v_max` cinemático até o `19` fechar.

## Comments

- 2026-10-18, do `12`: o tier 0 agora declara o envelope **derivado** do tier 1 (`v_max` 3.30 m/s,
  `a_max` 2.30 m/s², isotrópicos), não o medido. Quando este ticket fechar, o tier 1 deve alcançar
  pelo menos 3.30 m/s em x e 2.30 m/s² nos dois eixos; se não alcançar, o `12` reabre.
- 2026-10-18, do `13`: hipótese (d) medida em parte (`../candidate-13/sweep_t1.txt`, comando de
  avanço cheio pela env, 2 s). `v` final 1.90 / 2.005 / 1.666 / 1.633 / 1.192 m/s a 0.25 / 0.5 / 1 /
  2 / 4 ms: o passo mexe ~15 % e de forma não monotônica, mas nem 0.25 ms chega perto de 3.30, então
  **o passo não é o mecanismo dominante**. O giro cheio também depende do passo (31.8 / 30.9 / 30.0 /
  29.2 rad/s), por isso `MAX_W` virou o teto cinemático 35.9. O passo agora é 1/960 s; medir as
  variantes deste ticket a ≤ 0.5 ms para tirar o passo da conta.
- 2026-10-18, do `14`: medido a 1/960 s, comando cheio pela env, 2 s
  ([`../candidate-14/accel.txt`](../candidate-14/accel.txt)): frontal 2.47 m/s², `v` 2.14 m/s;
  lateral 1.87 m/s², pico 0.78, **0.43 m/s em 2 s**. A aceleração passa no critério 5 do `14`; a
  velocidade é deste ticket. `tasks.wheel_command` dá o comando de rodas de qualquer `(vx, vy, ω)`
  para as variantes daqui.
