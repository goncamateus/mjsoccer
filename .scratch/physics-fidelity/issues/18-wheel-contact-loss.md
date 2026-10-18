# Por que as rodas do tier 1 perdem contato com o piso ao dirigir?

Type: task
Status: open
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
