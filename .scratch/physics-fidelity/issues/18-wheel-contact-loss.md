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
