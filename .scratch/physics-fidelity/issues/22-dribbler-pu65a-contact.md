# Dar ao contato bola↔rolete a maciez do PU 65A

Type: task
Status: resolved
Blocked by: —

## Pergunta

Por que o robô perde a bola girando a ≥1 rad/s, e o que, dentro da física (sem junta, snap
cinemático ou restrição prendendo a bola, como grSim/rSim/Sumatra fazem), devolve a retenção?

## Diagnóstico

Todas as sondas eram edições em memória do `MjModel` compilado (ou um overlay de `models/` via
`MJSOCCER_MODELS`), num `Dribbling-v0` com a bola assentada por `seat_ball`; o protótipo foi
descartado depois, os números ficam aqui. Métrica: `holds()` em 120 passos de controle girando no
lugar a `w` rad/s, pior das seeds 0-2 (tier 0 salvo indicação).

1. **Sintoma.** `a = 0.5`: w 0.25/0.5/1/2 → 117/104-108/29/12; `a = 1`: 117/104/85/14. Tier 1
   igual (w 1 → 26-82, w 2 → 13-14). Andar reto a 1 m/s segura (107-120). A doc antiga só media
   ≥10 rad/s.
2. **Como a bola sai** (traço a 1 rad/s, `a = 0.5`): toca **só o rolete** (nenhum casco, placa ou
   para-choque), escorrega 56 mm/s de lado contra 94 mm/s de giro pedido, e o rolete cai de 157
   para 73-95 rad/s sob carga. No robô real também é o dribbler que amortece e conduz, então não
   há segundo contato a modelar.
3. **Não é geometria.** Visual `Dribblerschnecke.stl`: tubo reto (r 3.0/6.5 mm, 72 mm, 0 de 208
   faces oblíquas) — não há hélice no CAD. Colisão trocada por barra em V (cones, r 9 → 11/13 mm
   nas pontas) e hélice dupla (cápsulas r 1.5 mm, passo 24 mm, 2 entradas, mão oposta por
   metade, as duas mãos), `a = 1`:

   | | w 1 | w 2 | parado |
   |---|---|---|---|
   | cilindro | 85 | 14 | 120 |
   | V 2 / 4 mm | 22 / 55 | 13 / 16 | 118 / 120 |
   | hélice + / − | 37 / 70 | 26 / 16 | 85 / 95 |

4. **Não é atrito nem solver.** `a = 1`, w 1 / w 2: μ do par 2.5 → 5: 86 / 14; → 10: 87 / 66;
   `impratio` 10: 81 / 13; `cone` elíptico: 85 / 14; `noslip_iterations` 10/50: 90 / 14. O μ da
   geom é ignorado (o `<pair>` manda) e a estrutura do dribbler nem encosta na bola.
5. **Não é rotação.** Alvo 1500/3000/6000/9000 rpm, motor ×1 (real sob carga ~790/2100/4800/7000)
   e ×10 (~1400/2800/5500/6600): w 1 → 31/85/61/12 e 81/64/55/17; w 2 nunca passa de 17. Mais
   rpm piora: mais cone vai na direção do giro.
6. **É o quique.** A 960 Hz, parado: contato bola↔rolete em 83 % dos substeps, normal p10/p90
   0.44 / 11.2 N (peso da bola 0.45 N), |f_t|/μf_n mediana 0.40 (saturado só em ~10 %). Um parâmetro
   por vez, `a = 1`:

   | contato | duty | w 1 | w 2 |
   |---|---|---|---|
   | `solref 0.004 0.5` (antes) | 83 % | 85 | 14 |
   | dampratio 1.0 | 98 % | 88 | 67 |
   | timeconst 8 ms | 97 % | 97 | 80 |
   | timeconst 16 ms | 100 % | 102 | 93 |
   | amortecimento do braço ×10 | 80 % | 24 | 19 |

   (Braço ×100 e raio 9 → 6.5 mm invalidam o assentamento: a bola não chega ao rolete.)
7. **Material.** Pesquisa `03`: o rolete de referência é PU Shore 65A Ø13 mm (RobôCIn ETDP 2026);
   Shore → módulo (Gent / Qi) → Hertz dá ~0.2 mm de afundamento a 3 N, ou seja timeconst 7-9 ms
   com dampratio 1 (14-17 ms com 0.5) — o contato antigo era ~10× mais duro e subamortecido.

## Decisão

`solref="0.008 1"` nos dois pares bola↔rolete (`r_col_axisDribbler`, `r_col_barDribbler`) de
`scene.xml` e `scene_rollers.xml`. Fonte: rolete de PU Shore 65A (RobôCIn ETDP 2026 §2.2) →
módulo (Gent / Qi) → Hertz: ~0.2 mm de afundamento a 3 N → timeconst 7-9 ms com dampratio 1
(research `03` §5.5). O par `0.016 0.5`, também consistente com o afundamento, falha no tier 1
a 2 rad/s (15/120) e fica de fora .

## Resultado

`holds()` em 120 passos, giro no lugar, `dribble = 1`, pior das seeds 0-2 :

| | 0.5 | 1 | 2 | 3 rad/s |
|---|---|---|---|---|
| tier 0, antes → depois | 104 → 107 | 85 → 100 | 14 → 81 | 10 → 11 |
| tier 1, antes → depois | 94 → 103 | 82 → 96 | 13 → 73 | 11 → 11 |

Contato bruto parado: 74 → 97 % (tier 0), 65 → 98 % (tier 1). Posse parada
(`test_seated_ball_is_possessed`): 120/118/117/119 → 120/120/118/120. Fixado por
`test_seated_ball_survives_a_2_rad_s_turn`.

## Fica aberto

3 rad/s ainda derruba a bola nos dois tiers; as equipes publicadas seguram 3-7 rad/s por área
de contato (rolete macio/dividido), não por atrito (research `03` §1.3, §3.1). Um contato
esfera-cilindro do MuJoCo é um ponto só; o próximo passo físico seria área de contato
(`flexcomp` como luva, ou rolete dividido se o hardware for). O raio de colisão (9 mm) segue
maior que o real (6.5 mm): reduzir só o raio tira a bola do alcance do rolete, então pede
reposicionar o rolete junto.
