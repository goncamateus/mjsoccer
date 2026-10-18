# Decisão: o orçamento da malha de colisão

Type: grilling
Status: resolved
Blocked by: 06

## Pergunta

O tier 1 carrega **209 geoms de colisão** e 84 corpos extras por robô (medido): uma decomposição do
chassi do CAD em cascos convexos, quatro cubos de roda e 80 roletes girando livres. O rSim carrega um
cilindro, quatro cilindros de roda e uma caixa soldada, e testa exatamente sete pares de colisão
(`pworld.cpp:287-305`).

Sob fidelidade ao hardware, a versão do mjsoccer é a *mais* correta, e não é obviamente viável em
custo. Três perguntas:

1. **Manter o chassi de 209 cascos?** É por causa dele que os robôs enroscam nos para-choques e
   carcaças de motor uns dos outros e geram torques de contato em múltiplas alturas que um cilindro
   não consegue (`B1`). Também é o principal fornecedor do impulso que converte um giro cônico
   marginal em queda. A fidelidade diz manter; os impulsos de tropeço são reais, não artefatos.
   Decidir se algum casco é *ruído de CAD* em vez de uma superfície externa real — esse é o único
   critério de poda defensável.
2. **Manter os contatos bola↔rolete?** A bola é `contype="7" conaffinity="7"` e os roletes são
   `contype="4" conaffinity="4"`, então o bit 4 casa e uma bola rolando por baixo de um robô é
   chutada para fora por um rolete (`B6`). O rSim nunca colide bola com roda. Fisicamente a bola
   *bate* nas rodas. Manter — a menos que o efeito medido seja dominado por artefatos da malha.
3. **Manter cubo↔parede?** Os cubos de roda são `contype="1" conaffinity="1"` e alcançam os cascos da
   parede (`C7`). Mesmo raciocínio.

Também resolver, para registro, a discrepância de contagem entre as duas auditorias: 209 geoms de
colisão (documento de pesquisa) vs "315 descendant geoms" (documento de análise) — só colisão contra
colisão+visual. Dizer qual número a documentação deve citar.

Resolvido por uma medição de tempo de relógio contra a linha de base do `03` mais a atribuição de
impulsos de contato do harness do `14`: quais contatos de fato fornecem impulsos de tropeço em jogo
normal.

## Answer

**Manter a malha inteira, menos 16 cascos enterrados.** Os três pares ficam (chassi multicasco,
bola↔rolete, cubo↔parede); podam-se só os 16 cascos que nenhum contato alcança sem penetração. Aplicado
em `models/robot_rollers.xml` e `models/robot_kinematic.xml` (as 16 linhas são idênticas nos dois
tiers), com o teste `test_no_buried_collision_hulls`. Medições em
[`../candidate-07/`](../candidate-07/), contra os modelos pós-`06`. O harness do `14` ainda não existe;
`attribution.py` faz a parte de atribuição de impulso que este ticket precisava.

**Método.** Um hook em cada `mj_step` atribui a cada classe de par de contato o impulso angular
horizontal (roll/pitch) que ele aplica em torno do CoM de cada robô; suportes (rolete|piso, bola|piso)
só são contados. Cenários: ação aleatória nos seis ids, tier 1, seeds 0-4 (`attr_random.txt`); robô a
comando cheio contra robô parado, desvio lateral 0/60/120 mm × yaw 0/45/90/135/180°
(`attr_robot.txt`); comando cheio contra a parede a 15/45/90° (`attr_wall.txt`); bola rolada a
1/2/4 m/s contra robô parado, por trás e de lado, desvio −60/0/+60 mm (`attr_ball.txt`).

**Quem fornece impulso de tropeço em jogo normal: ninguém.** Com o atuador do `04` e μ 0.25 do `05`:

- Ação aleatória: **zero** contatos de casco — corpo, cubo, robô↔robô, parede — em 6 ids × 5 seeds.
  O único contato não-suporte é bola|mecanismo (dribbler/chute). 0 tombamentos; pico de inclinação
  5.6-6.6°, que vem da direção, não de contato.
- Robô↔robô a comando cheio (~1.9 m/s, o `v` final de `../candidate-06/measure_full.txt`; a
  velocidade de impacto não foi registrada aqui): pico de inclinação **≤ 3.1°** em todos os 15 casos. corpo|corpo domina
  (impulso angular por step até 58 mN·m·s), corpo|cubo em segundo (até 83 mN·m·s com desvio de
  60 mm), rolete|rolete e cubo|cubo marginais. Os enroscos de `B1` existem — os impulsos são de
  múltiplos cascos em alturas diferentes — mas não chegam perto de tombar o robô.
- Parede: pico **≤ 1.7°**; corpo|parede domina a 90°, cubo|parede leva o maior impulso por step a 15°
  (72 mN·m·s, 0.83°).
- A conversão "giro cônico marginal → queda" que o ticket atribuía ao chassi não tem mais o que
  converter: o `04` zerou o modo (0 tombamentos/min).

Portanto o orçamento não se decide por tropeço; decide-se por fidelidade e custo.

1. **Chassi: manter, podar 16.** Critério de poda: um casco rígido do corpo `robot` cuja superfície
   inteira (vértices + grade baricêntrica em cada face) fica dentro da união dos outros cascos rígidos,
   com tolerância 0 (`buried.py`, `buried.txt`). São 16 de 87: os 14 do motor e engrenagens do
   dribbler (`motorDribbler_c01-c06`, `gearMotorDribbler_c00-c03`, `gearMiddleDribbler_c00-c03`) e o casco `c02` de cada para-choque (`bumper{Left,Right}_1_c02`). Um teste independente por raios (`exposure.py`, 200k raios) os marca todos como nunca atingidos primeiro;
   ele marca também mais 9 (`chassis_1_c23`, `chassis_2_c06`, `chassis_2_c19`, `motor1-4_c01`) que ficam,
   porque o teste por raios conta cubos e roletes como oclusores e eles giram, e porque com 1M raios
   `chassis_2_c06`/`c19` passam a ser atingidos — o teste exato, só com oclusores rígidos, é o que vale. Zero contatos neles em
   todos os cenários (`buried_hull_contacts=0`). Eram `density="0"`: massa e inércia inalteradas. Com tolerância de 0.5 mm entrariam mais 7 (5 cascos de para-choque rentes ao chassi e
   `motorDribbler_c00`/`c07`) — expostos, ficam. Ganho de vazão: +0.3/+2.4/+1.1 % em GoToBall/ContestedPossession/StaticDefenders (`throughput_*.txt`,
   uma rodada de melhor-de-3 por variante; o `06` viu ±1.3 % entre variantes equivalentes e aqui
   `noballroller` dá −1.1 %, então é ruído ou perto dele). A poda é por correção, não por custo. As 15 malhas `r_c_*` que ficaram sem
   geom continuam no `<asset>`, como as órfãs que o exportador do CAD já deixava (DribblerBar,
   Dribblerschnecke).
2. **Bola↔rolete: manter.** Quem desvia a bola é o **cubo**, não o rolete: impulso bola|cubo 0.03-0.19 N·s
   contra bola|rolete ≤ 0.017. Sem o contato com os roletes (bola `contype/conaffinity` 7 → 3,
   `attr_ball_noballroller.txt`) a saída da bola muda ≤ 0.04 m/s em 18 casos. Custo: ruído. O `B6` é
   físico (a bola bate na roda) e o rolete só contribui atrito; o efeito não é dominado por artefato
   da malha.
3. **Cubo↔parede: manter.** Carrega impulso real de parede, de robô e de bola (acima). Desligar os
   cubos por completo (limite superior do custo) dá +2.0/+5.3/+6.4 % de vazão — dentro dos 10 % do
   `14`, mas tirá-los perde bola↔roda e o contato com o piso do `06`. Não há como desligar só
   cubo↔parede sem um bit novo; não vale.

**Contagem para a documentação.** As duas auditorias estavam certas, contando coisas diferentes: 209 era
o número de geoms de **colisão** por robô (87 cascos rígidos do corpo + 24 de cubo + 80 roletes + 18 do
mecanismo, que só tocam a bola), 315 o total **colisão + visual** (106 visuais). Depois da poda: **193
de colisão, 299 no total**, 71 cascos rígidos do corpo; os candidatos ao piso do `06` caem de 111 para
**95**. A documentação deve citar o número de colisão (**193**), que é o que custa. Corpos: 89 por robô
(chassi + dribbler_arm, dribbler, kicker, chipper + 4 cubos + 80 roletes); os "84 extras" são cubos +
roletes.
