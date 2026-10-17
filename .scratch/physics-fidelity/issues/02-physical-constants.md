# Constantes físicas: a bola, o carpete e o livro de regras

Type: research
Status: resolved
Blocked by: —

## Pergunta

Quais são as constantes físicas reais e citáveis para uma bola da SSL sobre o carpete da SSL, e quais
limites do livro de regras restringem o simulador?

Os dois simuladores discordam em cada uma delas e nenhum cita fonte:

| grandeza | mjsoccer | rSim |
|---|---|---|
| massa da bola | 0.046 kg (`models/ball.xml:7`) | 0.043 kg (`sslconfig.h:123`) |
| raio da bola | 0.0215 m | 0.0215 m |
| atrito de deslizamento bola↔piso | 0.35 (`models/scene_rollers.xml:23`) | 1.0 (`sslworld.cpp:154`) |
| restituição | **nenhuma** — `solref="0.006 1"`, criticamente amortecido | `bounce = 0.5` acima de `bounce_vel = 0.1` |
| desaceleração de rolamento | ≈0.29 m/s², a partir de `μ_roll = 0.0009` (derivado, `P4`) | 0.49 m/s² Coulomb constante + corte forçado abaixo de 0.01 m/s |
| atrito do piso | 0.8 (`models/field.xml:42`) | 1.0 |

Entregar, cada item citado em uma fonte primária:
1. A bola **regulamentar**: o tipo especificado pelo livro de regras da SSL, e a massa e o diâmetro
   reais dessa bola. Uma bola de golfe tem 45.93 g / 42.67 mm pelas regras da R&A e da USGA —
   estabelecer se o livro de regras de fato a especifica e, portanto, se 0.046 ou 0.043 kg está
   correto.
2. Seu **coeficiente de restituição** sobre piso duro, e no que ele se transforma no carpete em que
   a SSL joga. Bola de golfe tem COR publicado; o carpete é a incógnita. O 0.5 do rSim precisa ser
   checado contra qualquer coisa medida — trabalho publicado de calibração da SSL/grSim conta, post
   de blog não.
3. A **desaceleração de rolamento** da bola no carpete da SSL em m/s², de qualquer fonte medida
   (artigos de calibração de equipes, valores publicados de grSim/ER-Force/TIGERs, logs de visão da
   SSL).
4. **Limites do livro de regras** que restringem o simulador: velocidade máxima de chute (rasteiro e
   cavadinha/chip), dimensões máximas do robô, limite de cobertura da bola pelo dribbler, e se a
   velocidade do robô tem algum teto.
5. O **atrito de deslizamento** do carpete contra um rolete de borracha de omniwheel, se existir
   alguma fonte.

Dizer claramente quais dos cinco não têm fonte primária, porque esses viram "aproximação deliberada,
registrada" em vez de "justificado" — essa distinção é todo o destino.

## Answer

Resolvido em 2026-09-09. Desenvolvimento completo: [`research/02-physical-constants.md`](../research/02-physical-constants.md).

**Nenhum entregável voltou totalmente SEM FONTE.** Duas subquestões voltaram, sinalizadas no lugar:
restituição oblíqua e bola-saindo-do-robô (só existem valores de configuração sem respaldo, e os dois
valores publicados de redirecionamento diferem em 3×), e especificamente um rolete de *borracha* da
SSL sobre carpete de *feltro* da SSL.

| grandeza | modelo hoje | recomendado | classe |
|---|---|---|---|
| massa da bola | 0.046 kg | **manter** — R&A Regra 4.2 limita em 45.93 g | RULEBOOK |
| raio da bola | 0.0215 m | **manter** — R&A Regra 4.3 impõe mínimo de 42.67 mm | RULEBOOK |
| COR da bola no carpete | **0** (criticamente amortecido) | **0.50** (faixa medida 0.47–0.56) | MEASURED |
| desaceleração de rolamento da bola | ≈0.293 m/s² | **manter** `μ_roll = 0.0009` — faixa 0.26–0.31 | MEASURED |
| μ de deslizamento bola↔carpete | 0.35 | **manter** — bate com o 0.35 empírico da ER-Force | MEASURED |
| inércia da bola β = I/mr² | **0.4** (padrão de esfera sólida do MuJoCo) | **0.5** | MEASURED |
| velocidade máxima de chute | sem teto | **6.5 m/s, rasteiro e chip igualmente** (§8.4.2) | RULEBOOK |
| cobertura do dribbler | não modelada | **≤20 % ⇒ ≤10.9 mm de penetração no casco** (§3.2.4) | RULEBOOK |
| μ rolete↔piso | **1.0 efetivo** | **0.25** | MEASURED |

**Dois números do modelo se revelam corretos e nunca foram justificados**: atrito de deslizamento
bola↔piso 0.35 e `μ_roll = 0.0009`. Ambos agora têm fonte. Manter e citar.

**Três descobertas que invertem o que as auditorias diziam.**
1. O `ballMass = 0.043` do rSim é um **erro de transcrição**, não uma escolha de modelagem: o
   *diâmetro em metros* da §2.2 do livro de regras colado no campo de massa — e o seu
   `ballRadius = 0.0215` é esse mesmo 0.043 dividido por dois. O `B9` deve ser reclassificado de
   "uma diferença de massa de 7 %" para "erro de digitação upstream".
2. A desaceleração de bola de 0.49 m/s² do rSim é um **hack rastreável**: o `simball.cpp` da
   ER-Force calcula `hackFactor = 1.4; rollingDeceleration = hackFactor * 0.35` sob um comentário
   chamando isso de "quite a hack" — um contorno de estabilidade do Bullet. O rSim copiou o
   resultado, e o ramo de corte em 0.01 m/s, para o ODE, onde a instabilidade não existe. O `P4`
   dava a entender que a bola do mjsoccer é viva demais; a faixa medida diz que **o mjsoccer está
   certo e o rSim está errado**.
3. `bounce = 0.5` é a **única constante de contato defensável** do rSim. O `C2` se inverte: a bola
   com COR zero aqui é a lacuna de fidelidade, não o bounce do upstream.

**Contradição entre tickets com o `01`, verificada.** O único coeficiente medido de rolete sobre
carpete é **0.25**, não o 1.0 que o `01` assumiu para o seu orçamento de atrito (o modelo declara
`friction="1.0"` no rolete contra 0.8 no piso, e o MuJoCo usa o máximo elemento a elemento, então
1.0 é o que vale hoje). Com μ = 0.25 o orçamento é `0.25 · 2.69207 · 9.81 = 6.602` N contra os
8.16 N de força frontal no chão recomendados pelo `01`. Logo:

- O **`forcerange` do `01` se mantém** — vem do motor, não do piso.
- A **conclusão "limitado por torque, 31 % do orçamento" do `01` não se mantém.** O robô seria
  limitado por *atrito* em `μ·g = 2.45` m/s², não limitado por torque em 2.98.
- O **propósito do envelope continua valendo**: 2.45 m/s² está ainda mais abaixo do limiar de
  tombamento de 7.86 m/s² do que 2.98 estava. Aceleração em linha reta não conseguiria tombar o
  robô de jeito nenhum.
- Isso torna o `04` e o `05` uma **decisão conjunta**: qual dos dois, torque ou atrito, é dominante
  não é determinado por nenhum dos tickets sozinho.
