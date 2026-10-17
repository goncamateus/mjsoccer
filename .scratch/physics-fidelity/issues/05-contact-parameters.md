# Decisão: parâmetros de contato defensáveis em vez de ajustados a gosto

Type: grilling
Status: resolved
Blocked by: —

## Pergunta

Quais coeficientes de contato mudam, para quê, e com base em qual autoridade?

Os candidatos, com o que há de errado em cada um hoje:

| contato | hoje | problema |
|---|---|---|
| rolete ↔ piso | `friction="1.0 0.005 0.0001"`, `condim="4"`, `solref="0.004 1"` (`models/robot_rollers.xml:311`) | μ = 1.0 está no topo da faixa para borracha sobre carpete; ele define todo o teto de aceleração (9.81 m/s², acima do limiar de tombamento de 7.86). O valor tangencial do rSim é 0.8 |
| bola ↔ piso | `friction="0.35 0.35 0.005 0.0009 0.0009"`, `solref="0.006 1"` → **sem quique** (`models/scene_rollers.xml:23`) | uma bola real quica (`C2`); desaceleração de rolamento ≈0.29 m/s² contra 0.49 do rSim (`P4`); μ de deslizamento 0.35 contra 1.0 do rSim (`C3`) |
| bola ↔ parede | `solref="0.004 1"`, sem restituição (`models/field.xml:44-55`) | idem: um chute errado morre na parede em vez de ricochetear |
| paredes, para-choques | `solref="0.004 1"` | 4 ms é um contato muito rígido para um para-choque de plástico; o padrão do modelo é `0.02 1` |
| bola ↔ dribbler | `friction="2.5 …"`, `solref="0.004 0.5"` (subamortecido) (`models/scene_rollers.xml:21-22`) | μ = 2.5 é deliberadamente maior que 1 para a barra segurar a bola. Existe uma explicação física de retenção, ou é um ajuste forçado que deveria ser registrado como tal? |
| robô ↔ robô / parede | 0.5 / 0.6, sem fonte | ver "Not yet specified" no mapa |

**O `02` agora deu fonte para a maior parte disso. As decisões restantes são mais estreitas do que a
tabela acima sugere:**

| linha | veredito do `02` | o que falta decidir |
|---|---|---|
| μ rolete ↔ piso | medido **0.25** (Williams et al., IEEE T-RA 18(3) 2002, mesa inclinável) | se adotar 0.25 com base em uma medição de 2002 cujo composto do rolete e cujo carpete não são especificados — a única medição desse tipo que existe. Adotá-lo deixa o robô limitado por atrito em 2.45 m/s² e reabre a restrição dominante do `01` |
| μ de deslizamento bola ↔ piso | **0.35 está certo** — bate com o μ_k empírico da ER-Force | nada. Manter e citar |
| desaceleração de rolamento da bola | **`μ_roll = 0.0009` está certo** — implica 0.293 m/s², faixa 0.26–0.31 | nada. Manter e citar. O 0.49 do rSim é um contorno do Bullet copiado |
| COR da bola | modelo tem **0**, faixa medida **0.47–0.56** | a razão de amortecimento do `solref` que dá COR ≈ 0.50, e se bola↔parede recebe o mesmo |
| inércia da bola | modelo tem **β = 0.4** (padrão de esfera sólida do MuJoCo, verificado), medido **0.5** | se declarar `diaginertia` explicitamente em `models/ball.xml:7`. Um atributo; afeta transferência de spin, retenção de backspin e todo chute |
| `solref` de paredes, para-choques | sem fonte | sem mudança — continua sendo uma decisão de julgamento |
| μ bola ↔ dribbler = 2.5 | sem fonte | sem mudança — decidir se é uma explicação física de retenção ou um ajuste forçado registrado. A §3.2.4 do livro de regras limita a cobertura da bola a 20 %, o que é uma restrição real sobre o mecanismo |
| robô ↔ robô / parede | sem fonte | sem mudança |

Decidir, por linha: o valor, a classe da fonte (medido / livro de regras / fabricante /
**aproximação deliberada, registrada**), e se é um `<pair>` ou fica para a mistura de geoms — o `C9`
nota que a maioria dos contatos bola/robô hoje são médias emergentes, só os dois pares do dribbler são
explícitos.

Atenção ao acoplamento: baixar o μ do rolete baixa o teto de aceleração, o que interage com o `04`;
aumentar o atrito bola↔piso e adicionar restituição mudam todo chute, passe e drible, e é por isso que
o `09` espera por este. O momento de inércia da bola também pertence aqui, mesmo sendo uma propriedade
do corpo e não do contato: é a mesma decisão — um valor medido substituindo um padrão nunca examinado.
`impratio` (hoje 1) também pertence aqui — aumentá-lo torna o atrito mais duro e o escorregamento
menos abrupto.

Restrição dos critérios de aceitação: a duração da posse em `Dribbling-v0` não pode piorar mais de 5%
em relação a `docs/index.md:199-212`. Uma mudança de contato que conserta o tombamento arruinando a
retenção não é uma correção.

## Answer

Resolvido em 2026-09-10, por grilling (`mattpocock-skills:grilling`), citando ids de achados de
`docs/research/rsoccer-vs-mjsoccer.md`.

| linha | decisão | classe |
|---|---|---|
| μ rolete↔piso | 1.0 → **0.25** (Williams et al., IEEE T-RA 18(3) 2002, mesa inclinável) — reabre a restrição dominante do `01`: o robô passa a ser limitado por atrito em 2.45 m/s² em vez de limitado por torque em 2.98, e ainda passa pelo limiar de tombamento de 7.86 m/s². `C1`, `W3`, `L1`. | MEASURED |
| COR bola↔piso | 0 (criticamente amortecido) → alvo **0.50** (faixa 0.47–0.56). O MuJoCo não tem um coeficiente de restituição literal — o quique emerge da razão de amortecimento do `solref` (confirmado no MuJoCo 3.1.1 instalado; `docs/research/02-physical-constants.md:94-95`) — então a dampratio real do `solref` é encontrada empiricamente, não derivada aqui. `C2`. | MEASURED (alvo); dampratio via novo ticket de tarefa |
| COR bola↔parede | mesmo alvo **0.50**, mesma calibração | MEASURED (alvo); mesmo ticket |
| inércia da bola | β 0.4 (padrão de esfera sólida do MuJoCo) → **declarar `diaginertia` explicitamente** em `models/ball.xml:7` para β=0.5. Sem id de auditoria — uma lacuna na própria auditoria, não uma rederivação; o documento de pesquisa do `02` já sinaliza isso como "not fixable without an explicit inertia declaration" (`research/02-physical-constants.md:453`). | MEASURED |
| `solref` de paredes/para-choques | sem mudança (`0.004 1`) — nem esse nem o padrão `0.02 1` do MuJoCo têm fonte. Precedente mais próximo `C8` (Low): a complacência por par é "comparable in spirit, not in value" mesmo no upstream. | DELIBERATE APPROXIMATION |
| μ bola↔dribbler = 2.5 | sem mudança — sem explicação física de retenção, deliberadamente maior que 1 para a barra segurar (`C4`). O valor fica; o mecanismo em si é escopo do `08`, e a restrição de cobertura da §3.2.4 é acompanhada separadamente no `map.md`. | DELIBERATE APPROXIMATION |
| μ robô↔robô / parede (0.5 / 0.6) | sem mudança. Sem id de auditoria — a própria tabela da auditoria lista os dois valores sem sinalizar nenhum como achado (o do rSim também é só um padrão `PSurface` nunca examinado). O valor fica em aberto para um ticket futuro. | DELIBERATE APPROXIMATION |
| `impratio` | sem mudança (1) — um ajuste do solver do MuJoCo sem equivalente no ODE/rSim, então não poderia existir id de achado nem fonte física. Revisitar só se o harness de validação do `14` mostrar que as mudanças de μ do rolete/COR acima tornam o escorregamento abrupto demais frente ao critério de duração de posse do Dribbling-v0. | DELIBERATE APPROXIMATION |

**Levado adiante.** O `04` herda como fato a conclusão de limitado por atrito (não por torque). Um novo
ticket de tarefa, [`17-ball-restitution-calibration.md`](17-ball-restitution-calibration.md), fica
responsável por encontrar a dampratio do `solref` que atinge a faixa de COR nas duas superfícies.
Todas as edições aqui continuam condicionadas à verificação de duração de posse do Dribbling-v0 do
`14` antes de entrar.
