Type: study
Source: [`05-contact-parameters.md`](05-contact-parameters.md)

# Guia de estudo: Parâmetros de contato defensáveis em vez de ajustados a gosto

Dois dos números deste ticket estão prontos para serem digitados hoje: a aderência do rolete no piso
estava 4x alta demais (ninguém tinha medido, então ficou no padrão "alto o bastante para nunca
escorregar"), e o momento de inércia da bola está silenciosamente errado desde que o modelo foi
construído (o MuJoCo assume uma esfera sólida e uniforme a menos que você diga o contrário, e uma
bola real não é isso). Um terceiro número — o quanto a bola quica — recebeu um *alvo*, não um valor,
porque o MuJoCo não tem uma configuração literal de "coeficiente de restituição"; esse vira um
projeto prático próprio no ticket `17`. Este guia cobre os dois que você pode de fato editar agora,
mais o suficiente do modelo de contato do MuJoCo para que o "porquê" de cada um não seja uma caixa
preta.

## Antes de mexer em qualquer coisa

- **A 5-tupla `friction` e o `condim`.** O atrito de uma geom no MuJoCo não é um número só: é
  `(sliding, sliding, torsional, rolling, rolling)`, e quantos deles de fato se aplicam é definido
  pelo `condim` do contato (1 = sem atrito, 3 = só deslizamento, 4 = + torcional, 6 = + rolamento).
  O contato rolete↔piso aqui usa `condim="4"` — deslizamento e torcional, sem resistência ao
  rolamento modelada nesse contato — então mudar o *primeiro* número é o que de fato muda o orçamento
  de aderência. [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **Como os parâmetros de contato de duas geoms se combinam quando não há `<pair>` explícito.** Este
  modelo depende da mistura padrão para os contatos robô↔bola e robô↔robô (só os dois pares do
  dribbler e bola↔piso são elementos `<pair>` explícitos em `models/scene_rollers.xml`) — o MuJoCo
  usa o *máximo elemento a elemento* dos valores de `friction` das duas geoms e uma média ponderada
  por `solmix` do `solref`, não uma média de tudo. [Modeling — MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **Os dois números do `solref` são uma constante de tempo e uma razão de amortecimento, não uma
  rigidez e uma restituição.** `dampratio = 1` é criticamente amortecido (o que este modelo usa quase
  em todo lugar, incluindo o contato do rolete que você vai editar) — ele assenta no contato sem
  ricochete. Restituição/quique é o que acontece quando você vai *abaixo* de 1 (subamortecido), que é
  exatamente o controle de que a calibração do `17` vai precisar, mas não é o que você mexe neste
  ticket. [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **O fator de momento de inércia, β = I / (m·r²).** Para uma esfera sólida uniforme β = 0.4
  exatamente — é o que o MuJoCo calcula automaticamente a partir de `mass` + `size` em uma geom
  esférica sem elemento `inertial`, que é silenciosamente o que esta bola tem hoje. Uma bola real (em
  camadas, com covinhas, densidade não uniforme) tem outra distribuição de massa e, portanto, outro β;
  declarar `diaginertia` explicitamente é como você sobrescreve a suposição automática de esfera
  sólida. [Moment of inertia factor — Wikipedia](https://en.wikipedia.org/wiki/Moment_of_inertia_factor)
  (o mesmo β usado para planetas e luas — a física é idêntica, só o corpo muda)
- **Definir `diaginertia` à mão.** O MuJoCo compila a inércia de um corpo a partir das suas geoms
  automaticamente, a menos que você forneça um elemento `<inertial>` explícito com `mass` e
  `diaginertia` (ou `fullinertia`) — quando você faz isso, os seus números valem integralmente, sem
  média com o valor implícito da geom. [Auto Inertia Computation discussion — mujoco#318](https://github.com/google-deepmind/mujoco/discussions/318)

## Exemplo real

**[dm_control: `soccer_ball.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/soccer/soccer_ball.py)** —
o próprio ambiente de futebol multiagente em MuJoCo do Google DeepMind. O módulo da bola fixa no
código `_REGULATION_RADIUS = 0.117`, `_REGULATION_MASS = 0.45`, `_DEFAULT_FRICTION = (0.7, 0.05, 0.04)`
e `_DEFAULT_DAMP_RATIO = 0.4`, cada um comentado como **"FIFA regulation parameters for a size 5
ball"** — ou seja, com fonte no próprio livro de regras do esporte, exatamente o movimento que o
`02`/`05` fazem para esta bola da SSL. O `damp_ratio` é passado direto para
`solref=[0.02, damp_ratio]` com o comentário "lower implies less dampening upon contacts" — a mesma
relação dampratio-controla-o-quique em que a discussão de COR deste ticket (e a calibração do `17`)
se apoia, dita nas palavras do próprio DeepMind em vez das deste projeto.

## Passos práticos

1. Abra `models/robot_rollers.xml` e encontre as geoms de colisão dos roletes (pesquise por
   `friction="1.0 0.005 0.0001"` — são 80, uma por rolete, ex.: `col_wheel1_roller_1_c00`). Mude o
   primeiro número de cada uma de `1.0` para `0.25`. Sim, todas as 80 — é um bom momento para usar um
   localizar/substituir no projeto inteiro em vez de editar uma por uma, mas olhe algumas antes para
   saber exatamente o que está substituindo.
2. Depois carregue o modelo (o carregador de cena do próprio repositório, ou um
   `mujoco.viewer.launch` puro) e dirija o robô em linha reta com um comando de roda saturante.
   Compare a aceleração obtida com os dois números do ticket: antes desta mudança ela deve saturar
   perto de `μ·g = 9.81` m/s² (com o antigo μ=1.0 efetivo pela mistura de geoms); depois, perto de
   `2.45` m/s². Se você não vir a queda, verifique se os contatos de parede/robô ainda são tratados
   pela mistura de geoms em vez do par que você editou — a mistura usa o *máximo* do atrito das duas
   geoms, então uma geom não relacionada com valor maior pode mascarar a sua edição.
3. Abra `models/ball.xml` e olhe a única `<geom>` da bola — repare que ela tem `mass="0.046"` mas
   nenhum elemento irmão `<inertial>`, e é por isso que o MuJoCo preenche hoje o padrão de esfera
   sólida. Adicione um elemento `<inertial>` com `pos="0 0 0"`, `mass="0.046"`, e `diaginertia`
   calculado a partir de `β = 0.5` (faça a aritmética você mesmo: `I = β·m·r²` com `r = 0.0215` m,
   mesmo valor nas três entradas da diagonal, já que uma esfera não tem eixo preferido).
4. Verifique que a inércia realmente mudou: `mujoco.MjModel.from_xml_path` no modelo da bola, rode
   `mj_forward`, e leia `model.body_inertia` do corpo da bola — agora deve mostrar o seu valor, não
   `0.4·m·r²`. Depois pense no que uma inércia rotacional maior (relativa à massa) *faz* com a bola
   fisicamente — significa que é preciso mais torque para fazê-la girar, e ela retém backspin/sidespin
   por mais tempo depois que ganha algum. Esse é o comportamento que esta mudança de fato compra.

## Se você travar

`docs/research/02-physical-constants.md` tem a cadeia completa de fontes para β = 0.5 e para a medição
de μ = 0.25 do rolete (Williams et al., estudo com mesa inclinável) se o "por que 0.25 e não outra
coisa" não ficar claro só pelo ticket. Para a mecânica do MuJoCo especificamente, a aba *Discussions*
do GitHub do `google-deepmind/mujoco` (não Issues) tem os mantenedores respondendo perguntas de
contato/inércia diretamente — pesquise por `diaginertia` ou `condim` antes de abrir uma thread nova.
