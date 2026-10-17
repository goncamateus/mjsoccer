Type: study
Source: [`01-drivetrain-envelope.md`](01-drivetrain-envelope.md)

# Guia de estudo: Envelope do drivetrain — o que os motores deste robô realmente conseguem fazer?

Hoje toda roda deste robô tem `forcerange="-3 3"` N·m — torque suficiente para empurrar 399 N pelo
contato do pneu, 15x mais do que o atrito jamais deixará passar. Isso não modela um motor, modela
"torque infinito, limitado por atrito". O ticket derivou o número real a partir do datasheet de um
Maxon EC45 flat 50W e da própria geometria do robô; este guia é sobre *como* sair de um datasheet de
motor e chegar a uma tag de atuador do MuJoCo por conta própria, para você poder refazer a derivação e
editar o XML com as próprias mãos.

## Antes de mexer em qualquer coisa

- **O atuador `velocity` do MuJoCo e seus atributos `kv`/`forcerange`/`ctrlrange`** — `kv` define um
  ganho proporcional (`force = kv · (ctrl − joint_velocity)`), e `forcerange` satura o que quer que
  esse ganho calcule. São controles independentes: `ctrlrange` diz qual velocidade você pode *pedir*,
  `forcerange` diz com quanta força o servo *pode empurrar* para chegar lá. [XML Reference — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **Como `gainprm`/`biasprm` de fato produzem essa força**, se você quiser a matemática por baixo em
  vez de só os atributos de atalho. [Modeling — MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
- **O motor real**: o próprio modelo deste robô assume um Maxon EC 45 flat 50 W, part 251601 —
  sensores Hall, 24 V, 6700 rpm em vazio, 84.3 mNm de torque nominal (contínuo), 822 mNm de stall.
  Esses são os números dos quais a derivação do ticket parte; confira-os contra a sua própria leitura
  da aritmética do ticket. [Maxon EC 45 flat 251601 datasheet (PDF)](https://mat.transtechnik.fr/LINMOT/documentation/Moteurs/EC%20Motors/Maxon/Maxon%20-%20EC_45_flat_251601.pdf)
- **Orçamento de atrito**: a força horizontal máxima que um contato sustenta antes de escorregar é
  `μ · N` (atrito de Coulomb) — esse é o teto contra o qual o ticket compara o torque da roda para
  decidir se o robô é *limitado por torque* (o motor cede primeiro) ou *limitado por atrito* (o pneu
  escorrega primeiro). [Friction — Wikipedia](https://en.wikipedia.org/wiki/Friction)
- **Tombamento como equilíbrio de momentos**: um robô tomba quando o momento de tombamento (aqui, o
  deslocamento fora do eixo do CoM sob yaw) supera o momento restaurador do próprio peso em torno da
  borda de contato — a mesma ideia de estabilidade estática usada em toda a literatura de robôs com
  rodas, só que simplificada para 2D aqui porque o deslocamento do CoM é pequeno. [Tip-over stability analysis of a three-wheeled mobile robot (IISc, PDF)](https://mecheng.iisc.ac.in/~asitava/mimt2011_tharak.pdf)
- **`kv` não é livre para subir** — aumentar o ganho de um atuador de velocidade para compensar um
  `forcerange` pequeno pode desestabilizar o solver (NaN/Inf) em vez de só rastrear mais devagar.
  Saiba disso antes de mexer em `kv` no `04`. [`kv` above 1 causes instability — mujoco#184](https://github.com/google-deepmind/mujoco/issues/184)

## Exemplo real

**[mujoco_menagerie: kuka_iiwa_14 README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/kuka_iiwa_14/README.md)** —
a própria coleção curada de modelos MJCF do Google DeepMind. As notas de conversão URDF→MJCF dizem
simplesmente: *"Added forcelimits to match the torque limits in the [spec sheet]"* — ou seja, foram
ao datasheet real do fabricante e definiram `forcerange` para bater com ele, exatamente o movimento
que este ticket faz. Eles também não mostram a aritmética (a spec sheet em PDF já traz os números
diretamente), o que é normal: o passo "derivar o número do datasheet" costuma ser um cálculo pontual
que você faz uma vez e guarda, não algo que o arquivo do modelo precise rederivar toda vez que é lido.

## Passos práticos

1. Abra `docs/research/01-drivetrain-envelope.md` e encontre onde ele transforma a corrente nominal e
   a constante de torque do datasheet em um *torque nominal contínuo na roda* — essa é a derivação do
   `forcerange`. Refaça a aritmética você mesmo usando os números do datasheet acima (eles diferem
   um pouco dos do ticket — 84.3 mNm nominal / 822 mNm stall aqui vs. 83.4 / 780 no ticket —
   descubra se é uma edição diferente do datasheet ou uma escolha de arredondamento, e em qual você
   confiaria).
2. Abra `models/robot_rollers.xml` e encontre os quatro atuadores `wheel*_vel` (hoje
   `forcerange="-3 3"`, `ctrlrange="-200 200"`, os quatro idênticos). Mude o `forcerange` de uma
   roda só para o seu valor derivado e deixe as outras três em `-3 3` — carregue o modelo (por
   exemplo via `mujoco.viewer` ou o carregador de cena do próprio repositório) e observe o que um
   drivetrain assimétrico faz com um comando em linha reta. É um passo intermediário
   deliberadamente errado, só para ver a saturação do atuador em ação antes de se comprometer com a
   edição simétrica real.
3. Abra `src/mjsoccer/base.py` perto de `_drive_robot` (mais ou menos linhas 143–156) e siga como uma
   ação em `[-1, 1]` vira `data.ctrl` — repare no comentário dizendo que `ctrlrange` é o único lugar
   onde o limite pode morar, conforme `docs/index.md:79-82`. Se mais tarde você mexer em `ctrlrange`
   no `04`, esse é o caminho de código cujo contrato você estaria mudando.
4. Com as quatro rodas no seu `forcerange`/`ctrlrange` derivado, escreva um script de varredura de
   comando uniforme nas rodas (o próprio método de medição do ticket: `place` o robô na origem, rode
   alguns segundos de física sem nenhum outro corpo em contato, registre o pico de taxa de yaw e a
   inclinação do chassi em alguns comandos de velocidade de roda). Verifique se o início da sua
   inclinação cerca o `ω_tip = 47.7 rad/s` do ticket do mesmo jeito que a medição do ticket cercou
   (42.1 rad/s com comando de roda de 120 rad/s, 55.6 rad/s com 160). Se os seus números discordarem,
   esse é o resultado interessante — vá descobrir o porquê antes de assumir que o ticket está certo.

## Se você travar

Releia `docs/research/01-drivetrain-envelope.md` inteiro — ele mostra cada número intermediário, não
só o `forcerange`/`ctrlrange` final. Se o confuso for o próprio modelo de atuador do MuJoCo, a aba
GitHub Discussions do `google-deepmind/mujoco` (não Issues) é onde a maioria das perguntas do tipo
"por que meu atuador se comporta como X" é respondida diretamente pela equipe do MuJoCo — pesquise
por `velocity actuator` ou `forcerange` antes de abrir uma thread nova.
