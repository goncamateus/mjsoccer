Type: study
Source: [`17-ball-restitution-calibration.md`](17-ball-restitution-calibration.md)

# Guia de estudo: encontrar o `solref` que dá à bola o seu COR medido

Este é o ticket mais mão-na-massa dos três: ainda não há `## Answer` para explicar, porque a resposta
é um número que ninguém mediu *neste* modelo. O alvo físico está resolvido — uma bola real da SSL
rebate com cerca de metade da velocidade de impacto (COR ≈ 0.50) — mas o MuJoCo não tem uma
configuração literal de "restituição", então chegar lá significa rodar o mesmo experimento que
físicos usam em uma bola real, dentro do simulador, e ajustar o `solref` até a bola simulada bater.
É uma tarefa de calibração genuinamente aberta, e este guia é sobre o ponto de verificação que prova
que você de fato a fechou: um número entra, o rebote certo sai.

## Antes de mexer em qualquer coisa

- **O método físico de teste de queda que você está reproduzindo na simulação.**
  `e = sqrt(h_rebound / h_drop)` — porque as velocidades de impacto e de rebote são cada uma
  proporcionais à raiz quadrada das respectivas alturas, e o COR é a razão entre a velocidade de saída
  e a de entrada. É exatamente o método que a ER-Force usou para obter o valor de 0.56 de onde vem a
  faixa alvo deste ticket (ver `docs/research/02-physical-constants.md` §2.2), e é padrão o bastante
  para ter o seu próprio roteiro de laboratório de física introdutória. [Laboratory 4: Coefficient of Restitution (U. Alabama, PDF)](https://pleclair.ua.edu/ph105/Labs/old/coef_rest_lab.pdf)
- **Por que não existe uma fórmula de `solref` que simplesmente funcione.** Os dois números do
  `solref` são uma constante de tempo de referência e uma razão de amortecimento para um solver de
  restrições, não "restituição" — os próprios mantenedores do MuJoCo confirmaram que não existe
  parâmetro embutido de coeficiente de restituição nem um mapeamento exato em forma fechada para um.
  [`Coefficient of restitution` — mujoco#2081](https://github.com/google-deepmind/mujoco/discussions/2081)
- **Uma aproximação de ponto de partida, para você não varrer às cegas.** A mesma discussão acima dá
  uma aproximação de mola amortecida: fixe uma rigidez `k` (o exemplo deles usa `k = 2500`), depois
  defina o amortecimento `b = -2·sqrt(k·m_eff)·ln(ce)/π` para uma restituição alvo `ce`, onde `m_eff`
  é a média harmônica das duas massas em colisão (para bola vs. um piso/parede efetivamente infinito,
  `m_eff` → a própria massa da bola). O autor chama isso de "accurate enough" nos próprios testes, não
  exato — e é exatamente por isso que este ticket pede para você *verificar* com um teste de queda em
  vez de confiar na fórmula de cara.
- **Ler a velocidade de um corpo na API Python**, já que o seu harness de teste de queda precisa medir
  a velocidade de impacto e de rebote, não só olhar um viewer. `data.qvel` guarda o vetor de
  velocidade completo; para um corpo com junta livre como a bola, a parte linear são as três primeiras
  entradas da sua fatia. [Python bindings — MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/python.html)

## Exemplo real

**[`Coefficient of restitution` — google-deepmind/mujoco Discussion #2081](https://github.com/google-deepmind/mujoco/discussions/2081)** —
alguém esbarrou exatamente neste problema (querer um COR específico de um contato do MuJoCo) e recebeu
uma resposta da comunidade com números que funcionam: `k = -solref[0]`,
`b = -2·sqrt(k·m_eff)·ln(ce)/π`, com a ressalva *"you need to consider inertia matrices and jacobians
for the colliding bodies if there are joints on any of the bodies or if they are not point masses."*
Essa ressalva importa aqui: a bola é um corpo livre (sem juntas, efetivamente uma massa pontual para
este propósito), mas o *piso* é um plano fixo e a *parede* é uma coleção de malhas que teria as suas
próprias considerações de inércia se não fosse estática — vale entender por que as suposições da
fórmula valem ou não para cada uma das suas duas superfícies alvo antes de confiar no mesmo par `k`/`b`
nas duas.

## Passos práticos

1. Escreva um script autônomo pequeno (não precisa mexer no maquinário de env Gymnasium deste
   repositório — um laço puro de `mujoco.MjModel`/`mujoco.MjData` sobre `models/ball.xml` mais um
   plano de piso basta) que coloque a bola a uma altura conhecida acima do piso de `models/field.xml`
   (`z = 0.02` mais a sua altura de queda), velocidade zero, e avance a física até detectar um máximo
   local de altura depois do primeiro quique.
2. Calcule o COR medido a partir das alturas de queda e de rebote registradas usando a fórmula
   `sqrt(h/H)` acima. Rode primeiro uma vez contra o modelo *atual* (`solref="0.006 1"`, criticamente
   amortecido) — você deve medir COR ≈ 0, confirmando que o seu harness de fato detecta um quique
   antes de sair procurando um que não existe.
3. Abra `models/ball.xml` e encontre o `solref="0.006 1"` da geom da bola. Teste a fórmula de
   aproximação da discussão para um alvo `ce = 0.50` para obter um primeiro par candidato
   `(timeconst, dampratio)` — note que o `solref` do MuJoCo não é literalmente `(k, b)`, é
   `(timeconst, dampratio)`, então você vai precisar converter, ou usar a convenção documentada de
   `solref` negativo do MuJoCo, em que os dois números *são* rigidez/amortecimento diretos em vez de
   timeconst/dampratio (confira na referência XML qual convenção de sinal a sua versão do MuJoCo
   espera).
4. Varra em torno desse candidato, rodando o harness de novo a cada vez, até o COR medido cair em
   0.47–0.56 no piso. Depois aponte o mesmo harness para um chute horizontal em direção às geoms de
   parede de `models/field.xml` (`friction="0.6 …"`, `solref="0.004 1"` hoje) e verifique se o seu
   valor do piso também cai na faixa ali, ou se a parede precisa do próprio par — essa é a pergunta
   em aberto do ticket sobre se um único valor serve às duas superfícies.
5. Ponto de verificação: você terminou a metade de medição deste ticket quando conseguir reportar,
   para as duas superfícies, o par `solref` usado e o COR medido que ele produziu, com o seu script de
   harness ainda executável (não números pontuais que você olhou no olho e jogou fora) — essa
   repetibilidade é o que o critério de aceitação do ticket está pedindo.

## Se você travar

`docs/research/02-physical-constants.md` §2 tem o quadro físico completo (por que 0.47–0.56, e a
ressalva de dependência da velocidade — o COR cai conforme a velocidade de impacto sobe, então anote em
que altura/velocidade de queda você calibrou). Se o ponto de travamento for o lado do MuJoCo, pesquise
na aba *Discussions* do GitHub do `google-deepmind/mujoco` por `solref` ou `restitution` — várias
pessoas esbarraram nessa mesma parede de "o MuJoCo não tem ajuste de restituição" e as respostas dos
mantenedores ali são mais atuais do que qualquer coisa escrita de memória.
