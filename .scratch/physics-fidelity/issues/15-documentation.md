# Documentação: registrar os números e corrigir o desvio

Type: task
Status: open
Blocked by: 04, 05, 08, 10, 11, 12, 13

## Pergunta

Último ticket. Nada a decidir; escrever o que o mapa resolveu, e corrigir o que já está errado.

**Desvio existente, independente de todos os outros tickets:**
1. As seis páginas de ambiente dizem que o step de tombamento paga `-10.0`
   (`docs/environments/go_to_ball.md:136` e a mesma linha nas outras cinco). A constante era 10.0
   quando o `3e19dc4` entrou e hoje é **100.0**, mudada no `cd495d7`.
2. `docs/index.md` nunca menciona tombamento: ausente da coluna "terminates on" do índice de ambientes
   (`docs/index.md:37-47`) e das seções Timing e Observation. Adicionar em `docs/index.md:46`.
3. `docs/index.md:135-137` afirma que os valores de observação "sit in roughly [−1, 1]" e atribui as
   excursões a "a tier-1 chassis pushed by another robot". Falso hoje: o próprio acionamento do tier 1
   ultrapassa `MAX_V` e `MAX_W` (`L9`). Se a frase passa a ser verdadeira ou é corrigida depende do
   `11`.
4. A matriz roda-chassi do tier 1 **nunca é declarada em lugar nenhum** — `docs/index.md:69-77` dá só
   o `ctrlrange`. Adicioná-la, com o `r_eff = 30.089` mm e o `R = 0.0796` m medidos, os ângulos das
   rodas 60/135/225/300°, e uma nota de que a convenção de sinais é o espelho da do rSim (`W2`), então
   um vetor de rodas portado anda para trás.

**Material novo deste mapa:** o envelope derivado do drivetrain e a aceleração que ele implica (`01`,
`04`) declarados junto da tabela de ações, como `docs/index.md:79-82` promete; os coeficientes de
contato e a classe de suas fontes (`05`); a semântica da ação do dribbler (`08`); o tratamento de
capotamento e o que o `info` reporta (`10`); os normalizadores de observação e a referência da altura
da bola (`11`); o envelope igualado do tier 0 e o que ele *não* iguala (`12`); a justificativa do
timestep e a taxa de controle real (`13`).

**Correções devidas aos próprios documentos de auditoria** (encontradas ao resolver o `01`):
- O `W3` de `docs/research/rsoccer-vs-mjsoccer.md` e a tabela de limites da §7 dão o teto de
  aceleração do rSim como 4.67 m/s². Isso supõe que as quatro rodas projetam força unitária; a
  geometria 60/135/225/300° limita a projeção a `Σ|sin α| = 3.14626` frontal e `Σ|cos α| = 2.41421`
  lateral. O valor real é **3.68 m/s² frontal / 2.82 lateral** — o que confirma o próprio comentário
  "reduced to 3m/s2 accel" do rSim no eixo fraco. A conclusão do `W3` não é afetada.
- Os mesmos fatores de projeção se aplicam a qualquer declaração futura da aceleração *deste* robô,
  então declará-los uma vez junto da matriz de rodas em vez de a cada número.
- O `B9` chama a bola de 0.043 kg do rSim de "a 7 % mass difference". O `02` mostra que é um **erro
  de transcrição** — o diâmetro em metros da §2.2 do livro de regras colado no campo de massa, com o
  `ballRadius` derivado dividindo-o por dois. Reclassificar.
- O `P4` dá a entender que a bola do mjsoccer é viva demais frente aos 0.49 m/s² do rSim. O `02`
  rastreia esse 0.49 até o contorno do Bullet `hackFactor = 1.4` da ER-Force e coloca a faixa medida
  em 0.26–0.31, então **o 0.293 do mjsoccer está certo e o rSim está errado**. Inverter o
  enquadramento.
- O `C2` apresenta a diferença de quique de forma neutra. O `02` mede o COR no carpete em 0.47–0.56,
  então o `bounce = 0.5` do rSim é a sua única constante de contato defensável e **a bola com COR
  zero aqui é a lacuna de fidelidade**. Inverter.
- A comparação de CoM de `docs/analysis/ssl-simulator-rsoccer-comparison.md` (ver as Notes do mapa).

Também decidir onde os dois documentos de auditoria vivem a longo prazo: `docs/research/` e
`docs/analysis/` estão ambos fora do controle de versão hoje. Eles são a justificativa de todo número
deste mapa, então deveriam ser commitados, e a comparação de CoM do documento de análise corrigida
(ver as Notes do mapa).

## Comments

- 2026-10-17, do `07`: citar **193 geoms de colisão por robô** (299 no total com os visuais), não 209
  nem 315 — 16 cascos enterrados foram podados. `docs/research/rsoccer-vs-mjsoccer.md:203,215,352`
  dizem 209; o `B6` deve dizer que quem desvia a bola é o cubo da roda, não o rolete.
- 2026-10-17, do `08`: o dribbler é `314 * max(a, 0)` — `a ≤ 0` desligado, ação zero não dribla.
  314 rad/s é suposição, não setpoint do firmware; documentar como tal. `docs/index.md` já atualizado.
- 2026-10-18, do `09`: chute/chip pedem velocidade, `6.5 * a` m/s, meia faixa; `ctrlrange` 0..500 N;
  substeps no step de disparo; bloqueio de 250 ms é aproximação deliberada. `docs/index.md` e as
  seis páginas de ambiente já atualizadas. Fora do `09`: as tabelas de ação das páginas de ambiente
  ainda dão as rodas como ±200 rad/s (é ±95 desde o `04`); e
  `docs/research/rsoccer-vs-mjsoccer.md:66,524,525,558` ainda diz pulso de 200 N e "6-7 m/s
  emergente (`base.py:35`)".
