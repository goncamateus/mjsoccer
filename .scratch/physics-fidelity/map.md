# Mapa: fidelidade física do simulador SSL mjsoccer

## Destino

Uma **especificação de correções** resolvida e ordenada: todo parâmetro do MuJoCo com efeito relevante
no comportamento em `models/*.xml` e toda constante de atuação/observação em `src/mjsoccer/` ou
justificada por hardware/primeiros princípios, ou explicitamente registrada como aproximação
deliberada — com os critérios de aceitação que comprovam isso. O mapa está pronto quando não sobrar
nada a decidir e alguém puder executar as edições sem reabrir nenhuma pergunta.

Chegar lá **não** significa paridade com o rSoccer. Significa que os números são defensáveis.

## Notes

**Domínio.** Robô omni da Divisão A da RoboCup SSL em MuJoCo (`mjsoccer`), 6 tarefas Gymnasium, dois
tiers de física. Auditorias de referência, ambas citadas na fonte, validadas uma contra a outra:
- `docs/research/rsoccer-vs-mjsoccer.md` — 80 diferenças vs `robocin/rSoccer` + `robocin/rSim`, ids `P*`/`U*`/`B*`/`M*`/`C*`/`W*`/`L*`/`O*`/`T*`/`R*`/`X*`/`TO*`. Cite esses ids.
- `docs/analysis/ssl-simulator-rsoccer-comparison.md` — auditoria independente. Correta, exceto pela
  comparação de CoM, que omite o degrau de 20 mm do piso (`models/field.xml:42`, plano do piso em
  z=0.02): o CoM do mjsoccer fica 53.99 mm acima do carpete contra 72.25 mm do rSim, ou seja,
  **18.3 mm mais baixo**, não 2 mm mais alto. Os dois achados dela que a outra auditoria perdeu são
  reais e verificados: o mapa de ação afim do dribbler, e o dribbler com trava do rSim.
- Clones upstream para citação: `/tmp/rsim-upstream` (`b413932`), `/tmp/rsoccer-upstream` (`3adf7c3`).

**Skills que toda sessão deve consultar.** `mattpocock-skills:grilling` e
`mattpocock-skills:domain-modeling` para os tickets `grilling`; `mattpocock-skills:research` para os
tickets `research`. CodeGraph (`codegraph_*`) antes de ler código-fonte à mão.

**Restrições permanentes, resolvidas durante o mapeamento (2026-09-08).**
1. **O destino é fidelidade ao hardware, não paridade com o rSoccer.** rSoccer/rSim é só uma
   referência diferencial. Dos números do rSim, só dois têm fonte física —
   `wheelMotorMaxTorque = 0.070` e `wheelMotorMaxRPM = 1557` (`sslconfig.h:159-160`, "Maxon
   EC45 flat 50w with 18:60 gear ratio, reduced to 3m/s2 accel"). Todo o resto do rSim é contorno do
   ODE e não é alvo.
2. **Envelope do motor: Maxon EC 45 flat 50 W, acionamento direto.** ~~redução 18:60~~ —
   **substituído pelo `01`**, que afinal encontrou uma especificação real do drivetrain. O ETDP 2024
   da RobôCIn, §1.2, registra a mudança para acionamento direto em uma roda de 62 mm a ~10 % do torque
   de stall, e as malhas de roda deste CAD se chamam `Direct_Transmission_*` com `r_eff = 30.089` mm =
   uma roda de 60.18 mm. Os 0.070 N·m e 1557 rpm do rSim descrevem a geração *anterior, com redução*.
   A resposta do mapeamento escolheu 18:60 só porque se achava que não existia especificação; a
   especificação existe, então esta é a atualização que aquela resposta pedia em primeiro lugar.
   Reversível — é só dizer e o `01` rederiva com 18:60.
3. **Políticas já treinadas ficam invalidadas.** Sem tier de compatibilidade, sem segunda física de
   tier 1. Os checkpoints de `models/` e `runs/` são retreinados.
4. **O tier 0 é mantido** como tier de depuração com cinemática ideal e recebe o *mesmo envelope de
   velocidade/aceleração* do tier 1 (não o mesmo mecanismo), para que uma cabeça de ação continue
   significando uma coisa só.

**Mode.** Planejamento. Os tickets produzem decisões; as edições acontecem depois que o mapa fechar.
As exceções são os tickets `task`, que rodam medições — esses podem mexer em `models/` só como
diagnóstico revertido, e precisam reverter.

## Decisions so far

<!-- uma linha por ticket fechado -->

- [Constantes físicas](issues/02-physical-constants.md): massa da bola 0.046 kg e raio 0.0215 m são
  fixados pelo livro de regras e já estão certos; μ bola↔piso 0.35 e `μ_roll` 0.0009 estão corretos
  pela medição e só não tinham justificativa; a bola precisa de **COR ≈ 0.50** (não tem nenhum) e
  **β = 0.5** (tem o padrão 0.4 do MuJoCo); μ rolete↔piso deve ser **0.25**, não 1.0; chutes são
  limitados a **6.5 m/s** pela §8.4.2. A massa da bola do rSim é um erro de digitação, o arrasto da
  bola é um hack do Bullet copiado, o bounce é o seu único número certo. Desenvolvimento em
  [`research/02-physical-constants.md`](research/02-physical-constants.md).
- [Envelope do drivetrain](issues/01-drivetrain-envelope.md): `forcerange="-0.078 0.078"` N·m,
  `ctrlrange="-95 95"` rad/s → `a_max` 2.98 m/s², `v_max` 3.30 m/s, `ω_max` 35.9 rad/s; passa pelo
  limiar estático de tombamento com 62 % de folga e por `ω_tip` com 25 %. **O robô é de acionamento
  direto, não 18:60** — então os números do rSim descrevem a geração anterior do robô e nunca se
  transferiram. Desenvolvimento em
  [`research/01-drivetrain-envelope.md`](research/01-drivetrain-envelope.md).
- [Parâmetros de contato](issues/05-contact-parameters.md): μ rolete↔piso 1.0 → **0.25** (reabre a
  restrição dominante do `01`: limitado por atrito em 2.45 m/s², não por torque em 2.98, e ainda passa
  pelo limiar de tombamento); alvo de COR da bola **0.50** no piso e na parede, dampratio do `solref`
  adiada para a nova tarefa `17`; a inércia da bola ganha um `diaginertia` explícito para **β = 0.5**.
  `solref` de paredes/para-choques, μ bola↔dribbler = 2.5 e μ robô↔robô/parede ficam inalterados,
  registrados como aproximações deliberadas. `impratio` fica em 1, revisitar só se o `14` encontrar
  problemas de escorregamento.
- [Diagnóstico CoM/giro](issues/03-com-spin-diagnostic.md): o deslocamento de 3.459 mm baixa o
  limiar de giro (CAD anda na borda a partir de ~54 rad/s; CoM centralizado, ~62), mas **não é
  necessário** — o modo existe com o CoM no eixo; zerar `ipos` x/y nem centraliza o CoM (vira 3.28 mm
  para +x). Linha de base com ação aleatória, tier 1: **35.9 tombamentos/min**, inclinação p50/p95/p99
  12.3/44.5/59.8° (p99 censurado pelo término a 60°), yaw ≤ 34 rad/s — outro mecanismo, não o giro.
  Números em
  [`baseline/`](baseline/).
- [Enrolamento do motor](issues/16-motor-winding.md): **251601** (24 V, `k_M` 33.5 mN·m/A),
  confirmado pela equipe. A suposição do `01` vira fato; nenhum número muda.
- [Atuadores das rodas](issues/04-wheel-actuator-decision.md): `kv="0.103696" ctrlrange="-95 95"
  forcerange="-0.078 0.078"`, **sem clip em Python**. Contra o `03`, com μ 0.25: yaw de comando cheio
  29.2 rad/s (< `MAX_W`), inclinação 0.55°; ação aleatória **0 tombamentos/min** (era 35.9),
  inclinação p50/p95/p99 0.59/3.04/4.06°. O tombamento da linha de base era autoridade de torque, não
  giro. `kv` fica como aproximação da malha de velocidade do firmware. Achado lateral: as rodas perdem
  contato com o piso ao dirigir (`v` real ~1.7-2 m/s de 3.30 em x, 0.17 de 4.04 em y) → novo [`18`](issues/18-wheel-contact-loss.md).
  Números em [`candidate-04/`](candidate-04/).

## Not yet specified

- **Lacunas sim-to-real que nenhum dos simuladores trata** (`O6`, `R6`): sem ruído de observação, sem
  latência de atuação ou de visão, sem randomização de domínio — e `scene.py:187-189` nota que o
  `MjModel` em `lru_cache` bloqueia a randomização até o cache ser descartado. Dentro do escopo de
  "parâmetros defensáveis", mas não formulável como decisão até o envelope determinístico estar
  resolvido.
- **Fidelidade de escorregamento por roda.** Se algo a jusante precisa de forças normais e orçamentos
  de atrito físicos por roda. Decide-se sozinho quando o `07` fixar a topologia de colisão; só importa
  se uma correção de capotamento baseada em restrições voltar à mesa.
- **Atrito robô↔robô e robô↔parede** (`friction="0.5"` em 209 cascos, 0.6 nas paredes). Nenhuma fonte
  para nenhum dos dois valores. O `05` registrou isso como aproximação deliberada, sem mudança, e
  estabeleceu a taxonomia de classes de fonte (RULEBOOK > MEASURED > VENDOR > DELIBERATE
  APPROXIMATION) que as outras linhas agora usam. Revisitar só se aparecer uma fonte.
- **Sobrecorrente transitória.** Um `forcerange` constante não consegue expressar um ESC que permite
  várias vezes `I_nom` por dezenas de milissegundos; um envelope de torque real conseguiria.
  Registrado pelo `01` como aproximação deliberada. Torna-se especificável se alguém precisar que a
  aceleração de arrancada esteja certa.
- **Restituição oblíqua e bola-saindo-do-robô.** O `02` encontrou só valores de configuração sem
  respaldo, e os dois valores publicados de redirecionamento diferem em 3× (TIGERs 0.2, ER-Force
  ~0.60). Nada a decidir até alguém medir ou aparecer uma fonte.
- **Conformidade da cobertura da bola pelo dribbler.** A §3.2.4 do livro de regras limita a cobertura a
  20 %, que o `02` converte em ≤10.9 mm de penetração no casco. Se este modelo está em conformidade não
  foi medido, e isso restringe a geometria do dribbler em vez de algum parâmetro — então ainda não é
  uma decisão.
- **Custo de escala.** 100 DOF e 209 geoms de colisão por robô × até 7 robôs
  (`StaticDefenders-v0`). Se a fidelidade é viável em tamanhos de partida; precisa primeiro dos
  números do `07` e do `13`.
- **A queda de 1 mm no spawn** (`X8`): impacto determinístico de 0.14 m/s a cada reset.
  Provavelmente inofensivo, não quantificado.

## Fora de escopo

O destino exclui estes itens deste esforço. Só voltam se o destino for redesenhado.

- **Adotar o campo, teste de gol, tarefas, recompensas ou horizontes do rSoccer.** Campo de 6×4 m, gol
  de 0.70 m, teste de gol 2-D `ball.x > half_len`, terminações por regra em cada env, horizontes de
  25/30/120 s (`U1`, `U2`, `X3`, `X4`, `T1`). A Divisão A fica.
- **Reproduzir o modelo de manipulação de bola do rSim.** Posse por solda com hinge (`L3`), dribbler
  booleano, chute por atribuição de velocidade (`L4`), o callback de supressão do chassi (`C5`). Todos
  são contornos do ODE; os mecanismos do mjsoccer são o lado de maior fidelidade.
- **Capotamento por teletransporte** (`TO1`, §12.3 do documento de pesquisa): descontínuo, não
  markoviano, explorável, invisível.
- **Igualar a convenção de sinais da matriz de rodas do rSim** (`W2`). A matriz do tier 1 é
  autoconsistente; a inversão só importa para portar, o que não é objetivo. Ela é *documentada*, não
  alterada.
- **Comparabilidade de retorno com resultados publicados do rSoccer.** Decorre do destino (1a).
