# Spec: backend mjlab do mjsoccer

Status: ready-for-agent
Origem: [decisões do grilling](decisions.md) (itens 1-16, fechados em 2026-11-14),
[pesquisa](research/01-mjlab-fit.md), [resultados do protótipo](research/02-prototype-results.md)
(branch `prototype/mjlab`, `45b25d5`, fonte primária, não mergeado).
ADRs: [0002 — só MJX-Warp](../../docs/adr/0002-mjx-warp-only.md), a ser substituída pela 0003
(ticket deste spec). Spec anterior: [port GPU](../gpu-port/spec.md). Vocabulário:
[`CONTEXT.md`](../../CONTEXT.md).

## Problem Statement

O mjsoccer treina na GPU só pelo MJX-Warp com Brax PPO, em JAX. Isso deixa de fora o ecossistema
torch que o mjlab traz pronto: domain randomization por evento na GPU, ruído e atraso de
observação nativos, crítico assimétrico, export ONNX de política e o runner RSL-RL. Hoje uma
política treinada na GPU só volta para o backend CPU e para o viewer pelo checkpoint do Brax. O
pesquisador quer treinar também no mjlab, sem perder o MJX-Warp e sem afrouxar a fidelidade
física garantida pelo oráculo CPU.

O protótipo mostrou que o custo de desempenho não é obstáculo. Na ContestedPossession com kick, o
mjlab empata com o MJX-Warp (16 588 contra 16 550 decisões/s a 4 096 mundos, ambos no mujoco 4.0.1)
e vai a 16 384 mundos, contra o OOM do bench MJX a 8 192. O overhead dos managers fica em 6-7 %.

## Solution

O mjlab entra como **terceiro backend**. CPU, MJX-Warp e mjlab coexistem, com um catálogo só e um
contrato de env só:

1. **Subir o `main` para mujoco 4.0.1** (CPU, `mujoco-mjx[warp]` e o novo grupo `mjlab`) e rodar
   de novo a paridade CPU↔MJX nessa versão.
2. **Escrever a lógica das seis tasks em torch**, como funções puras sobre tensores do estado
   físico, presas à CPU por um teste de task-logic parity, como o MJX já é.
3. **Expor `mjsoccer.mjlab.make(id)`** sobre o `ManagerBasedRlEnv` do mjlab, no modelo base exato,
   com os mesmos ids, horizontes, espaços e layout de obs e ação dos outros backends.
4. **Passar o mjlab pelos mesmos critérios e limiares de paridade** do MJX-Warp, com
   `--backend mjlab` no sweep de validação.
5. **Treinar com RSL-RL** num script com as mesmas flags do `ppo_mjx`, exportar a política em ONNX
   e assisti-la no backend CPU, no viewer Studio.

O usuário treina com `uv run --group mjlab <script de treino mjlab> --env <id>` e assiste com
`--play <dir>`.

## User Stories

### Pesquisador treinando políticas

1. Como pesquisador, quero treinar PPO nas seis tasks no mjlab com um script só (`--env <id>`), para não manter um script por task.
2. Como pesquisador, quero que o script do mjlab aceite as mesmas flags do `ppo_mjx` (`--env`, `--seed`, `--save`, `--play`, `--eval`, `--render`, `--repeat-action`, `--num-envs`, `--obs-noise`, `--action-latency`, `--epochs`/`--timesteps`, `--num-eval-envs`, `--eval-randomize`), para trocar de backend sem reaprender a CLI.
3. Como pesquisador, quero que os hiperparâmetros PPO do script do mjlab partam dos defaults do `ppo_mjx` (learning rate, entropia, desconto, GAE, clipping, unroll, minibatches, updates por batch), para a comparação RSL-RL × Brax começar do mesmo ponto.
4. Como pesquisador, quero que o `--num-envs` default seja o maior `nworld` estável medido na minha GPU para cada cena (2 e 7 robôs), para não descobrir o lote por tentativa e erro.
5. Como pesquisador, quero sobrescrever o `--num-envs`, para experimentar lotes até 16 384 mundos.
6. Como pesquisador, quero ver no log decisões/s, retorno de eval e as taxas de `diverged`, `overflow` e `reset_overlap`, para saber se o treino está saudável.
7. Como pesquisador, quero que um mundo que diverge (NaN/inf) ou estoura o buffer do Warp termine com recompensa 0 e resete sem derrubar o treino, para um mundo ruim não matar um run de horas.
8. Como pesquisador, quero que só os mundos com bit de overflow terminem num lote misto, para um mundo ruim não resetar os bons.
9. Como pesquisador, quero que o eval standalone aborte com mensagem clara (bits de overflow por tipo) se algum mundo divergir ou estourar, para não confiar em número corrompido.
10. Como pesquisador, quero que o checkpoint saia num diretório de run com a config e a política exportada em ONNX, para reproduzir e assistir depois.
11. Como pesquisador, quero `--play <dir>` rodando a política ONNX no backend CPU, no viewer Studio, para ver a política com a física do oráculo.
12. Como pesquisador, quero `--eval <dir>` rodando a política no mjlab em lote e reportando retorno e taxas, para avaliar sem treinar.
13. Como pesquisador, quero que o primeiro estado de cada mundo seja sorteado de um pool de estados já assentados, refeito a cada época, para ter diversidade de starts igual à do MJX.
14. Como pesquisador, quero comparar RSL-RL e Brax PPO por retorno de eval, com o mesmo orçamento de decisões e seeds 0-4, para saber se o treino no mjlab é tão bom quanto no MJX.

### Pesquisador usando domain randomization, ruído e latência

15. Como pesquisador, quero passar ao `mjsoccer.mjlab.make` os mesmos parâmetros e ranges de DR do MJX, para treinar com a mesma distribuição de física nos dois backends.
16. Como pesquisador, quero que a DR do mjlab sorteie por mundo, na mesma granularidade do MJX, para os dois backends serem comparáveis.
17. Como pesquisador, quero o ruído de observação N(0, σ²) com a mesma semântica do `--obs-noise` do MJX (mesma seed dá o mesmo ruído e a primeira obs pós-reset também é ruidosa), para a flag significar a mesma coisa nos dois backends.
18. Como pesquisador, quero a latência de ação contada em passos de controle, aplicada à decisão (target pose) antes do controlador, como no MJX, para a flag `--action-latency` significar a mesma coisa nos dois backends.
19. Como pesquisador, quero `--eval-randomize` também no mjlab, para avaliar com e sem DR.

### Usuário das tasks (catálogo e contrato)

20. Como usuário, quero `mjsoccer.mjlab.make(id, **kw)` lendo o mesmo `ENV_IDS`, com ids e horizontes iguais, para os três backends terem um catálogo só.
21. Como usuário, quero os mesmos espaços de observação e ação e o mesmo layout de obs nos três backends, para uma política treinada num rodar nos outros.
22. Como usuário, quero a ação de 7 dims (target pose + kick/chip/dribble) com o controlador de navegação fixo a 40 Hz, como nos outros backends, para o mjlab não mudar o problema de controle.
23. Como usuário, quero `repeat_action` com a mesma semântica dos outros backends, para o horizonte em decisões ser o mesmo.
24. Como usuário, quero que `import mjsoccer` nunca importe o mjlab nem o torch, para o pacote base continuar leve.
25. Como usuário, quero um erro claro ao fazer `mjsoccer.mjlab.make` sem o grupo `mjlab` instalado ou sem CUDA, para saber o que falta.
26. Como usuário, quero que um id ainda não portado ao mjlab dê `NotImplementedError`, como no MJX, para o port ser incremental.

### Pesquisador de física (oráculo e paridade)

27. Como pesquisador de física, quero que o backend CPU continue sendo o oráculo da física, para todo backend de GPU ter contra o que provar fidelidade.
28. Como pesquisador de física, quero rodar de novo a paridade CPU↔MJX no mujoco 4.0.1 antes de qualquer código mjlab, para saber que a subida de versão não mexeu na física.
29. Como pesquisador de física, quero que o mjlab rode o modelo base exato, com a cena inteira vinda do `scene.build`, os mesmos nomes do CPU e o `<option>` do `scene.xml` (timestep 1/960, `cone="elliptic"`, Newton, `implicitfast`), para a paridade comparar a mesma física.
30. Como pesquisador de física, quero que o mjlab passe cada critério de aceitação com os mesmos limiares do CPU (envelope, kick/chip × `KICK_TABLE_MS`, COR e inércia da bola, posse inclusive no giro de 2 rad/s, `spin`, `kick_recoil`, critérios do controlador), para o terceiro backend não afrouxar a fidelidade.
31. Como pesquisador de física, quero paridade determinística mjlab × CPU (≤ 2 % em envelope, kick e COR; ≤ 5 % no tempo de chegada; ≤ 1 cm / 1° no erro de chegada), com o mesmo estado inicial escrito em `qpos`/`qvel`.
32. Como pesquisador de física, quero paridade com seed mjlab × CPU (mediana mjlab dentro do [p25, p75] da CPU, seeds 0-4) para `random`, `competent`, taxa de posse e retorno.
33. Como pesquisador de física, quero o `hold_turn` (giro de 2 rad/s, mediana de 64 cópias perturbadas) também no mjlab, para o caso caótico do ticket 16 ter o mesmo juiz.
34. Como pesquisador de física, quero rodar `scripts/validation_sweep.py --backend mjlab`, com a coluna `backend` nos CSVs, para comparar os três backends no mesmo formato.
35. Como pesquisador de física, quero que o passo de disparo do kick use o mesmo esquema de 48 iterações com `opt.timestep` por mundo do MJX, para o kick ter a mesma física nos dois backends de GPU.
36. Como pesquisador de física, quero que o mundo ocioso fique bit a bit nas iterações a `timestep = 0`, para o passo de disparo não mexer em quem não disparou.
37. Como pesquisador de física, quero saber se a observação do mjlab vê o estado no mesmo instante que a do CPU, para a obs mais fresca do mjlab (`forward` antes da obs) não virar diferença escondida.
38. Como pesquisador de física, quero que kick e drible continuem só físicos no mjlab (sem attach, snap ou joint de solda), para nenhum backend trapacear a física.

### Mantenedor do código

39. Como mantenedor, quero a lógica das tasks em torch como funções puras sobre tensores do estado (qpos, qvel, sensordata, xpos e afins) e do estado por mundo, sem depender do sim do mjlab, para testá-la em torch na CPU, sem GPU.
40. Como mantenedor, quero os termos de manager do mjlab como cola fina que só chama essas funções, para a lógica ter um lugar só.
41. Como mantenedor, quero que toda constante e config de task seja lida do lado CPU, como o MJX faz, para os três backends não divergirem por cópia.
42. Como mantenedor, quero um task-logic parity torch × CPU nas 6 tasks, seeds 0-4, estado por estado, para segurar a terceira implementação da lógica.
43. Como mantenedor, quero um smoke do mjlab nas 6 tasks que pula sozinho sem CUDA ou sem o grupo `mjlab`, para `uv run pytest` rodar em qualquer máquina.
44. Como mantenedor, quero o grupo opcional `mjlab` (mjlab pinado no git, torch ≥ 2.14 cu130, onnxruntime) separado do `gpu`, resolvendo no mesmo `uv lock`, para a GPU torch não entrar na base.
45. Como mantenedor, quero saber que o lock universal sobe também o torch do grupo `rl` (SB3), e os testes do `rl` continuam passando, para o efeito colateral não pegar ninguém de surpresa.
46. Como mantenedor, quero a decisão de stream torch × Warp tomada e documentada, para não haver corrida silenciosa entre os dois.
47. Como mantenedor, quero entender a diferença entre a memória do device (10,9 GB) e o pico do mempool do Warp (1,3 GB) a 16 384 mundos, para o `nworld` máximo não estar limitado por desperdício.
48. Como mantenedor, quero a ADR 0003 "MJX-Warp e mjlab" substituindo a 0002 e o glossário atualizado, para a decisão de três backends ficar registrada.
49. Como mantenedor, quero que o port comece pela ContestedPossession (o caminho do protótipo) e GoToBall (7 robôs), para pegar cedo o risco do #1263 (ITERATIONS) nas cenas grandes.

## Implementation Decisions

### Fase 0 — mujoco 4.0.1 no `main`

- Subir `mujoco`, `mujoco-mjx[warp]` (→ `mujoco-warp` 4.0.1, `warp-lang` 1.18.0) para 4.0.1. Rodar a
  suíte padrão, `-m validation` e a paridade CPU↔MJX (determinística, com seed e `hold_turn`). Se
  algum critério cair, parar e reportar antes de qualquer código mjlab.
- O `scene.build` se divide em construir o `MjSpec` e compilá-lo (como no protótipo), para o mjlab
  receber o spec inteiro. O CPU e o MJX continuam chamando `build` e não mudam.

### Fase 1 — Lógica das tasks em torch

- Novo subpacote `mjsoccer.mjlab`, nunca importado por `import mjsoccer`.
- A lógica é um port linha a linha do `mjsoccer.mjx` para torch: controlador de navegação (função
  pura, quirks do original verbatim), kick/chip com `KICK_TABLE_MS` e trava do plunger (cooldown
  10), posse por sensor `contact` + grace, `sample_pose` + `reset_overlap`, robôs scripted, obs,
  recompensa e término das seis tasks.
- O estado por mundo (grace, plunger lock, steps, RNG, estado da task) vive num container de
  tensores por mundo, o equivalente torch do pytree em `State.info` do MJX.
- As funções recebem tensores e devolvem tensores; não tocam o sim. É isso que deixa o task-logic
  parity rodar em torch-CPU.

### Fase 2 — Env mjlab

- **Entrada**: `mjsoccer.mjlab.make(id, *, nworld, repeat_action=1, randomize=None, **kw)` sobre o
  `ENV_IDS`. Devolve o env mjlab (`ManagerBasedRlEnv`) configurado.
- **Cena**: inteira via `SceneCfg.spec_fn` a partir do `MjSpec` do `scene.build`, sem `Entity` por
  robô. O action term aponta para uma entidade-âncora vazia, exigida pelo `ActionTermCfg` (o único
  corpo extra: `nbody` 13 contra 12 do CPU). `MujocoCfg` espelha o `<option>` do `scene.xml`.
  O aviso de conflito de `<option>` no `attach` é esperado e inofensivo (o `MujocoCfg` sobrescreve);
  silenciá-lo é opcional.
- **Passo de disparo** (definitivo; o protótipo empatou com o MJX nele):
  `decimation = 48 × repeat_action`, `timestep = 1/960`; o `opt.timestep` do modelo Warp é trocado
  por um array por mundo e o grafo é recapturado. A cada substep, a agenda por mundo é escrita
  in-place: 32 × dt/4 + 16 × dt se o mundo disparou, 24 × dt + 24 × 0 se não. O controlador roda
  uma vez por decisão. O `step_dt` que o mjlab reporta fica errado e não é usado: horizonte,
  recompensa e time-out são nossos (`scale_rewards_by_dt=False`, `episode_length_s` infinito,
  time-out por passos da task).
- **Ordem de carga**: o objeto da task existe antes dos managers (o termo de obs é dimensionado no
  load e o mjlab reseta no init), via hook em `load_managers`, como no protótipo.
- **Reset**: pool de primeiros estados assentados (`place` + o mesmo loop físico do `seat_ball`, em
  lote), do tamanho de `nworld`, gerado no `startup` e refeito a cada época pelo script de treino.
  O evento `reset` copia o slot do mundo (qpos, qvel e estado por mundo). Substitui o
  `seat_ball` por reset, que o mjlab não comporta (`sim.step` avança todos os mundos).
- **Falhas por mundo**: termination própria lendo `Data.overflow` (NEFC, BROADPHASE, ITERATIONS e
  afins) e a `nan_detection` do mjlab; ambos terminam como `terminated`, com obs e recompensa
  zeradas antes do gradiente. Métricas `diverged`, `overflow`, `reset_overlap` pelo
  `MetricsManager`. O `reset_data` mascarado do mjlab limpa só os bits dos mundos resetados
  (medido).
- **Buffers**: `nconmax` 64 e `njmax` 256 por mundo nas cenas de 2 robôs, o mesmo dimensionamento
  do MJX; as cenas de 7 robôs são dimensionadas pelos candidatos do broadphase, como no MJX, e
  medidas no ticket da GoToBall.
- **Streams**: o default é o do protótipo, tudo dentro do stream do Warp. Um ticket mede se
  rodar sem isso corrompe dados; se não corromper, a decisão fica documentada assim mesmo.
- **Observação**: a obs tem de ver o mesmo estado que a do CPU. O task-logic parity e a paridade
  com seed decidem se o `forward` antes da obs do mjlab precisa ser compensado.
- **DR, ruído e latência**: mesmos parâmetros, ranges e granularidade do MJX (sorteio por mundo).
  A DR usa os eventos do mjlab (`set_const` por grupo); `pair_solref`/`pair_solimp` dos contatos da
  bola (que são `<pair>`) pedem evento próprio. O ruído usa o `NoiseCfg` do mjlab com a semântica
  do `ObsNoiseWrapper`. A latência é da decisão, em passos de controle, no action term (a latência
  nativa do mjlab é do atuador, em passos de física, e não serve).

### Fase 3 — Treino

- Script novo, irmão do `ppo_mjx`, sobre o `MjlabOnPolicyRunner` (RSL-RL 5.5.2). Mesmas flags e
  defaults de PPO do `ppo_mjx`; o refazer do pool por época substitui o `--num-resets-per-eval`.
- Export ONNX no fim do treino e em cada checkpoint salvo. `--play` carrega o ONNX com
  onnxruntime e roda no backend CPU (Gymnasium), no viewer Studio, em processo próprio como o
  `ppo_mjx`. `--eval` roda no mjlab.
- `--num-envs` default: o maior `nworld` estável medido por cena (2 e 7 robôs) num treino longo,
  não só nos ~40 s do protótipo (que rodou 16 384 com 2 robôs).
- A comparação RSL-RL × Brax é um ticket: retorno de eval nas duas tasks de referência
  (ContestedPossession, GoToBall), mesmo orçamento de decisões, seeds 0-4. Relatório, não gate.

### Dependências

- Grupo opcional `mjlab`: mjlab pinado no git `6ab9846` (a release do PyPI v1.6.0 prende mujoco
  3.11), torch ≥ 2.14 do índice `pytorch-cu130`, onnxruntime. Resolve no mesmo `uv lock` que o
  `gpu`, sem `conflicts` (medido); torch e JAX importam no mesmo processo.
- O torch do grupo `rl` (SB3) passa a 2.14.1+cu130 pelo lock universal.

### Ordem do port

ContestedPossession primeiro (refaz o caminho do protótipo, agora com o task-logic parity), depois
GoToBall (7 robôs, risco de ITERATIONS do #1263, buffers e `nworld` próprios), depois as quatro
restantes. Decisões/s ponta a ponta registradas contra o MJX-Warp nas duas primeiras, como
relatório.

## Testing Decisions

**Um bom teste** verifica comportamento externo: o que o env devolve (obs, recompensa, término,
métricas) e o que a física faz a partir de uma ação. A paridade física é estatística, pelos
critérios de aceitação; só o task-logic parity é exato, estado por estado.

**Seams** (as mesmas três do port GPU; nenhuma nova):

1. **API pública do env**: `mjsoccer.mjlab.make`. Recebe o smoke do mjlab e o smoke do script de
   treino.
2. **Sweep de validação**: `episode(..., backend="mjlab")`. Toda checagem de critério e de paridade
   passa por ela.
3. **Snapshot `MjData` → tensores torch** para o task-logic parity. As funções da lógica recebem
   tensores puros, então o teste roda em torch-CPU, sem Warp nem GPU.

**Camadas**:

- **Suíte padrão (`uv run pytest`)**; o mjlab é pulado sem o grupo `mjlab`.
  - **Task-logic parity torch × CPU**: `importorskip("torch")`, torch na CPU. Snapshots de rollouts
    CPU com seeds 0-4; com a mesma ação, compara saída do controlador, obs, recompensa, término,
    posse e estado por mundo, na tolerância de float32. Cobre `_place` com candidatos injetados e
    `plunger_lock`/`fired`/cooldown 10. Cobre as 6 tasks e aborta se algum contador de falha for
    > 0.
  - **Smoke do mjlab**, via o runner de subprocesso de GPU do repo (pula sem CUDA): `make` + `reset`
    + ~10 passos nas 6 tasks, `nworld` pequeno, um kick disparado; `diverged == overflow == 0` e obs
    finita; mundo ocioso bit a bit nas iterações a `timestep = 0`; um lote misto com buffers
    subdimensionados em parte dos mundos onde só esses terminam.
  - **Smoke do treino**: poucas iterações, checkpoint e ONNX gravados, `--eval` e `--play`
    (headless) carregando o resultado.
- **`-m validation`**: os testes de critério, `test_parity_deterministic`, `test_parity_seeded` e o
  `hold_turn` ganham `mjlab` no parâmetro `backend`; os casos mjlab pulam sem CUDA.
- **Fora dos testes**: throughput, memória e a comparação RSL-RL × Brax são relatórios, nunca
  critério.

**Prior art**:

- o task-logic parity do MJX (snapshot → `put_data`, JAX-CPU forçado no topo do módulo);
- o Warp smoke das 6 tasks e o runner de subprocesso de GPU;
- o smoke do `ppo_mjx` (incluindo `--eval`/`--play`);
- os testes parametrizados por `backend` em `tests/test_validation.py`;
- os scripts do protótipo em `.scratch/mjlab-port/prototype/` (bench de decisões/s, mundo ocioso,
  overflow) e o código do branch `prototype/mjlab`.

**Execução**: sem CI, tudo local. Jobs GPU e o sweep rodam sob
`systemd-run --user --scope -p MemoryMax=...`, documentado, não em código.

## Out of Scope

- Apagar ou congelar o MJX-Warp ou o Brax: os três backends coexistem.
- O `step()` próprio com 24 substeps + 24 só quando algum mundo dispara (otimização a medir depois).
- Reamostragem de DR por episódio (o mjlab permite; fica para depois, mantendo os backends
  comparáveis). DR no backend CPU. Escolha de parâmetros e ranges de DR.
- Uma `Entity` por robô, o embrulho mocap e o modelo variante.
- Multi-agente, crítico assimétrico ligado por default, multi-GPU, cluster.
- Tuning de hiperparâmetros e resultados de treino além da comparação RSL-RL × Brax.
- Contribuições upstream ao mjlab ou ao `mujoco_warp` (reportar issue é ok).
- Espectador de um mundo do lote mjlab no viewer; o viewer nativo/viser do mjlab.
- Sim-to-real e CI.

## Further Notes

- **Medições do protótipo** (`research/02-prototype-results.md`): decisões/s mjlab × MJX por
  `nworld`; overhead dos managers 6-7 % a 4-8 k mundos (aproximado: a coluna "só física" usa a
  última agenda); pool de primeiros estados em 3,6 s a 8 192 mundos; 16 384 mundos usam 10,9 GB do
  device e 1,3 GB de pico no mempool do Warp (diferença não investigada; ticket próprio).
- **Riscos abertos**: #1263 do mjlab (ITERATIONS/LS_ITERATIONS) nas cenas de 7 robôs; o mjlab não
  tem release com mujoco 4.x (pin de git) e está há três meses sem release; frescor da obs.
- **Branch `prototype/mjlab`**: fonte primária, não mergeado. O código dele serve de referência,
  não de base; o port reescreve com o task-logic parity desde o início.
- **ADR 0003 e glossário**: ticket próprio, cedo (registra os três backends e o fim do "só
  MJX-Warp"; termos novos como pool de primeiros estados e backend mjlab no `CONTEXT.md`).
- **Memórias do usuário**: venvs e clones no disco, nunca em `/tmp`; sweeps com seeds 0-4; nada de
  sim cheats no kick/drible.
