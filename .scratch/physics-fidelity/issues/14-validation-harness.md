# Harness de validação e critérios de aceitação

Type: task
Status: open
Blocked by: 04, 05, 18

## Pergunta

Nada a decidir além do formato. Construir a coisa que comprova todos os outros tickets, e então
declarar os critérios.

Métricas (§13.1 do documento de pesquisa): taxa de tombamento por minuto de episódio; tempo até tombar;
roll/pitch do chassi média/máx/p99; distribuição da taxa de yaw e fração de steps acima de `MAX_W`;
impulso de contato por step p50/p95/p99/máx atribuído ao par de contato; os 0.5 s de histórico de
velocidade antes de cada tombamento; duração de posse média/p95; retorno do episódio; steps até o
sucesso em `GoToBall-v0` e `Dribbling-v0`; steps/s.

Cenários (§13.2), cada um uma sequência roteirizada determinística contra o env sem wrappers, sem
política: giro uniforme saturado; robô-robô de frente a 1/2/3/4 m/s; raspão na parede a 2/4/6 m/s e
5°/15°/30°; curva fechada driblando com yaw comandado de 10/20/30/50 rad/s; recuo do chute com o
chassi a 0/2/4 m/s; rollout com ações aleatórias em todos os seis ids e nos dois tiers; e
`goto_ball_policy` como controle de controlador competente.

Seeds 0-49 por (cenário × configuração), compartilhadas entre configurações para que as comparações
sejam pareadas. 5 seeds para as curvas de aprendizado, só em `GoToBall-v0` e `Dribbling-v0`. Medianas e
IQRs, não médias, para qualquer coisa assimétrica.

Formato: estender o teste de tombamento existente em `tests/test_envs.py` como modelo
(`gym.make(id, tier=1).unwrapped`, escrever `data.qpos`, step, assert sobre `info`) como uma suíte
marcada; um script que emite CSV ao lado de `scripts/make_gifs.py` para as distribuições; os exemplos
sb3 existentes como treinador. Nenhuma dependência nova.

Os critérios, da §13.5, a serem confirmados quando o `04` e o `05` fornecerem números reais: taxa de
tombamento com ações aleatórias abaixo de 1 por minuto de episódio em todos os seis ids e zero com um
controlador competente; p99 da taxa de yaw abaixo de `MAX_W`; p99 do roll do chassi abaixo de 15°; p50
da posse dentro de 5% de `docs/index.md:199-212`; aceleração alcançável dentro de 1.5× do valor do
motor do `01`; retorno e eficiência amostral não piores que a linha de base do `03` com o mesmo número
de steps; steps/s dentro de 10% da linha de base; e, se existir um momento endireitador, saturação em
menos de 0.1% dos steps.

Entregar o plano do harness e os valores finais dos critérios em `## Answer`. Escrever os arquivos é
execução, e acontece depois que o mapa fechar.

## Comments

- 2026-10-17, do `04`: (1) **afirmar** yaw de comando cheio (`w = [95]*4`) < `MAX_W = 30` rad/s
  como critério — medido 29.2, margem de 2.6 %, dependente do escorregamento com μ 0.25; o `04`
  decidiu não cortar yaw em Python e conta com este teste para pegar regressão. (2) Linha de base
  do candidato do `04`: 0 tombamentos/min, inclinação p50/p95/p99 0.59/3.04/4.06°
  ([`../candidate-04/summary.csv`](../candidate-04/summary.csv)) — a métrica de tombamentos/min do
  `03` satura em zero, então o critério precisa ser a inclinação p95/p99. (3) Não usar `v_max`/`a_max`
  cinemáticos até o `18` fechar (o chassi chega a ~1.7-2 m/s em x, não 3.30, e 0.17 m/s em y, não 4.04).
- 2026-10-17, do `05`: edições do `04`+`05` aplicadas em `models/`. Nova linha de base de posse
  parada (`holds()`/120, 8 seeds, tier 0 / tier 1): Dribbling-v0 119 / 114, ContestedPossession-v0
  82-93 / 89-96. O critério "p50 da posse dentro de 5% de `docs/index.md:199-212`" deve usar estes
  números (a seção já foi atualizada).
