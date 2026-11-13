# Reset em lote e estado por mundo

Type: grilling
Status: resolved
Blocked by: —

## Pergunta

A lógica JAX das tasks ([ticket 10](10-shared-task-logic.md)) precisa de reset e estado por mundo
dentro do `vmap`, com o auto-reset por `Data.where` que o
[smoke do Brax](12-brax-ppo-warp-smoke.md) validou. Decidir:
- **`sample_pose`** (`src/mjsoccer/base.py:336`): rejeição com `raise`. Em JAX: N candidatos fixos
  e pega o primeiro válido? `while_loop`? E o que acontece se nenhum serve (hoje `RuntimeError`)?
- **`seat_ball`** (`base.py:301`): assenta a bola por um loop de passos. No lote: mesmo loop
  (custo de física no reset), pose assentada pré-calculada, ou outro.
- **Auto-reset**: `Data.where` sobre o estado inteiro por mundo, reset a cada passo ou acumulado;
  o wrapper do Brax/Playground ou o nosso.
- **Estado por mundo**: posse com graça (`POSSESSION_GRACE`), `_plunger_lock`, `steps`, contadores e
  timers das tasks — tudo em `State.info`? Que parte é compartilhada com o CPU e coberta pelo teste
  de task-logic parity.
- **Paridade do reset**: a distribuição de poses iniciais precisa bater com a do CPU, ou basta a
  mesma regra (margem, folga)?

## Answer

Resolvido em grilling (2026-11-13), todas as recomendações aceitas.

- **Auto-reset**: o default do Brax/Playground, estado inicial em cache por mundo, re-amostrado entre
  avaliações por `num_resets_per_eval` (reset completo fora do step; refresh a cada eval). Wrapper
  `wrap_for_brax_training` do Playground com `mjx.Data.where` no `where_done`; sem wrapper nosso, sem
  `full_reset` (sob `vmap` o `reset` rodaria em todo passo de controle). Pool de M estados
  pré-amostrados fica como upgrade se a diversidade de starts medir como problema.
- **`seat_ball`**: o mesmo loop físico (0.35 s = 336 passos de física, dribbler girando) dentro do
  `reset` JAX, sem seat pré-calculado. Barato porque o reset saiu do step. CPU intacto.
- **`sample_pose`**: 200 candidatos vetorizados por objeto, sequencial entre objetos (`fori_loop`/scan
  sobre `taken`), primeiro válido por `argmax` — a mesma distribuição do rejection sampling do CPU,
  por construção. Nenhum válido: cai no candidato com maior distância mínima aos já colocados e soma
  em `State.metrics["reset_overlap"]`; sem abort (o tratamento por mundo é do ticket de falhas).
- **Estado por mundo**: um pytree em `State.info` com `grace (n_robots,)`, `plunger_lock (n_robots, 2)
  int32`, `steps`, a chave RNG do mundo e o estado da task (`prev`, `prev_goal`, `count`,
  `last_ball`, `stopped`). O snapshot do task-logic parity inclui esse pytree, comparado contra os
  atributos `_x` do CPU. O env é dono do `steps` (em passos de controle, como o CPU) e do
  `truncation`; o `EpisodeWrapper` recebe `episode_length = max_steps / repeat_action` só para concordar.
- **Paridade do reset**: basta a mesma regra (margem, folga, 200 candidatos); sem teste estatístico
  de poses. O task-logic parity testa `_place` com candidatos injetados (mesmos `x, y` → mesmas poses).
