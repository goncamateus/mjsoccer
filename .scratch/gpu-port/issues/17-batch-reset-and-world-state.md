# Reset em lote e estado por mundo

Type: grilling
Status: open
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
