# Qual derivado do PPO, e como se estruturam os scripts de treino

Type: grilling
Status: open
Blocked by: 02, 04

## Pergunta

Com o levantamento do 02 e um ambiente que roda (04):

- Que derivado do PPO, e sobre que base de código (adotar uma stack, vendorizar um arquivo único como
  o `examples/sac_cleanrl.py` fez para o SAC, ou escrever do zero)?
- Um script por task ou um script com a task como argumento? Onde ficam (`examples/`, `scripts/`)?
- Como o treino escolhe o motor (`impl="jax"` / `impl="warp"`), o `nworld` e o `repeat_action`.
- Logging, checkpoint, e avaliação: avaliar no backend CPU periodicamente, ou só no MJX?
