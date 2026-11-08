# Qual derivado do PPO, e como se estruturam os scripts de treino

Type: grilling
Status: open
Blocked by: 02, 04

## Pergunta

Com o levantamento do 02 e um ambiente que roda (04):

- Que derivado do PPO, e sobre que base de código (adotar uma stack, vendorizar um arquivo único como
  o `examples/sac_cleanrl.py` fez para o SAC, ou escrever do zero)?
- Um script por task ou um script com a task como argumento? Onde ficam (`examples/`, `scripts/`)?
- Como o treino escolhe o `nworld` (limitado pelo pool de contatos do Warp, 05) e o `repeat_action`
  (o motor é só o MJX-Warp, ADR 0002).
- Logging, checkpoint, e avaliação: avaliar no backend CPU periodicamente, ou só no MJX?
