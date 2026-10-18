# Decisão: o que uma ação de dribbler igual a 0 deve significar

Type: grilling
Status: resolved
Blocked by: —

## Pergunta

`base.py:166` mapeia a ação do dribbler de forma afim:

```python
self.data.ctrl[r.dribble] = scene.DRIBBLER_OMEGA * (a[k + 2] + 1.0) * 0.5
```

Então `a = 0` gira a barra a **157 rad/s**, e a barra só fica desligada exatamente em `a = −1`. O
comentário justifica a não negatividade ("a bar driven backwards throws the ball out of the mouth"),
o que faz sentido, mas não o ponto neutro. Consequências: todo rollout com `action_space.sample()`
dribla continuamente a uma média de 157 rad/s, e uma política precisa empurrar ativamente para −1
para parar de driblar.

Três opções:
1. **Manter o mapa afim.** Controle contínuo da velocidade da barra, neutro = meia velocidade.
   Defensável se o firmware real recebe uma velocidade, não um liga/desliga.
2. **Mapa de meia faixa**: `a ≤ 0` → desligado, `a ∈ (0, 1]` → `[0, 314]`. Neutro significa
   desligado; continua contínuo. Custa metade da resolução da ação.
3. **Booleano**, `a > 0`. O que todo env do rSoccer faz — mas o do rSim também é um bug de trava
   (`sslworld.cpp:320` só chama `setDribbler(true)`, então uma vez ligado fica ligado até o mundo ser
   reconstruído), então isso não é um endosso.

O que resolve: o que a interface do dribbler do robô real aceita — um duty cycle/velocidade ou um
liga/desliga — e se 314 rad/s é em si um valor com fonte (`scene.py:43`) ou um chute. Note que isso
muda o contrato de ações, então precisa entrar antes que o `15` o documente, e interage com o `05`
(velocidade da barra e μ do contato definem juntos o orçamento de retenção).

## Comments

- 2026-10-17, do `05`: com a inércia medida da bola (β = 0.5), uma bola chegando a ≥ 2 m/s de
  200 mm ricocheteia na barra girando e não é presa (com β 0.4 era). A 1.0-1.5 m/s é presa. Isso
  limita a recepção de passes; ver os Comments do `05`. `seat_ball` agora rola a 1 m/s.

## Answer

Resolvido em 2026-10-17, com a equipe: o firmware do dribbler recebe uma **velocidade** (duty
cycle), não um liga/desliga, e **314 rad/s não tem fonte** — é um chute (~3000 rpm).

Escolhida a opção 2, **meia faixa**: `ctrl = DRIBBLER_OMEGA * max(a, 0)` (`base.py`). `a ≤ 0` →
desligado, `(0, 1]` → `(0, 314]`. Neutro agora é desligado, e `action_space.sample()` dribla metade
do tempo em vez de sempre. Continua contínuo porque o hardware recebe velocidade; o custo é metade
da resolução da ação. `scene.py:43` e `docs/index.md` registram o 314 como suposição declarada.

Consequência: o `test_seated_ball_is_possessed` media retenção com ação zero, que até aqui girava a
barra a 157 rad/s. Ele agora segura `dribble = 0.5` (= 157 rad/s exato), então os números de
retenção do `05` em `docs/index.md` ficam bit a bit iguais e não precisaram ser remedidos. Novo
`test_dribbler_neutral_is_off` fixa o mapa. O `15` pode documentar o contrato como está.
