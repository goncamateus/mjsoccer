# Decisão: o que uma ação de dribbler igual a 0 deve significar

Type: grilling
Status: open
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
