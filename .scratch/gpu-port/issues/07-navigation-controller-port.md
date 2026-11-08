# O port do controlador de navegação: o que vai verbatim e o que se adapta

Type: grilling
Status: resolved
Blocked by: —

## Pergunta

O `go_to_point_new` (`envs/navigation.py` do Planning-the-path-with-rl, commit `2287531`) entra
verbatim sobre o envelope de hardware atual. O que exatamente isso significa no `mjsoccer`?

- **Quirks do original**: muta `entry` in place; rotaciona posição e velocidade do agente para o eixo
  da velocidade-alvo; `MAX_DV` é por passo (0.1 m/s a 40 Hz); a frenagem planeja com
  `MAX_ACCEL = 1` m/s²; zona morta de 2.5° no giro; `ω = 2·Δθ` sem limite próprio. Quais ficam
  (inclusive os que parecem bugs), quais se corrigem?
- **De onde vêm as entradas**: posição, velocidade e heading do robô no referencial do campo, a
  partir do estado do chassi cinemático.
- **Overrides por task**: o PassEndurance hoje zera `vx, vy` do robô controlado ("keep the spin",
  `src/mjsoccer/tasks.py:307-313`) — o que vira com target pose?
- **Robôs scripted** (amarelos parados, o companheiro do PassEndurance): passam pelo controlador ou
  seguem como hoje?
- **Observações**: alguma task precisa observar a última target pose comandada?
- **O `goto_ball_policy`** dos testes (`src/mjsoccer/tasks.py:343`) e os scripted policies de
  aceitação: reescritos como target poses?

## Answer

Resolvido em grilling (2026-11-08). Vale para o backend CPU e para o MJX-Warp.

1. **Quirks verbatim, menos um.** Ficam como no original: `MAX_DV = 0.1` por passo de controle
   (4 m/s² pedidos; quem limita a aceleração é o servo, `forcerange` ≈ 1.30 m/s²);
   frenagem planejada com `MAX_ACCEL = 1`; sem banda morta de posição (chatter de ±0.1 m/s no alvo,
   amortecido pelo servo a ~3 cm/s por passo); zona morta de 2.5° no giro; `ω = 2·Δθ` sem limite
   próprio (só o `ctrlrange` de 37.1 rad/s corta); `max_velocity` por eixo no referencial
   rotacionado. **Só sai** a mutação de `entry`: o controlador vira função pura (sem mudar o
   comportamento, porque o original cria um `entry` novo por passo).
2. **`max_velocity = MAX_V = 3.3` m/s** (o envelope), não os 2.2 do original. As outras
   constantes ficam como no original.
3. **Entradas = estado real**: posição, heading e velocidade vêm do `qpos`/`qvel` das juntas
   `robot_x/y/yaw` (referencial do campo), não da observação. A saída no referencial do corpo passa
   pela rotação body→world que o `_drive_robot` já tem, até os servos de velocidade do mundo. O clamp
   por eixo do `ctrlrange` (diagonal até 3.3·√2) fica como está.
4. **PassEndurance**: o controlador roda inteiro e depois `vx, vy` vão a 0; fica só o `ω`.
   Nessa task o `x, y` da ação é ignorado, e a ação continua com 7 dims.
5. **Robôs scripted fora do controlador**: os amarelos seguem com `ctrl = 0` (servo segurando
   velocidade zero), e o companheiro do PassEndurance só liga o dribbler. O controlador roda só
   sobre os `n_controlled`.
6. **Nenhuma observação nova.** O controlador não guarda estado; pose e velocidade do robô já
   estão na observação.
7. **Policies scripted em target pose**: `goto_ball_policy` devolve alvo = bola e heading =
   ângulo robô→bola, o que muda `tests/test_envs.py`, `scripts/make_gifs.py` e
   `scripts/validation_sweep.py`. O `wheel_command` sai com o tier 1. Fica um **caminho interno
   de velocidade crua** (a parte de velocidade do `_drive_robot` atual), fora do `action_space`, só
   para os testes de envelope do `test_validation`.
