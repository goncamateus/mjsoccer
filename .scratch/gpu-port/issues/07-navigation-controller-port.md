# O port do controlador de navegação: o que vai verbatim e o que se adapta

Type: grilling
Status: open
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
