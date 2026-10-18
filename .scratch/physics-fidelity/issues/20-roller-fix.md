# Aplicar a correção dos roletes do `19` em `models/`

Type: task
Status: open
Blocked by: —

## Pergunta

O [`19`](19-wheel-axial-scrub.md) fechou o arrasto axial: três coisas freiam o giro do rolete, e
tirar as três leva o tier 1 a 3.29 / 4.02 m/s (x / y) com arrasto ~0. A variante é a `ellip_c3_d0`
de [`../candidate-19/variants.py`](../candidate-19/variants.py). Em `models/robot_rollers.xml`, nos
80 roletes:

1. Colisão: o casco `r_c_roller_00` (`maxhullvert 32`) vira elipsoide `size="0.0063 0.01377 0.0063"`
   (semi-eixo longo no eixo do rolete), com `contype="0" conaffinity="4"`.
2. `condim="4"` → `condim="3"` nas geoms de colisão dos roletes.
3. `damping="1e-6"` → `damping="0"` nas juntas `roll_wheel*`.

Fazer:
- Aplicar, e rodar a suíte e o harness do `14`. Registrar o que muda em contagem de contatos e em
  steps/s (o `07` tem o orçamento de colisão).
- Medir de novo o critério 5 do `14` e o `accel.py` dele, e conferir que a bola ainda bate nos roletes
  (`contype 7` da bola contra `conaffinity 4`).
- Passar ao `12` o envelope medido. O esperado pelo `19` é `v_max` 3.29 / 4.02 m/s medidos (cinemáticos 3.30 /
  4.04) e `a_max` sustentado entre as rampas 1.6–2.0 / 1.2–1.4 m/s² (limites analíticos 1.95 / 1.33).
