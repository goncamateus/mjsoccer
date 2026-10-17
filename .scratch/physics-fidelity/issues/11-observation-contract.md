# Decisão: tornar honesto o contrato de observação

Type: grilling
Status: open
Blocked by: 04

## Pergunta

Três defeitos, e todos precisam do envelope final do `04` antes de poderem ser corrigidos:

1. **`MAX_W = 30.0` não é o máximo do tier 1.** Está documentado como "the tier-0 chassis servo
   `ctrlrange`" (`base.py:34`, `docs/index.md:133`), mas o tier 1 chega a 59.4 rad/s medidos, então o
   canal `omega` rotineiramente lê ≈2.0 sob o próprio acionamento do robô. `docs/index.md:135-137`
   afirma que os valores "sit in roughly [−1, 1]" e culpa as excursões em "a tier-1 chassis pushed by
   another robot" — falso hoje (`L9`). Depois do `04` isso pode se resolver sozinho; confirmar, e
   decidir se `MAX_V`/`MAX_W` devem ser *derivados do modelo* em vez de declarados como constantes.
2. **O canal de altura da bola tem um erro de referência.** `base.py:220` divide o `bp[2]` bruto do
   mundo por `GOAL_HEIGHT = 0.155`, mas o plano do piso está em z = 0.02 (`models/field.xml:42`),
   então o canal carrega um deslocamento sistemático de `+0.02/0.155 = +0.129` e mede a altura acima
   da origem do mundo, não acima do carpete (`U4`). Corrigir a observação, ou mover o piso para
   z = 0? Mover o piso mexe em todo z de spawn do modelo; corrigir a observação é uma subtração.
   Decidir qual, e se mais alguma coisa lê um z bruto.
3. **O `Box` é `(−inf, inf)` e nada faz clip** (`base.py:79`). O rSoccer faz clip de todo canal em
   ±1.2 (`ssl_gym_base.py:27,213-220`). Sob fidelidade, um espaço de observação ilimitado é
   correto-mas-inconveniente, ou o `Box` declarado deveria ao menos ser *verdadeiro*?

Resolvidos juntos porque os três são a mesma pergunta: o contrato de observação descreve o simulador,
ou aspira a ele.
