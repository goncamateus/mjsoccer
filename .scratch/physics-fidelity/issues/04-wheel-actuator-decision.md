# Decisão: o que vai nos atuadores das rodas

Type: grilling
Status: open
Blocked by: 01, 03, 05, 16

## Pergunta

Dado o envelope derivado no `01` e a linha de base medida no `03`, o que exatamente é escrito em
`models/robot_rollers.xml:823-830`, e algo também é cortado (clip) em Python?

Subdecisões:
1. **`forcerange`.** O valor derivado do motor, ou esse valor com uma margem de segurança declarada?
   Um acionamento real também tem um limite de corrente e um controlador que não é um servo de
   velocidade puro.
2. **`ctrlrange`.** Hoje ±200 rad/s = 6.018 m/s de velocidade de superfície. Manter, ou cortar para a
   velocidade em vazio do drivetrain? Eles interagem: um `ctrlrange` que o `forcerange` nunca alcança
   é uma mentira nos normalizadores de observação (`L9`).
3. **`kv`.** `kv = 0.103696` com um erro de 200 rad/s exige 20.7 N·m e entrega 3, então o servo fica
   permanentemente saturado em erros grandes (`W7`). Quando o `forcerange` cair ~40×, `kv` passa a
   determinar toda a resposta de rastreamento. Reajustar, ou deixar e aceitar a nova constante de
   tempo?
4. **O *comando* de yaw também é cortado?** Mesmo um envelope limitado por roda admite um comando
   uniforme cujo único efeito é yaw. Se o envelope do `01` ainda passar de `MAX_W = 30` rad/s,
   decidir entre um `ctrlrange` menor e um clip em `_drive_robot` (`base.py:143-156`).
   `docs/index.md:79-82` promete que os limites vêm do modelo, não de uma constante Python — um clip
   quebra essa promessa e precisa ser justificado.

Resolvido por: os números do `01` mais uma nova execução da varredura do `03` contra o candidato,
mostrando que a taxa de yaw alcançável cai abaixo de `MAX_W` e o pico de inclinação abaixo de 15°.
