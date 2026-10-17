# Decisão: timestep, taxa de controle e os 60 Hz que não são 60 Hz

Type: grilling
Status: open
Blocked by: 07

## Pergunta

Física com `dt = 0.002` s, `decimation = round(1/(60 · 0.002)) = 8`, então a taxa de controle é
**62.5 Hz**, não os 60 pedidos (`base.py:68-69`, e `docs/index.md:302-305` diz isso). Um episódio de
"20 s" tem 1250 steps, não 1200 (`base.py:71`).

Decidir:
1. **2 ms é o passo de física certo?** Para um modelo de 100 DOF com contatos de rolete alternando e um
   pulso de chute de 16 ms, plausivelmente sim — mas nada estabeleceu isso. O teste é uma varredura de
   resolução: os resultados devem variar menos de 5% entre `dt` = 1 ms e 4 ms. A decisão de geoms do
   `07` muda o custo dessa resposta, e é por isso que este espera.
2. **`control_hz` deve ser exato?** Ou aceitar 62.5 e documentar (já feito), ou fazer de `control_hz`
   um pedido que se ajusta a um valor alcançável e o *reporta*, ou escolher um `dt` que divida o alvo
   exatamente (`dt = 1/480` dá 60 Hz com `decimation = 8`).
3. **O horizonte continua em segundos?** Horizontes são declarados em segundos e convertidos
   (`__init__.py:20-27`), então uma mudança de `dt` muda silenciosamente a contagem de steps de todo
   episódio e a taxa por segundo de toda recompensa por step. Decidir se essa conversão é o contrato ou
   um risco.
4. **A que taxa a SSL real controla?** A visão é em torno de 60 Hz; os laços de controle do firmware
   são bem mais rápidos. Se a taxa do env pretende representar a taxa da *política*, 60-62.5 Hz está
   certo e os substeps representam o firmware. Dizer qual, porque isso determina se `decimation` é um
   ajuste de fidelidade ou de desempenho.
