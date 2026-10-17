# Decisão: alvos de calibração do chute e do chip

Type: grilling
Status: open
Blocked by: 02, 05

## Pergunta

Que distribuição de velocidade de saída o chutador deve produzir, e os parâmetros do mecanismo
precisam ser reajustados para atingi-la?

Hoje o chute é um mecanismo de verdade: `KICK_FORCE = 200` N aplicados por exatamente um step de
controle (16 ms = 8 substeps) a um êmbolo de 0.1413 kg em um trilho `range="0 0.03"` contra
`stiffness="2000"` / `damping="2.0"`, com um bloqueio por software de 250 ms por êmbolo
(`base.py:84-90,160-163`, `models/robot_rollers.xml:270-296`). A velocidade de saída é *emergente* —
`base.py:35` documenta 6-7 m/s para um chute rasteiro. O chipper é o mesmo mecanismo de novo.

Pontos em aberto:
1. **6-7 m/s é legal e realista?** O `02` fornece o teto do livro de regras. Se o livro de regras diz
   6.5 m/s, um mecanismo que emergentemente o ultrapassa é um defeito.
2. **A velocidade de saída sobrevive ao `05`?** Massa da bola, atrito bola↔placa e complacência de
   contato da placa alimentam o impulso. Medir de novo depois do `05`, não calibrar antes.
3. **Uma força fixa de 200 N é a interface certa?** Hoje a ação é um gatilho booleano (`a > 0`,
   depois uma força fixa). O firmware real da SSL comanda uma *velocidade* de chute. Decidir se a
   ação deve escalar a força e, se sim, como o pedido de uma política se mapeia em uma carga de
   capacitor.
4. **O bloqueio de 250 ms é físico?** Ele é estrutural (`base.py:84-87`: sem ele, uma ação de chute
   mantida prende o êmbolo contra o batente e a bola nunca é liberada), mas o número deveria vir de
   um tempo real de recarga do capacitor, não do que fez o êmbolo se comportar.
5. **Recuo.** O chassi absorve a reação (`L4`), algo que o rSim não tem equivalente. Confirmar que
   tem a magnitude certa em vez de ser um artefato dos 0.1413 kg do êmbolo — os êmbolos são 0.2785 kg
   dos 2.692 kg do robô e deslizam à frente do centro (`M3`).

Medido contra o cenário 5 da §13.2 do documento de pesquisa (recuo do chute com o chassi a 0/2/4 m/s)
e sua métrica de velocidade de saída.
