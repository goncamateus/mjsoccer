# Onde moram os testes do backend MJX-Warp

Type: grilling
Status: open
Blocked by: 16

## Pergunta

O MJX-Warp precisa de GPU CUDA e do grupo opcional `gpu`; o teste de task-logic parity
([ticket 10](10-shared-task-logic.md)) roda em JAX-CPU. Decidir, para as checagens do
[ticket de aceitação e paridade](09-acceptance-and-parity.md) e as do kick
([ticket do kick no lote](16-kick-in-batch.md)):
- o que entra na suíte padrão (pulado sem `gpu`/sem CUDA) e o que fica `-m validation`;
- se a paridade MJX × CPU vira `scripts/validation_sweep.py --backend mjx` ou um script próprio;
- se o teste de task-logic parity entra na suíte padrão (pulado sem o grupo `gpu`);
- como o CI (se houver) e a máquina local rodam cada camada.
