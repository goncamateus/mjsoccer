# Decisão: tratamento de capotamento quando a atuação estiver limitada

Type: grilling
Status: open
Blocked by: 04

## Pergunta

O `04` remove a *causa* do capotamento (um yaw alcançável de 59 rad/s agindo sobre um deslocamento do
CoM de 3.459 mm). Qual deve ser então o *tratamento*?

Hoje: `TIPPED_COS = 0.5` (60°), `TIPOVER_PENALTY = 100.0`, terminar o episódio, só para robôs
controlados, `info["tipped"] = True` (`base.py:41-42,129-132,170-182`). O tier 0 é estruturalmente
imune.

Decidir:
1. **Manter terminar-e-penalizar?** A recomendação do documento de pesquisa é sim — como *rede de
   segurança* com uma taxa limitada e registrada, não como mecanismo principal. Confirmar, ou
   argumentar por registrar-e-continuar agora que um tombamento significa que algo deu realmente
   errado.
2. **100.0 ainda está certo?** Na taxa atual, um tombamento apaga o retorno de 5-10 episódios bem
   sucedidos, o que deixa ruidoso o gradiente de todo o resto. Quando o `04` derrubar a taxa, uma
   penalidade menor pode ser suficiente e mais bem condicionada. **Não** baixar para suavizar o
   treino antes de o `04` entrar — isso ensina à política que tombar é barato.
3. **Manter 60°?** Não o 0.9/25.84° do rSim (que dispararia em um giro cônico marginal), mas 60° é a
   fronteira recuperável/irrecuperável, ou é arbitrário? `base.py:37-41` argumenta a favor; checar o
   argumento contra a distribuição de inclinação medida no `03`.
4. **O que é registrado?** `info["tipped"]` existe; adicionar o ângulo de inclinação e o índice do step
   para que a taxa e a distribuição do tempo até tombar sejam mensuráveis a partir do env, e não de um
   harness feito sob medida (§12.5 passo 4).
5. **Algum dia se constrói um momento endireitador?** Só se o capotamento persistir depois do `04`,
   `05` e `07`. Se persistir: saturação rígida, uma zona morta acima da inclinação normal de direção,
   uma constante de tempo de primeira ordem, e o impulso aplicado registrado — um torque endireitador
   não saturado é um momento de reação grátis contra o mundo e uma política vai explorá-lo. Saturar em
   mais de 0.1% dos steps significa que a causa real continua sem correção.

Explicitamente fora de escopo: teletransportar, soldar roll/pitch, e trocar a junta livre por uma
cadeia planar (isso é o tier 0, e o `W5` mostra que o tier 0 não é equivalente).
