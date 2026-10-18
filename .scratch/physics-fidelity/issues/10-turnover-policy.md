# Decisão: tratamento de capotamento quando a atuação estiver limitada

Type: grilling
Status: resolved
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

## Answer

Resolvido em 2026-10-18, com a equipe. Medições em [`candidate-10/`](../candidate-10/), no `main` com
`04`-`09` aplicados.

**Taxa atual.** Ação aleatória, tier 1, seeds 0-49, os seis ambientes
(`candidate-10/baseline.py <env>` → [`summary.csv`](../candidate-10/summary.csv)): **0 tombamentos em
101 min de episódio** (era 35.9/min na linha de base do `03`). Inclinação p50/p95/p99/máx
**0.36/2.46/3.60/8.32°**; 250 de 300 episódios chegam ao truncamento. O máximo nunca chega aos 9.5° em
que o chassi começa a encostar no carpete (`06`).

1. **Manter terminar-e-penalizar**, como rede de segurança. Com taxa ~0, ela só dispara quando algo deu
   errado de verdade; e passar de 60° é irrecuperável (item 3), então continuar seria só tempo morto de
   lado até o truncamento.
2. **`TIPOVER_PENALTY` fica em 100.0.** O argumento do ruído ("um tombamento apaga 5-10 episódios")
   pressupunha 35.9/min; com taxa ~0 o custo no gradiente some, e baixar só enfraqueceria a dissuasão.
3. **60° fica, e agora tem fonte medida**: fica a ~1° da fronteira de recuperação estática.
   `candidate-10/recover.py` → [`recover.txt`](../candidate-10/recover.txt) inclina o robô θ em torno de
   um eixo horizontal de azimute φ (passo de 15°), apoia no carpete parado, ctrl zero, 2 s: o menor θ
   que **não** volta de pé fica em **59-67°**, mediana 61°. Então 60° é conservador em 0-7° em 22 de
   24 azimutes (um robô *parado* a 60-62° ainda voltaria) e 1° permissivo em φ = 225° e 315° (59°).
   Aceito: o ensaio é estático, e um robô que passa de 60° caindo carrega momento angular para o lado
   do tombamento, então a fronteira dinâmica fica abaixo da estática. Não é arbitrário.
4. **Registro: só no step do tombamento**, junto de `info["tipped"]`: `info["tilt_deg"]` (pior robô
   controlado, graus) e `info["tip_step"]` (steps de controle desde o reset; tempo até tombar =
   `tip_step * dt`). Taxa e distribuição do tempo até tombar saem do env, sem harness sob medida.
   Fixado no `test_tipover_ends_episode`; documentado nas tabelas de `info` das seis páginas de
   ambiente, que também passam a dizer `-100.0` (item 1 do `15`).
5. **Momento endireitador: fechado, nunca construir.** A condição do ticket (persistir depois de `04`,
   `05`, `07`) não se cumpriu. Reabrir só se uma política treinada mostrar taxa de `tipped` não nula —
   e aí os requisitos do item 5 da pergunta (saturação rígida, zona morta, constante de tempo, impulso
   registrado, alarme acima de 0.1 % dos steps saturados) continuam valendo.
