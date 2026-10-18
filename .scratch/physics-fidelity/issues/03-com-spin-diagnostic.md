# Diagnóstico: confirmar o mecanismo de tombamento por deslocamento do CoM e medir a linha de base

Type: task
Status: resolved
Blocked by: —

## Pergunta

Nada a decidir. Rodar a medição da qual os tickets seguintes dependem, depois reverter.

A afirmação sob teste (`M1`, `TO2`): o tier 1 tomba porque o centro de massa do chassi do CAD fica
**3.459 mm fora do eixo de giro** (verificado: `|subtree_com − xpos|` no modelo compilado, com
`subtree_com = [·, ·, 0.0739943]` e o plano do piso em z = 0.02), e o espaço de ações permite
59.4 rad/s de yaw. O limiar previsto é
`ω_tip = sqrt(g · d_edge / (h · d_com)) = sqrt(9.81 · 0.04325 / (0.05399 · 0.003459)) = 47.7 rad/s`.

Fazer:
1. **Reproduzir a varredura de giro.** `place` um robô tier-1 na origem, comandar
   `w = [k,k,k,k]` para `k` em `[0.2 … 1.0]` do `ctrlrange`, 5 s de física, nenhum outro corpo em
   contato. Registrar pico de taxa de yaw e pico de inclinação por `k`. Esperado pelo documento de
   pesquisa: 42.1 rad/s → 6.0°; 55.6 → 43.2°; 59.4 → 46.6°.
2. **Falseá-la.** Zerar temporariamente as componentes x/y de `ipos` em
   `models/robot_rollers.xml:153` e rodar de novo. Se a inclinação colapsar, o `M1` está confirmado.
   **Reverter a edição na mesma sessão** — `git diff` precisa estar vazio quando este ticket fechar.
   Isto é um diagnóstico, nunca uma mudança entregue (§12.4(a)).
3. **Taxa de tombamento de linha de base.** `env.action_space.sample()` em todos os seis ids, tier 1,
   seeds 0-49: tombamentos por minuto de episódio, distribuição do tempo até tombar, e p50/p95/p99 da
   inclinação do chassi. Esse é o número contra o qual toda correção futura será medida, então
   registrar o comando exato e o commit.
4. **Linha de base de tempo de relógio** em steps/s por env, para comparar com `docs/index.md:157-160`.

Escrever os números no `## Answer` deste ticket, e as linhas brutas em
`.scratch/physics-fidelity/baseline/`. Não modificar nada em `src/`, `models/`, `examples/`,
`scripts/` ou `tests/` exceto o diagnóstico revertido de `ipos`.

## Answer

Resolvido em 2026-10-17. Commit medido: `6b66c61`, MuJoCo 3.11.0, gymnasium 1.3.0, numpy 2.5.2,
AMD Ryzen 5 5600X. Linhas brutas e scripts em [`../baseline/`](../baseline/).

**Resumo: o `M1` é só parcialmente confirmado, e não explica a linha de base.** O deslocamento de
3.459 mm baixa o limiar de giro, mas o chassi entra no mesmo modo de "andar na borda" (~40-49°
sustentados) com o CoM **exatamente** no eixo. E as ações aleatórias tombam ~36 vezes por minuto com
pico de yaw ≤ 34 rad/s — abaixo de qualquer limiar de giro medido —, então o tombamento da linha de
base vem de outro mecanismo, não do giro. Hipótese **não testada** aqui: aceleração/jerk das rodas
sob comando aleatório a ±200 rad/s.

### 1. Varredura de giro (CAD)

`uv run python .scratch/physics-fidelity/baseline/sweep.py cad > .scratch/physics-fidelity/baseline/sweep_cad.csv`
→ [`sweep_cad.csv`](../baseline/sweep_cad.csv) (todas as linhas, k = 0.2 … 1.0).
Um robô tier 1 na origem, bola em (4, 3), `w = k·200` nas quatro rodas, 5 s, `dt = 0.002`. Confirmado
`|subtree_com − xpos|_xy = 3.459 mm`, `subtree_com.z = 0.0739942`. Inclinação = `arccos(R[2,2])`;
yaw = componente z de `R · qvel[3:6]` (a velocidade angular do freejoint é no referencial do corpo).

| k | w (rad/s) | pico de yaw | yaw em regime (último 1 s) | pico de inclinação | t do pico | inclinação final |
|---|---|---|---|---|---|---|
| 0.2 | 40 | 14.6 | 14.3 | 4.7° | 0.14 s | 0.1° |
| 0.4 | 80 | 28.6 | 28.2 | 12.5° | 0.20 s | 0.1° |
| 0.6 | 120 | 42.6 | 41.6 | 4.9° | 0.24 s | 0.3° |
| 0.7 | 140 | 50.1 | 48.3 | 10.7° | 0.15 s | 0.6° |
| 0.75 | 150 | 57.1 | 54.3 | 37.6° | 4.89 s | 34.9° |
| 0.8 | 160 | 62.8 | 61.5 | 41.6° | 1.92 s | 40.8° |
| 1.0 | 200 | 71.3 | 68.6 | 42.0° | 4.22 s | 35.1° |

Para k ≤ 0.7 o pico é transiente de arranque (< 0.25 s) e o chassi volta a ~0°. Para k ≥ 0.75 o
chassi entra num regime estável de ~35-45° apoiado na borda — **nunca passa dos 60° do `TIPPED_COS`**,
então o env não chamaria isso de tombamento. Limiar CAD entre **48.3 e 54.3 rad/s** (yaw em regime).
O previsto `ω_tip = 47.7` fica ~1 rad/s abaixo do último ponto estável (48.3 em regime, 50.1 de pico):
mesma ordem, prevê um pouco cedo demais.

Os números esperados do documento de pesquisa só se reproduzem em parte: k = 0.6 bate (42.6 vs 42.1
rad/s de pico, 4.9° vs 6.0°). O "55.6 → 43.2°" fica mais perto de k = 0.75 aqui (57.1 de pico, 54.3 em
regime, 37.6°) do que de k = 0.8 — o mapeamento de k do documento não é explícito. O "59.4 → 46.6°" é
o comando cheio (k = 1.0), que hoje dá 71.3 de pico / 68.6 em regime e 42.0° — esse não se
reproduz. O pico de yaw ainda fica abaixo dos 75.6 rad/s cinemáticos.

### 2. Falseamento

Feito sobre uma **cópia** de `models/` no scratchpad, apontada com `MJSOCCER_MODELS` — o arquivo
versionado nunca foi tocado (`git diff` vazio o tempo todo). Cada variante é um patch salvo em
`baseline/robot_rollers.<variante>.patch`. Para reproduzir:

```
cp -r models /tmp/models_com0 && patch /tmp/models_com0/robot_rollers.xml .scratch/physics-fidelity/baseline/robot_rollers.com0.patch
MJSOCCER_MODELS=/tmp/models_com0 uv run python .scratch/physics-fidelity/baseline/sweep.py com0 > .scratch/physics-fidelity/baseline/sweep_com0.csv
```

Varredura com k em `0.2, 0.4, 0.6, 0.65 … 1.0`.

| variante | `|Δ_xy|` da subárvore | estável até (yaw em regime) | primeiro k que anda na borda |
|---|---|---|---|
| `cad` | 3.459 mm (−x) | k 0.70 → 48.3 rad/s | k 0.75 → 54.3 rad/s |
| `ipos0`: `ipos` x/y = 0 (pedido pelo ticket) | **3.278 mm (+x)** | k 0.90 → 58.6 | k 0.95 → 63.5 |
| `com0`: `ipos` = (−3.876, 0.014) mm, CoM na linha do eixo | 0.000 mm | k 0.80 → 55.5 | k 0.85 → 62.6 |
| `com0diag`: `com0` + produtos de inércia do chassi = 0 | 0.000 mm | k 0.75 → 51.9 | k 0.80 → 59.0 |

- **Zerar `ipos` x/y não centraliza o CoM**: rodas, dribbler e kicker colocam a subárvore 3.278 mm
  para +x. O falseamento do jeito que o ticket escreveu não testa a hipótese; por isso a variante
  `com0`, que resolve `ipos` para `subtree_com_xy = xpos_xy`.
- Com CoM no eixo, k = 0.8 (55.5 rad/s) deixa de andar na borda (41.6° → 10.1° transiente, 0.1°
  final), então o deslocamento **baixa** o limiar ~7 rad/s. Mas em k ≥ 0.85 (≥ 62 rad/s) o modo volta
  com 45-47°: **o deslocamento não é necessário**. Os produtos de inércia do chassi também não são a
  causa residual (`com0diag` não fica mais estável).
- `ipos0` (3.278 mm, sinal oposto) é **mais** estável que `com0`; o limiar não escala como
  `1/sqrt(d_com)`. A fórmula de `ω_tip` não é preditiva fora do caso CAD, e o sentido do deslocamento
  importa (as rodas não são simétricas em x).
- Ressalva: o início do modo às vezes chega tarde (4.9 s em `cad` k = 0.75), então "estável" num
  ensaio de 5 s é cota inferior, não prova.
- Implicação para o `01`: o `ω_max = 35.9` rad/s proposto continua abaixo de todos os limiares
  medidos (o menor regime instável é 54.3 rad/s), mas a margem de 25 % não deve ser lida como
  derivada de um mecanismo único.

### 3. Taxa de tombamento de linha de base (ação aleatória, tier 1, seeds 0-49)

```
for e in GoToBall-v0 GoToBallShoot-v0 Dribbling-v0 StaticDefenders-v0 ContestedPossession-v0 PassEndurance-v0; do
  uv run python .scratch/physics-fidelity/baseline/baseline.py $e; done   # rodado com os 6 em paralelo
```

Cada execução imprime uma linha de [`summary.csv`](../baseline/summary.csv) (cabeçalho escrito à mão;
a última coluna é steps/s com 6 processos concorrentes, não usar como throughput) e grava
`episodes_<env>.csv` e `tilt_<env>.npy`. `env.reset(seed=s)` + `env.action_space.seed(s)`.
Inclinação amostrada a cada step de controle (`dt = 8 · 0.002 = 0.016` s, 62.5 Hz) em todos os robôs
controlados.

Duas ressalvas de leitura:
- O p99 ≈ 60° é **censurado**: o episódio termina quando a inclinação passa de 60° (`TIPPED_COS`), então
  a cauda da distribuição é o limiar de término, não uma propriedade do chassi. Só p50/p95 dizem algo.
- Como quase todo episódio termina tombado, tombamentos/min ≈ 1 / duração média do episódio. Serve
  como linha de base, mas satura: uma correção só aparece neste número quando episódios passam a
  terminar por outro motivo.

| env | min de episódio | tombamentos | **por min** | t até tombar min / mediana / máx | inclinação p50 / p95 / p99 |
|---|---|---|---|---|---|
| GoToBall-v0 | 1.48 | 50/50 | 33.8 | 0.56 / 1.54 / 5.97 s | 12.4 / 43.9 / 59.4° |
| GoToBallShoot-v0 | 1.42 | 50/50 | 35.2 | 0.56 / 1.54 / 5.70 s | 12.4 / 44.4 / 59.4° |
| Dribbling-v0 | 0.74 | 31/50 (17 ball_out, 2 outros) | 41.8 | 0.27 / 0.77 / 1.90 s | 12.2 / 45.8 / 60.8° |
| StaticDefenders-v0 | 1.53 | 50/50 | 32.7 | 0.56 / 1.54 / 5.98 s | 12.2 / 44.8 / 59.2° |
| ContestedPossession-v0 | 1.21 | 50/50 | 41.2 | 0.32 / 1.18 / 5.31 s | 12.1 / 45.2 / 60.6° |
| PassEndurance-v0 | 1.44 | 50/50 | 34.8 | 0.56 / 1.54 / 5.02 s | 12.2 / 43.3 / 59.4° |
| **todos** | 7.82 | 281/300 | **35.9** | min/q1/mediana/q3/máx 0.27 / 0.75 / 1.20 / 2.06 / 5.98 s | **12.3 / 44.5 / 59.8°** |

Ação aleatória não é dirigir: o robô passa a mediana do tempo a 12° de inclinação e quase todo
episódio acaba tombado em ~1-2 s. Pico de `|yaw|` por episódio em `GoToBall-v0` (mesmas seeds):
p50 18.7, p95 31.3, máx 34.4 rad/s, yaw no referencial do mundo
(`uv run python .scratch/physics-fidelity/baseline/random_yaw.py` →
[`random_yaw_GoToBall.txt`](../baseline/random_yaw_GoToBall.txt))
— o giro nunca chega perto do limiar, então centralizar o CoM sozinho não deve mover este número.

### 4. Tempo de relógio

`uv run python .scratch/physics-fidelity/baseline/throughput.py > .scratch/physics-fidelity/baseline/throughput.csv`
→ [`throughput.csv`](../baseline/throughput.csv). 2000 steps aleatórios por (env, tier), processo
único, resets contados. Método diferente do `docs/index.md` (100 steps, e um espaço de ações mais
antigo), então a comparação é indicativa.

| env | tier 0 | tier 1 | `docs/index.md` t0 / t1 |
|---|---|---|---|
| GoToBall-v0 | 2 892 | 949 | 3 300 / 650 |
| GoToBallShoot-v0 | 3 208 | 947 | — |
| Dribbling-v0 | 2 636 | 686 | — |
| StaticDefenders-v0 | 892 | 153 | 820 / 88 |
| ContestedPossession-v0 | 1 642 | 396 | — |
| PassEndurance-v0 | 2 033 | 494 | — |

Tier 1 hoje é mais rápido que o documentado, mas os episódios aleatórios no tier 1 tombam a cada
~1.5 s, então uma fração não-desprezível do tempo é `reset` (inclui o `seat_ball` onde houver); não é
um número de steps/s em regime de treino.
