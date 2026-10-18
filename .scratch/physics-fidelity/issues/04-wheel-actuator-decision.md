# Decisão: o que vai nos atuadores das rodas

Type: grilling
Status: resolved
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

## Answer

Resolvido em 2026-10-17, por grilling, depois do `16` (enrolamento 251601 confirmado pela equipe).
Commit medido: `78281c7`. Scripts, patches do candidato e saídas brutas em
[`../candidate-04/`](../candidate-04/).

**O que vai em `models/robot_rollers.xml:827-830`, nas quatro rodas:**

```xml
<velocity name="wheelN_vel" joint="wheelN_spin" kv="0.103696" ctrlrange="-95 95" forcerange="-0.078 0.078" />
```

**Nada é cortado em Python.** `_drive_robot` continua escalando pelo `ctrlrange` do modelo, e a
promessa de `docs/index.md:79-82` fica de pé.

| subdecisão | decisão | classe |
|---|---|---|
| 1. `forcerange` | **±0.078 N·m, sem margem.** `I_nom · k_M` do 251601 já é a especificação contínua (10 % do stall de 780 mN·m), não o pico. O sobrecorrente transitório do ESC continua aproximação deliberada (`map.md`, *Sobrecorrente transitória*). | VENDOR |
| 2. `ctrlrange` | **±95 rad/s, mantido** (escolha de projeto do `01`). As rodas alcançam o comando — 90-95 rad/s em regime no giro cheio (`track_c95.txt`) — então o normalizador não mente sobre a roda (`L9`). O que não chega em `v_max` cinemático é o chassi em linha reta, por perda de contato das rodas (abaixo), e isso não é do atuador. | DELIBERATE APPROXIMATION (escolha de projeto) |
| 3. `kv` | **0.103696, mantido.** Satura acima de 0.75 rad/s de erro: P rígido + limite de corrente ≈ malha de velocidade do firmware com clamp de corrente. O `kv` derivado do motor, `1 / (8.77 rpm/mN·m) = 1.09e-3` N·m·s/rad, modela o motor em tensão constante sem controlador; medido, as rodas assentam em 76-81 de 95 rad/s e o yaw cai para 23.3 (`track_c95kvm.txt`) — erro em regime de 14-19 rad/s que nenhum acionamento em malha fechada tem. | DELIBERATE APPROXIMATION |
| 4. clip de yaw | **Sem clip.** O yaw de comando cheio medido é 29.2 rad/s < `MAX_W = 30`. A margem é de 2.6 % de `MAX_W` e depende do escorregamento com μ 0.25 (cinemático seria 35.9), então o `14` precisa **afirmar** yaw de comando cheio < `MAX_W` como critério, para que uma mudança de μ/contato que a quebre seja pega. | — |

### Verificação: varredura do `03` contra o candidato

Candidato `c95` = o atuador acima + μ rolete↔piso 0.25 do `05` (aplicado como 0.25 no rolete **e** no
piso, porque o MuJoCo usa o máximo elemento a elemento; bola↔piso continua 0.35 pelo valor da própria
bola). Patch em `candidate.c95.patch`, rodado com `MJSOCCER_MODELS` apontando para a cópia — o
`models/` versionado não foi tocado. Para reproduzir (idem `c79`, `c95kvm`, `c95mu1`):

```
cp -r models /tmp/models_c95 && (cd /tmp/models_c95 && patch -p1 < $OLDPWD/.scratch/physics-fidelity/candidate-04/candidate.c95.patch)
MJSOCCER_MODELS=/tmp/models_c95 uv run python .scratch/physics-fidelity/candidate-04/sweep.py c95 > .scratch/physics-fidelity/candidate-04/sweep_c95.csv
```

Os outros scripts de `candidate-04/` têm a linha de execução no docstring.
A aprovação vale com o μ 0.25 do `05` aplicado; não houve varredura de giro com o atuador novo e
μ 1.0.

`sweep_c95.csv` (um robô, `w = k·95` nas quatro rodas, 5 s):

| k | w | pico de yaw | yaw em regime | pico de inclinação |
|---|---|---|---|---|
| 0.2 | 19 | 6.3 | 6.2 | 0.24° |
| 0.6 | 57 | 18.0 | 17.7 | 0.36° |
| 0.8 | 76 | 23.2 | 22.9 | 0.48° |
| 1.0 | 95 | **29.2** | 28.5 | **0.55°** |

Critério do ticket atendido: yaw alcançável < `MAX_W` (30) e pico de inclinação < 15°, com 25+ rad/s
de folga até o primeiro regime de borda medido no `03` (54.3 rad/s). `sweep_c79.csv` (±79, yaw
cinemático = 30) dá 24.1 / 0.49° — não foi escolhido, porque ±95 já passa.

### Verificação: linha de base aleatória do `03` contra o candidato

Mesmo comando e seeds do `03` (`baseline.py`, seis envs, tier 1, seeds 0-49) →
[`summary.csv`](../candidate-04/summary.csv):

| | linha de base `03` | candidato `c95` |
|---|---|---|
| min de episódio | 7.82 | 101.1 |
| tombamentos | 281/300 | **0/300** (250 truncados, 39 `ball_out`, 11 outros) |
| tombamentos/min | 35.9 | **0** |
| inclinação p50 / p95 / p99 / máx | 12.3 / 44.5 / 59.8° / — | **0.59 / 3.04 / 4.06 / 6.81°** |
| pico de \|yaw\| por episódio, GoToBall p50/p95/máx | 18.7 / 31.3 / 34.4 | 2.77 / 3.51 / 3.95 |

Isso responde a hipótese que o `03` deixou não testada: o tombamento da linha de base vinha da
autoridade de torque das rodas (±3 N·m, 99.7 N de força no chão por roda, 399 N nas quatro) sob comando aleatório, não
do giro. Com o atuador do motor real, o mesmo comando aleatório não tira o chassi de 7°.

### Achado lateral, levado para o `18`

Dirigindo em linha reta com o comando cinemático cheio (`line.py`, `contact.py`), as rodas **perdem
contato com o piso** boa parte do tempo (\|v\| abaixo é de `contact.py`; `line.py` dá 1.67 m/s em x
para `c95`):

| modelo | fração de steps com contato roda↔piso, rodas 1-4, avançando | \|v\| final (cinemático 3.30) |
|---|---|---|
| atual (±200, ±3 N·m, μ 1.0) | 0.11 / 0.06 / 0.06 / 0.08 | 0.30 m/s |
| `c95mu1` (atuador novo, μ 1.0) | 0.72 / 0.49 / 0.47 / 0.72 | 2.00 m/s |
| `c95` (atuador novo, μ 0.25) | 0.42 / 0.27 / 0.28 / 0.42 | 1.94 m/s |

Parado, todas as rodas ficam em contato 99 % do tempo. O atuador novo melhora muito, mas o robô ainda
quica, e `v_max` real é ~1.7-2 m/s em x, não 3.30. **Em y é pior:** `line_c95.txt` dá 0.17 m/s
contra 4.04 cinemáticos, aceleração ~0 e pico de yaw 6.6 rad/s — o robô praticamente não anda de
lado. Isso não muda nenhuma decisão acima — é a geometria de
contato dos roletes, não o atuador — mas invalida qualquer critério de aceitação do `14` que use
`v_max` ou `a_max` cinemáticos até ser resolvido. Novo ticket
[`18-wheel-contact-loss.md`](18-wheel-contact-loss.md).
