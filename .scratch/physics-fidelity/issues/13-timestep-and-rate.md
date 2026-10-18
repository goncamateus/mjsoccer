# Decisão: timestep, taxa de controle e os 60 Hz que não são 60 Hz

Type: grilling
Status: resolved
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

## Answer

Resolvido em 2026-10-18, por grilling, no `main` com `04`-`12` aplicados. Medições em
[`candidate-13/`](../candidate-13/): `sweep.py` (varredura de resolução, período de controle fixo em
16 ms, `dt` = 0.25 / 0.5 / 1 / 2 / 4 ms → `sweep_t0.txt`, `sweep_t1.txt`, `sweep_t1_025.txt`),
`held.py` (retenção e giro cheio no passo novo → `held.txt`), o `calibrate.py` do `09` rodado de novo
(→ `calibrate.txt`) e o `extremes.py`/`bounds.py` do `11` (→ `extremes_t*.csv`, `bounds_t*.csv`).

1. **2 ms não é o passo certo.** As constantes de tempo de contato são 4-6 ms (`solref`), e 2 ms as
   resolve em dois steps (4 ms prende tudo no limite `2·dt` do MuJoCo). Tier 1, erro contra 0.25 ms:

   | métrica | 0.25 (ref) | 0.5 | 1 | 2 ms |
   |---|---|---|---|---|
   | retenção CP (holds/120) | 120 | −1 % | −5 % | **−24 %** |
   | chip a 100 % | 6.40 m/s | −1 % | −4 % | **−11 %** |
   | aceleração frontal | 1.59 m/s² | −1 % | +1 % | **−12 %** |
   | giro cheio | 31.8 rad/s | −3 % | −5.6 % | −8 % |
   | `v` frontal final | 1.90 m/s | +5.5 % | −12 % | −14 % |
   | vazão | 0.14× | 0.27× | 0.51× | 1× |

   Tier 0: retenção CP 120 / 119 / 87 / 0 a 0.5 / 1 / 2 / 4 ms. Rolagem da bola, envelope do tier 0,
   chute (com o substep de disparo) e inclinação com ação aleatória ficam estáveis até 2 ms. A
   varredura **inclui o step de disparo** (roda a `dt / FIRING_SUBSTEPS`, como no `main`; `kick1` é
   sem ele). **Decisão: passo global 1/960 s** (~1.04 ms): retenção, chip e aceleração dentro de 5 %
   por metade da vazão. Ficam como resíduos declarados o giro (−5.6 %) e a `v` frontal (−12 %, não
   monotônica), os dois pelo contato dos roletes do `18` (Comment lá).
2. **`control_hz` exato.** `dt = 1/960` dá `decimation = 16` e **60 Hz exatos**: 20 s = 1200 steps.
   Uma taxa que não é número inteiro de passos de física levanta `ValueError` em vez de ser
   arredondada (`test_unreachable_control_rate_raises`); valem os divisores de 960 Hz.
3. **Segundos são o contrato.** Horizontes, o bloqueio de 250 ms e a carência de 50 ms continuam em
   segundos. O risco era o silêncio: `test_horizon_is_seconds_at_exactly_60_hz` prende o `max_steps`
   de cada env (1200 / 1800 / 2400). Recompensas por step são por step de 1/60 s.
4. **A taxa do env é a taxa da política** (a visão da SSL, ~60 Hz). Os 16 substeps representam o
   firmware e a integração: os servos de velocidade das rodas e do chassi agem em cada um. `control_hz`
   é ajuste de fidelidade (quando a política decide); o passo é ajuste de resolução.

Consequências aplicadas:
- `FIRING_SUBSTEPS` 8 → **4**: o step de disparo fica em ~0.26 ms. A tabela do `09` foi medida de
  novo no passo novo e trocada (o chip do tier 1 tinha caído 9 % a 300 N). O teto agora pede ~285-365 N.
- **`ω_max` do tier 0 = 35.9 rad/s** (teto cinemático das rodas, `01`), `MAX_W` = 35.9. O giro cheio
  do tier 1 no passo resolvido passa de 30 (30.1 a 1/960 s, 31.8 a 0.25 ms), então o limite medido do
  `12` não servia; o derivado não se move com `dt` nem μ. `goto_ball_policy` usa `kw` 1.25 para manter
  o ganho físico de antes; com o mesmo ganho, ele resolve 53/60 seeds a 2 ms e a 1/960 s, e o limite
  do `test_goto_ball_is_solvable` foi de 18 para 16/20 (18 estava numa janela de seeds de sorte).
- **Retenção:** CP 114-118 (tier 0) / 112-118 (tier 1), Dribbling 119 / 115-117. O "caos" do `12`
  era resolução (Comment lá). Limites do `test_seated_ball_is_possessed`: CP 60 → 106, Dribbling
  106 → 109 (5 % abaixo do pior seed).
- Ação aleatória (`extremes_t*.csv`): `|v|`/`MAX_V` e `|ω|`/`MAX_W` chegam a 0.45/0.14 no tier 1 e
  0.42/0.45 no tier 0. `bounds.py`: 0 observações fora do `Box`.
- Docs: seção Timing de `docs/index.md` reescrita, mais a tabela de ação, `MAX_W`, a retenção e a
  vazão; as seis páginas de ambiente. Comments no `15`, `17` (calibrar o `solref` a 1/960 s; o COR
  depende do passo) e `18`.

## Comments
- 2026-10-18, do `09`: 2 ms **não** resolve o impacto da placa do chutador (±35 % na velocidade de
  saída, não monotônica). O `09` contornou rodando só o step de disparo a `dt / 8`
  (`scene.FIRING_SUBSTEPS`, 1.64-1.72× de custo com ação aleatória). Se o dt global cair para ≤ 0.25 ms,
  apagar o contorno; a varredura de resolução daqui deve declarar se inclui o step de disparo.
