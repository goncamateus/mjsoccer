# mjlab — decisões do grilling (2026-11-14)

Pesquisa: [`research/01-mjlab-fit.md`](research/01-mjlab-fit.md) (mjlab `main` `6ab9846`, mujoco_warp
`v4.0.1`). ADR vigente: [0002 — só MJX-Warp](../../docs/adr/0002-mjx-warp-only.md).

## Papel

1. **Terceiro backend.** CPU, MJX-Warp (+ Brax) e mjlab coexistem. O MJX não é apagado.
2. **Um catálogo, um contrato.** `mjsoccer.mjlab.make(id)` lê o mesmo `ENV_IDS`: mesmos ids,
   horizontes, espaços e layout de observação/ação. Política exportada em ONNX roda no CPU e no
   viewer.
3. **CPU segue oráculo.** O mjlab passa os mesmos critérios e limiares do MJX-Warp (determinístico
   ≤ 2 % envelope/kick/COR, ≤ 5 % chegada, ≤ 1 cm / 1°; com seed nas seeds 0-4),
   `validation_sweep.py --backend mjlab`, e um task-logic parity em torch. Entra no spec.
4. **Treino: RSL-RL 5.5.2 nativo**, partindo dos hiperparâmetros do `ppo_mjx`. Comparação com Brax
   por retorno de eval, mesmo orçamento de decisões, seeds 0-4 — ticket do spec.
5. **Dependências:** grupo opcional `mjlab` (torch) separado do `gpu`; declarados em conflito se o
   `uv lock` não resolver os dois juntos.
6. **ADR 0003** "MJX-Warp e mjlab" substituindo a 0002, escrita só depois do protótipo.

## Protótipo (antes do spec)

7. Branch `prototype/mjlab`, sessão separada via `/handoff`. **Sem gate numérico**: só informa; o
   usuário decide se segue para o spec.
8. Versões sobem só no branch: `mujoco` 4.0.1, `mujoco-mjx[warp]` 4.0.1, mjlab git `6ab9846`,
   torch ≥ 2.14. O `main` só sobe se o mjlab for adotado (aí o 1º ticket roda de novo a paridade
   CPU↔MJX na 4.0.1).
9. **Escopo:** só ContestedPossession, **com kick**, sem treino e sem paridade.
10. **Passo de disparo:** `decimation = 48 × repeat_action`, `timestep = 1/960` no `MujocoCfg`,
    `opt.timestep` por mundo trocado à mão (`wp.array(nworld)` + `sim.create_graph()`), agenda
    escrita in-place por substep (mesmo esquema do ticket 08 do MJX). Horizonte, recompensa e
    time-out são nossos (`scale_rewards_by_dt=False`). `step()` próprio com 24+24 condicional fica
    para depois.
11. **Cena:** inteira via `SceneCfg.spec_fn`, modelo base exato, nomes iguais ao CPU, sem `Entity`;
    action term aponta para âncora vazia. `MujocoCfg` igual ao `<option>` do `scene.xml`
    (`timestep` 1/960, `cone="elliptic"`).
12. **Reset:** `seat_ball` vira pool de primeiros estados assentados (mesmo loop físico, em lote),
    tamanho `nworld`, refeito por época — equivalente ao cache do Brax + `--num-resets-per-eval`.
13. **Medir** (itens 1-4, 6, 7 do §7 da pesquisa):
    - decisões/s ponta a ponta com o kick disparando, contra ~16,4 k do MJX-Warp na 3060
      (`.scratch/gpu-port-impl/10-bench/results.txt`);
    - memória GPU e `nworld` máximo estável (MJX: 8 192 com 2 robôs);
    - mundo ocioso bit a bit nas iterações a `timestep=0` sob o grafo capturado;
    - `Data.overflow` limpo pelo `reset_data` do mjlab; termination própria lendo `overflow`;
    - `uv lock` com torch + JAX num env só.

## Depois do protótipo

Timestep do kick definitivo, paridade, treino RSL-RL vs Brax, ADR 0003 → `/to-spec` com os números.

## Depois do protótipo (2026-11-14)

Resultados: [`research/02-prototype-results.md`](research/02-prototype-results.md) (branch
`prototype/mjlab`, `45b25d5`; o branch fica como fonte primária e não é mergeado).

14. **Segue para o spec** com o mjlab como terceiro backend: throughput empata com o MJX-Warp
    (16 588 × 16 550 dec/s a 4 096 mundos, mujoco 4.0.1) e vai a 16 384 mundos.
15. **Memória do device × mempool do Warp** (10,9 GB × 1,3 GB a 16 384) vira ticket do spec, junto
    com a sincronia de streams torch × Warp; não bloqueia (8 192 cabe com folga).
16. Spec e tickets em `.scratch/mjlab-port/` no `main`, apontando para o branch do protótipo.
