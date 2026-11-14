# 14: Domain randomization por mundo no MJX

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** `mjsoccer.mjx.make(id, randomize=sampler)` com `sampler(mj_model, rng) -> None`. Para cada slot, uma vez no início do run: mutar o `MjModel` em cache, `mj_setConst`, copiar campos variados e derivados para uma pilha, restaurar o nominal; nunca copiar o `MjModel`. O diff contra o nominal vira o `in_axes`, adaptado ao `randomization_fn` do `brax ppo.train`; RNG derivado da seed de treino, determinístico por slot. Recusa sampler que mude `opt.timestep`; campos estruturais ficam fora. Docstring avisa que os contatos da bola são `<pair>`. `randomize=None` = modelo nominal. O `examples/ppo_mjx.py` avalia no nominal por default; `--eval-randomize` avalia sob DR. O CPU fica sem DR.

**Blocked by:** 13 (Script de treino `examples/ppo_mjx.py`)

**Status:** done

- [x] Sampler que varia massa produz mundos com massa e `body_invweight0` coerentes por slot
- [x] Mesma seed → mesmas variantes
- [x] Sampler que muda `opt.timestep` levanta erro
- [x] Treino com DR roda; `--eval-randomize` funciona; eval default no nominal

## Comments

**Feito em 2026-11-14** (branch `gpu-port-14-mjx-dr`).

- **Módulo** `src/mjsoccer/mjx/randomize.py`; `MjxEnv.randomization_fn` (property) é
  `partial(randomize.randomization_fn, mj_model, sampler)`, ou `None` sem sampler. Para cada
  chave: `sampler(m, np.random.default_rng(key_data))` no `MjModel` em cache, recusa se o
  `opt.timestep` mudou, `mj_setConst`, compara com o nominal e guarda só os campos que
  diferem (backfill nominal dos slots anteriores), restaura. Nunca copia o `MjModel`.
- **Campos variáveis**: os públicos do `mjx.Model` que são `jax.Array` **e** `True` no
  `_BATCH_DIM` do MJX-Warp, mais os float de `opt` com `opt__<campo>` batchado (`impratio`
  fica fora: o Warp lê `impratio_invsqrt`). `dof_M0` muda no `setConst`, não é batchado e
  nenhum dos motores o lê fora de `types.py`: restaurado por slot, nunca empilhado.
- **Estruturais**: tudo fora do conjunto variável é checado no fim por crc32 (os ~300 MB de
  mesh/textura com 7 robôs não são copiados; os arrays ≤ 1 MB são, e voltam se o sampler os
  tocou). Mudou → `ValueError` com o nome do campo. Limitação: um sampler que muda um
  estrutural e o desfaz no slot seguinte passa (só o estado final é checado).
- **`wrap`** (`mjx/wrappers.py`, uma linha): o `randomization_fn` só vale para um env feito
  com `randomize=`. O `ppo.train` passa o mesmo fn ao env de treino e ao de eval; o eval feito
  sem sampler fica nominal.
- **`examples/ppo_mjx.py`**: `--randomize module:sampler` (`pkgutil.resolve_name`) e
  `--eval-randomize` (exige `--randomize`); `make_envs(args)` monta treino/eval. As chaves são
  as do Brax (`key_env` da `--seed`, `num_envs` para o treino, `eval_key` para o eval). O
  `--eval` standalone e o `--play` seguem nominais.
- **Custo** (JAX-CPU, sampler de verificação abaixo): 8 192 variantes com 2 robôs em 3.1 s
  (0.37 ms/variante); 2 048 com 7 robôs em 1.5 s (0.74 ms). Variaram `body_mass`,
  `body_inertia`, `pair_friction` e os derivados `body_subtreemass`, `body_invweight0`,
  `dof_invweight0`, `actuator_acc0`, `cam_poscom0`, `light_poscom0`.
- **Testes** (`tests/test_mjx_randomize.py`, JAX-CPU + um filho no Warp): massa por slot e
  `body_invweight0`/`dof_invweight0` contra `mj_setConst` numa cópia fresca do modelo;
  `in_axes` 0 só nos campos variados; modelo em cache nominal bit a bit depois; mesma chave →
  mesma variante (e por slot: o slot 1 de um lote = slot 0 do lote que começa na chave dele);
  `opt.timestep` e `geom_contype` recusados, com o nominal restaurado; env sem sampler não
  chama o fn; lote randomizado no Warp pelo `DomainRandomizationVmapWrapper` do Brax, 5
  passos finitos, sem falhas, massas distintas. `tests/test_ppo_mjx.py`: plumbing do
  `--randomize`/`--eval-randomize`.
- **Treinos curtos** (sampler de verificação no scratchpad: massa/inércia de cada robô e
  `pair_friction` × U(0.8, 1.2); não é uma escolha de DR). Contadores 0 em todas as épocas:
  - ContestedPossession, 8 192 mundos, DR, eval nominal: 4 épocas, treino 15.7 k decisões/s
    na última (GPU sozinha). **GPU 8 266 MB plana da época 2 à 4** (6 094 na 1), ~3.5 GB
    livres com outro processo de ~0.5 GB. O nominal na mesma árvore, 3 épocas: **6 174 MB**
    (igual ao ticket 13). A DR custa ~2.1 GB nesse tamanho (a pilha em si é de KB; causa não
    investigada). A folga de 2 GB do `--num-envs` ainda vale (~4 GB livres sem vizinhos), mas
    é bem menor.
  - ContestedPossession, 2 048 mundos, DR **com `--eval-randomize`**: 4 épocas, GPU 1 926 MB
    plana da época 2.
  - GoToBall, 2 048 mundos, DR: 3 épocas, GPU 5 176 MB plana da época 2 (nominal no 13: 5 156),
    ~1.1-1.7 k decisões/s com a GPU dividida com outro agente.
- **Verificação** (`gpu-port-impl` já contido, merge sem mudanças): `uv sync --group gpu` + `systemd-run ... uv run pytest` 241 passed; `uv sync && uv run pytest` 98 passed, 5 skipped. Nenhum processo meu na GPU; nada de `runs/` no git.
