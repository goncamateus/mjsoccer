# Instalar o MJX nesta máquina e rodar o modelo base e a variante

Type: task
Status: resolved
Blocked by: —

## Pergunta

Nada a decidir: o 05 e o 08 precisam de um ambiente que funcione nesta RTX 3060 (driver 595).

Num env isolado (fora do `uv` do projeto, sem tocar `pyproject.toml`):

**Segurança de memória (obrigatório — a primeira tentativa derrubou a máquina por OOM).** O env fica
em disco, nunca em `/tmp` (tmpfs = RAM). Todo processo MJX/JAX roda com limite de RAM
(`systemd-run --user --scope -p MemoryMax=8G ...`) e `XLA_PYTHON_CLIENT_PREALLOCATE=false` /
`XLA_PYTHON_CLIENT_MEM_FRACTION`. Cena de 2 robôs antes da de 7.

1. Instalar `mujoco==4.0.0`, `mujoco-mjx[warp]==4.0.0` (→ `warp-lang==1.18.0`, restrição 12 do mapa)
   e JAX com CUDA (sem pin upstream: começar pelo mais recente, recuar se o FFI do Warp falhar).
   Registrar se o FFI do Warp importa com cada JAX e o CUDA do wheel do `warp-lang`. O Playground
   `main` (`rss2025`, `warp>=1.18`) passa a ser compatível.
   Na 3.11 o Warp deu `CUDA error 700: illegal memory access` na cena de 2 robôs (provável overflow
   de `naconmax` default): passar `naconmax`/`njmax` explícitos desde o primeiro teste.
2. Na CPU (4.0), sem MJX: remedir a tabela de fatos do mapa (`nv`, `ncon`, pares), `model.nmesh` das duas cenas (confirma a duplicação de meshes por robô) e quantos
   pares o `geom_pairs` do MJX enumeraria.
3. Compilar a cena de 2 robôs e a de 7 robôs (`scene.build`, modelo cinemático) e:
   - `mjx.put_model(..., impl="warp")` + alguns `mjx.step` — deve funcionar no modelo base;
   - `mjx.put_model(..., impl="jax")` no modelo base, **só na cena de 2 robôs**: o sphere↔cylinder
     SDF existe na 3.11 (pesquisa 01), então deve carregar; medir tempo e pico de RAM do compile;
   - na cena de 7 robôs, `impl="jax"` **só** com uma variante descartável que corta chassi↔chassi
     (via `MjSpec`, só no script), nunca com o modelo base.
4. Registrar versões exatas, comandos de instalação, onde o env ficou, e as mensagens de erro.

## Resultado esperado

A resposta registra o env, as versões, e os resultados acima (incluindo pico de RAM e o `nmesh`). Script e env não entram em `src/`.

## Answer

Resolvido em 2026-11-08. Script: [`research/04-smoke/smoke.py`](../research/04-smoke/smoke.py) (um estágio por
processo); logs dos MJX-JAX ao lado (`jax2*.log`, `jax7nomesh.log`).

**Env.** `~/.venvs/mjsoccer-gpu` (Python 3.12.13, em disco). O projeto entra por `PYTHONPATH=src`,
sem instalar. Instalação:

```bash
uv venv -p 3.12 ~/.venvs/mjsoccer-gpu
VIRTUAL_ENV=~/.venvs/mjsoccer-gpu uv pip install "mujoco==4.0.0" "mujoco-mjx[warp]==4.0.0" "jax[cuda12]" gymnasium numpy
```

Todo processo: `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 …`.
O cgroup funcionou: os três OOMs abaixo mataram só o processo, e a máquina seguiu de pé.

**Versões resolvidas.** `mujoco 4.0.0`, `mujoco-mjx 4.0.0`, `warp-lang 1.18.0` (wheel com toolkit
CUDA 13.4; driver reporta 13.2, roda), `jax`/`jaxlib 0.12.0.1` + `jax-cuda12-plugin 0.12.0.1`
(CUDA 12.9 do pip), `numpy 2.5.4`. O FFI do Warp importa e roda com o JAX mais recente; não foi
preciso recuar. JAX em CUDA 12 e Warp em CUDA 13 convivem no mesmo processo. `mujoco-mjx` também
puxa `mujoco-studio`, `trimesh`, `scipy`.

**CPU (4.0), sem MJX.**

| cena | `nv` | `ngeom` | `nmesh` | `nmeshvert` | meshes que colidem | `npair` | pares do `geom_pairs` do MJX |
|---|---|---|---|---|---|---|---|
| 2 robôs | 20 | 424 | 423 | 325 665 | 202 | 17 | 9 366 (mesh↔mesh 9 017) |
| 7 robôs | 55 | 1 399 | 1 088 | 867 125 | 637 | 27 | 120 926 (mesh↔mesh 119 777) |

- Os meshes **são duplicados por robô**: (1088 − 423)/5 = 133 meshes a cada robô a mais; o campo tem
  ~157. Por robô colidem 87 meshes; no campo, 28 (paredes e gols).
- O mesh↔mesh se divide em robô↔robô (5 041 com 2 robôs, 105 861 com 7, igual à 3.11) **e robô↔campo**:
  28 × 71 = 1 988 por robô (3 976 com 2, 13 916 com 7). Essa parte a tabela de fatos do mapa não contava.
- **Correção da tabela do mapa**: os 12 `<pair>` mesh↔sphere são **bola × parede**
  (`col_f_walls_1_c00..c11`), não chassi × bola. A bola × chassi vai por contype/conaffinity contra
  os 87 meshes do robô (202 = 28 + 2·87 sphere↔mesh no `geom_pairs`). Por robô, os `<pair>` explícitos
  são só os 2 cilindros do dribbler.
- `ncon`/`nefc` com 3 000 passos de ação aleatória por task (seed 0, CPU):

  | task | `nv` | `ncon` médio / máx | `nefc` médio / máx |
  |---|---|---|---|
  | GoToBall | 55 | 1.01 / 11 | 10.6 / 42 |
  | GoToBallShoot | 13 | 1.00 / 1 | 7.2 / 9 |
  | Dribbling | 41 | 0.85 / 11 | 8.3 / 41 |
  | StaticDefenders | 55 | 1.00 / 1 | 9.9 / 21 |
  | ContestedPossession | 20 | 0.99 / 2 | 7.6 / 12 |
  | PassEndurance | 20 | 0.16 / 3 | 3.3 / 16 |

  Ação aleatória quase não produz contato robô↔robô; o 05 deve medir também com o controlador
  scriptado e com colisões forçadas.

**MJX-Warp, modelo base** (`nworld` 64, 24 passos por chamada, `njmax=256`):

| cena | `naconmax`/mundo | compile 1ª chamada | `overflow` | `nacon` | `ncollision` (broadphase) | smoke SPS |
|---|---|---|---|---|---|---|
| 2 robôs | 64 | 62 s (cache frio de kernels Warp; ~2 s depois) | 0 | 1/mundo | 35/mundo | ~15 k |
| 7 robôs | 64 | 13 s | **4 (broadphase)** | — | 120/mundo | — |
| 7 robôs | 256 | 1.7 s | 0 | 1/mundo | 120/mundo | ~9 k |

- O `CUDA error 700` da 3.11 não apareceu na 4.0 com `naconmax`/`njmax` explícitos.
- **O `naconmax` se dimensiona pelos candidatos do broadphase** (`ncollision`, ~120 por mundo com 7
  robôs parados a 0.5 m), não pelo `max(ncon)` da CPU. Com 64/mundo dá
  `broadphase overflow - please increase nconmax beyond 64 or naconmax beyond 4096`.
- GPU < 1 GB em `nworld` 64; RSS do host 2.5–3.1 GB. SPS aqui é de smoke (`nworld` 64), não benchmark.

**MJX-JAX** (`nworld` 8 nos OOMs, 64 nos que rodaram):

| cena | modelo | pares mesh↔mesh | resultado |
|---|---|---|---|
| 2 robôs | base | 9 017 | `put_model` 1.9 s; **OOM no JIT** (8 GB do host, ~11.5 min de compile) |
| 2 robôs | corte chassi↔chassi (`<exclude>` entre raízes) | 3 976 (todos robô↔campo) | **OOM no JIT** (8 GB, ~11.7 min) |
| 2 robôs | `nomesh`: nenhum mesh de robô colide | 0 (33 pares no total) | roda: compile 13.7 s, pico 2.6 GB, finito, ~770 SPS |
| 7 robôs | `nomesh` | 0 (43 pares) | roda: compile 15.1 s, pico 3.1 GB, finito, ~780 SPS |

- O estouro é de memória do **host durante a compilação XLA** (o `put_model` termina em 2 s). A
  pesquisa 01 previa memória de GPU por mundo. Com 2 robôs, só o robô↔campo já não compila em 8 GB.
- **Consequência para o mapa**: cortar chassi↔chassi **não basta**. O MJX-JAX precisa que **nenhum
  mesh do robô** fique em par de colisão, então a variante troca a colisão de robô inteira
  (robô↔robô, robô↔parede/gol, **e bola↔chassi**, hoje 87 meshes) por primitivas ou poucos cascos
  convexos. É o conteúdo do ticket da variante.
- `nomesh` é só prova de que o caminho MJX-JAX compila e passa. Sem a bola × chassi e sem robô ×
  parede, ele não serve como variante.
- O MJX-JAX avalia todo par estático a cada passo (`ncon` = 33/43 por mundo, sempre). O SPS de smoke é
  ~20× menor que o do Warp no mesmo `nworld`; os números de verdade ficam para o 05.

**Não feito aqui (vai para o 05):** `mujoco.rollout` na CPU, varredura de `nworld`, `WARP` ×
`WARP_STAGED`, precisão do SDF sphere↔cylinder.
