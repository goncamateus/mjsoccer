# Rodar o `brax ppo.train` sobre o MJX-Warp nesta máquina

Type: task
Status: resolved
Blocked by: 08

## Pergunta

O 08 adotou o `brax ppo.train`, mas o 02 deixou sem verificar: Brax com `jax 0.12.0.1` + MJX-Warp
(`impl="warp"`), o `pmap` incondicional do Brax numa GPU só, e o SPS na RTX 3060.

No env isolado `~/.venvs/mjsoccer-gpu` (em disco; job com `MemoryMax`; sem mexer em `src/`,
`models/`, `pyproject.toml`): instalar o Brax compatível, rodar `ppo.train` sobre um env MJX-Warp de
brinquedo (o modelo base de 2 robôs com recompensa trivial serve) com reset por `mjx.Data.where`, e
registrar:

- a versão do Brax que instala com o `jax` atual (o pin do grupo `gpu`);
- se o `pmap` com 1 dispositivo roda com Warp, ou se o fallback do 08 (vendorizar `ppo/train.py` com
  `jit`) é necessário;
- SPS de treino ponta a ponta contra o SPS bruto do 05, a 1 024 e 8 192 mundos, e a memória de GPU.
- com DR por slot ([ticket 11](11-dr-capacity.md)): um `randomization_fn` que varia um campo (ex.
  massa da bola) com `in_axes`, se o grafo CUDA do Warp recaptura a cada passo (modo default vs
  `WARP_STAGED`) e o SPS com e sem DR.

## Answer

Resolvido em 2026-11-08. Detalhe e tabelas em
[`research/12-brax-ppo-warp-smoke.md`](../research/12-brax-ppo-warp-smoke.md); script e saída bruta em
[`research/12-brax/`](../research/12-brax/). Env de brinquedo de 2 robôs (robô 0 persegue a bola),
auto-reset por `mjx.Data.where` dentro do `vmap`, `njmax` 256 e `naconmax` 128/mundo.

- **Versão**: o `brax 0.14.2` do PyPI quebra no `jax 0.12` (`jax.device_put_replicated` removido).
  Pin do grupo `gpu`: `brax @ git+https://github.com/google/brax@c8605294264a7b7546268adbb4beabc9eb624000`
  (`main` de 2026-10-20, já corrigido), até uma release > 0.14.2. Não mexe em `mujoco`/`jax`/`warp`.
- **`pmap` com 1 GPU roda com Warp** (treino e reset). O fallback de vendorizar `train.py` sai.
- **SPS de treino** (decisões/s, sem eval): `WARP` 24.4 k a 1 024 (91 % da física bruta do 05) e
  31.3 k a 4 096 (88 %); `WARP_STAGED` 13.2 k / 16.4 k; `NONE` 17.8 k a 4 096, 25.2 k a 8 192.
- **Recaptura**: `WARP` recaptura o grafo **a cada época** (não a cada passo), e a memória das
  capturas não volta, nem com `graph_cache_max=2`: **OOM na época 3 a 8 192 e na época 6 a 4 096**.
  `WARP_STAGED` (5 capturas, plano) e `NONE` (zero) ficaram estáveis por 30 épocas a 4 096;
  `WARP_STAGED` dá OOM a 8 192 logo no início. Com `NONE`, a eval de 128 mundos fica ~4× mais lenta.
- **DR por slot** (`in_axes` sobre `body_mass` no `DomainRandomizationVmapWrapper` do Brax): roda nos
  modos testados, custo de SPS zero, não acrescenta capturas.
- **Consequência**: o `--num-envs` 8 192 do 08 não roda estável nesta GPU com nenhum modo de grafo
  rápido. A escolha de modo e de `nworld` default virou o ticket
  [Modo de grafo do Warp e `--num-envs` default do treino](13-graph-mode-and-num-envs.md).
