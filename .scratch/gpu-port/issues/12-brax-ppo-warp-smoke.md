# Rodar o `brax ppo.train` sobre o MJX-Warp nesta máquina

Type: task
Status: open
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
