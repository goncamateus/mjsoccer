# Uma implementação da lógica das tasks para os dois backends, ou duas?

Type: grilling
Status: resolved
Blocked by: 01

## Pergunta

A lógica das tasks — observação, recompensa, término, reset (posicionamento, `seat_ball`), posse,
controlador de navegação, oponentes — tem hoje uma implementação em numpy sobre `MjData`
(`src/mjsoccer/base.py`, `src/mjsoccer/tasks.py`). O MJX precisa dela funcional e vetorizada em JAX.

- **Uma implementação** compartilhada (ex.: código JAX/array-API que também dirige o backend CPU, ou
  o backend CPU embrulhando a versão JAX) ou **duas** (numpy no CPU, JAX no MJX), com a paridade
  verificada por testes?
- Se forem duas: o que garante que não divergem quando uma task muda?
- A API do lado MJX: a interface funcional `brax.envs.Env` é obrigatória (ticket 08, treino por
  `brax ppo.train`); falta decidir se também há uma classe Gymnasium vetorizada por cima.

## Answer

Resolvido em grilling (2026-11-08).

1. **Duas implementações.** O backend CPU fica em numpy sobre `MjData`, como está. O lado MJX é JAX
   funcional novo. Uma só em JAX poria JAX na base (restrição 11), e uma genérica por `xp` não cobre
   o controle de fluxo (rejeição com `raise`, loop do `seat_ball`, `if` por task).
2. **Contra divergência, o teste de task-logic parity por snapshot.** Rollouts CPU (seeds 0-4) geram
   snapshots de `MjData` depois do `mj_forward`. Cada um vai para `mjx.Data` por `put_data`; com a
   mesma ação, os dois lados têm de dar a mesma observação, recompensa, término, posse e saída do
   controlador de navegação, na tolerância de float32. É `jnp` pura sobre os campos do `Data`, então
   roda em JAX-CPU, sem GPU. Cobre as 6 tasks. Mudar uma task obriga a mexer nos dois lados. É
   separado da paridade estatística da física do ticket 09.
3. **Fonte única de números no CPU.** O lado MJX importa as constantes de módulo (`base`, `scene`,
   `tasks`) e lê a config de cada task dos atributos da classe CPU correspondente (`n_yellow`,
   `energy_steps`, ...). Nenhum número é copiado.
4. **Posse pelo sensor `contact` nos dois backends.** O CPU deixa de varrer `data.contact`
   (`src/mjsoccer/base.py:237`) e lê o mesmo sensor geom-geom do MJX (pesquisa 01). O sensor não é
   descrição física (restrição 7 intacta). Os testes de posse do CPU rodam de novo.
5. **API: só `brax.envs.Env`.** Não há classe Gymnasium vetorizada; o `--play` e o render usam o
   backend CPU (ticket 08).
6. **Layout**: subpacote `src/mjsoccer/mjx/` com `base.py` (env base, controlador, posse, observação
   comum), `tasks.py` (as 6 tasks) e `wrappers.py` (equivalente do `wrap_for_brax_training`, com
   `Data.where`/`full_reset`; o desenho do reset fica na fog "Reset em lote"). `import mjsoccer`
   nunca importa `mjsoccer.mjx`.
7. **Entrada**: `mjsoccer.mjx.make(id, **kw)` lê o mesmo `ENV_IDS` (ids e horizontes iguais), sem
   `envs.register_environment` do Brax.

Termo novo no `CONTEXT.md`: **task-logic parity**.
