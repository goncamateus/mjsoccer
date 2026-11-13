# Levar o `render_mode="human"` e o `--play` para o `mujoco.studio`: como

Type: grilling
Status: resolved
Blocked by: 20

## Pergunta

O dev quer a visualização no `mujoco.studio`. A [pesquisa do studio](20-mujoco-studio.md) mostrou que
ele substitui o `mujoco.viewer.launch_passive` (`launch_native.launch` + `handle.sync`), já vem com o
`mujoco` 4.0, mas não grava vídeo e renderiza offscreen 4× mais lento que o `mujoco.Renderer`/EGL.
Decidir:
- o que migra: `render_mode="human"` e o `--play`; se `rgb_array` e `scripts/make_gifs.py` ficam no
  `mujoco.Renderer` (a pesquisa recomenda);
- o `gfx` default (`opengl_headless` nativo × `web` sobre `ssh -L`, sem autenticação) e se é kwarg;
- o `_frame_field` virando um `ViewerPlugin` que posiciona a câmera na abertura;
- se o `--play` mostra um mundo do lote MJX-Warp (cópia `qpos/qvel/time` + `mj_forward`, ~1 ms) ou só
  o backend CPU, como o [ticket do PPO](08-ppo-training-scripts.md) fixou.

## Answer

- **Só o `render_mode="human"` migra** para o Studio (`launch_native.launch` + `handle.sync`); o
  `--view` de `tasks.py`/`random.py` e o `--play` de `ppo_agent.py`/`sac_agent.py` vêm junto, porque
  passam pelo `render()`. `rgb_array`, `scripts/make_gifs.py` e `examples/visualize_sweep.py` ficam no
  `mujoco.Renderer`/EGL. Sem fallback para o `mujoco.viewer`.
- **`gfx`**: kwarg `viewer_gfx` do env, aceita `opengl_headless` (default) e `web`; sem flag nova nos
  scripts. O docstring avisa que o `web` escuta em **todas as interfaces, sem autenticação**
  (`self._host = '::'` fixo em `mujoco/studio/web_viewer.py:224` no 4.0.0; há um TODO upstream para
  loopback).
- **Câmera**: o `_frame_field` vira um `ViewerPlugin` em `base.py` que aplica a pose de hoje
  (`lookat` 0, `distance` 2.2 × `FIELD_HALF_X`, `azimuth` 90, `elevation` −35) uma vez, no primeiro
  frame; câmera livre, mouse livre.
- **`--play`** continua só no backend CPU (ticket 08); o espectador de um mundo do lote MJX-Warp fica
  fora de escopo.
- **`sync`**: o env ignora o `(model, data)` devolvido e segue dono do estado; o viewer é espectador
  (edições na UI não voltam). Janela fechada (`not handle.is_running()`) não derruba o episódio; o
  próximo `render()` reabre.
