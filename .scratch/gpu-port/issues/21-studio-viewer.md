# Levar o `render_mode="human"` e o `--play` para o `mujoco.studio`: como

Type: grilling
Status: open
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
