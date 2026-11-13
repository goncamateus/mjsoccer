# 06: Viewer `mujoco.studio` para o `render_mode="human"`

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** O `render_mode="human"` (e com ele o `--view` e o `--play` dos exemplos) usa `launch_native.launch` + `handle.sync`, sem fallback para o `mujoco.viewer`. Kwarg `viewer_gfx`: `opengl_headless` (default) ou `web`; o docstring avisa que o `web` escuta em todas as interfaces sem autenticação. A pose de câmera de hoje vira um `ViewerPlugin` aplicado só no primeiro frame. O env ignora o `(model, data)` do `sync`; janela fechada não derruba o episódio e o próximo `render()` reabre. `rgb_array`, `make_gifs` e `visualize_sweep` ficam no `mujoco.Renderer`.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** ready-for-agent

- [ ] `render_mode="human"` abre o Studio com a câmera de hoje e câmera livre depois
- [ ] `viewer_gfx="web"` funciona e o aviso de segurança está no docstring
- [ ] Fechar a janela no meio do episódio não levanta erro; o próximo `render()` reabre
- [ ] `rgb_array` e `make_gifs` inalterados; `uv run pytest` verde
