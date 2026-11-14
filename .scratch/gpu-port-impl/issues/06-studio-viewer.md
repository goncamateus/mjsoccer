# 06: Viewer `mujoco.studio` para o `render_mode="human"`

Spec: [port GPU](../../gpu-port/spec.md)

**What to build:** O `render_mode="human"` (e com ele o `--view` e o `--play` dos exemplos) usa `launch_native.launch` + `handle.sync`, sem fallback para o `mujoco.viewer`. Kwarg `viewer_gfx`: `opengl_headless` (default) ou `web`; o docstring avisa que o `web` escuta em todas as interfaces sem autenticação. A pose de câmera de hoje vira um `ViewerPlugin` aplicado só no primeiro frame. O env ignora o `(model, data)` do `sync`; janela fechada não derruba o episódio e o próximo `render()` reabre. `rgb_array`, `make_gifs` e `visualize_sweep` ficam no `mujoco.Renderer`.

**Blocked by:** 02 (MuJoCo 4.0 no backend CPU, com kick K = 32)

**Status:** done

- [x] `render_mode="human"` abre o Studio com a câmera de hoje e câmera livre depois
- [x] `viewer_gfx="web"` funciona e o aviso de segurança está no docstring
- [x] Fechar a janela no meio do episódio não levanta erro; o próximo `render()` reabre
- [x] `rgb_array` e `make_gifs` inalterados; `uv run pytest` verde

## Comments

**Feito em 2026-11-14.**

- `render_mode="human"` agora abre o Studio por `launch_passive.launch` (que despacha para
  `launch_native` ou `launch_web` pelo `gfx`) com `studio_app.App()` e um `FrameField`, num
  módulo novo `src/mjsoccer/viewer.py`, importado só quando o viewer abre. Fora do `base.py`
  porque o `from __future__ import annotations` dele quebra o registro dos handlers do Studio
  (que lê a anotação do evento como classe). No `base.py` ficam o kwarg `viewer_gfx`
  (validado no construtor, docstring com o aviso de segurança do `web`), o `render()` e o
  `close()`. Sem `StepControl`: o `sync` não passa nem pausa a física.
- `FrameField` aplica a pose de hoje no primeiro `PostModelEvent` (prioridade `INTERNAL - 1`,
  depois do `App` alinhar a câmera dele no mesmo evento) e nunca mais.
- **Achado upstream (mujoco 4.0.0):** só um `WebViewer` funciona por processo; o segundo trava
  em `HeadlessUi.new_frame()` (nunca desenha), o `close()` estoura o timeout de 5 s e o
  servidor HTTP continua escutando até o processo sair (`gc.collect()` não resolve; é estado
  nativo do NetImgui). O env recusa um segundo `web` no processo com `RuntimeError` em vez de
  vazar a porta. Basta para `--view`/`--play` (um env, um viewer); o nativo reabre sem limite.
- Testes (`tests/test_viewer.py`): `viewer_gfx` inválido recusado; `web` real escuta enquanto
  aberto, para de escutar no `close()` e recusa o segundo; reabertura após janela fechada com
  um `ViewerHandle` falso na fronteira do Studio; `FrameField` aplicado uma vez via
  `PluginRegistry` real. Nenhum teste deixa porta aberta (conferido com `ss -ltn`).
- Verificado à mão no `opengl_headless` (Wayland/XWayland, RTX 3060): janela abre; câmera em
  `lookat` 0, `distance` 13.2, `azimuth` 90, `elevation` -35 após 120 frames, com o `App` no
  modo tumble (mouse livre); `ExitEvent` do lado do viewer (o que o botão fechar faz) no meio
  do episódio não levanta, o próximo `step` reabre outra janela e o `close()` não deixa thread.
- `rgb_array` / `make_gifs` / `visualize_sweep` intocados; `rgb_array` conferido com
  `MUJOCO_GL=egl` (720x1280). Sem EGL, um `mujoco.Renderer` qualquer cai nesta sessão GNOME
  com erro do GLib (`xsettings ... antialiasing`), independente deste ticket.
- `docs/index.md` (seção Rendering) atualizado.
