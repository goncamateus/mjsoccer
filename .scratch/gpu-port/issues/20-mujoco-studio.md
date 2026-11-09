# O que o `mujoco.studio` oferece para viewer, vídeo e GIF

Type: research
Status: resolved
Blocked by: —

## Pergunta

O dev quer trocar a visualização e a gravação de vídeo/GIF pelo `mujoco.studio` (MuJoCo 4.0), mesmo
com o `--play` do treino já rodando no backend CPU ([ticket 08](08-ppo-training-scripts.md)).
Levantar, nas fontes primárias (`google-deepmind/mujoco` tag 4.0.0: docs, `python/mujoco/`, changelog):
- o que é o `mujoco.studio` no 4.0 (app, módulo Python, API), como abre/roda uma cena a partir de
  `MjModel`/`MjData` controlados por código (passo externo, como o `mujoco.viewer.launch_passive`);
- gravação: vídeo e GIF/frames offscreen, headless (sem display), resolução, câmeras;
- se mostra um mundo de um lote MJX-Warp (via `mjx.get_data` de um índice) e a que custo;
- requisitos (GPU/OpenGL/EGL, extras de pip) e se cabe na base (`mujoco`) ou num grupo opcional;
- o que substitui hoje: `mujoco.viewer`, `mujoco.Renderer` e a gravação de GIF do repo.

## Answer

- `mujoco.studio` é o viewer interativo novo (Filament + Dear ImGui + plugins), sucessor do
  `mujoco.viewer`, que continua no 4.0 sem depreciação. O análogo do `launch_passive` é
  `launch_native.launch(cfg, plugins=[studio_app.App()])` com `model, data = handle.sync(model, data)`
  e passo/ritmo do usuário; o `sync` manda o estado de integração e o viewer roda o próprio `mj_forward`.
- Nesta máquina (Wayland + XWayland) só `gfx='opengl_headless'` e `gfx='web'` rodam; `''`/`opengl`/
  `vulkan` abortam (`Target not found`). O fallback de Wayland do doc só existe no launcher C++.
  `web` roda sem display e sem GPU no host (o browser rasteriza), sem autenticação.
- **O Studio não grava vídeo nem GIF.** O vídeo do 4.0 é o sample C++ `render --duration --fps`
  (fora do pip). Offscreen em Python com Filament (`mjrf` + `ModelScene` + `Renderer`) funciona, mas
  pede `import mujoco.studio.window`, só roda headless com Vulkan, e custa 2.2/4.3 ms por frame
  (640×360/1280×720) contra 0.5/1.1 ms do `mujoco.Renderer` com EGL, que segue no 4.0.
- Um mundo de um lote MJX-Warp: sem suporte a lote; copiar `qpos`/`qvel`/`time` do mundo `i` para um
  `MjData` + `mj_forward` custa ~1 ms (o `mjx.get_data` completo de um mundo, 51 ms), e o `sync()`
  0.17 ms. Serve como espectador para o `--play`/progresso.
- Dependências: o meta-pacote `mujoco` 4.0 já instala `mujoco-studio` (+`websockets`), então o
  `mujoco>=3.11.0` da base já traz o Studio; nenhum grupo novo é necessário.
- Troca: `mujoco.Renderer` (`rgb_array`) e `make_gifs.py` ficam como estão; o `human`/`--play` pode
  passar ao Studio (nativo `opengl_headless` ou `web`).

Detalhes, medidas e fontes: [research/20-mujoco-studio.md](../research/20-mujoco-studio.md).
