# O que o `mujoco.studio` oferece para viewer, vídeo e GIF (ticket 20)

Fontes: `google-deepmind/mujoco` tag `4.0.0` (commit `ac60eae`, clonado em `~/src/mujoco-4.0.0`):
`doc/changelog.rst`, `doc/studio/*.rst`, `doc/python.rst`, `doc/programming/rendering.rst`,
`doc/programming/samples.rst`, `doc/skills/{studio,rendering}/SKILL.md`, `python/mujoco/studio/`,
`python/mujoco/rendering/`, `studio/hal/`, `src/render/filament/core/`. O pacote instalado em
`~/.venvs/mjsoccer-gpu` (`mujoco` 4.0.0) bate com a tag. Medidas em 2026-11-09 na RTX 3060, sessão
Wayland (com XWayland, `DISPLAY=:0`). Scripts e saída em [`20-studio/`](20-studio/)
(`studio_probe.py`, `native_probe.py`, `results.jsonl`, `log*.txt`). Fora de `src/`.

## 1. O que é

- **Viewer interativo novo, não gravador.** "MuJoCo Studio is a Python and C++ SDK for building
  interactive simulations [...] the next iteration of the `simulate` application and
  `mujoco.viewer`" (`doc/studio/index.rst`). Troca o renderer OpenGL clássico pelo **Filament** (PBR),
  os painéis `mjUI` por Dear ImGui, e o laço acoplado por plugins (`ViewerPlugin`/`SimPlugin`) que
  conversam por mensagens.
- Três formas: app C++ standalone (`bin/mujoco_studio`, só no tarball do release); CLI Python
  `python -m mujoco.studio.viewer --model=x.xml [--gfx=web]` (o pip instala `mujoco-studio` como
  console script); e **API Python** em `mujoco.studio` (`launch_native`, `launch_web`,
  `launch_passive`, `app`, `viewer_protocol`, `step_control`, `sim_history`, `plugin`).
- **Equivalente do `launch_passive`** = `launch_native.launch(config, plugins=[studio_app.App()])`.
  É um context manager que sobe o viewer numa thread daemon e devolve um `ViewerHandle`; o laço do
  usuário fica assim (`doc/studio/native_viewers.rst`, "Non-Blocking Mode"):

  ```python
  with launch_native.launch(cfg, plugins=[studio_app.App()]) as handle:
      while handle.is_running():
          model, data = handle.sync(model, data)   # sempre religar: pode voltar objeto novo
          mujoco.mj_step(model, data)
  ```

  Sem `StepControl()` nos plugins o `sync()` não avança a física nem dorme — o passo e o ritmo são
  do usuário. Diferenças para o `mujoco.viewer`: não há `lock()` nem `MjData` compartilhado; o
  `sync()` copia o estado (`mj_getState(mjSTATE_INTEGRATION)`) num `StateSnapshot` e o viewer faz
  `mj_setState` + `mj_forward` na própria cópia (`viewer_handle.py:384`, `viewer_protocol.py:251`).
  A câmera fica no viewer (`viewer.camera`), mexida por um `ViewerPlugin`, não pelo laço.
- `launch_passive.launch(config)` escolhe web ou nativo por `config.gfx`; `.run(...)` é a versão
  bloqueante (o Studio dirige o laço, com `*studio_app.SIM_PLUGINS`).
- **O `mujoco.viewer` antigo continua no 4.0**, documentado em `python.rst` (`PyViewer`) sem aviso
  de depreciação; o `mjx-viewer` do próprio MJX ainda usa `mujoco.viewer.launch_passive` +
  `mjx.get_data_into`. O Studio é a recomendação para coisa nova, não uma troca obrigatória.

## 2. Modos gráficos e o que roda nesta máquina

`ViewerConfig.gfx` ∈ `''`, `opengl`, `opengl_headless`, `opengl_software`, `vulkan`,
`vulkan_software`, `web`, `webgl` (`viewer_protocol.GFX_MODES`).

| modo | o que é (fonte) | aqui (Wayland + XWayland) |
|---|---|---|
| `''` / `opengl` / `vulkan` | janela SDL com swapchain do Filament | **aborta**: `ValueError: Target not found: ''` no `NativeViewer.sync`, depois `Engine::shutdown() called from the wrong thread!` (core dump) |
| `opengl_headless` | Filament renderiza em textura e o SDL faz *blit* por software numa janela (`studio/hal/window.cc:155`) | **roda**: 455 syncs em 5 s, `sync()` 0.15 ms |
| `web` | `WebViewer`: só CPU no host, rasteriza no browser (WASM/WebGL) por um porto HTTP/WS (`doc/studio/web_viewers.rst`) | **roda**, sem display: `sync()` 0.17 ms; imprime `http://localhost:8099/`; **sem autenticação** (o próprio banner avisa) |

- O doc diz que "in a Wayland session, omitting `gfx` falls back to `'opengl_headless'`", mas
  esse fallback só existe no launcher C++ (`studio/launcher.cc:42`); o `Window` do Python mapeia `''`
  para `FilamentOpenGl` (`python/mujoco/studio/window.cc:66`). **No Python, numa sessão Wayland,
  passar `gfx='opengl_headless'` explicitamente** (ou `web`).
- `opengl_headless` ainda abre uma janela (precisa de display); headless de verdade é o `web` ou o
  renderer offscreen da seção 3.

## 3. Gravação: vídeo, GIF, frames offscreen

- **O Studio não grava.** Nada de screenshot/vídeo/GIF em `python/mujoco/studio/` nem em
  `studio/*.cc`; o único "record" é o `SimHistory` (buffer de estados para voltar no tempo, não de
  imagens). Os docs do Studio mandam offscreen para o skill `mujoco-rendering`.
- **Vídeo "oficial" do 4.0** é o sample C++ `render` (changelog 4.0 item 41: `record.cc` removido,
  `render model.xml out.rgb --duration=T --fps=F`, ou `-` para mandar RGB24 ao `ffmpeg`,
  `--backend=filament|classic`). É binário de sample, não está no pip, e só renderiza o modelo
  rodando sozinho (não um estado vindo do nosso laço) — não serve para os GIFs do repo.
- **Offscreen em Python com Filament** = `mjrf.Context` + `mujoco.rendering.filament.ModelScene` +
  `renderer.Renderer` (`python.rst`, `PyFilamentRender`; `sample/studio/filament_render.py`):
  `ms.update(data)` a cada frame, `r.render([View(scene, target, camera=mjv_camera2GLCamera(...))])`
  enche um `np.ndarray` (H, W, 3) qualquer — sem o limite `offwidth/offheight` do clássico.
  Câmeras: qualquer `MjvCamera` (fixa do MJCF, livre, `trackcom`) via `mjv_camera2GLCamera`.
  Pegadinhas medidas:
  - **o exemplo do `python.rst` não roda como está**: sem `import mujoco.studio.window` antes do
    contexto, sai `ERROR: Failed to open filament asset 'pbr.filamat'` e o processo morre (a
    docstring de `mujoco/rendering/filament/__init__.py` avisa: o import registra os provedores de
    material);
  - **sem display, só Vulkan**: `GRAPHICS_API_OPENGL` sem `DISPLAY` → `Failed to open X display.
    (exiting)` (a fábrica de plataforma open source não escolhe EGL,
    `src/render/filament/core/filament_platform_factory.cc`). `GRAPHICS_API_DEFAULT` já é Vulkan no
    Linux e roda headless na RTX 3060 (driver NVIDIA). `MUJOCO_GL=egl` não afeta o Filament.
- Custo por frame (campo `overview`, mediana de 55 frames, `models/scene.xml`):

  | renderer | 640×360 | 1280×720 |
  |---|---|---|
  | `mujoco.Renderer` clássico, `MUJOCO_GL=egl` | 0.5 ms | 1.1 ms |
  | Filament Vulkan offscreen | 2.2 ms | 4.3 ms |

  Os dois folgam para GIF a 20 fps. Visual: o Filament sai mais escuro (o modelo não tem luz com
  `intensity`, então cai no modo "compatível com o clássico", `rendering.rst`), com serrilhado nas
  linhas finas do campo e linhas da grande área sumindo a essa distância; o clássico desenha as
  linhas limpas. Para os GIFs do repo o Filament é mais caro e não melhora nada sem retocar materiais
  e luzes no MJCF.
- **GIF continua sendo nosso**: nenhum dos dois escreve GIF/MP4; o `pillow` do grupo `dev` (ou
  `imageio`, que o 4.0 só puxa no extra `sysid`) segue necessário.

## 4. Um mundo de um lote MJX-Warp

Não há suporte a lote no Studio (nada de `nworld`/`mjx`/`warp` em `mujoco/studio/`); o viewer
recebe um `MjModel` + `MjData` de CPU. O caminho é copiar um mundo para um `MjData` e dar `sync()`.
Medido com 1 024 mundos, modelo de 2 robôs (`scene.build(1, 1, 0)`), `WARP_STAGED`
([`results.jsonl`](20-studio/results.jsonl)):

| o quê | ms |
|---|---|
| `mjx.get_data` de um mundo (fatiar todas as folhas do `dx` em `i`, depois `get_data`) | 51 |
| cópia enxuta: `qpos`, `qvel`, `time` do mundo `i` → `MjData` + `mj_forward` na CPU | 1.0 |
| enxuta + frame Filament 640×360 | 4.0 |
| `handle.sync()` do `WebViewer` (depois da cópia) | 0.17 |

- O viewer só lê o estado de integração e roda o próprio `mj_forward`, então o `get_data` completo
  (contatos, `efc_*`) é desperdício: a cópia de `qpos`/`qvel` custa ~1 ms por frame e não mexe no
  lote. Visualizar o mundo `i` a ~30 fps custa uns 30 ms/s de host — dá para fazer entre épocas ou
  num callback de progresso, não dentro do `scan` do treino.
- Só funciona como espectador: perturbar com o mouse ou o reset do viewer mudam a cópia de CPU, não
  o lote.

## 5. Requisitos e dependências

- Desde o 4.0 (changelog item 3) o pip separa `mujoco-core` (bindings + lib, só `numpy`),
  `mujoco-studio` (Studio + bindings Filament `mujoco._render_filament`; depende de `websockets>=13`)
  e `mujoco` (meta, instala os dois e ainda `glfw` + `pyopengl`). No venv: `mujoco-studio` ocupa
  34 MB, `mujoco-core` 30 MB.
- **O `mujoco>=3.11.0` do `pyproject.toml` já traz o Studio no 4.0** — não precisa de grupo
  opcional. O `mujoco-mjx` depende de `mujoco-core[rendering]`, não do Studio. Para enxugar, a
  base poderia trocar por `mujoco-core` e deixar `mujoco-studio` num grupo `viewer`, mas aí o
  `render_mode="human"` atual (`mujoco.viewer`, que mora no core e pede `glfw`) também precisa do
  extra `rendering`.
- GPU: nativo e Filament offscreen pedem Vulkan ou OpenGL com display; `web` não pede GPU nem
  display no host (o browser rasteriza). macOS: `launch_native.launch` exige `mjpython`
  (`native_viewers.rst`); `launch_web` não.

## 6. O que substitui hoje

| hoje no repo | no 4.0 | troca vale? |
|---|---|---|
| `mujoco.viewer.launch_passive` em `base.py` (`render_mode="human"`) | `launch_native.launch(ViewerConfig(gfx='opengl_headless'), plugins=[studio_app.App()])`, laço `model, data = h.sync(model, data)` | possível; o enquadramento inicial (`_frame_field`) passa a ser um `ViewerPlugin` que mexe em `viewer.camera` no `ViewerInitEvent`. Ganho: UI nova, `web` para máquina remota/headless. Custo: `gfx` explícito por causa do Wayland e o crash dos outros modos aqui |
| `mujoco.Renderer` em `base.py` (`rgb_array`) | `mujoco.Renderer` continua igual (`mujoco/renderer.py` agora reexporta `rendering.classic.renderer`); a alternativa é `rendering.filament` | **não**: mesma API do Gymnasium, 2–4× mais rápido, sem as pegadinhas do Filament |
| `scripts/make_gifs.py` (EGL + `mujoco.Renderer` + pillow) | nada no Studio; Filament offscreen exigiria Vulkan, o import de `mujoco.studio.window` e retoque de luzes | **não** |
| `--play` do treino (ticket 08) | `launch_web` ou `launch_native` alimentado pela cópia enxuta de um mundo (seção 4) | é aqui que o Studio acrescenta algo: assistir um mundo do lote GPU, inclusive de máquina sem display via `web` + `ssh -L` |
