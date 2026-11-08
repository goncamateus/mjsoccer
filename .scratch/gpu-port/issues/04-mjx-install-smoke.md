# Instalar o MJX nesta máquina e rodar o modelo base e a variante

Type: task
Status: open
Blocked by: —

## Pergunta

Nada a decidir: o 05 e o 08 precisam de um ambiente que funcione nesta RTX 3060 (driver 595).

Num env isolado (fora do `uv` do projeto, sem tocar `pyproject.toml`):

1. Instalar `mujoco-mjx[warp]` e JAX com CUDA, nas versões que o 01 apontar (ou nas mais
   recentes compatíveis com `mujoco 3.11.0`, se o 01 ainda não fechou).
2. Compilar a cena de 2 robôs e a de 7 robôs (`scene.build`, modelo cinemático) e:
   - `mjx.put_model(..., impl="warp")` + alguns `mjx.step` — deve funcionar no modelo base;
   - `mjx.put_model(..., impl="jax")` no modelo base — deve **recusar** o par sphere↔cylinder;
   - uma variante descartável com os dois cilindros do dribbler trocados por cápsulas (via
     `MjSpec`, só no script) em `impl="jax"` — deve funcionar.
3. Registrar versões exatas, comandos de instalação, onde o env ficou, e as mensagens de erro.

## Resultado esperado

A resposta registra o env, as versões, e os três resultados acima. Script e env não entram em `src/`.
