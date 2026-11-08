# Uma implementação da lógica das tasks para os dois backends, ou duas?

Type: grilling
Status: open
Blocked by: 01

## Pergunta

A lógica das tasks — observação, recompensa, término, reset (posicionamento, `seat_ball`), posse,
controlador de navegação, oponentes — tem hoje uma implementação em numpy sobre `MjData`
(`src/mjsoccer/base.py`, `src/mjsoccer/tasks.py`). O MJX precisa dela funcional e vetorizada em JAX.

- **Uma implementação** compartilhada (ex.: código JAX/array-API que também dirige o backend CPU, ou
  o backend CPU embrulhando a versão JAX) ou **duas** (numpy no CPU, JAX no MJX), com a paridade
  verificada por testes?
- Se forem duas: o que garante que não divergem quando uma task muda?
- A API do lado MJX: classe Gymnasium vetorizada, interface funcional `reset(key) / step(state,
  action)` no estilo Playground/Brax, ou as duas?
