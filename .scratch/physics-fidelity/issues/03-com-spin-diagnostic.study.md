Type: study
Source: [`03-com-spin-diagnostic.md`](03-com-spin-diagnostic.md)

# Guia de estudo: Diagnóstico — confirmar o mecanismo de tombamento por deslocamento do CoM

O chassi do robô tem o centro de massa 3.459 mm fora do eixo de giro (uma realidade de
fabricação/CAD, não um bug). Gire o robô rápido o bastante no lugar e esse deslocamento minúsculo
gera força lateral suficiente para tombá-lo — como o tambor desbalanceado de uma máquina de lavar
que sai andando pelo chão quando acelera. Este ticket ainda não pede para você consertar nada: pede
para você provar que essa história é de fato o que está acontecendo, e medir o quão ruim está hoje,
antes que alguém mexa no modelo.

## Antes de mexer em qualquer coisa

- **Centro de massa da subárvore no MuJoCo (`subtree_com`)** — é o número que o ticket compara com
  `xpos` para obter o deslocamento de 3.459 mm; é calculado ponderando pela massa a posição de cada
  corpo subindo a árvore cinemática. [MuJoCo Computation docs](https://mujoco.readthedocs.io/en/stable/computation.html)
- **O elemento `inertial`/`ipos`** — o atributo que o ticket manda zerar temporariamente
  (`models/robot_rollers.xml:153`) para falsear o mecanismo. Se um corpo não tem um filho `inertial`
  explícito, o MuJoCo infere massa/inércia a partir das suas geoms — bom saber antes de sair editando.
  [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- **Tombamento giroscópico/centrífugo por uma massa fora do eixo durante o giro** — a física geral
  por trás de por que uma massa girando fora do centro produz um torque de tombamento, explicada
  através de robôs de combate com armas giratórias (mesmo fenômeno, escala mais intuitiva).
  [Designing Around the Gyroscopic Effect — T.i. Combat Robotics](https://runamok.tech/insanity/gyroscopiceffect.html)

**Nenhuma fonte primária encontrada:** não consegui achar uma derivação publicada que bata exatamente
com o `ω_tip = sqrt(g·d_edge/(h·d_com))` do ticket — parece uma análise de tombamento ad hoc (torque
restaurador da gravidade em torno da borda de contato vs. torque centrífugo da massa fora do centro),
plausível mas não uma fórmula de livro-texto com nome. Vale a pena rederivá-la você mesmo a partir do
torque da gravidade em torno da borda de contato vs. a força centrípeta `m·ω²·d_com·h` — bom exercício
de mecânica antes de rodar o experimento.

## Exemplo real

**[Designing Around the Gyroscopic Effect](https://runamok.tech/insanity/gyroscopiceffect.html)** —
construtores de robôs de combate esbarram exatamente nessa classe de problema: uma massa girando cujo
eixo não está perfeitamente alinhado com a estrutura cria um torque que briga com o chassi. O caminho
de correção deles tem o mesmo formato do diagnóstico deste ticket: identificar a direção do torque
pelo produto vetorial entre o momento angular e a rotação imposta, confirmar experimentalmente, e
então projetar a distribuição de massa (ou a resposta de controle) em torno disso em vez de chutar.

> "The gyroscopic effect is what causes robots with vertically spinning weapons to tip up when
> turning... the vector cross product of the torque from gravity acting through the center of mass
> with the force establishes a gyroscopic torque acting tangentially."

## Passos práticos

1. Abra `models/robot_rollers.xml` perto da linha 153 — encontre o elemento `inertial` (ou o corpo
   sem um) e leia o `ipos` atual. Não mude nada ainda, só confirme que bate com os 3.459 mm citados
   no ticket.
2. Escreva um script pequeno (Python + `mujoco` + o seu `env`) que faça `place` de um robô tier-1 na
   origem e comande `w = [k,k,k,k]` para cada `k` em `[0.2, 0.4, 0.6, 0.8, 1.0]` do `ctrlrange`,
   5 s de física a cada vez, nenhum outro corpo por perto. Registre pico de taxa de yaw e pico de
   inclinação por `k` — confira se fica perto de 42.1 rad/s → 6.0°, ..., 59.4 → 46.6°.
3. Zere temporariamente as componentes x/y de `ipos` (edição local, não faça commit) e rode a mesma
   varredura de novo. Se a inclinação quase desaparecer, a hipótese `M1` está confirmada — reverta o
   XML imediatamente (`git diff` precisa voltar limpo).
4. Rode `env.action_space.sample()` no tier 1, seeds 0–49, e conte tombamentos por minuto de
   episódio, além de p50/p95/p99 da inclinação do chassi — essa é a linha de base contra a qual os
   tickets futuros serão medidos.
5. Meça os steps/s do seu ambiente durante esse processo, para comparar com o número já documentado
   em `docs/index.md:157-160`.
6. Preencha a seção `## Answer` do ticket original com os seus números — essa parte é o seu
   entregável, não o deste guia.

## Se você travar

- Se a varredura não bater com os graus esperados: verifique se o `ctrlrange` do seu atuador é o que
  o ticket assume, e se a "inclinação" está sendo medida do mesmo jeito (ângulo do eixo z do chassi
  em relação à vertical, não yaw).
- Para entender melhor `subtree_com` vs `xipos` vs `xpos`, a seção de frames do doc MuJoCo
  Computation linkado acima cobre as diferentes convenções de referencial que o MuJoCo usa.
- Para implementações comparáveis, pesquise "MuJoCo center of mass offset tipping" no GitHub
  Discussions do `google-deepmind/mujoco` — é onde esse tipo de pergunta de modelagem costuma
  aparecer.
