# Por que o tier 1 não alcança `v_max` cinemático: arrasto axial dos roletes e colapso lateral em yaw

Type: task
Status: open
Blocked by: —

## Pergunta

O `18` mostrou que a perda de contato não é o que trava a velocidade: a força normal média continua
mg, e curar o contato rende +0.4 m/s
([`../candidate-18/`](../candidate-18/), seção *Velocidade* do [`18`](18-wheel-contact-loss.md)).
Sobram dois mecanismos.

1. **x: platô em ~2.3 de 3.30 m/s** (`terminal.txt`). Mesmo a 0.66 m/s o regime fica 15 % abaixo,
   com o arrasto axial cancelando o empurrão das rodas. Cada roda leva 0.8–0.9 N do piso **ao longo do eixo**, o que um rolete livre
   deveria zerar. Cada rolete novo entra no contato a 4–8 rad/s quando precisa de ~40, e leva ~8 ms
   para chegar (`spinup.txt`). Não muda com `armature`, atrito torcional, amortecimento, `impratio` nem
   cone. O casco de rolete em resolução cheia (`hull`) e o rolete elipsoide (`ellip`) levam o terminal
   a 2.76–2.80, mas o `ellip` ainda tem 0.7 N axiais.
2. **y: colapso em yaw** a k ≥ 0.5 (`v` ~0, yaw 4–9 rad/s) em todas as variantes do `18`, inclusive
   com contato a 0.7–0.9. A hipótese não medida é a assimetria frente/trás (rodas a 60° e a 45°): com
   atrito saturado desigual, o momento não se cancela.

Fazer:
1. Fechar o mecanismo do arrasto axial. Testar: por que um rolete liso ainda arrasta (transição entre
   roletes vizinhos sobrepostos? rolete que já sai do contato ainda girando?), e se o torque de `Fn`
   fora do eixo do casco de 32 vértices é o que segura o rolete na base.
2. Medir o momento de yaw por roda no comando lateral e confirmar ou derrubar a hipótese
   frente/trás.
3. Dizer se a correção é de malha (`07`: `maxhullvert` do rolete, forma do rolete), de contato ou de
   atuação, e qual `v_max` e `a_max` lateral o tier 1 alcança depois dela. O `12` espera esse número.
