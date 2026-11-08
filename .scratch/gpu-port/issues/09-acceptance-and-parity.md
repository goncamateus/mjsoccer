# Que critérios de aceitação sobrevivem, e com que tolerâncias de paridade

Type: grilling
Status: resolved
Blocked by: 05

## Pergunta

Sem o tier 1 e com o controlador de navegação entre a política e o chassi:

1. Quais critérios de aceitação de hoje (`tests/test_validation.py`, `scripts/validation_sweep.py`,
   `docs/index.md`) continuam valendo para o robô cinemático, quais caem com o tier 1, e quais
   critérios novos o controlador pede (ex.: tempo e erro de chegada a uma target pose)?
2. Para cada critério que fica, a **tolerância de paridade** MJX-Warp × CPU no modelo base (a
   checagem variante × base saiu com o ADR 0002), à luz do ruído de float32 e das diferenças de
   solver que o 05 mostrou (contato da barra igual a 1e-5 mm, então a diferença vem da dinâmica).
3. Com que amostra (seeds, episódios) cada checagem roda. Sweeps de validação usam `--seeds 0-4`,
   nunca mais que 5 seeds.

## Answer

Resolvido em grilling (2026-11-08).

1. **Tilt e tombamento saem com o tier 1.** O chassi cinemático é planar: "sem tombar", tilt p99
   < 15°, a taxa de tombos por minuto e os percentis de tilt agrupados caem, junto com
   `test_tipover_ends_episode` e o `pretip.csv`. Parte mecânica da remoção; não voltam como paridade.
2. **Critérios que ficam** (cada um checado nos dois backends):
   - **Envelope**, pelo caminho interno de velocidade crua (ticket do controlador): velocidade de topo
     3.30 m/s, aceleração 1.30 m/s² (banda de 1.5×), yaw p99 < 37.1 rad/s, aceleração angular 55 rad/s².
   - **Kick e chip**: velocidade de saída pela tabela.
   - **Bola**: COR por superfície, inércia.
   - **Posse**: bola assentada segura, inclusive no giro de 2 rad/s.
   - Cenários scripted: ficam `spin` e `kick_recoil` (reescritos em target pose ou velocidade crua);
     `head_on`, `wall`, `dribble_turn` saem. `random` e `competent` (agora `goto_ball_policy` em
     target pose) ficam como cenários de task.
3. **Critérios novos do controlador de navegação**, do repouso, alvos a 0.5 / 2 / 6 m e giros de
   0 / 90 / 180°:
   - **chegada**: erro final < 5 cm e < 2.5° (zona morta do giro);
   - **tempo**: ≤ 1.25× o perfil trapezoidal analítico (3.3 m/s, sobe a 1.30, freia a 1.0);
   - **chatter no alvo**: RMS de velocidade no último segundo < 0.05 m/s;
   - **teto**: `ω` comandado nunca acima de 37.1.
   Os números são ponto de partida: medir primeiro na CPU, fixar o limiar no medido mais margem
   (método dos tickets do tier 1).
4. **Paridade = (b)**: o MJX-Warp passa todo critério com os mesmos limiares da CPU **e** cada
   métrica escalar fica dentro de uma tolerância entre backends.
   - Cenários determinísticos (mesmo estado inicial nos dois): |MJX − CPU| / CPU ≤ 2 % em
     envelope, kick, COR; ≤ 5 % no tempo de chegada; erro de chegada ≤ 1 cm / 1° absoluto.
   - Cenários com seed (`random`, `competent`, taxa de posse, retorno): mediana do MJX dentro do
     [p25, p75] da CPU. Sem pareamento seed a seed (RNG do JAX ≠ NumPy).
   - Sem testes de distribuição (KS etc.): sem poder com 5 seeds. Tolerâncias revisáveis depois da
     primeira medição; contatos compostos (posse) só têm a checagem estatística.
5. **Amostra**: determinísticos rodam uma vez por parâmetro, estado inicial escrito inteiro em
   `qpos`/`qvel`, igual nos dois. Com seed: `--seeds 0-4` nos dois, um episódio (CPU) / um mundo
   (MJX) por seed por env — amostras simétricas.
6. **Throughput não é paridade**: o "dentro de 10 % do arquivo" do `docs/index.md` fica só da CPU;
   a velocidade do MJX é benchmark (decisões/s, ticket 05), nunca critério de aceitação.
