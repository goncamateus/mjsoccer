# Diagnóstico: confirmar o mecanismo de tombamento por deslocamento do CoM e medir a linha de base

Type: task
Status: open
Blocked by: —

## Pergunta

Nada a decidir. Rodar a medição da qual os tickets seguintes dependem, depois reverter.

A afirmação sob teste (`M1`, `TO2`): o tier 1 tomba porque o centro de massa do chassi do CAD fica
**3.459 mm fora do eixo de giro** (verificado: `|subtree_com − xpos|` no modelo compilado, com
`subtree_com = [·, ·, 0.0739943]` e o plano do piso em z = 0.02), e o espaço de ações permite
59.4 rad/s de yaw. O limiar previsto é
`ω_tip = sqrt(g · d_edge / (h · d_com)) = sqrt(9.81 · 0.04325 / (0.05399 · 0.003459)) = 47.7 rad/s`.

Fazer:
1. **Reproduzir a varredura de giro.** `place` um robô tier-1 na origem, comandar
   `w = [k,k,k,k]` para `k` em `[0.2 … 1.0]` do `ctrlrange`, 5 s de física, nenhum outro corpo em
   contato. Registrar pico de taxa de yaw e pico de inclinação por `k`. Esperado pelo documento de
   pesquisa: 42.1 rad/s → 6.0°; 55.6 → 43.2°; 59.4 → 46.6°.
2. **Falseá-la.** Zerar temporariamente as componentes x/y de `ipos` em
   `models/robot_rollers.xml:153` e rodar de novo. Se a inclinação colapsar, o `M1` está confirmado.
   **Reverter a edição na mesma sessão** — `git diff` precisa estar vazio quando este ticket fechar.
   Isto é um diagnóstico, nunca uma mudança entregue (§12.4(a)).
3. **Taxa de tombamento de linha de base.** `env.action_space.sample()` em todos os seis ids, tier 1,
   seeds 0-49: tombamentos por minuto de episódio, distribuição do tempo até tombar, e p50/p95/p99 da
   inclinação do chassi. Esse é o número contra o qual toda correção futura será medida, então
   registrar o comando exato e o commit.
4. **Linha de base de tempo de relógio** em steps/s por env, para comparar com `docs/index.md:157-160`.

Escrever os números no `## Answer` deste ticket, e as linhas brutas em
`.scratch/physics-fidelity/baseline/`. Não modificar nada em `src/`, `models/`, `examples/`,
`scripts/` ou `tests/` exceto o diagnóstico revertido de `ipos`.
