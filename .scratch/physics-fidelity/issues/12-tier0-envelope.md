# Decisão: o que "o mesmo envelope" significa para o tier 0

Type: grilling
Status: open
Blocked by: 04

## Pergunta

Restrição 4 do mapa: o tier 0 mantém o seu *mecanismo* de cinemática ideal, mas recebe o *envelope*
de velocidade/aceleração do tier 1, para que uma cabeça de ação signifique a mesma coisa nos dois.

Tier 0 hoje: três juntas nos eixos do mundo (`slide-x`, `slide-y`, `hinge-z`) com servos de
velocidade com `ctrlrange` ±4 m/s e ±30 rad/s, `forcerange` ±19.1261 N (x, y) e ±3.427 N·m (yaw)
(`models/robot_kinematic.xml:152-154,389-391`). O chassi nunca toca o piso, então não há limite de
tração, nem escorregamento, nem tombamento (`W5`). 19.1261 N sobre 2.69207 kg dá 7.1 m/s².

Decidir:
1. **Quais grandezas são igualadas?** `v_max`, `ω_max` e `a_max` são as três óbvias. Escorregamento,
   tração e transferência de peso não podem ser igualados — esse é o ponto do tier. Dizer isso
   explicitamente, porque "o mesmo envelope" precisa ser um termo definido, não uma impressão.
2. **`forcerange` nos trilhos (slides).** Definir como `m · a_max` do `04`? Isso torna o teto de
   aceleração do tier 0 igual ao teto *limitado por atrito e torque* do tier 1, que é a
   correspondência mais próxima disponível, mas o tier 0 vai alcançá-lo em todas as direções ao mesmo
   tempo, onde o tier 1 não consegue.
3. **O servo de yaw.** ±3.427 N·m contra o `Izz` do chassi dá ao tier 0 uma aceleração angular sem
   análogo em torque de roda. Igualar à autoridade de yaw das quatro rodas do `04`.
4. **Os números redondos ficam?** ±4 m/s e ±30 rad/s parecem escolhidos por arrumação. Se o `04`
   derivar outra coisa, o tier 0 deve seguir a derivação, não a arrumação.
5. **Onde a diferença é documentada?** O `README.md` afirma que uma cabeça de ação se transfere entre
   tiers; `docs/index.md:145-150,199-212` explica as diferenças entre os tiers. Depois deste ticket,
   ambos precisam da declaração de envelope, o que é trabalho do `15`.
