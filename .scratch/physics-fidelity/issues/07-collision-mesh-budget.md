# Decisão: o orçamento da malha de colisão

Type: grilling
Status: open
Blocked by: 06

## Pergunta

O tier 1 carrega **209 geoms de colisão** e 84 corpos extras por robô (medido): uma decomposição do
chassi do CAD em cascos convexos, quatro cubos de roda e 80 roletes girando livres. O rSim carrega um
cilindro, quatro cilindros de roda e uma caixa soldada, e testa exatamente sete pares de colisão
(`pworld.cpp:287-305`).

Sob fidelidade ao hardware, a versão do mjsoccer é a *mais* correta, e não é obviamente viável em
custo. Três perguntas:

1. **Manter o chassi de 209 cascos?** É por causa dele que os robôs enroscam nos para-choques e
   carcaças de motor uns dos outros e geram torques de contato em múltiplas alturas que um cilindro
   não consegue (`B1`). Também é o principal fornecedor do impulso que converte um giro cônico
   marginal em queda. A fidelidade diz manter; os impulsos de tropeço são reais, não artefatos.
   Decidir se algum casco é *ruído de CAD* em vez de uma superfície externa real — esse é o único
   critério de poda defensável.
2. **Manter os contatos bola↔rolete?** A bola é `contype="7" conaffinity="7"` e os roletes são
   `contype="4" conaffinity="4"`, então o bit 4 casa e uma bola rolando por baixo de um robô é
   chutada para fora por um rolete (`B6`). O rSim nunca colide bola com roda. Fisicamente a bola
   *bate* nas rodas. Manter — a menos que o efeito medido seja dominado por artefatos da malha.
3. **Manter cubo↔parede?** Os cubos de roda são `contype="1" conaffinity="1"` e alcançam os cascos da
   parede (`C7`). Mesmo raciocínio.

Também resolver, para registro, a discrepância de contagem entre as duas auditorias: 209 geoms de
colisão (documento de pesquisa) vs "315 descendant geoms" (documento de análise) — só colisão contra
colisão+visual. Dizer qual número a documentação deve citar.

Resolvido por uma medição de tempo de relógio contra a linha de base do `03` mais a atribuição de
impulsos de contato do harness do `14`: quais contatos de fato fornecem impulsos de tropeço em jogo
normal.
