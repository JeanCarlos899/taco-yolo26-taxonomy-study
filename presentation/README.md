# Apresentação do artigo

Projeto Beamer em formato 16:9 para uma apresentação curta do estudo sobre granularidade semântica no TACO.

## Compilação

Execute a partir da raiz do repositório:

```powershell
Push-Location presentation
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -jobname=slides_clean -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -jobname=slides_clean -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
Copy-Item build/slides_clean.pdf slides_taco_yolo26.pdf -Force
Pop-Location
```

O PDF final é copiado para `presentation/slides_taco_yolo26.pdf`.

## Estrutura sugerida de fala

A apresentação possui 16 slides, organizados para explicar o sentido do experimento antes de apresentar a estatística.

## Divisão da fala

| Pessoa | Páginas | Conteúdo |
| --- | --- | --- |
| Carol (Karielly) | 1 a 6 | Pergunta, base, anotações, experimento e classes |
| Jean | 7 a 12 | Métricas, resultados, reavaliação e estatística |
| Marcos | 13 a 16 | Omissões, confusões, casos reais e conclusões |

## Roteiro de apoio

### Carol: páginas 1 a 6

1. Apresentar o tema e os autores.
2. Explicar a pergunta: quanto da dificuldade vem de localizar o resíduo e quanto vem de distinguir seu tipo? O experimento mantém dados e configuração, alterando a taxonomia.
3. Mostrar o tamanho do TACO e explicar cauda longa: poucas classes frequentes e muitas com poucos exemplos.
4. Mostrar imagem, polígonos e caixas. As caixas COCO constam no JSON oficial. O pipeline converte as coordenadas para YOLO.
5. Explicar os três modelos e as três sementes por modelo, totalizando nove treinamentos, com partição fixa.
6. Usar a garrafa para explicar Fine, Material e Binary. Binary possui uma classe de objeto, além do fundo.

Transição: Jean mostra como medir o efeito dessas exigências diferentes.

### Jean: páginas 7 a 12

7. AP50 resume precisão e recuperação ao variar a confiança. F1 combina precisão e recall em um ponto de operação. Valores maiores indicam desempenho melhor segundo a tarefa avaliada.
8. Comparar os modelos treinados. Binary tem os maiores valores numa tarefa menos detalhada. Material melhora a média em relação a Fine. O sinal ± mostra desvio-padrão entre sementes, não intervalo de confiança.
9. Explicar o exemplo ilustrativo: lata prevista como aerossol erra o tipo, mas acerta metal. A caixa permanece igual, desde que tenha sobreposição suficiente.
10. Mostrar que apenas agrupar as saídas do Fine já aumenta AP50. Isso evidencia o custo da classificação detalhada. Não há novo treinamento. Esta tabela usa avaliador próprio e não deve ser comparada diretamente à tabela nativa do slide 8.
11. Distinguir repetições de treinamento de reamostragens do teste. O bootstrap usa as mesmas imagens nos dois métodos. Um IC95% do ganho acima de zero sustenta ganho positivo. Incluir zero é inconclusivo, não prova igualdade.
12. Explicar que 3 de 3 conta execuções com IC95% do ganho acima de zero. Material treinado tem esse suporte em F1 nas três execuções, mas em AP50 só em uma. Agrupar as saídas do Fine tem suporte nas três. Isso não é probabilidade de acerto nem IC agregado das sementes.

Transição: Marcos mostra o que permanece difícil mesmo com rótulos mais amplos.

### Marcos: páginas 13 a 16

13. O denominador é o total de 637 objetos. Em média, 404,7 não recebem detecção com sobreposição suficiente. Mudar classes não cria previsões ausentes. Contagens fracionárias resultam da média de três sementes.
14. O denominador agora é apenas o conjunto de associações espaciais. Cerca de 43% acertam a classe, 23% erram dentro do material e 34% erram entre materiais. Somam 100%. Não somar às omissões do slide anterior.
15. Mostrar acerto, confusão dentro do material, confusão entre materiais e omissão. Ler GT como anotação de referência e identificar a classe prevista. Exemplos da semente 42 ilustram situações, não estimam frequências.
16. Concluir que classes amplas facilitam a tarefa e perdem informação. Material oferece compromisso com ganho sustentado em F1. Mencionar a partição fixa, teste de 225 imagens e agrupamento adotado como limites.

## Artefatos de referência

- Modelos treinados: results/multiseed_summary.csv.
- Reavaliação: results/hierarchical_multiseed_summary.csv.
- Bootstrap: results/bootstrap_multiseed_summary.csv.
- Erros: results/multiseed_error_summary.csv.

Tempo sugerido: Carol 3 minutos, Jean 4 minutos, Marcos 3 minutos.
