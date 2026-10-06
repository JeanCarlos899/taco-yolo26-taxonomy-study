# Apresentação do artigo

Projeto Beamer em formato 16:9 para uma apresentação curta do estudo sobre granularidade semântica no TACO.

## Compilação

Execute a partir da raiz do repositório:

```powershell
python scripts/24_unify_reported_evaluation.py
python scripts/23_generate_hierarchy_gain_table.py
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
| Karielly | 1 a 4 | Apresentação, pergunta, base e anotações |
| Jean | 5 a 10 | Experimento, classes, métricas, resultados e reavaliação |
| Marcos | 11 a 16 | Estatística, erros, casos reais e conclusões |

## Roteiro de apoio

### Karielly: páginas 1 a 4

1. Apresentar o tema e os autores.
2. Explicar a pergunta: quanto da dificuldade vem de localizar o resíduo e quanto vem de distinguir seu tipo? O experimento mantém dados e configuração, alterando a taxonomia.
3. Mostrar o tamanho do TACO e explicar cauda longa: poucas classes frequentes e muitas com poucos exemplos.
4. Mostrar imagem, polígonos e caixas. As caixas COCO constam no JSON oficial. O pipeline converte as coordenadas para YOLO.

Transição: Jean apresenta o experimento e os resultados.

### Jean: páginas 5 a 10

5. Explicar os três modelos e as três sementes por modelo, totalizando nove treinamentos, com partição fixa.
6. Usar a garrafa para explicar Fine, Material e Binary. Binary possui uma classe de objeto, além do fundo.


7. AP50 resume precisão e recuperação ao variar a confiança. F1 combina precisão e recall em um ponto de operação. Valores maiores indicam desempenho melhor segundo a tarefa avaliada.
8. Comparar os modelos treinados. Binary tem os maiores valores numa tarefa menos detalhada. Material melhora a média em relação a Fine. O sinal ± mostra desvio-padrão entre sementes, não intervalo de confiança.
9. Explicar o exemplo ilustrativo: lata prevista como aerossol erra o tipo, mas acerta metal. A caixa permanece igual, desde que tenha sobreposição suficiente.
10. O mesmo Fine pode ser julgado pelo tipo exato, pelo material ou pela presença de resíduo. A tabela mostra o ganho relativo de AP50: +65,8% por material e +351,6% por presença, tomando como referência o AP50 médio do tipo exato no avaliador controlado. O cálculo usa a diferença entre médias dividida pela média Fine. Não é porcentagem de objetos corretos, não envolve novo treinamento e usa a mesma referência Fine de 0,103 do slide 8. A agregação também altera a macro-média.

Transição: Marcos explica a incerteza dos ganhos e os erros que persistem.

### Marcos: páginas 11 a 16

11. Distinguir repetições de treinamento de reamostragens do teste. O bootstrap usa as mesmas imagens nos dois métodos. Um IC95% do ganho acima de zero sustenta ganho positivo. Incluir zero é inconclusivo, não prova igualdade.
12. Explicar que 3 de 3 conta execuções com IC95% do ganho acima de zero. Material treinado tem esse suporte em F1 nas três execuções, mas em AP50 só em uma. Agrupar as saídas do Fine tem suporte nas três. Isso não é probabilidade de acerto nem IC agregado das sementes.


13. O denominador é o total de 637 objetos. Em média, 404,7 não recebem detecção com sobreposição suficiente. Mudar classes não cria previsões ausentes. Contagens fracionárias resultam da média de três sementes.
14. O denominador agora é apenas o conjunto de associações espaciais. Cerca de 43% acertam a classe, 23% erram dentro do material e 34% erram entre materiais. Somam 100%. Não somar às omissões do slide anterior.
15. Mostrar acerto, confusão dentro do material, confusão entre materiais e omissão. Ler GT como anotação de referência e identificar a classe prevista. Exemplos da semente 42 ilustram situações, não estimam frequências.
16. Concluir que classes amplas facilitam a tarefa e perdem informação. Material oferece compromisso com ganho sustentado em F1. Mencionar a partição fixa, teste de 225 imagens e agrupamento adotado como limites.

## Artefatos de referência

- Modelos treinados: results/controlled_multiseed_summary.csv. O histórico nativo permanece em results/multiseed_summary.csv.
- Reavaliação: results/hierarchical_multiseed_summary.csv.
- Bootstrap: results/bootstrap_multiseed_summary.csv.
- Erros: results/multiseed_error_summary.csv.

Tempo sugerido: Karielly 2 minutos, Jean 4 minutos, Marcos 4 minutos.

O slide 10 usa ganhos observados por semente, gerados por scripts/23_generate_hierarchy_gain_table.py. A tabela de bootstrap do artigo usa diferenças médias das reamostragens, que não são necessariamente idênticas à diferença observada no teste fixo.
