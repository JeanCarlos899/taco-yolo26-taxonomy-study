# Apresentação do artigo

Projeto Beamer em formato 16:9 para uma apresentação curta do estudo sobre granularidade semântica no TACO.

## Compilação

Execute a partir da raiz do repositório:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=presentation/build presentation/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=presentation/build presentation/main.tex
```

O PDF final é copiado para `presentation/slides_taco_yolo26.pdf`.

## Estrutura sugerida de fala

A apresentação possui 12 slides e foi dimensionada para aproximadamente 8 a 10 minutos. O roteiro progride da pergunta central e da estrutura das anotações para o desenho experimental, resultados, diagnóstico de erros e conclusões.
