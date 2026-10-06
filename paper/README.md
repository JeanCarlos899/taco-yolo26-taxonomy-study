# Artigo SBC

O manuscrito principal está em `main.tex`; o PDF compilado está em `main.pdf`.
Todas as métricas reportadas utilizam o avaliador controlado comum. Os resultados nativos Ultralytics são preservados como histórico, fora das tabelas principais.

## Reproduzir tabelas, figuras e verificações

Na raiz do projeto, com o ambiente virtual ativo:

```powershell
.\.venv\Scripts\python.exe scripts\24_unify_reported_evaluation.py
.\.venv\Scripts\python.exe scripts\17_generate_paper_assets.py
.\.venv\Scripts\python.exe scripts\18_validate_paper_claims.py
```

O primeiro comando reavalia as predições salvas com o avaliador comum e gera os
resultados padronizados. O segundo recalcula as tabelas, os gráficos vetoriais,
a prancha qualitativa, o manifesto de proveniência e os fatos do artigo. O terceiro
confronta os números centrais do texto com CSVs e JSONs derivados das predições.
Esses comandos requerem as predições brutas locais, que não são versionadas;
consulte `PIPELINE.md` para gerar os treinamentos e avaliações necessários.

## Compilar

Dentro de `paper`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Os arquivos `sbc-template.sty`, `sbc.bst` e `caption2.sty` acompanham o projeto para
que a compilação não dependa de uma cópia externa do template SBC.
