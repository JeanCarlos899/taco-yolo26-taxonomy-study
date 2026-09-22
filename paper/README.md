# Artigo SBC

O manuscrito principal está em `main.tex`; o PDF compilado está em `main.pdf`.
Antes da submissão, substitua a autoria, a afiliação e o e-mail marcados no preâmbulo.

## Reproduzir tabelas, figuras e verificações

Na raiz do projeto, com o ambiente virtual ativo:

```powershell
.\.venv\Scripts\python.exe scripts\17_generate_paper_assets.py
.\.venv\Scripts\python.exe scripts\18_validate_paper_claims.py
```

O primeiro comando recalcula as tabelas, os gráficos vetoriais, a prancha qualitativa,
o manifesto de proveniência e os fatos do artigo. O segundo confronta os números
centrais do texto com CSVs e JSONs derivados das predições.

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
