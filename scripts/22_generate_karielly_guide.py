"""Generate a four-page study guide for presentation slides 1-4."""
from pathlib import Path
import subprocess
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output' / 'pdf' / 'guia_karielly_slides_1_a_4.pdf'
GREEN = colors.HexColor('#006633')
PAGE_SIZE = landscape(A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='GuideTitle', fontName='Helvetica-Bold', fontSize=21, leading=25, textColor=GREEN, spaceAfter=12))
styles.add(ParagraphStyle(name='GuideHeading', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=GREEN, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name='GuideBody', fontName='Helvetica', fontSize=10.5, leading=14, spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name='GuideTalk', parent=styles['GuideBody'], leftIndent=8, spaceAfter=12))
styles.add(ParagraphStyle(name='GuideSmall', fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#657078'), spaceAfter=8))

PAGES = [
    ('1', 'Apresentação do trabalho', [
        ('O que este slide apresenta', 'O título, os autores e a instituição. O trabalho estuda o custo de exigir classes mais detalhadas na detecção de resíduos. Aqui, custo significa dificuldade e perda de desempenho segundo a avaliação, e não gasto financeiro ou consumo de energia.'),
        ('Conceitos que Karielly precisa conhecer', '<b>Detecção de objetos:</b> localizar um objeto na imagem e atribuir uma classe a ele.<br/><b>Granularidade semântica:</b> nível de detalhe do rótulo. "Garrafa plástica" é mais específico que "plástico", que é mais específico que "resíduo".<br/><b>TACO:</b> base de imagens e anotações de resíduos usada no estudo.<br/><b>YOLO26n:</b> detector usado nos experimentos. O sufixo n identifica a variante nano. Não é necessário explicar sua arquitetura nesta abertura.'),
        ('Sugestão de fala', '"Bom dia! Somos Jean Carlos, Karielly e Marcos, da Universidade Federal do Piauí, campus Picos. Nosso trabalho investiga como o nível de detalhe das classes afeta a detecção de resíduos. Para isso, usamos a base TACO e o detector YOLO26n."'),
        ('Se perguntarem: qual é a contribuição?', 'O estudo compara diferentes exigências de classificação mantendo os dados e a configuração experimental. Também reavalia as mesmas saídas do modelo detalhado com rótulos agrupados, para investigar a penalização causada pela classe prevista. Os resultados serão apresentados pelos demais integrantes.'),
        ('Ligação com o próximo slide', '"Para entender o experimento, primeiro precisamos separar a localização do objeto da identificação de sua classe."'),
    ]),
    ('2', 'Pergunta central da pesquisa', [
        ('O que a pergunta significa', 'A pesquisa pergunta como o nível de detalhe das classes afeta o desempenho em uma base pequena e com distribuição desigual de exemplos. O detector pode encontrar um objeto, mas confundir seu tipo. Por isso, localização e classificação devem ser compreendidas separadamente.'),
        ('Exemplo para entender', 'Imagine uma lata de bebida reconhecida como aerossol. Se a caixa estiver bem posicionada, o modelo localizou o objeto, mas errou a classe específica. No agrupamento adotado pelo projeto, os dois pertencem a metal. Assim, uma avaliação por material pode aceitar essa mesma previsão. O exemplo explica o critério e não representa, sozinho, um resultado estatístico.'),
        ('O que permanece controlado', 'Os experimentos usam as mesmas imagens, caixas e divisão entre treino, validação e teste, com o mesmo detector e configuração de treinamento. Os rótulos mudam conforme a taxonomia. Há treinamentos separados: as redes aprendidas e suas previsões não são idênticas. Existe também uma análise adicional que reutiliza as previsões do Fine.'),
        ('Sugestão de fala', '"Nossa pergunta é quanto o desempenho muda quando o modelo precisa reconhecer o tipo específico do resíduo, o material ou apenas a presença de resíduo. Detectar envolve encontrar o objeto e atribuir o rótulo esperado. Mantemos os dados e a configuração experimental para investigar o efeito do nível de detalhe das classes."'),
        ('Se perguntarem: o que é taxonomia?', 'É a organização das classes usadas para rotular os objetos. O projeto trabalha com 60 classes detalhadas, seis grupos por material e uma classe de resíduo. No último caso, o fundo continua existindo como ausência de objeto, mas não é uma segunda classe anotada.'),
        ('Ligação com o próximo slide', '"Essa comparação é especialmente relevante no TACO, porque muitas classes têm poucos exemplos."'),
    ]),
    ('3', 'TACO e o problema de cauda longa', [
        ('Os números que precisam estar claros', 'A versão da base usada contém <b>1.500 imagens, 4.784 objetos anotados e 60 classes</b>. Imagens e objetos são unidades diferentes: uma fotografia pode conter vários resíduos. Esses números descrevem a base completa, antes da divisão experimental.'),
        ('O que é cauda longa', 'Uma distribuição de cauda longa tem poucas classes com muitos exemplos e muitas classes com poucos exemplos. É um desbalanceamento entre categorias. Classes menos representadas oferecem menos exemplos para aprender suas variações. Isso ajuda a contextualizar a dificuldade, mas não explica sozinho todos os erros do detector.'),
        ('Como ler o gráfico', 'O eixo horizontal ordena as classes da mais frequente para a menos frequente. O eixo vertical mostra o número de objetos por classe, em escala logarítmica. A curva descreve a <b>partição de treinamento</b>, não a base completa. Cada posição representa uma classe, e não uma época de treinamento. A queda indica a redução da frequência ao longo das classes ordenadas.'),
        ('Sugestão de fala', '"O TACO tem 1.500 imagens de ambientes reais, com 4.784 objetos em 60 classes. Há um desbalanceamento chamado cauda longa: poucas classes concentram muitos exemplos e várias têm poucas amostras. O gráfico mostra essa distribuição no treino. Isso torna mais difícil aprender as categorias menos frequentes."'),
        ('Se perguntarem: por que ambientes reais importam?', 'As fotografias mostram os resíduos em suas cenas, com fundos e condições visuais variados. O modelo precisa reconhecer objetos em meio a esse contexto. Não é uma coleção apenas de produtos isolados em fundo uniforme.'),
        ('Ligação com o próximo slide', '"Além das fotografias, a base fornece anotações que identificam cada objeto. O próximo slide mostra como elas são representadas."'),
    ]),
    ('4', 'Como o TACO representa os objetos', [
        ('O que aparece nas três imagens', '<b>Original:</b> a fotografia, sem sobreposição das anotações.<br/><b>Segmentação:</b> polígonos que contornam cada instância de resíduo.<br/><b>Caixas COCO:</b> retângulos que delimitam os objetos. As cores e os rótulos sobrepostos foram produzidos para visualizar as anotações. A fotografia não vem com esses desenhos incorporados.'),
        ('O que vem da base e o que o projeto faz', 'O arquivo oficial de anotações, annotations.json, já fornece os campos segmentation, bbox e category_id. A bbox COCO registra x e y do canto superior esquerdo, largura e altura. O pipeline converte essas coordenadas para o centro, largura e altura normalizados exigidos pelo YOLO. Neste estudo, o treinamento usa <b>caixas para detecção</b>. As máscaras ilustram as anotações originais; não há treinamento de segmentação.'),
        ('O exemplo mostrado no slide', 'A fotografia contém três objetos e três classes: copo de papel (Paper cup), tampa plástica (Plastic lid) e saco de papel (Paper bag). O copo recebe Paper cup no Fine, Paper/Cardboard no Material e Litter no Binary. A tampa recebe Plastic lid, Plastic e Litter. Os seis grupos por material são uma escolha do projeto, e não devem ser apresentados como a taxonomia oficial completa do TACO.'),
        ('Sugestão de fala', '"Aqui temos a mesma fotografia em três representações: original, segmentação e caixas delimitadoras. O JSON oficial já fornece os contornos, as caixas e as classes. Nosso pipeline converte as caixas para YOLO e agrupa os rótulos conforme o experimento. Por exemplo, a tampa mantém sua classe específica no Fine, vira plástico no Material e resíduo no Binary."'),
        ('Se perguntarem: fomos nós que criamos as caixas?', 'As coordenadas já vêm nas anotações oficiais. O projeto converte o formato e desenha as caixas para visualização. Também remapeia as classes. A conversão não cria novos objetos anotados.'),
        ('Passagem para Jean', '"Com a base e suas anotações explicadas, Jean vai mostrar como o experimento foi organizado e como os resultados foram avaliados."'),
    ]),
]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GREEN)
    canvas.line(1.5*cm, 1.3*cm, PAGE_SIZE[0]-1.5*cm, 1.3*cm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#657078'))
    canvas.drawString(1.5*cm, 0.9*cm, 'Guia de apoio de Karielly | TACO e YOLO26n | Slides 1 a 4')
    canvas.drawRightString(PAGE_SIZE[0]-1.5*cm, 0.9*cm, str(doc.page))
    canvas.restoreState()

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    previews = ROOT / 'tmp' / 'pdfs' / 'karielly' / 'slides'
    previews.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-f', '1', '-l', '4', '-scale-to', '1600', '-png', str(ROOT / 'presentation' / 'slides_taco_yolo26.pdf'), str(previews / 'slide')], check=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=PAGE_SIZE, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.1*cm, bottomMargin=1.65*cm, title='Guia visual de Karielly - Slides 1 a 4', author='Equipe TACO - UFPI')
    story = []
    for index, (number, title, sections) in enumerate(PAGES):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(f'SLIDE {number} / GUIA DE ESTUDO', styles['GuideSmall']))
        story.append(Paragraph(title, styles['GuideTitle']))
        slide_file = next(previews.glob(f'slide-{int(number):02d}.png'))
        width = 12.7*cm
        slide_frame = Table([[Image(str(slide_file), width=width, height=width*9/16)]], colWidths=[width])
        slide_frame.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#7A8580')),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        left = [slide_frame, Spacer(1, 8)]
        right = []
        left_indices = {0, 2} if index == 3 else {0, 1}
        for section_index, (heading, content) in enumerate(sections):
            column = left if section_index in left_indices else right
            column.append(Paragraph(heading, styles['GuideHeading']))
            column.append(Paragraph(content, styles['GuideTalk'] if heading == 'Sugestão de fala' else styles['GuideBody']))
        layout = Table([[left, right]], colWidths=[13.35*cm, 13.35*cm])
        layout.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0.65*cm), ('TOPPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 0)]))
        story.append(layout)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)

if __name__ == '__main__':
    main()
