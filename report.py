"""Build the two-page research note from measured aggregates."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).parent

def main():
    groups=json.loads((ROOT/'results/aggregates.json').read_text())
    def row(cond,kind,cp='0.01'):
        return next(g for g in groups if g['condition']==cond and g['kind']==kind and g['checkpoint']==cp)
    def pct(cond,kind,cp='0.01'):
        return f"{100*row(cond,kind,cp)['test_accuracy_mean']:.1f}%"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#243746')))
    styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=10,spaceAfter=6,textColor=colors.HexColor('#176B87')))
    styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=22,leading=25,spaceAfter=8,textColor=colors.HexColor('#142D3B')))
    styles.add(ParagraphStyle(name='CaptionCustom',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=7,textColor=colors.HexColor('#52616B')))
    story=[]; md=[]
    def p(text): story.append(Paragraph(text,styles['BodyCustom']));md.append(text+'\n')
    def h(text): story.append(Paragraph(text,styles['SectionCustom']));md.append('## '+text+'\n')
    title='When Does Factorization Enable OCR?'
    story.append(Paragraph(title,styles['TitleCustom']));md.append('# '+title+'\n')
    p('An optimizer stress test | Controlled fixed-attention experiment | October 10, 2026')
    h('1. Question')
    p('At comparable training cross-entropy, does the advantage of a factorized output-value map persist under full-batch gradient descent (GD) and Adam? Does decoupled weight decay change that advantage? This project follows the fixed-attention setting of Huang et al. [1], rather than reproducing their complete transformer experiments. The paper already reports AdamW use; this is not a first test of OCR under AdamW.')
    h('2. Setup')
    p('We use 20 fact groups, four subjects per group, and one subject per group with an observed implication: 80 facts plus 20 implications for training; 60 hidden implications for testing; 40 candidate answers. Subjects have opaque, permuted one-hot IDs. Inputs are x = e_s + e_r. We compare logits Wx with AB<super>T</super>x. W is 40 x 82; the factor width is 40, sufficient to represent every such W, so there is no hard rank restriction.')
    p('For each of five seeds (0-4), W0 has independent N(0, 0.01<super>2</super>) entries. Its SVD gives balanced factors A = U sqrt(S), B = V sqrt(S). All conditions share the same initial function per seed. Training uses CPU float64, full batches and 20,000 updates. A separate seed-100 pilot (5,000 updates per candidate) selects the lowest final training CE among stable candidates: GD learning rate 1.0; Adam 0.01, for both parameterizations. AdamW inherits 0.01. All 12 pilot outcomes are saved; test performance is never evaluated in the pilot.')
    p('The main comparison has zero decay. AdamW additionally uses decay 0.01 and 0.03. Ten zero-decay AdamW runs check equivalence to Adam. We save the first crossing of training CE 0.1, 0.03 and 0.01, and the final checkpoint; training continues after crossing. All metrics use only active answer rows and input columns. Spectral metrics use output-row centering, Wc = W - row_mean(W).')
    h('3. Results')
    story.append(Image(str(ROOT/'figures/01_accuracy_vs_training_loss.png'),width=500,height=178))
    story.append(Paragraph('Figure 1. Each thin line is one seed. Dots mark mean OCR at first CE-threshold crossings. The horizontal axis decreases to the right. Vertical dotted lines mark CE 0.1, 0.03 and 0.01. Curves below CE 10^-5 are outside the plotted range.',styles['CaptionCustom']))
    md.append('![Figure 1](figures/01_accuracy_vs_training_loss.png)\n')
    story.append(PageBreak())
    h('3. Results (continued)')
    table=[['Condition','Direct OCR','Factorized OCR','Reached CE <= 0.01']]
    for key,label in [('gd','GD'),('adam','Adam'),('adamw0.01','AdamW, decay 0.01'),('adamw0.03','AdamW, decay 0.03')]:
        table.append([label,pct(key,'direct'),pct(key,'factorized'),f"{row(key,'direct')['reached']}/5 and {row(key,'factorized')['reached']}/5"])
    md.append('\n'.join(['| ' + ' | '.join(table[0]) + ' |', '|---|---|---|---|'] + ['| ' + ' | '.join(v) + ' |' for v in table[1:]]) + '\n')
    t=Table(table,colWidths=[175,90,110,125],hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF1')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#176B87'))]));story.append(t);story.append(Spacer(1,9))
    p('Table 1. Mean hidden-implication accuracy across five seeds at the first CE <= 0.01 checkpoint. Figure 2 in the repository shows every seed at matched fit and at the fixed update budget; Figure 3 relates centered spectral ratio to OCR at matched fit.')
    p(f"<b>Observation 1.</b> At matched fit, factorized GD reaches {pct('gd','factorized')} OCR and factorized Adam {pct('adam','factorized')}; direct GD reaches {pct('gd','direct')} and direct Adam {pct('adam','direct')}. These comparisons control initial predictions and attained training loss, but not optimizer step size or parameter-space geometry.")
    p(f"<b>Observation 2.</b> With AdamW decay 0.01 / 0.03, factorized OCR is {pct('adamw0.01','factorized')} / {pct('adamw0.03','factorized')}, versus {pct('adamw0.01','direct')} / {pct('adamw0.03','direct')} for direct models. At 20,000 updates, factorized Adam reaches {pct('adam','factorized','final')} and direct Adam {pct('adam','direct','final')}. Matched-fit and fixed-budget results answer different questions.")
    g=row('gd','factorized');a=row('adam','factorized')
    p(f"<b>Observation 3.</b> At CE <= 0.01, mean centered nuclear/Frobenius ratio for factorized GD is {g['spectral_ratio_mean']:.3f}, compared with {a['spectral_ratio_mean']:.3f} for factorized Adam. This statistic is scale invariant; its square is one definition of spectral effective dimension. Similar OCR need not imply identical learned spectra. The five seeds, not individual time checkpoints, are the replication units.")
    h('4. Interpretation')
    p('Factorization helps under both GD and Adam, but Adam has a 12.3 percentage-point OCR deficit at CE <= 0.01 that disappears with continued training. Adam also has the lower spectral ratio at this checkpoint: lower ratio does not imply better OCR. This is an association, not a causal mechanism. The code records margins and ||Wc||*/gamma for positive minimum training margin gamma, allowing comparisons at unit training margin.')
    p('The nuclear-norm account in [1] motivates this analysis, but no numerical SVM reference or closed-form solution alignment is computed here. Nonzero AdamW decay changes regularization as well as dynamics. Decaying both factors by (1 - learning_rate * decay) shrinks their product by the square of that factor, whereas direct W receives a single factor. Equal decay coefficients therefore do not match functional shrinkage.')
    h('5. Limitations')
    p('Attention is fixed; the data are small, noiseless and structurally balanced. Five initialization seeds use one fixed data mapping, and the pilot uses one separate seed with a limited learning-rate grid. First-crossing CE ranges from 0.00964 to 0.01000 at the 0.01 target, so fit is approximate. Finite-step GD and random matched initialization differ from asymptotic or symmetry assumptions in theory. There is no intervention proving causality, no theorem verification, and no evidence here about realistic LLM hallucination. Conclusions are restricted to this setup.')
    story.append(Paragraph('[1] Huang et al. Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers. arXiv:2506.10887v4 (2026), Sections 4.1, A.3 and E. <link href="https://arxiv.org/html/2506.10887v4">https://arxiv.org/html/2506.10887v4</link>',styles['CaptionCustom']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52616B'));canvas.drawString(48,25,'OCR / OPTIMIZER STRESS TEST');canvas.drawRightString(564,25,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'research_note.pdf'),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=35,bottomMargin=40)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'research_note.md').write_text('\n'.join(md))
    print('Created research_note.pdf and research_note.md')
if __name__=='__main__': main()
