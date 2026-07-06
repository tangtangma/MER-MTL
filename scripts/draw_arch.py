import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


fig, ax = plt.subplots(figsize=(30, 21))
ax.set_ylim(0, 21); ax.axis('off')
fig.patch.set_facecolor('white')

# ── colours ──
C_TEXT_IN  = {'fc': '#B8CCE4', 'ec': '#4A78A8'}
C_AUDIO_IN = {'fc': '#E4D8A8', 'ec': '#B8A040'}
C_VIDEO_IN = {'fc': '#E4B0B0', 'ec': '#C05050'}
C_TEXT_ENC = {'fc': '#8CAAC8', 'ec': '#3868A0'}
C_AUDIO_ENC= {'fc': '#C8BC80', 'ec': '#988828'}
C_VIDEO_ENC= {'fc': '#C89090', 'ec': '#A04040'}
C_TT  = {'fc': '#C0D0E0', 'ec': '#5070A0'}
C_MP  = {'fc': '#D0DCE8', 'ec': '#6080A8'}
C_CA  = {'fc': '#D4CCE8', 'ec': '#6858A0'}
C_SH  = {'fc': '#E0D4BC', 'ec': '#988060'}
C_PR  = {'fc': '#E4DCC4', 'ec': '#A08860'}
C_UNC = {'fc': '#B8D8E0', 'ec': '#408098'}
C_CON = {'fc': '#D0DCE8', 'ec': '#6080A8'}
C_TOT = {'fc': '#D0C0E0', 'ec': '#403068'}
C_AUX = [
    {'fc': '#D8D0C0', 'ec': '#888050'},
    {'fc': '#C8D8C8', 'ec': '#588858'},
    {'fc': '#C0D0D8', 'ec': '#487888'},
    {'fc': '#D0C8D8', 'ec': '#685888'},
]
BG_IN  = '#E8F0F8';  BG_ENC = '#F0E8D8';  BG_CA  = '#F0ECF8'
BG_PR  = '#F0D0E0';  BG_AUX = '#D0F0D0';  BG_O2 = '#D8D0F0'
BG_SH  = '#F0E4D0';  BG_PRi = '#E8DCC8'
FC = '#2A2A2A';  AC = '#303030'

# ── layout ─
C1, C2, C3 = 5.5, 15.0, 24.5
MW = 4.2;  MH = 0.95;  G = 0.05

yIN = 18.80;  hIN = 1.20
yEN = 17.10;  hEN = 1.20
yTT = 14.90;  ttH = 1.10
yBU = 14.20
yCA = 12.45;  caHb = 0.95
ySH = 10.70;  rpH = 0.65
yPR =  9.30

yO1 =  5.30;  hO1 = 3.60
yO2 =  0.70;  hO2 = 4.20

coY =  7.50;  coH = 0.65
prY =  5.95;  prH = 0.65
auW = 2.40;  auH = 0.55;  auG = 0.50
auTW = 4*auW + 3*auG
auX0 = 15.0 + (14.0 - auTW)/2
auBY = 7.00

# L_total block: increased height 0.60 → 0.85
loW = 20.0;  loH = 0.85;  loX = 15.0-loW/2;  loY = 3.10
# Adjust tY and meY to maintain arrow ratios
loT = loY + loH          # 3.95
tY  = loT + 0.65         # 4.60 (was 4.35)
meY = tY + 0.60          # 5.20 (was 4.70)
diY =  2.40
suW = 4.0;  suH = 0.70
suX0 = 3.60;  suGA = 0.70
suC = [suX0 + i*(suW+suGA) for i in range(5)]
suY =  1.00

# ─ helpers (thinner lines, v58b) ──
def bx(x,y,w,h,fc,ec,lw=1.2,ls='-',zo=3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0",
        facecolor=fc,edgecolor=ec,lw=lw,linestyle=ls,zorder=zo))

def tx(x,y,s,fs=12,bo=False,ha='center',va='center',c=None,z=5,**kw):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,
        color=c if c else FC,
        fontweight='bold' if bo else 'normal',
        fontfamily='serif',zorder=z,**kw)

def _o(x,y,d,g):
    return {'up':(x,y+g),'down':(x,y-g),'left':(x-g,y),'right':(x+g,y)}.get(d,(x,y))

def aw(x1,y1,x2,y2,lw=1.8,d1=None,d2=None,g1=G,g2=G):
    sx,sy = _o(x1,y1,d1,g1) if d1 else (x1,y1)
    ex,ey = _o(x2,y2,d2,g2) if d2 else (x2,y2)
    ax.annotate('',xy=(ex,ey),xytext=(sx,sy),zorder=4,
        arrowprops=dict(arrowstyle='->',color=AC,lw=lw,
            shrinkA=0,shrinkB=0,mutation_scale=16))

def ln(x1,y1,x2,y2,lw=2.0,zo=4):
    ax.plot([x1,x2],[y1,y2],color=AC,lw=lw,zorder=zo,solid_capstyle='round')

def sh(x,y,s,fs=15):
    ax.text(x,y,s,ha='left',va='center',fontsize=fs,c='#304878',
        fontweight='bold',fontstyle='italic',fontfamily='sans-serif',zorder=5)

def rb(x,y,w,h,fc,a=0.50):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec='#999999',
        lw=1.0,ls=':',alpha=a,zorder=2))

# ═════════════════════════════════════════════════════
tx(15, 21.50, 'MER-MTL', fs=22, bo=True)

# ══ SECTION BACKGROUNDS ═══
rb(1.0, yIN+0.05, 28.0, hIN+0.15, BG_IN, a=0.40)
rb(1.0, yEN+0.02, 28.0, hEN+0.15, BG_ENC, a=0.40)
rb(4.5, yCA-0.10, 21.0, caHb+0.20, BG_CA, a=0.40)

# ══ INPUT MODULE ═══
sh(1.3, yIN+hIN-0.10, 'Input Module')
for cx,(n,d),s,ic in zip([C1,C2,C3],
    [('Text','768-d'),('Audio','768-d'),('Video','768-d')],
    ['BERT Token IDs','wav2vec / HuBERT','FaceFormer / 3DMM'],
    [C_TEXT_IN,C_AUDIO_IN,C_VIDEO_IN]):
    x=cx-MW/2; y=yIN+0.15
    bx(x,y,MW,MH,ic['fc'],ic['ec'],lw=1.4)
    tx(cx,y+MH*0.68,n,fs=20,bo=True)
    tx(cx,y+MH*0.28,d,fs=14)
    tx(cx,y+MH+0.15,s,fs=11,c='#506080')

# ══ ENCODERS ═══
sh(1.3, yEN+hEN-0.08, 'Encoders', fs=14)
for cx,l,v,ec in zip([C1,C2,C3],
    ['Text Encoder','Audio Encoder','Video Encoder'],
    ['Conv1D -> 50-d']*3,
    [C_TEXT_ENC,C_AUDIO_ENC,C_VIDEO_ENC]):
    x=cx-MW/2; y=yEN+0.10
    bx(x,y,MW,MH,ec['fc'],ec['ec'])
    tx(cx,y+MH*0.68,l,fs=18,bo=True)
    tx(cx,y+MH*0.28,v,fs=13)

# ═══ TT / MP ═══
ttW,mpW=3.8,3.8; gTT=0.80
ttX=C1-ttW-gTT/2; mpX=C1+gTT/2
ttCX=ttX+ttW/2; mpCX=mpX+mpW/2
bx(ttX,yTT,ttW,ttH,C_TT['fc'],C_TT['ec'],lw=1.4)
tx(ttCX,yTT+ttH*0.68,'TT',fs=19,bo=True)
tx(ttCX,yTT+ttH*0.28,'Transformer',fs=13)
bx(mpX,yTT,mpW,ttH,C_MP['fc'],C_MP['ec'],lw=1.4,ls='--')
tx(mpCX,yTT+ttH*0.68,'MP',fs=19,bo=True)
tx(mpCX,yTT+ttH*0.28,'Mean Pool + MLP',fs=13)

# ═══ CROSS-ATTENTION ═══
caW=20.0; caX=15-caW/2; caY=yCA
bx(caX,caY,caW,caHb,C_CA['fc'],C_CA['ec'])
tx(caX+caW/2, caY+caHb*0.65, 'Pairwise Cross-Attention', fs=19, bo=True)
tx(caX+caW/2, caY+caHb*0.30,
   'L<->A  .  L<->V  .  A<->V    (10 heads)', fs=14)

# ══ SHARED / PRIVATE (expanded backgrounds) ══
# Backgrounds expanded: padding 0.15/0.30 → 0.30/0.60
rb(1.0, ySH-0.30, 28.0, rpH+0.60, BG_SH)
rb(1.0, yPR-0.30, 28.0, rpH+0.60, BG_PRi)
tx(1.8, ySH+rpH+0.05, 'Shared Representations', fs=13, ha='left', c='#605040', fontstyle='italic')
tx(1.8, yPR+rpH+0.05, 'Private Representations', fs=13, ha='left', c='#605040', fontstyle='italic')
for cx,m in zip([C1,C2,C3],'LAV'):
    bx(cx-2.25, ySH, 4.5, rpH, C_SH['fc'], C_SH['ec'])
    tx(cx, ySH+rpH/2, f'Shared: s_{m}  (50-d)', fs=14)
for cx,m in zip([C1,C2,C3],'LAV'):
    bx(cx-2.25, yPR, 4.5, rpH, C_PR['fc'], C_PR['ec'])
    tx(cx, yPR+rpH/2, f'Private: c_{m}  (50-d)', fs=14)

# ═══ OM1 LEFT ═══
HW = 14.0
rb(1.0, yO1, HW, hO1, BG_PR)
o1cx = 1.0 + HW/2
sh(1.3, yO1+hO1-0.50, 'Output Module 1: Emotion Recognition')
cW=11.0
bx(o1cx-cW/2, coY, cW, coH, C_CON['fc'], C_CON['ec'])
tx(o1cx, coY+coH*0.68, 'Concat(s_L, s_A, s_V)', fs=15)
tx(o1cx, coY+coH*0.28, '-> Dual Pooling (mean + max)', fs=13)
bx(o1cx-cW/2, prY, cW, prH, C_UNC['fc'], C_UNC['ec'])
tx(o1cx, prY+prH/2, '7-class + Binary (+/-)', fs=15, bo=True)

# ═══ OM1 RIGHT (Aux) ═══
rb(1.0+HW, yO1, HW, hO1, BG_AUX)
axCX = 1.0+HW + HW/2
sh(1.0+HW+0.3, yO1+hO1-0.50, 'Auxiliary Tasks')
# Aux labels: LaTeX math font with subscripts
auLabs_latex = [
    r'$\mathcal{L}_{rec}$: Recon',
    r'$\mathcal{L}_{cyc}$: Cycle',
    r'$\mathcal{L}_{mar}$: Marginal',
    r'$\mathcal{L}_{ort}$: Orthog.',
]
auCXs = []
for i,(ac,al) in enumerate(zip(C_AUX, auLabs_latex)):
    x=auX0+i*(auW+auG)
    bx(x, auBY, auW, auH, ac['fc'], ac['ec'])
    cx=x+auW/2; auCXs.append(cx)
    tx(cx, auBY+auH/2, al, fs=10, bo=False, z=7)

# ═ OM2 ═══
rb(1.0, yO2, 28.0, hO2, BG_O2)
sh(1.3, yO2+hO2-0.18, 'Output Module 2: Uncertainty-Aware Loss')

# σ blocks: LaTeX math font with \mathcal{L} subscripts
sig_latex = [
    r'$\sigma_0$  ($\mathcal{L}_{task}$)',
    r'$\sigma_1$  ($\mathcal{L}_{rec}$)',
    r'$\sigma_2$  ($\mathcal{L}_{cyc}$)',
    r'$\sigma_3$  ($\mathcal{L}_{mar}$)',
    r'$\sigma_4$  ($\mathcal{L}_{ort}$)',
]
for i, scx in enumerate(suC):
    bx(scx-suW/2, suY, suW, suH, C_UNC['fc'], C_UNC['ec'], lw=1.4, zo=6)
    tx(scx, suY+suH/2, sig_latex[i], fs=11, bo=False, z=7)

bx(loX, loY, loW, loH, C_TOT['fc'], C_TOT['ec'], lw=1.6, zo=3)
tx(15.0, loY+loH/2,
   r'$\mathcal{L}_{total} = \sum_k \left[ \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k \right]$',
   fs=16, bo=True)

ln(suC[0], diY, suC[-1], diY)
meCX = (auCXs[0]+auCXs[-1])/2
ln(auCXs[0], meY, auCXs[-1], meY)

# ═══ EDGES ══
iB = yIN+0.15;       eT = yEN+0.10+MH;  eB = yEN+0.10
ttT= yTT+ttH;        ttB= yTT
caT= caY+caHb;       caB= caY
sT = ySH+rpH;        sB = ySH
pT = yPR+rpH;        pB = yPR
cT = coY+coH;        cB = coY
prT= prY+prH;        prB= prY
aT = auBY+auH;       aB = auBY
suT= suY+suH;        suB= suY
loT_val= loY+loH;    loB = loY

# ═══ ARROWS ═══

# 1  Input -> Encoder
for cx in [C1,C2,C3]:
    aw(cx,iB, cx,eT, d1='down',d2='up')

# 2  Text Encoder -> TT & MP
b2 = ttT + 0.65
ln(C1,eB, C1,b2)
ln(ttCX,b2, mpCX,b2)
aw(ttCX,b2, ttCX,ttT, d2='up')
aw(mpCX,b2, mpCX,ttT, d2='up')

# 3  TT + MP + AudioEnc + VideoEnc -> bus -> CA
ln(ttCX, ttB, ttCX, yBU)
ln(mpCX, ttB, mpCX, yBU)
ln(C2,   eB, C2,   yBU)
ln(C3,   eB, C3,   yBU)
ln(ttCX, yBU, C3,  yBU)
aw(15.0, yBU, 15.0, caT, d2='up')

# 4  CA -> Shared
b4 = caB - 0.30
ln(15.0,caB, 15.0,b4)
ln(C1,b4, C3,b4)
for cx in [C1,C2,C3]:
    aw(cx,b4, cx,sT, d2='up')

# 5  Shared -> Private
for cx in [C1,C2,C3]:
    aw(cx,sB, cx,pT, d1='down',d2='up')

# 6  Private -> Concat (left) + Aux (right)
b6 = pB - 0.50
ln(C1,pB, C1,b6)
ln(C3,pB, C3,b6)
ln(C1,b6, C3,b6)
aw(o1cx,b6, o1cx,cT, d2='up')
l6 = b6 - 0.50
ln(axCX,b6, axCX,l6)
ln(auCXs[0],l6, auCXs[-1],l6)
for cx in auCXs:
    aw(cx,l6, cx,aT, d2='up')

# 7  Concat -> Prediction
aw(o1cx,cB, o1cx,prT, d1='down',d2='up')

# 8  Aux -> merge -> L_total -> dist -> sigma
for cx in auCXs:
    ln(cx, aB, cx, meY)
ln(auCXs[0], meY, auCXs[-1], meY)
ln(meCX, meY, meCX, tY)
ln(meCX, tY, 15.0, tY)
aw(15.0, tY, 15.0, loT_val, d2='up')
aw(15.0, loB, 15.0, diY, d1='down',d2='up')
for i in range(5):
    aw(suC[i],diY, suC[i],suT, d1='down',d2='up')

# ═══ LEGEND ═══
yL = 0.15
for i,(cs,ll) in enumerate([
    (C_TEXT_IN,'Text'),(C_AUDIO_IN,'Audio'),
    (C_VIDEO_IN,'Video'),(C_CA,'Cross-Attn'),
    (C_SH,'Shared'),(C_PR,'Private'),
    (C_UNC,'Output'),(C_AUX[0],'Aux Loss'),
    (C_UNC,'Uncertainty'),(C_TOT,'Total Loss')]):
    lx=1.5+i*2.7
    bx(lx,yL,0.45,0.35,cs['fc'],cs['ec'],lw=0.8)
    tx(lx+0.65,yL+0.18,ll,fs=10,ha='left')

# ═══ SAVE ═══
plt.tight_layout(pad=0.5)
import os as _os
_ob_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'figures')
_os.makedirs(_ob_dir, exist_ok=True)
ob = _os.path.join(_ob_dir, 'fig_architecture')
plt.savefig(ob+'.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.savefig(ob+'.pdf',dpi=300,bbox_inches='tight',facecolor='white')
plt.close()
print("arch done!")
