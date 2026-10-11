"""Generate publication figures and LaTeX numbers, without manual transcription."""
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT
from analyze import DATASETS,ARCHS,METHODS
PAPER=ROOT/'paper';FIG=PAPER/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':9,'legend.fontsize':9})
LABELS={'nominal':'Nominal','context':'Context','bonferroni':'Bonf.','corrected_maxrank':'CMR','envelope':'Envelope'}
COLORS=['#777777','#bd4c35','#6a58a5','#007e87','#d99200']

def main(incomplete=False):
 s=json.loads((ROOT/'results'/'summary.json').read_text());index={(r['dataset'],r['arch'],r['mode'],r['method']):r for r in s}
 if not incomplete:
  assert len(s)==75 and all(r['nseeds']==5 for r in s), 'All 45 primary models must be complete.'
 high=json.loads((ROOT/'results_highres'/'summary.json').read_text()) if (ROOT/'results_highres'/'summary.json').exists() else []
 hi={(r['mode'],r['method']):r for r in high}
 if not incomplete:assert len(high)==10 and all(r['nseeds']==5 for r in high), 'All five higher-resolution models must be complete.'
 def get(d,a,m='context',mode='target'):return index.get((d,a,mode,m))
 def val(d,a,k,m='context',mode='target'):
  r=get(d,a,m,mode);return r[k] if r and r['nseeds']==5 else np.nan
 def save(fig,name):
  fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.04);fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.04);plt.close(fig)
 labels=[];means=[];joints=[];msd=[];jsd=[]
 for d,short in zip(DATASETS,['P','B','D']):
  for a,label in zip(ARCHS[:2],['C','R']):
   labels.append(short+'/'+label);means.append(val(d,a,'mean_context_coverage_mean')*100);joints.append(val(d,a,'simultaneous_coverage_mean')*100);msd.append(val(d,a,'mean_context_coverage_sd')*100);jsd.append(val(d,a,'simultaneous_coverage_sd')*100)
 if hi.get(('target','context'),{}).get('nseeds')==5:
  r=hi[('target','context')];labels.append('P/18');means.append(r['mean_context_coverage_mean']*100);joints.append(r['simultaneous_coverage_mean']*100);msd.append(r['mean_context_coverage_sd']*100);jsd.append(r['simultaneous_coverage_sd']*100)
 fig,ax=plt.subplots(figsize=(3.32,2.0));x=np.arange(len(labels))
 ax.errorbar(x-.1,means,yerr=msd,fmt='o',ms=4,capsize=2,color='#007e87',label='Mean over contexts');ax.errorbar(x+.1,joints,yerr=jsd,fmt='s',ms=4,capsize=2,color='#bd4c35',label='All contexts')
 for i in x:ax.plot([i-.1,i+.1],[means[i],joints[i]],color='.65',lw=.8,zorder=0)
 ax.axhline(90,color='.4',ls='--',lw=.8);ax.set_xticks(x,labels);ax.set_xlabel('Dataset / model');ax.set_ylabel('True-label coverage (%)');ax.set_ylim(min(50,np.nanmin(joints)-4) if np.isfinite(joints).any() else 50,103);ax.legend(frameon=False,loc='lower left');ax.grid(axis='y',alpha=.15);fig.tight_layout(pad=.4);save(fig,'coverage')
 fig,ax=plt.subplots(figsize=(3.32,2.0))
 offsets=[(5,-9),(5,-14),(-40,5),(3,-14),(-18,-29)]
 for m,c,off in zip(METHODS,COLORS,offsets):
  r=get(DATASETS[0],'resnet_bn',m)
  if r:
   ax.errorbar(r['mean_size_mean'],r['simultaneous_coverage_mean']*100,xerr=r['mean_size_sd'],yerr=r['simultaneous_coverage_sd']*100,fmt='o',ms=4,capsize=2,color=c);ax.annotate(LABELS[m],(r['mean_size_mean'],r['simultaneous_coverage_mean']*100),xytext=off,textcoords='offset points',fontsize=9,color=c)
 ax.axhline(90,color='.4',ls='--',lw=.8);ax.set(xlim=(.95,2.07),ylim=(50,104),xlabel='Mean prediction-set size (2 classes)',ylabel='Simultaneous coverage (%)');ax.grid(alpha=.15);fig.tight_layout(pad=.4);save(fig,'tradeoff')
 rr=list(csv.DictReader((ROOT/'results'/'split_sensitivity.csv').open()));fig,ax=plt.subplots(figsize=(3.32,1.85))
 vals=[np.array([float(r['simultaneous_coverage'])*100 for r in rr if r['dataset']=='breastmnist' and r['arch']=='cnn_bn' and r['method']==m]) for m in METHODS]
 if all(len(v) for v in vals):
  bp=ax.boxplot(vals,whis=(5,95),showfliers=False,patch_artist=True,tick_labels=[LABELS[m] for m in METHODS],widths=.5)
  for patch,c in zip(bp['boxes'],COLORS):patch.set_facecolor(c);patch.set_alpha(.65)
 else:ax.text(.5,.5,'Experiment running',ha='center',transform=ax.transAxes)
 ax.axhline(90,color='.4',ls='--',lw=.8);ax.set_ylabel('Simultaneous coverage (%)');ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.15);fig.tight_layout(pad=.4);save(fig,'splits')
 # Full comparison, compact cells joint coverage / deployed size.
 tex=['\\begin{table*}[t]','\\centering','\\caption{Current-batch BN: simultaneous coverage (\\%) / mean deployed set size. Each cell is the mean across five training seeds. Per-context calibration targets each context separately; simultaneous methods target the joint event. Full seed SDs, accuracy/AUC, and per-image outputs are released.}','\\label{tab:main}','\\begin{tabular}{llrrrrrr}\\toprule','Dataset & Model & Flip (\\%) & Nominal & Per-context & Bonferroni & CMR & Envelope\\\\\\midrule']
 for d,label in zip(DATASETS,['PneumoniaMNIST','BreastMNIST','DermaMNIST-C']):
  for a,aname in zip(ARCHS[:2],['CNN-BN','ResNet-BN']):
   cells=[]
   for m in METHODS:
    r=get(d,a,m);cells.append(f"{r['simultaneous_coverage_mean']*100:.1f} / {r['mean_size_mean']:.2f}" if r and r['nseeds']==5 else 'pending')
   flip=val(d,a,'class_flip_rate_mean');tex.append(' & '.join([label,aname,f'{flip*100:.1f}' if np.isfinite(flip) else 'pending',*cells])+r'\\')
 if hi.get(('target','context'),{}).get('nseeds')==5:
  cells=[f"{hi['target',m]['simultaneous_coverage_mean']*100:.1f} / {hi['target',m]['mean_size_mean']:.2f}" for m in METHODS]
  tex.append(' & '.join(['PneumoniaMNIST','RN18-224',f"{hi['target','context']['class_flip_rate_mean']*100:.1f}",*cells])+r'\\')
 tex.extend([r'\bottomrule\end{tabular}',r'\end{table*}']);(PAPER/'main_table.tex').write_text('\n'.join(tex))
 macros={};r=get(DATASETS[0],'resnet_bn')
 for name,key,m in [('PneuContextMean','mean_context_coverage_mean','context'),('PneuContextJoint','simultaneous_coverage_mean','context'),('PneuCMRJoint','simultaneous_coverage_mean','corrected_maxrank'),('PneuEnvelopeJoint','simultaneous_coverage_mean','envelope'),('PneuCMRSize','mean_size_mean','corrected_maxrank'),('PneuEnvelopeSize','mean_size_mean','envelope'),('PneuContextSize','mean_size_mean','context')]:
  v=val(DATASETS[0],'resnet_bn',key,m);macros[name]=f'{v*100:.1f}\\%' if 'coverage' in key else f'{v:.2f}'
  if not np.isfinite(v):macros[name]=r'\textit{pending}'
 flips=[val(d,a,'class_flip_rate_mean')*100 for d in DATASETS for a in ARCHS[:2]]
 macros['FlipRange']=f'{np.nanmin(flips):.1f}--{np.nanmax(flips):.1f}\\%'
 boot=list(csv.DictReader((ROOT/'results'/'paired_bootstrap.csv').open())) if (ROOT/'results'/'paired_bootstrap.csv').exists() else []
 gap=next((r for r in boot if r['dataset']==DATASETS[0] and r['arch']=='resnet_bn' and r['effect']=='context_mean_minus_joint'),None)
 macros['PneuGapCI']=f"{float(gap['mean'])*100:.1f} [{float(gap['lower'])*100:.1f}, {float(gap['upper'])*100:.1f}]" if gap else r'\textit{pending}'
 (PAPER/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()))
 allr=list(csv.DictReader((ROOT/'results'/'all_metrics.csv').open()));sec=[]
 for a,aname in zip(ARCHS[:2],['CNN-BN','ResNet-BN']):
  hits={r['mode']:float(r['class_flip_rate'])*100 for r in allr if r['dataset']==DATASETS[0] and r['arch']==a and r['seed']=='0' and r['condition']=='clean' and r['method']=='nominal' and r['batch']=='16'}
  if all(k in hits for k in ['target','mixed','first']):sec.append(f"{aname}: {hits['target']:.1f}\\% full BN, {hits['mixed']:.1f}\\% mixed moments, and {hits['first']:.1f}\\% first-layer-only")
 text='Seed-0 normalization controls on PneumoniaMNIST yield class-flip rates of '+('; '.join(sec))+'. The fixed 50:50 moment mixture is a control, not a full TTN implementation. Complete secondary results are released, including a matched affine-intensity condition and batch sizes 8 and 32.' if len(sec)==2 else 'Secondary experiments are running.'
 (PAPER/'secondary_text.tex').write_text(text)
 aps=list(csv.DictReader((ROOT/'results'/'score_sensitivity.csv').open())) if (ROOT/'results'/'score_sensitivity.csv').exists() else []
 def apsval(method,key):
  selected=[r for r in aps if r['dataset']==DATASETS[0] and r['arch']=='resnet_bn' and r['score']=='randomized_APS' and r['method']==method]
  return float(np.mean([float(r[key]) for r in selected])) if len(selected)==5 else np.nan
 ac=apsval('context','simultaneous_coverage');am=apsval('corrected_maxrank','simultaneous_coverage');az=apsval('corrected_maxrank','mean_size')
 if np.isfinite(ac):
  text=f"An exploratory randomized APS sensitivity \\cite{{aps}} shares one uniform variate across each query's contexts: the PneumoniaMNIST residual model gives {ac*100:.1f}\\% simultaneous coverage with context-specific calibration and {am*100:.1f}\\% with CMR (size {az:.2f}). Deterministic APS makes CMR and the envelope return both labels on this task. Score choice therefore changes the coverage--size trade-off; all score diagnostics are released."
 else:text='Score sensitivity is running.'
 (PAPER/'score_text.tex').write_text(text)
 if hi.get(('target','context'),{}).get('nseeds')==5:
  r=hi['target','context'];c=hi['target','corrected_maxrank'];src=hi['source','nominal']
  text=f"The 224-pixel pretrained RN18 extension has {src['accuracy_mean']*100:.1f}\\% source-statistics accuracy. Current-batch BN gives a {r['class_flip_rate_mean']*100:.1f}\\% companion-induced class-flip rate; context-specific calibration yields {r['mean_context_coverage_mean']*100:.1f}\\% mean and {r['simultaneous_coverage_mean']*100:.1f}\\% simultaneous coverage. CMR gives {c['simultaneous_coverage_mean']*100:.1f}\\% simultaneous coverage with mean set size {c['mean_size_mean']:.2f}. This extension uses its own predefined fine-tuning protocol and does not establish clinical generalization."
 else:text='Higher-resolution extension is running.'
 (PAPER/'highres_text.tex').write_text(text)
 vp=list(csv.DictReader((ROOT/'results'/'validation_pool.csv').open())) if (ROOT/'results'/'validation_pool.csv').exists() else []
 v=next((r for r in vp if r['dataset']==DATASETS[0] and r['arch']=='resnet_bn' and r['method']=='context'),None)
 if v:
  text=f"A seed-0 validation-companion sensitivity gives {float(v['mean_context_coverage'])*100:.1f}\\% mean versus {float(v['simultaneous_coverage'])*100:.1f}\\% simultaneous coverage for the compact pneumonia residual model. These companions were used for checkpoint selection but excluded from gradient training; results for all six configurations are released."
 else:text='Validation-companion sensitivity is running.'
 (PAPER/'validation_text.tex').write_text(text)
 print('Generated figures and tables; incomplete mode =',incomplete)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args();main(a.allow_incomplete)
