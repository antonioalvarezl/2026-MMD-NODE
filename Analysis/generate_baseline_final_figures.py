from pathlib import Path
import sys,json,shutil,numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',required=True,type=Path)
parser.add_argument('--output-dir',required=True,type=Path)
args=parser.parse_args()
r=args.project_root.resolve();b=args.output_dir.resolve();b.mkdir(parents=True,exist_ok=True);
out=b/'overleaf/Pictures/AoAS';out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(r))
from Analysis import paper_figures as pf
from Analysis.baseline_projection import load_baseline_frames,baseline_points,project_onto_dictionary
cgm,qc,_=load_baseline_frames();cache=b/'d1_cache';cache.mkdir(exist_ok=True);D=json.loads((r/'Results_paper/CGM/run16/global_dictionary.json').read_text());mu=np.array(D['means']);cov=np.array(D['covariances']);sc=D['scaling'];ids=np.load(r/'figures_paper/d1/run16/global_moments/global_moments.npz')['patient_ids']
def work(pid):
 p=cache/f'{pid}.json'
 if p.exists():return json.loads(p.read_text())
 pts=baseline_points(pid,cgm,qc,'univariate')
 if pts is None:return None
 f=json.loads((r/f'Results_paper/CGM/run16/patient_{pid}/result_fit.json').read_text());n=json.loads((r/f'Results_paper/CGM/run16/patient_{pid}/node_1/result_node.json').read_text());cfg=f['config'];w=project_onto_dictionary(pts,mu,cov,sc['mean'],sc['std'],cfg['patient_bandwidth_reference'],cfg.get('lambda_ridge',0));record=dict(id=int(pid),arm=str(qc.set_index('PtID').loc[pid,'TrtGroup']),baseline_weights=w.tolist(),final_weights=np.array(n['trajectory']).reshape(-1,5)[-1].tolist(),final_week=float(n['trajectory_time_grid_weeks'][-1]));p.write_text(json.dumps(record));return record
records=[]
with ThreadPoolExecutor(max_workers=4) as ex:
 for f in as_completed([ex.submit(work,int(p)) for p in ids]):
  a=f.result()
  if a is not None:records.append(a);print('d1 projected',len(records),flush=True)
records.sort(key=lambda a:a['id']);(b/'d1_participants.json').write_text(json.dumps(records,indent=2));assert len(records)==85
folder=out/'d1/baseline_to_final';folder.mkdir(parents=True,exist_ok=True)
data=dict(trajectories=np.array([[a['baseline_weights'],a['final_weights']] for a in records]),labels=np.array([1 if a['arm']=='CLC' else 0 for a in records]),means_original_units=np.array(D['means_original_units']),covs_original_units=np.array(D['covariances_original_units']),means_standardized=mu,covs_standardized=cov,scaling=sc)
original_save=pf._save_figure
mapping={'density_initial_final_difference':'baseline_final_difference','density_initial_comparison_frechet':'baseline_comparison','density_final_comparison_frechet':'final_comparison','density_frechet_difference_measure_control':'control_difference','density_frechet_difference_measure_treatment':'treatment_difference'}
def save(fig,output,stem):
 if stem=='density_initial_final_difference':
  ymax=max(max(line.get_ydata()) for ax in fig.axes[:2] for line in ax.lines)*1.06
  for ax in fig.axes[:2]:ax.set_ylim(0,ymax)
 for ax in fig.axes:
  if ax.get_title()=='Initial':ax.set_title('Pre-randomization baseline')
  elif ax.get_title()=='Final':ax.set_title('Final follow-up')
  elif ax.get_title().startswith('Difference:'):ax.set_title(ax.get_title().replace('Difference:','Final minus baseline:'))
 return original_save(fig,output,mapping.get(stem,stem))
pf._save_figure=save;pf._plot_density_figure(data,folder,expected_dimension=1);pf._save_figure=original_save
print('Saved d1 figures',flush=True)
# Existing verified bivariate baseline-final outputs, preserving the same paired cohort.
source=r/'figures_paper/d2/baseline_to_final';target=out/'d2/baseline_to_final';target.mkdir(parents=True,exist_ok=True)
for p in source.glob('*.pdf'):shutil.copy2(p,target/p.name)
# Render individual baseline/final densities while retaining the follow-up signal and NODE panels.
nb=json.loads((r/'Analysis/Medical_DataAnalysis.ipynb').read_text());ns={'display':lambda *a:None}
for i in [2,4,21,22]:
 s=''.join(nb['cells'][i]['source'])
 if i==2:s=s.replace('CWD = Path.cwd().resolve()',f'CWD = Path({str(r)!r})').replace('OUTPUT_FOLDER = FIGURES_ROOT / DIMENSION / RUN_NAME',f'OUTPUT_FOLDER = Path({str(b/"notebook_scratch")!r})')
 exec(compile(s,f'notebook cell {i}','exec'),ns)
s=''.join(nb['cells'][23]['source']);s=s.replace('for ax, which, index in ((axes[1, 0], "Initial", 0), (axes[1, 1], "Final", -1)):', 'for ax, which, index in ((axes[1, 0], "Pre-randomization baseline", 0), (axes[1, 1], "Final follow-up", -1)):');s=s.replace('weights = node_weights_at_times(bundle, np.asarray([obs_time]))[0]','weights = bundle["baseline_weights"] if index == 0 else bundle["final_weights"]');s=s.replace('Glucose rate of change (mg/dL/time)','Glucose rate of change (mg/dL/min)');exec(compile(s,'paired individual composite','exec'),ns)
paired={a['id']:a for a in json.loads((source/'metadata.json').read_text())['participants']};cgmfollow=ns['load_cgm_table']() if 'load_cgm_table' in ns else ns['pd'].read_parquet(ns['PEDAP_CGM_FILE'])
for pid in [16,26,39]:
 bundle=ns['load_patient_bundle'](pid,ns['RUN_FOLDER']);bundle['baseline_weights']=np.array(paired[pid]['baseline_weights']);bundle['final_weights']=np.array(paired[pid]['final_weights']);raw,_=ns['_patient_raw_blocks_d2'](pid);pre=baseline_points(pid,cgm,qc,'bivariate');blocks=[pre,raw[-1]];dest=target/f'individual/patient_{pid}';dest.mkdir(parents=True,exist_ok=True);ns['plot_patient_composite_d2'](pid,bundle,blocks,cgmfollow,dest);print('Saved paired individual',pid,flush=True)
print('DONE ALL FIGURES',flush=True)
