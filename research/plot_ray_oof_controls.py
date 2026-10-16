"""Plot verified existing numbers only; no predictions or models recomputed."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
METHODS=[('recentered_base','Strong concat (hip recentered)'),
 ('free_no_ray_residual','OOF residual, no ray geometry'),
 ('constant_ray_gate','OOF gate, constant-depth rays'),
 ('radar_ray_gate','OOF gate, radar-depth rays'),
 ('free_constant_residual','OOF residual, constant-depth rays'),
 ('free_radar_residual','OOF residual, radar-depth rays')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='fig_ray_oof_controls');args=p.parse_args()
    source=ROOT/'artifacts/ray_oof_strict_controls_20261016/reporting/results.json'
    data=json.loads(source.read_text());assert data['status']=='complete'
    assert json.loads((source.parents[1]/'audit.json').read_text())['status']=='complete'
    out=source.parent/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    rows=[];exports=[]
    for quantity,xlabel,tag in [('clean_mm','Clean MPJPE (mm)','clean'),('S_mm','Five-condition subject-macro error S (mm)','score')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for i,(method,label) in enumerate(METHODS):
            summary=data['aggregate'][method][quantity]
            color='#15607A' if method=='free_radar_residual' else '#545B66'
            observations=sorted([r for r in data['per_initialization'] if r['method']==method],key=lambda r:r['seed'])
            assert len(observations)==3
            for j,row in enumerate(observations):
                ax.scatter(row[quantity],i+(j-1)*.09,s=20,facecolors='white',edgecolors=color,linewidths=.85,zorder=2)
                rows.append({'quantity':quantity,'method':method,'seed':row['seed'],'value_mm':row[quantity]})
            ax.errorbar(summary['mean'],i,xerr=summary['initialization_sd'],fmt='o',markersize=6,color=color,
                        elinewidth=1.35,capsize=3,zorder=3)
        ax.set_yticks(range(len(METHODS)),[label for _,label in METHODS]);ax.invert_yaxis()
        ax.set_xlabel(xlabel);ax.set_ylabel('Fusion variant')
        ax.grid(axis='x',color='#D8DDE2',linewidth=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.08,y=.1)
        fig.tight_layout()
        stem=out/(args.prefix+'_'+tag)
        for extension in ('png','pdf','svg'):
            target=stem.with_suffix('.'+extension);fig.savefig(target,dpi=300,bbox_inches='tight');exports.append(str(target))
        plt.close(fig)
    with (out/(args.prefix+'_source.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    caption=('mRI development validation: 12 training and 4 validation subjects, 96 frames per subject, three initialization seeds. '
        'Filled points and horizontal error bars show the mean and initialization standard deviation; hollow points show each initialization. '
        'The seeds are not independent subject splits. Correctors use the same subject-held-out predictions and training budget; '
        'the plain base has no additional OOF correction stage. Constant-depth controls retain the observed depth feature but fix depth inside '
        'the geometric operator. Final-test subjects were not evaluated. These figures support an empirical comparison, not a claim of novelty.')
    (out/(args.prefix+'_captions.md')).write_text('# Figure captions\n\nClean MPJPE. '+caption+'\n\nFive-condition score S. '+caption+'\n',encoding='utf-8')
    tex=[]
    for tag,description in [('clean','Clean MPJPE'),('score','Five-condition subject-macro score')]:
        tex.append('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.96\\linewidth]{'+args.prefix+'_'+tag+'.pdf}\n\\caption{'+description+'. '+caption+'}\n\\label{fig:ray_oof_'+tag+'}\n\\end{figure}\n')
    (out/(args.prefix+'_latex.tex')).write_text('\n'.join(tex),encoding='utf-8')
    receipt={'source_results_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plotter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exports':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in exports},'data_rows':len(rows),'visual_review':'pending'}
    (out/(args.prefix+'_receipt.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'figures':exports,'data_rows':len(rows)}))

if __name__=='__main__':main()
