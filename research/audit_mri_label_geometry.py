"""Descriptive triangulation conditioning, not calibrated label uncertainty.

Assume independent one-pixel camera measurements solely for a local sensitivity
calculation. No labels are corrected and no model is trained in this script.
"""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from audit_mri import load_labels
from cache_mri_action_windows import select_windows,DEV
ROOT=Path(__file__).resolve().parents[1]

def projection_jacobian(points,projection):
    # Same previously audited label-to-calibration convention, mm and reversed XY.
    homogeneous=np.concatenate([points*np.array([-1000.,-1000.,1000.]),np.ones(points.shape[:-1]+(1,))],axis=-1)
    q=homogeneous@projection.T
    denominator=q[...,2]
    if (abs(denominator)<1e-12).any():raise ValueError('Singular camera projection')
    jac=(projection[:2,:3]*denominator[...,None,None]-q[...,:2,None]*projection[2,:3])/(denominator[...,None,None]**2)
    jac=jac*np.array([-1000.,-1000.,1000.])
    return q[...,:2]/denominator[...,None],jac

def pct(v):return {k:float(x) for k,x in zip(['min','p50','p90','p99','max'],np.percentile(v,[0,50,90,99,100]))}

def main():
    out=ROOT/'artifacts/mri_label_geometry_20261016'
    if out.exists():raise ValueError('New output required')
    out.mkdir();rows=[]
    with zipfile.ZipFile(ROOT/'dataset_release.zip') as z:
        for s in DEV:
            labels=load_labels(z.read(f'dataset_release/aligned_data/pose_labels/subject{s}_all_labels.cpl'))
            rs,re=labels['radar_avail_frames'];gs,ge=labels['gt_avail_frames']
            ids,actions,windows=select_windows(labels['video_label'],max(rs,gs),min(re,ge))
            gt=labels['refined_gt_kps'][ids].transpose(0,2,1).astype(float)
            cameras=labels['camera_matrix'];projections=[];jacobians=[];residual=[]
            for name,side in [('cam1','r'),('cam2','l')]:
                p,j=projection_jacobian(gt,cameras[name+'_proj_mat']);projections.append(p);jacobians.append(j)
                residual.append(np.linalg.norm(p-labels['pose_2d_'+side][ids].transpose(0,2,1),axis=-1))
            jac=np.concatenate(jacobians,axis=-2)
            information=np.swapaxes(jac,-1,-2)@jac
            values,vectors=np.linalg.eigh(information)
            if (values< -1e-6).any():raise ValueError('Invalid information PSD')
            # Floored inversion is reported as an approximation, never as a
            # calibrated Gaussian confidence on a physically impossible label.
            inverse=1/np.maximum(values,1e-16)
            covariance=(vectors*inverse[...,None,:])@np.swapaxes(vectors,-1,-2)
            sensitivity_mm=1000*np.sqrt(np.trace(covariance,axis1=-2,axis2=-1))
            relative=gt-gt[:,[11,12]].mean(1)[:,None]
            radius=np.linalg.norm(relative,axis=-1).max(1)
            bad=radius>3
            np.savez_compressed(out/f'subject{s}.npz',frame_ids=ids,one_pixel_covariance_m2=covariance,
                one_pixel_sensitivity_mm=sensitivity_mm,reprojection_residual_pixels=np.stack(residual,axis=1),radius_m=radius)
            rows.append({'subject':s,'frames':len(ids),'radius_above_3m_frames':int(bad.sum()),
                'sensitivity_all_mm':pct(sensitivity_mm),'sensitivity_radius_le3m_mm':pct(sensitivity_mm[~bad]),
                'sensitivity_radius_gt3m_mm':pct(sensitivity_mm[bad]) if bad.any() else None,
                'reprojection_residual_px':pct(np.stack(residual))})
    report={'status':'complete','test_subjects_used':[],'subjects':rows,
        'definition':'Local inverse JtJ at published 3D, two cameras, fixed independent 1-pixel isotropic measurement noise. A descriptive conditioning score only; no measured/calibrated GT error distribution.',
        'limitations':['Published 2D detector errors can be biased/correlated','No independent motion capture labels exist here',
            'Local linearization may be invalid at extreme labels','No data correction, prediction, training or final-test access'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'complete','subjects':len(rows),'summary':[{k:r[k] for k in ['subject','radius_above_3m_frames','sensitivity_all_mm']} for r in rows]}))

if __name__=='__main__':main()
