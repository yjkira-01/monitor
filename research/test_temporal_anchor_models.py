"""Mechanism checks: real boundaries, observation masking and motion transport."""
import unittest
import numpy as np
import torch
from temporal_anchor_models import context_indices,AnchorTransport,TemporalControl,CONTEXT

def batch():
    b,t=2,CONTEXT
    return {'vf':torch.zeros(b,t,128),'rf':torch.zeros(b,t,128),
        'base':torch.zeros(b,t,17,3),'video':torch.zeros(b,t,17,3),'radar':torch.zeros(b,t,17,3),
        'confidence':torch.zeros(b,t,17),'vv':torch.zeros(b,t,dtype=torch.bool),
        'rv':torch.ones(b,t,dtype=torch.bool),'dt':torch.tensor([[0.]+[.1]*(t-1)]*b),
        'history_valid':torch.ones(b,t,dtype=torch.bool)}

class MechanismChecks(unittest.TestCase):
    def test_context_never_crosses_subject_window_gap_or_future(self):
        s=np.array([1,1,1,1,2,2]);w=np.array([0,0,0,1,1,1]);f=np.array([5,6,8,9,10,11])
        idx,valid=context_indices(s,w,f)
        for i in range(len(f)):
            self.assertTrue(np.all(idx[i,valid[i]]<=i))
            self.assertTrue(np.all(s[idx[i,valid[i]]]==s[i]))
            self.assertTrue(np.all(w[idx[i,valid[i]]]==w[i]))
        self.assertEqual(valid[2].sum(),1);self.assertEqual(valid[3].sum(),1);self.assertEqual(valid[4].sum(),1)

    def test_no_anchor_exact_base_even_with_large_network_outputs(self):
        x=batch();x['base'].normal_();m=AnchorTransport('anchor_transport')
        with torch.no_grad():m.motion[-1].bias.fill_(1000)
        out=m(x)
        torch.testing.assert_close(out['prediction'],x['base'][:,-1],atol=0,rtol=0)
        self.assertFalse(out['eligible'].any())

    def test_visible_current_joint_is_protected(self):
        x=batch();x['confidence'][:,-1]=.9;x['base'].normal_();m=AnchorTransport('anchor_transport')
        torch.testing.assert_close(m(x)['prediction'],x['base'][:,-1],atol=0,rtol=0)

    def test_known_constant_velocity_transports_individual_anchor(self):
        x=batch();x['confidence'][:,5]=.9;x['video'][:,5]=2.
        m=AnchorTransport('anchor_transport')
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m.motion[-1].bias.fill_(1.)
        out=m(x)
        torch.testing.assert_close(out['transported'],torch.full((2,17,3),2.3),atol=1e-6,rtol=0)
        torch.testing.assert_close(out['prediction'],torch.full((2,17,3),1.15),atol=1e-6,rtol=0)

    def test_missing_radar_cannot_invent_motion(self):
        x=batch();x['confidence'][:,5]=.9;x['video'][:,5]=2.;x['rv'].zero_()
        m=AnchorTransport('anchor_transport')
        with torch.no_grad():m.motion[-1].bias.fill_(999.)
        out=m(x);self.assertFalse(out['pair_valid'].any())
        torch.testing.assert_close(out['transported'],torch.full((2,17,3),2.),atol=0,rtol=0)

    def test_all_models_have_finite_backward_on_partial_history(self):
        for method in ['gru','temporal_mlp','anchor_hold','anchor_transport','anchor_no_motion_loss']:
            from temporal_anchor_models import make_model
            x=batch();x['confidence'][:,3]=.8;x['history_valid'][:,:2]=False
            m=make_model(method);loss=(m(x)['prediction']-1).square().mean();loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),method)

if __name__=='__main__':unittest.main()
