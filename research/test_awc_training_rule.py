import unittest
import torch
from fusion_models import FusionModel
from awc_training_rule import AWCConcat,empirical_fisher,shapley_two,pearson_sum,encoder_parameters,awc_regularizer

class AWCRuleChecks(unittest.TestCase):
    def inputs(self,n=3):
        torch.manual_seed(1)
        return (torch.randn(n,51),torch.randn(n,196,5),torch.ones(n,196,dtype=torch.bool),torch.ones(n,17))

    def test_model_exact_same_as_concat_including_missing_routes(self):
        torch.manual_seed(7);a=FusionModel('concat');b=AWCConcat();b.load_state_dict(a.state_dict())
        x=list(self.inputs());x[2][1]=False;x[3][2]=0
        for key in ('fused','video','radar'):
            torch.testing.assert_close(a(*x)[key],b(*x)[key],atol=0,rtol=0)

    def test_fisher_is_mean_of_individual_squared_gradients(self):
        m=AWCConcat();x=self.inputs();y=torch.randn(3,17,3)
        actual=empirical_fisher(m,x,y,batch_size=2)
        expected={n:torch.zeros_like(v) for n,v in encoder_parameters(m).items()}
        for i in range(3):
            m.zero_grad(set_to_none=True)
            (m(*(v[i:i+1] for v in x))['fused']-y[i:i+1]).norm(dim=-1).mean().backward()
            for n,p in encoder_parameters(m).items():expected[n]+=p.grad.square()/3
        for n in actual:torch.testing.assert_close(actual[n],expected[n],atol=1e-10,rtol=1e-4)

    def test_shapley_efficiency_for_two_modalities(self):
        m=AWCConcat();x=self.inputs(5);y=torch.randn(5,17,3)
        scores=shapley_two(m,x,y)
        vf=m.rgb_encoder(x[0]);_,rf=m.point_encoder(x[1],x[2])
        full=m.concat_head(torch.cat((vf,rf),-1)).reshape(-1,17,3)
        zero=m.concat_head(torch.zeros(5,256)).reshape(-1,17,3)
        torch.testing.assert_close(scores.sum(),pearson_sum(y,full)-pearson_sum(y,zero),atol=1e-6,rtol=1e-5)

    def test_regularizer_zero_at_epoch_anchor(self):
        m=AWCConcat();anchor={n:p.detach().clone() for n,p in encoder_parameters(m).items()}
        f={n:torch.ones_like(v) for n,v in anchor.items()}
        self.assertEqual(awc_regularizer(m,anchor,f,torch.tensor([1.,2.])).item(),0.)

if __name__=='__main__':unittest.main()
