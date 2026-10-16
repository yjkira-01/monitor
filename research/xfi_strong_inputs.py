"""X-Fi fusion-core adaptation to the same RTMO/point inputs as strong concat.

The pinned official X_Fusion body is unchanged: 512 channels, 32 tokens per
modality, 8 heads, depth 4 and original five-slot indexing. Modality encoders,
pooled-feature token adapters and exact learned fallback heads are local.
This is explicitly a core adaptation, not the official full architecture.
"""
import torch
from torch import nn
from fusion_models import FusionModel
from xfi_paper_baseline import official_modules

class XFiStrongCore(FusionModel):
    def __init__(self):
        # Same stem and branch-head initialization as the ordinary concat.
        super().__init__('concat')
        del self.concat_head
        self.video_token_projection=nn.Linear(128,512)
        self.radar_token_projection=nn.Linear(128,512)
        self.video_token_positions=nn.Parameter(torch.randn(32,512)*.02)
        self.radar_token_positions=nn.Parameter(torch.randn(32,512)*.02)
        _,_,fusion,self.official_definitions=official_modules()
        self.native_core=fusion['X_Fusion'](5,512,2,512,32,8,64,4)

    def forward(self,rgb,points,mask,confidence):
        vv,rv=(confidence>=.3).any(1),mask.any(1)
        vf=self.rgb_encoder(rgb)*vv[:,None]
        _,rf=self.point_encoder(points,mask)
        video=self.video_head(vf).reshape(-1,17,3)
        radar=self.radar_head(rf).reshape(-1,17,3)
        vt=self.video_token_projection(vf)[:,None]+self.video_token_positions[None]
        rt=self.radar_token_projection(rf)[:,None]+self.radar_token_positions[None]
        # Encode every batch jointly, like concat, then apply the same fallbacks.
        fused=self.native_core(torch.cat((vt,rt),1),[True,False,True,False,False])
        fused=torch.where(~rv[:,None,None],video,fused)
        fused=torch.where(~vv[:,None,None],radar,fused)
        return {'video':video,'radar':radar,'fused':fused,'video_valid':vv,'radar_valid':rv,'both':vv&rv}
