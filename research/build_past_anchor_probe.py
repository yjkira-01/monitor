"""Check if limiting correction to detector-missing joints hides opportunity."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'research/probe_anchor_transport_opportunity.py').read_text()
source=source.replace("'artifacts/anchor_opportunity_20261016'","'artifacts/past_anchor_opportunity_20261016'")
source=source.replace("            last=np.where(visible,np.arange(9)[None,:,None],-1).max(1)",
                      "            visible[:,-1]=False\n            last=np.where(visible,np.arange(9)[None,:,None],-1).max(1)")
source=source.replace("eligible=(last>=0)&(conf<.3)","eligible=(last>=0)")
source=source.replace('Compare last observed joint holding', 'Compare strictly past observed joint holding, also allowing correction of current detections,')
destination=ROOT/'research/probe_past_anchor_opportunity_20261016.py'
compile(source,str(destination),'exec');destination.write_text(source,encoding='utf-8')
