from pathlib import Path
import json,pickle
class ModelRegistry:
 def __init__(self,path='models'): self.path=Path(path); self.path.mkdir(exist_ok=True)
 def save(self,model,metadata,prefix='model'):
  versions=sorted(self.path.glob(f'{prefix}_v*.pkl')); v=len(versions)+1; fn=self.path/f'{prefix}_v{v:03d}.pkl';
  with open(fn,'wb') as f: pickle.dump(model,f)
  fn.with_suffix('.json').write_text(json.dumps(metadata,indent=2,default=str)); return fn
