import os
from config import make_cfg
from train.trainer import Trainer

cfg = make_cfg()
os.makedirs(cfg.model_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)
trainer = Trainer(cfg)
trainer.run()
