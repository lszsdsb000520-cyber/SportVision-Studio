from pathlib import Path
import torch
import yaml
from .vendor.blurball import BlurBall


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def config_object(value):
    if isinstance(value,dict):
        return Config({k:config_object(v) for k,v in value.items()})
    if isinstance(value,list):
        return [config_object(v) for v in value]
    return value


def load_model(root, weights, device='cpu'):
    config = config_object(yaml.safe_load((Path(root)/'models'/'blurball.yaml').read_text(encoding='utf-8')))
    model = BlurBall(config)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    return model.to(device)


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]
