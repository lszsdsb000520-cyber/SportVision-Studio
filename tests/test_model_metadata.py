import json
import numpy as np
from pingpong.detector import heatmap_point
from pingpong.paths import model_threshold


def test_probability_and_logit_models_have_identical_ball_localization():
    logits=np.full((20,30),-12.,np.float32);logits[8:11,13:16]=2.;logits[9,14]=5.
    probabilities=np.clip(1/(1+np.exp(-logits)),.001,.999)
    a=heatmap_point(logits,.5);b=heatmap_point(probabilities,.5,True)
    assert np.allclose(a[:2],b[:2],atol=1e-3)
    assert heatmap_point(np.zeros((20,30),np.float32),.5,True)[0] is None


def test_recommended_threshold_is_optional_and_validated(tmp_path):
    path=tmp_path/'detector.onnx'
    assert model_threshold(path)==.5
    path.with_suffix('.json').write_text(json.dumps({'threshold':.4}))
    assert model_threshold(path)==.4
    path.with_suffix('.json').write_text(json.dumps({'threshold':2}))
    assert model_threshold(path)==.5
    path.with_suffix('.json').write_text('[]')
    assert model_threshold(path)==.5
