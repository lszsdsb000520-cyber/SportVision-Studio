"""Export the upstream TT3D table segmenter, without Lightning at runtime."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT=Path(__file__).resolve().parents[1]


class TableModel(torch.nn.Module):
    def __init__(self,model,mean,std):
        super().__init__();self.model=model
        self.register_buffer('mean',mean);self.register_buffer('std',std)

    def forward(self,images):return self.model((images-self.mean)/self.std)


def main():
    torch.set_num_threads(4)
    checkpoint=torch.load(ROOT/'external/tt3d/weights/table_segmentation.ckpt',map_location='cpu',weights_only=True)
    model=smp.create_model('unetplusplus',encoder_name='timm-regnety_002',in_channels=3,classes=1,encoder_weights=None)
    model.load_state_dict({k[6:]:v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')},strict=True)
    wrapped=TableModel(model,checkpoint['state_dict']['mean'],checkpoint['state_dict']['std']).eval()
    cap=cv2.VideoCapture(str(ROOT/'sample/demo.mp4'));ok,frame=cap.read();cap.release();assert ok
    rgb=cv2.cvtColor(cv2.resize(frame,(640,320)),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]
    output=ROOT/'reports/v2';output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for factor,name in ((1.,'raw'),(1/255.,'unit')):
            mask=torch.sigmoid(wrapped(x*factor))[0,0].numpy()
            canvas=cv2.resize(frame,(960,540));m=cv2.resize(mask,(960,540))>.8
            canvas[m]=(canvas[m]*.5+np.array([0,190,80])*.5).astype(np.uint8)
            cv2.imwrite(str(output/f'table_{name}.jpg'),canvas)
            print(name,'mask_fraction',float((mask>.8).mean()),flush=True)
        reference=wrapped(x).numpy()
        torch.onnx.export(wrapped,x,str(ROOT/'models/table.onnx'),input_names=['image'],output_names=['mask'],
                          opset_version=17,dynamo=False)
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(ROOT/'models/table.onnx'),opts,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'image':x.numpy()})[0]
    error=float(np.max(np.abs(actual-reference)))
    if not np.allclose(actual,reference,atol=1e-3,rtol=1e-3):raise RuntimeError(f'Export mismatch {error}')
    report=dict(source='https://github.com/cogsys-tuebingen/tt3d',checkpoint='weights/table_segmentation.ckpt',
                architecture='UnetPlusPlus, timm-regnety_002',input='RGB float 0..255, 1x3x320x640; upstream preprocessing',
                max_abs_error=error,segmentation_models_pytorch=smp.__version__)
    (ROOT/'models/table.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
