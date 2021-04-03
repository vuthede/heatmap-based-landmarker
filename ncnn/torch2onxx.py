import sys
import os
import argparse
from torch.autograd import Variable
import torch
import onnxsim

sys.path.insert(0, '..')
from models.heatmapmodel import HeatMapLandmarker, HeatMapLandmarkerInference
from models.mobilenetv2faster import PFLDInference

def init_weights(m):
    try:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    except:
        print(f'this layer :{m} can not init weights')
    

# plfd_backbone = PFLD_Ultralight(width_factor=1, input_size=112, landmark_number=68)
# plfd_backbone = HeatMapLandmarkerInference(pretrained=True, model_url="https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1")
# plfd_backbone = HeatMapLandmarkerInference(alpha=0.5, use_author_mobv2=False)
plfd_backbone = HeatMapLandmarkerInference(alpha=0.5, use_author_mobv2=True, use_mobile_v3=False)
# plfd_backbone = HeatMapLandmarkerInference(alpha=1, use_author_mobv2=True, use_mobile_v3=False)

# plfd_backbone = HeatMapLandmarker()


# plfd_backbone.apply(init_weights)
# plfd_backbone  = PFLDInference(alpha=0.25)
# print(plfd_backbone)
# plfd_backbone(torch.randn(1, 3, 256, 256))

# checkpoint_path = "../ckpt/epoch_80.pth.tar"   # Aungment lighting/ fix translation/no histequal  --more epoch

# checkpoint_path = "../192/epoch_142.pth.tar"   # Aungment lighting/ fix translation/no histequal  --more epoch
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])


# params = list(plfd_backbone.parameters())
# for i in range(len(params)):
#     params[i].data = torch.round(params[i].data*10**4) / 10**4
# print("Loaded PFLD model")


# dummy_input = Variable(torch.randn(1, 3, 192, 192)) 
dummy_input = Variable(torch.randn(1, 3, 256, 256)) 
input_names = ["input_1"]
output_names = [ "output_1" ]
torch.onnx.export(plfd_backbone, dummy_input, "./model.onnx", verbose=True, input_names=input_names, output_names=output_names)
print("=====> converted pytorch model to onnx...")


import onnx
model = onnx.load("./model.onnx")
onnx.checker.check_model(model)
print("====> checked onnx model Success...")

from onnxsim import simplify
import onnxmltools
model_simp, check = simplify(model)
print(check)
onnxmltools.utils.save_model(model_simp, 'model_sim.onnx')