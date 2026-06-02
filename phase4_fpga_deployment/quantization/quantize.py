import os

import torch

from pytorch_nndct.apis import torch_quantizer

from phase2_extension.models.enhancement_net import EnhancementNet


DEVICE = torch.device("cpu")

MODEL_PATH = "../../outputs/checkpoints/latest_model.pth"

QUANT_MODE = "calib"

INPUT_SHAPE = (1, 3, 256, 256)


model = EnhancementNet(

    channels=48,

    num_rrdb_blocks=8

)

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)

model.load_state_dict(

    checkpoint["model_state_dict"]

)

model.eval()

dummy_input = torch.randn(

    INPUT_SHAPE

)

quantizer = torch_quantizer(

    quant_mode=QUANT_MODE,

    module=model,

    input_args=(dummy_input,)

)

quantized_model = quantizer.quant_model

with torch.no_grad():

    quantized_model(dummy_input)

quantizer.export_quant_config()

quantizer.export_xmodel(

    deploy_check=False

)

print(

    "\nQuantization Completed Successfully\n"

)