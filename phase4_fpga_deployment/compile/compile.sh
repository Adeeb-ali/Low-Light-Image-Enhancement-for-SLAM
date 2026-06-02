#!/bin/bash

vai_c_xir \
--xmodel ../quantization/quantize_result/EnhancementNet_int.xmodel \
--arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
--output_dir compiled_model \
--net_name enhancement_net