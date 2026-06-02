import cv2

import numpy as np

import vart

import xir


MODEL_PATH = "../compile/compiled_model/enhancement_net.xmodel"

IMAGE_PATH = "../../test_images/sample.png"


graph = xir.Graph.deserialize(

    MODEL_PATH

)

subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

dpu_subgraph = subgraphs[0]

runner = vart.Runner.create_runner(

    dpu_subgraph,

    "run"

)

input_tensor = runner.get_input_tensors()[0]

output_tensor = runner.get_output_tensors()[0]

input_shape = tuple(

    input_tensor.dims

)

output_shape = tuple(

    output_tensor.dims

)

image = cv2.imread(

    IMAGE_PATH

)

image = cv2.resize(

    image,

    (256, 256)

)

image = image.astype(

    np.float32

) / 255.0

image = np.transpose(

    image,

    (2, 0, 1)

)

image = np.expand_dims(

    image,

    axis=0

)

input_data = [image.astype(np.float32)]

output_data = [

    np.empty(

        output_shape,

        dtype=np.float32

    )

]

job_id = runner.execute_async(

    input_data,

    output_data

)

runner.wait(job_id)

output = output_data[0][0]

output = np.transpose(

    output,

    (1, 2, 0)

)

output = np.clip(

    output * 255.0,

    0,
    255

).astype(np.uint8)

cv2.imwrite(

    "fpga_output.png",

    output

)

print(

    "\nFPGA Inference Completed Successfully\n"

)