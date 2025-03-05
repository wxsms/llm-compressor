import torch
from compressed_tensors.quantization import  QuantizationScheme
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)
from compressed_tensors.utils import update_offload_parameter
from llmcompressor.pytorch.utils import (
    pseudo_quantize_tensor,
)

def dynamic_quantize(
    module: torch.nn.Module, value: torch.Tensor, args: QuantizationScheme, attach_q_params:bool = False
) -> torch.Tensor:

    q_weight, scale, zero_point = pseudo_quantize_tensor(value, args.weights.symmetric, args.weights.num_bits, args.weights.group_size)


    if attach_q_params:
        initialize_module_for_quantization(module, args)
        update_offload_parameter(module, "weight_scale", scale)
        update_offload_parameter(module, "weight_zero_point", zero_point)
    
    return q_weight


         
