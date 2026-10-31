import math

import torch
from torch.nn import functional as F

# The following code is adapted from DeepSpeed's helper.py
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/compression/helper.py
def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)

class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # Initialize LoRA parameters
        rows, columns = weight.shape
        self.lora_right_weight = torch.nn.Parameter(torch.zeros((columns, lora_dim)))
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((lora_dim, rows)))
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # Freeze original weight and bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def init_parameters(self):
        # Initialize LoRA parameters
        torch.nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias) + (
                input @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling


def convert_linear_layer_to_lora(model, include_module_names, exclude_module_names, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) and 
            any(include_name in name for include_name in include_module_names) and 
            not any(exclude_name in name for exclude_name in exclude_module_names)):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # Turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def get_lora_state_dict(model):
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k] for k in my_state_dict if 'lora_right_weight' in k or 'lora_left_weight' in k}