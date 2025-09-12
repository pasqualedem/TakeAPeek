

from peft import AdaLoraConfig, LoHaConfig, LoKrConfig, LoraConfig, XLoraConfig
from peft import get_peft_model as get_peft_model_from_peft


class DummyConfig:
    """
    A dummy class to represent a configuration.
    """
    def __init__(self, **kwargs):
        """
        Initialize the DummyConfig with given parameters.
        """
        self.params = kwargs
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.target_modules = kwargs.get("target_modules", "")


class FullRank(DummyConfig):
    """
    A class to represent a full Rank configuration.
    """

    def __repr__(self):
        return f"FullModelConfig({self.params})"
    
        
def fix_peft_config(params):
    """
    Fix the PEFT config parameters for LoHa.
    """
    if "lora_alpha" in params:
        params["alpha"] = params["lora_alpha"]
        params.pop("lora_alpha")

    if "lora_dropout" in params:
        params.pop("lora_dropout")

    if "bias" in params:
        params.pop("bias")
    return params


def get_peft_config(peft_type, params):
    """
    Get the PEFT config based on the peft_type.
    """
    if peft_type not in PEFT_CONFIGS:
        raise ValueError(f"Unsupported peft_type: {peft_type}")
    
    if peft_type in {"loha", "lokr"}:
        params = fix_peft_config(params)

    config_class = PEFT_CONFIGS[peft_type]
    return config_class(**params)


def get_peft_model(model, config):
    """
    Get the PEFT model based on the peft_type.
    """
    
    if isinstance(config, FullRank):
        
        target_modules = config.target_modules
        target_modules_names = []
        
        if target_modules == "decoder":
            print("Fine-tuning decoder only")
            parameters = model.decoder_params()
        else:
            print(f"Fine-tuning {target_modules} only")
            parameters = model.named_parameters()

        for name, param in parameters:
            if any(target_module in name for target_module in target_modules):
                target_modules_names.append(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Set requires_grad=True for the feature extractor

        setattr(model, "targeted_module_names", target_modules_names)
        return model
    
    return get_peft_model_from_peft(model, config)


PEFT_CONFIGS = {
    "lora": LoraConfig,
    "xlora": XLoraConfig,
    "adalora": AdaLoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
    "full": FullRank,
}