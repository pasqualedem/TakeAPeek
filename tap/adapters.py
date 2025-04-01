

from peft import AdaLoraConfig, LoHaConfig, LoKrConfig, LoraConfig, XLoraConfig

        
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


PEFT_CONFIGS = {
    "lora": LoraConfig,
    "xlora": XLoraConfig,
    "adalora": AdaLoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
}