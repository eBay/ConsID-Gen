import torch
from typing import Dict, Optional


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")


class GeneralCheckpointLoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    @staticmethod
    def _has_lora_keys(state_dict) -> bool:
        return any(".lora_A." in k or ".lora_B." in k for k in state_dict)

    @staticmethod
    def _module_checkpoint_prefixes(module, prefix):
        aliases = tuple(getattr(module, "_checkpoint_prefix_aliases", ()))
        return (prefix, *aliases)

    @staticmethod
    def _remap_module_state_dict_keys(module, sub_dict):
        key_aliases = getattr(module, "_checkpoint_key_aliases", None)
        if not key_aliases:
            return sub_dict

        remapped = {}
        for key, val in sub_dict.items():
            mapped_key = key
            for old_prefix, new_prefix in key_aliases.items():
                if mapped_key.startswith(old_prefix):
                    mapped_key = new_prefix + mapped_key[len(old_prefix):]
                    break
            remapped[mapped_key] = val
        return remapped

    def load(
        self,
        dit_module: torch.nn.Module,
        state_dict,
        module_mappings: Optional[Dict[str, torch.nn.Module]] = None,
        strict: bool = False,
    ):
        module_mappings = module_mappings or {}

        if self._has_lora_keys(state_dict):
            raise ValueError("[CheckpointLoader] Detected LoRA keys in checkpoint. Use GeneralLoRALoader for adapter merge, or export/load a full-finetune checkpoint.")

        # Load full/finetuned DiT weights.
        dit_state_dict = {}
        for key, val in state_dict.items():
            if any(key.startswith(prefix) for prefix in module_mappings):
                continue
            if key.startswith("pipe.dit."):
                key = key[len("pipe.dit."):]
            elif key.startswith("dit."):
                key = key[len("dit."):]
            elif key.startswith("pipe."):
                # Skip other pipeline-scoped modules not explicitly mapped.
                continue
            dit_state_dict[key] = val

        if dit_state_dict:
            missing, unexpected = dit_module.load_state_dict(dit_state_dict, strict=strict)
            print(f"[Checkpoint][DiT] Loaded tensors (tensors={len(dit_state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})")

        # Load extra modules by explicit prefix mapping.
        for prefix, module in module_mappings.items():
            if module is None:
                continue
            sub_dict = {}
            for candidate_prefix in self._module_checkpoint_prefixes(module, prefix):
                for key, val in state_dict.items():
                    if key.startswith(candidate_prefix):
                        sub_dict[key[len(candidate_prefix):]] = val
            sub_dict = self._remap_module_state_dict_keys(module, sub_dict)
            if sub_dict:
                missing, unexpected = module.load_state_dict(sub_dict, strict=strict)
                print(f"[Checkpoint][Module] Loaded into {module.__class__.__name__} (prefix={prefix}, tensors={len(sub_dict)}, missing={len(missing)}, unexpected={len(unexpected)})")
        
