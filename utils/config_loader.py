# config_loader.py


import os
import yaml

#  Self implement ConfigLoader, benefit: Higher maintainability
#


class ConfigLoader:
    def __init__(self, model_config_path):
        self.model_config = self._load_yaml(model_config_path)
        if self.model_config.get("inherit"):
            print("Loading from the xxx")

        self.config = {
            "model": self.model_config,
            # "training": self.training_config,
            # "inference": self.inference_config
        }

    def _load_yaml(self, path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, section, key=None, default=None):
        if key:
            return self.config.get(section, {}).get(key, default)
        return self.config.get(section, {})


if __name__ == "__main__":
    cfg_loader = ConfigLoader(
        "/Users/chenyujie/Desktop/NLP/llm-training-optimization/modelConfig/deepseekLight.yaml"
    )

    print(cfg_loader.get("model"))
    # print(cfg_loader.get("training"))
    # print(cfg_loader.get("inference"))
    print(cfg_loader.get("model", "hidden_size"))
