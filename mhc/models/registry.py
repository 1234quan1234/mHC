"""Factory registry for model creation from config."""


MODEL_REGISTRY = {}


def register_model(name: str, cls: type) -> None:
    MODEL_REGISTRY[name] = cls


def build_model(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]()
