from . import opt
from . import bloom

MODEL_REGISTRY = {
    'opt': opt.OPT,
    'bloom': bloom.BLOOM
}


def get_model(model_name):
    if 'opt' in model_name:
        return MODEL_REGISTRY['opt']
    elif 'bloom' in model_name:
        return MODEL_REGISTRY['bloom']
    return MODEL_REGISTRY[model_name]
