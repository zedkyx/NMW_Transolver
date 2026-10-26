import importlib


def get_model(args):
    name = args.model
    if name == 'Transolver_2D':
        return importlib.import_module('models.transformer.model.Transolver_Structured_Mesh_2D')
    if name == 'Transolver_3D':
        return importlib.import_module('models.transformer.model.Transolver_Structured_Mesh_3D')
    if name == 'Transolver_1D':
        return importlib.import_module('models.transformer.model.Transolver_Irregular_Mesh')
    if name == 'Transolver_plus':
        return importlib.import_module('models.transformer.model.Transolver_Irregular_Mesh')
    if name == 'SATO':
        return importlib.import_module('models.transformer.model.sato')
    raise ValueError(f'Unknown model name: {name}')


