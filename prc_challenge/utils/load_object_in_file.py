import importlib.util
import sys
import os

def load_object_in_file(file_path:str, namespace:str, object_name:str):

    # Temporarily setting cwd as the directory of this very file
    before_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    # Loading the object
    spec = importlib.util.spec_from_file_location(namespace, os.path.abspath(file_path))
    try:
        module = importlib.util.module_from_spec(spec)
    except AttributeError as exception:
        if exception.args[0] == "'NoneType' object has no attribute 'loader'":
            raise Exception("Provided file_path likely doesn't correspond to an actual file.")
    sys.modules[namespace] = module
    spec.loader.exec_module(module)

    # Getting back to the right working directory
    os.chdir(before_cwd)

    return getattr(module, object_name)
