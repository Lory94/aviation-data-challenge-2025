from .load_object_in_file import load_object_in_file


def load_objects_from_config(config):

    result = {}

    for object_role, object_payload in config.items():

        if not isinstance(object_payload, str):
            object = object_payload

        else:
            try:
                object = load_object_in_file(
                    file_path=f"../{object_role}/__init__.py", 
                    namespace=object_role, 
                    object_name=object_payload,
                )
            except FileNotFoundError as exception:
                print(exception.args)
                1/0

        result[object_role] = object

    return result
