import importlib


def create(cls):
    """expects a string that can be imported as with a module.class name"""
    module_name, class_name = cls.rsplit(".", 1)

    try:
        print("importing " + module_name)
        somemodule = importlib.import_module(module_name)
        print("getattr " + class_name)
        cls_instance = getattr(somemodule, class_name)
    except Exception as err:
        print(
            "Project factory error: {0} in {1}.{2}".format(err, module_name, class_name)
        )
        exit(-1)

    return cls_instance
