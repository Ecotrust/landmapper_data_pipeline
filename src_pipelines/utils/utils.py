# %%
import os
import sys
import yaml
from dotenv import dotenv_values

import numpy as np


# %%
class AttrDict(dict):
    """Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Stolen from: https://stackoverflow.com/a/48806603/1913361
    """

    def __init__(self, mapping=None):
        super(AttrDict, self).__init__()

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


class ConfigLoader:
    def __init__(self, root, config_file="config.yaml"):
        self.config = AttrDict({})
        self.root = root
        self.config_file = config_file

    def load_from_env(self):
        env = dotenv_values(os.path.join(self.root, ".env"))
        for key in env:
            self.config[key] = env[key]

    def load_from_file(self):
        with open(os.path.join(self.root, self.config_file), "r") as f:
            config = yaml.safe_load(f)
            self.config.update(config)

    def load(self):
        self.load_from_env()
        self.load_from_file()
        return self.config

    def get(self, key):
        return self.config.get(key, None)

# %%
def multithreaded_execution(function, parameters, threads=20):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import inspect
    import tqdm

    # Get parameter list
    fun_params = inspect.signature(function).parameters.keys()
    fun_progressbar_param = True if "progressbar" in fun_params else False

    n_items = len(parameters)
    assert n_items > 0, "Empty list of parameters passed."
    print("\n", "Processing {:,d} images".format(n_items))

    with tqdm.tqdm(
        total=n_items, bar_format="{l_bar}{bar:75}{r_bar}{bar:-50b}", file=sys.stdout
    ) as pbar:
        if fun_progressbar_param:
            _ = [p.update({"progressbar": pbar}) for p in parameters]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(function, **param) for param in parameters]
            results = []

            try:  # catch exceptions
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

            except Exception as e:
                print(f'Exception "{e}" raised while processing files.')
                raise e

        return np.array(results)
