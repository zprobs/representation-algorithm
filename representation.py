import numpy as np
import pygambit


def build_representation(normal_form: np.ndarray, title="EFG"):

    g = pygambit.Game.new_tree()
    g.title = title

    if normal_form.size == 0:
        return g.write()

    if normal_form.dtype not in ['int64', 'float64', 'i', 'u', 'f']:
        if normal_form.dtype == '0':
            raise ValueError("All specified arrays must have the same shape")
        else:
            raise ValueError("All specified payoffs must be numerical values")

    return g.write()


