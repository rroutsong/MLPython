import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def wave_data(n=100):
    """
    Taken from https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/datasets.py#L22

    To better annotate the data generation of these wave datasets
    """
    # Seed a Mersenne Twister pseudo-random number generator
    rnd = np.random.RandomState(42)

    # Make x a uniform distribution between [-3, 3) (include -3, exclude 3)
    #   - any value between the interval is equally likely to be given
    x = rnd.uniform(-3, 3, size=n)

    # Make y a diagnal parabolic sin distribution
    #   - https://www.wolframalpha.com/input/?i=sin%284x%29%2Bx
    y_no_noise = (np.sin(4 * x) + x)
    # add some noise
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    # x size: (n,) -> (n, 1)
    return x.reshape(-1, 1), y


