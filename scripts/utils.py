import random

import numpy as np


def batch_split(data_size, batch_size):
    """Returns a list of batches with sample indexes for a training epoch."""
    original_idxs = range(data_size)

    batch_list = []
    if batch_size <= 0 or batch_size >= data_size:
        batch_list = list(range(data_size))
    else:
        num_batches = int(np.ceil(data_size / batch_size))

        for i in range(num_batches):
            if len(original_idxs) > batch_size:
                batch_idxs = random.sample(original_idxs, batch_size)
                original_idxs = list(set(original_idxs) - set(batch_idxs))
            else:
                batch_idxs = original_idxs
            batch_list.append(batch_idxs)

    return batch_list
