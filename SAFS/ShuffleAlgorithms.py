import numpy as np


def cross_shuffle(d_features, n_replica):
    '''
        Arguments:
            d_features {Int} -- the dimension of the features
            depth {Int} -- the depth of the expected output

        Returns:
            indexes {list} -- replicated and shuffled indexes, length = d_features*depth

    '''
    index = np.arange(0, d_features).tolist()

    if d_features % 2 == 1:
        dim_odd = 1
    else:
        dim_odd = 0
    return odd_even_shuffle(index, n_replica, dim_odd)


def odd_even_shuffle(index, depth, dim_odd=1):
    indexes = [] + index

    def odd_step(index):
        odds = index[0:-1:2]
        evens = index[1:-1:2]
        last = [index[-1]]
        return evens + odds + last

    def even_step(index):
        first = [index[0]]
        evens = index[1::2]
        odds = index[2::2]
        if dim_odd:
            return first + odds + evens
        else:
            return first + evens + odds

    while (depth > 1):
        if depth % 2 == 1:
            index = odd_step(index)
        else:
            index = even_step(index)

        indexes += index

        depth -= 1

    return indexes

