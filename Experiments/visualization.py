import numpy as np


def attention_between_2layers(shuffle_indexes, n_replica, kernel, stride, d_features, attn_value):
    '''
        Description:
            Each output features attend to feature combos of the replicas (shuffled origin features), this function
            calculates the attention values of the output features on all origin features.
        Arguments:
            shuffle_indexes {ndarray, shape [n_replica * d_features, ]} -- shuffled and replicated indexes.
            n_replica {Int} -- how many replicates.
            kernel {int} -- kernel size.
            stride {int} -- stride.
            d_features {int} -- feature dimension.
            attn_value {ndarray, shape [d_out, n_replica * n_candidate]} -- attention values, d_out is the dimension of
            the output features, n_candidate represents how many feature combos in each replica.

        return:
            attention {ndarray, shape [d_out, d_features]} -- the attention values of the output features on all origin
            features.
    '''

    shuffle_indexes = shuffle_indexes.reshape(n_replica, d_features)
    n_candidate = np.ceil(np.divide(d_features, stride)).astype(int)

    def per_replica(shuffle_index, attn):
        '''
            Description:
                This function calculates the attention value (of one output feature) on one replica, and map the value
                back to the origin order.

            Arguments:
                shuffle_index {ndarray, shape [d_features, ]} -- shuffled index of one replica.
                attn {ndarray, shape [n_candidate, ]} -- attention on one replica.

            Return:
                attention {ndarray, shape [d_features, ]} -- attention value on each position of the replica.
        '''

        index = np.arange(d_features)
        # get the reverse index of the shuffled indexes.
        reverse_index = [z[1] for z in sorted(zip(shuffle_index, index))]

        height = kernel
        width = n_candidate * stride - stride + kernel
        # empty attention value matrix
        attention = np.zeros([height, width])

        for i in np.arange(attention.shape[0]):
            # fill the attention value to the corresponding location
            # example:
            # 1 _ _ 4 _ _ 7 ... 97 _ _ ,
            # _ 2 _ _ 5 _ _ ... _ 98 _ ,
            # _ _ 3 _ _ 6 _ ... _ _ 99
            max = i+stride * (n_candidate - 1) + 1
            # assign attention values, note that here value should be divided by the kernel size.
            attention[i][i:max:stride] = attn / kernel
        # add up the attention value on each shuffled feature, then reverse to the order of the origin features.
        attention = attention.sum(axis=0)[reverse_index]
        return attention

    def addup_all_replicas(attns, indexes):
        '''
            Description:
                This function sum up the attention values (of one output feature) on each origin feature.

            Arguments:
                indexes {ndarray, shape [n_replica, d_features]} -- shuffled and replicated indexes.
                attn {ndarray, shape [n_replica * n_candidate]} -- attention on all regions(size = kernel) of all
                replicas.

            Return:
                attention {ndarray, shape [d_features, ]} -- attention on all original feature.
        '''
        attns = attns.reshape(n_replica, n_candidate)
        attention = [per_replica(indexes[i], attns[i]) for i in range(n_replica)]
        attention = np.array(attention).sum(axis=0)
        return attention

    attention = np.apply_along_axis(addup_all_replicas, 1, attn_value, shuffle_indexes)
    return attention

def end2end_attention(f_shuffle, attention_list, d_features, n_replica, kernel, stride=None):
    '''
        Description:
            Calculate attention between layers then apply matrix multiplication to get the final attention between
            output features and input features.
        Arguments:
            f_shuffle {function} -- shuffle algorithm.
            attention_list {List} -- attention(s) between layers
            d_features {int} -- origin feature dimension.
            n_replica {int} -- how many replicas.
            kernel {int} -- kernel size.
            stride {int} -- stride on the replicas.

        Return:
            attention_map {ndarray} -- the attention map between the output features and the input features.
    '''
    # Origin feature dimension
    d_in = d_features
    attention_between_layers = []
    for i in range(len(attention_list)):
        attn = attention_list[i]
        attn = np.array(attn.tolist())
        # each self_attention layer's output dimension
        d_out = attn.shape[-2]
        # generate the shuffled index
        shuffled_index = np.array(f_shuffle(d_in, n_replica))
        if stride is None:
            stride = np.ceil(np.divide(n_replica * d_in, attn.shape[-1])).astype(int)
        # calculate the attention between i th layer of features and i+1 th layer of features. (0 means origin features)
        attn = attention_between_2layers(shuffled_index, n_replica=n_replica,
                                         kernel=kernel, stride=stride, d_features=d_in,attn_value=attn)
        # the output dimension of this layer becomes the input dimension of the next layer
        d_in = d_out
        attention_between_layers.append(attn)
    # Apply Matmul to get the final attention map
    attn_latter = attention_between_layers.pop()
    while attention_between_layers:
        attn_former = attention_between_layers.pop()
        attn_latter = np.matmul(attn_latter, attn_former)
    attention_map = attn_latter
    return attention_map
