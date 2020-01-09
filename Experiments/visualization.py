import numpy as np


def visual_layer_attention(shuffle_indexes, n_replica, kernel, stride, d_features, d_out, attn_value):
    '''
    :param shuffle_indexes: shape [n_replica * d_features]
    :param n_replica: int
    :param kernel: int
    :param stride: int
    :param d_features: int
    :param attn_value: shape [d_out, n_replica * n_candidate]
    :return:
    '''
    shuffle_indexes = shuffle_indexes.reshape(n_replica, d_features)
    attn_value = attn_value.reshape(d_out, n_replica, -1)

    def per_replica(shuffle_index, attn):
        '''The attention of one region on one replica'''
        '''
            Arguments:
                shuffle_index:   shape [(1,) d_features]
                attn:            shape [(1,) n_candidate]
                
        '''

        index = np.arange(d_features)
        # get index to reorder shuffled feature
        restore_index = [z[1] for z in zip(shuffle_index, index)]

        height = kernel
        width = np.ceil(float(d_features) / float(stride))

        attention = np.zeros([height, width * stride - stride + kernel])

        for i in np.arange(attention.shape[0]):
            attention[i, attention[::stride] + i] = attn

        attention = attention.sum(axis=0)[restore_index]  # shape: [(1,) d_features]
        return attention

    def per_region(indexes, attns):
        '''
            Arguments:
                indexes:         shape [n_replica, d_features]
                attn:            shape [n_replica, n_candidate]
        '''
        attention = [per_replica(indexes[i], attns[i]) for i in range(n_replica)]
        attention = np.array(attention).sum(axis=0)
        return attention

    # attention = []
    # for i in range(attn_value.shape[0]):
    attention = np.apply_along_axis(per_region, axis=0, arr=attn_value, indexes=shuffle_indexes)

    return attention
