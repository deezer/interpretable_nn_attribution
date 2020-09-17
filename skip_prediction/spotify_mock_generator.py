'''
Mock generators for debugging purposes.
Logically, I should delete them for a public paper repository, but if someone
tries to adapt this code for another project, this may be very useful.
'''

import numpy as np
import random

def mock_gen(bs = 100, tile_num = 1):
    '''
    Mock feature to check model faster.
    -> to adapt to your debug needs

    * bs: batch size
    * tile_num: duplicate output
    '''
    while True:
        xid = np.random.randint(0, 2, (bs, 10, 1))   # random song ids - given half session
        xid2 = np.random.randint(0, 2, (bs, 10, 1)) + 2   # random song ids - half session to predict
        xdata = np.random.random((bs, 10, 44)).astype('float32') - 0.5  # random log data - given
        xdata2 = np.tile(
            np.linspace(0, 1, 10).reshape((1, 10, 1)),
            (bs, 1, 2),
            ).astype('float32')  # random reduced log data with just the session position and length
        xdata[:,:,-2:] = xdata2 - 1.1  # positional encoding
        xdata2[:,:,1] = 1.

        ydata = np.mean(xdata[:,:,:44], axis=2, keepdims=True) > 0
        ydata[:,3:6,0] = xid[:,7:,0] == 0  # needs attention
        ydata[:,6:,0] = xid2[:,6:,0] == 2
        ydata[:,-1:,0] = xid2[:,:1,0] == 3
        ydata = 2. * ydata.astype('float32') - 1
        if random.random() > 0.5:  # mock variable session lengths
            ydata[:,-1] = -2.
            if random.random() > 0.5:
                ydata[:,-2] = -2.
        if tile_num > 1:
            ydata = np.tile(ydata, (1, 1, tile_num))
        yield (xid, xdata, xid2, xdata2), ydata
