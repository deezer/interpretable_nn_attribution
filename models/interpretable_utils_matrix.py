'''
All of the above function allow to create interpretable models using mask
constraints. See interpretable_utils_list for a Layer list approach.
What is at stake is to compute the right matrices for connection, then
upsample them correctly to match the Dense layers of our model.


    ## How to computes custom connection matrix for Dense layers.

    M @ v_{k-1} = v_k

    Explanation for the connection numbers:

    We have an graph's adjancy matrix where (i, .) contain the connections (i,j).
    Then, for each node, the connection can be read from left to right as binary
    numbers. Example :

    i:    0 1 2 3
          | |\|/
    j:    0 1 2 3

    Adjancy matrix :

    [ 1 0 0 0        -> 0001 -> 1
    0 1 0 0        -> 0010 -> 2
    0 1 1 1        -> 1110 -> 6
    0 0 0 0 ]      -> 0000 -> 0

    Why even bother ?

    Because then we want to preserve the characteristic of those specialised nodes
    on the higher order nodes. Let's define:

    i:    0 1 2 3
          |\|\|/|     a more complex input graph
    j:    0 1 2 3

    How to we connect subsequent layers ?

    first solution:

          | | | |     to preserve the caracteristic of nodes, we connect to
    j+1:  0 1 2 3     similar nodes from previous layers. Ok.

    However, as 1 use data from 0, we may also connect to the higher order 0
    that only use data from 0. As for 2, we can also connect it to 3, but not to 1
    as 1 use data from 0 and 2 is not supposed to use data from 0.

    better solution:

          |\| |/|     to preserve the caracteristic of nodes, we connect to
    j+1:  0 1 2 3     similar nodes from previous layers. Ok.


    This is easier to modify without having to compute the transition matrix.
'''

import numpy as np


# --- Masked weights utilities

def generate_tr(S):
    '''
    Returns a boolean list of inclusions (slow method but easier to follow)
    * S: Boolean[][]
    '''
    assert len(S) > 0
    H = len(S)
    N = len(S[0])
    tr_mat = [[False for h in range(H)] for i in range(H)]

    for i in range(H):
        for j in range(H):
            is_included = True
            for k in range(N):
                if S[i][k] and not S[j][k]:
                    is_included = False
                    break
            tr_mat[i][j] = is_included
    return tr_mat


def generate_connection_graph(node_range, node_connect_type):
    '''
    Given a input graph (node_range, node_connect_type), build
    the input connection and transition matrices.

    Uses the binary trick described in the top of the file.

    * node_range: Tuple(int, int)[]
    * node_connect_type: int[]
    '''
    node_count = len(node_range)
    tr_count = len(node_connect_type)
    A_input = np.zeros((node_count, tr_count))   # X_i -> S_i
    A_tr = np.zeros((tr_count, tr_count))        # S_i -> S_j

    for i in range(node_count):
        for j in range(tr_count):
            if 1 << i & node_connect_type[j] > 0:  # input must be connected
                A_input[i, j] = 1

    for i in range(tr_count):
        for j in range(tr_count):
            if ~node_connect_type[i] & node_connect_type[j] == 0:  # data flow
                A_tr[j, i] = 1

    return A_input, A_tr


def expand_inmat(M, a, bl, bl_max):
    '''
    Use the template matrix and the corresponding input ranges
    to create the right connection matrix with the input size.

    * M: int[][]  -- template matrix
    * a: int  -- input layer feature multiplicity
    * bl: <Tuple(int, int) OR Tuple(int, int)[]>[]  -- output layer feature multiplicity list
    * bl_max: int  -- number of input features
    '''
    M_exp = np.zeros((bl_max, M.shape[1] * a))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 1:
                if type(bl[i]) == tuple:
                    k, l = bl[i]
                    M_exp[k:l, (j * a):((j + 1) * a)] = 1
                elif type(bl[i]) == list:
                    for bl_r in bl[i]:
                        k, l = bl_r
                        M_exp[k:l, (j * a):((j + 1) * a)] = 1
    return M_exp


def expand_trmat(M, a, b):
    '''
    Use template matrix and expand it to match intermediate layers matrices.

    * M: template matrix
    * a: input layer feature multiplicity
    * b: output layer feature multiplicity

    eg.  [ 1 1              [ 1 1 1 1 1 1
           0 1 ], 2, 3  =>    1 1 1 1 1 1
                              0 0 0 1 1 1
                              0 0 0 1 1 1 ]
    '''
    M_exp = np.zeros((M.shape[0] * a, M.shape[1] * b))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 1:
                M_exp[(i * a):((i + 1) * a), (j * b):((j + 1) * b)] = 1
    return M_exp
