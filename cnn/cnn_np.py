import numpy as np

def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

def conv_single_step(a_slice_prev, W, b):
    z = np.multiply(a_slice_prev, W) + b
    return np.sum(z)

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3) # H * W * C
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

print(conv_single_step(a_slice_prev, W, b))



def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1
    
    A_prev_pad = zero_pad(A_prev, pad)

    Z = np.zeros((m, n_H, n_W, n_C))


    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    Z[i, h, w, c] = conv_single_step(A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :], W[:, :, :, c], b[:, :, :, c])

    return Z

np.random.seed(1)
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2, "stride": 1}

Z = conv_forward(A_prev, W, b, hparameters)

print("Z's mean:", np.mean(Z))



def pool_forward(A_prev, hparameters, mode='max'):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    stride = hparameters['stride']
    f = hparameters['f']

    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)

    Z = np.zeros((m, n_H, n_W, n_C_prev))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C_prev):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    if mode == 'max':
                        Z[i, h, w, c] = np.max(A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c])
                    else:
                        Z[i, h, w, c] = np.average(A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c])

    return Z


np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride": 1, "f": 4}
A = pool_forward(A_prev, hparameters, 'max')
print('mode= max')
print("A=", A)


A = pool_forward(A_prev, hparameters, 'avg')
print('mode= avg')
print("A=", A)







    

