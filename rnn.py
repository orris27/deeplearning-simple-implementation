import numpy as np

#def softmax(input):
#    down = np.sum([np.exp(i) for i in input])
#    return np.array([np.exp(i) / down for i in input])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_next = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, a_prev) + ba)
    yt_pred = softmax(np.matmul(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

np.random.seed(1)
xt = np.random.randn(3, 10) # (input_size, batch_size)
a_prev = np.random.randn(5, 10) # (rnn_size, batch_size)
Waa = np.random.randn(5, 5) # (rnn_size, rnn_size)
Wax = np.random.randn(5, 3) # (rnn_size, input_size)
Wya = np.random.randn(2, 5) # (num_classes, rnn_size)
ba = np.random.randn(5, 1) # (rnn_size, 1)
by = np.random.randn(2, 1) # (num_classes, 1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)


def rnn_forward(x, a0, parameters):
    caches = list()
    n_x, m, T_x = x.shape # (input_size, batch_size, time_steps)
    n_y, n_a = parameters['Wya'].shape # (num_classes, rnn_size)

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0

    for t in range(T_x):
        #def rnn_cell_forward(xt, a_prev, parameters):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, caches
        

np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))


