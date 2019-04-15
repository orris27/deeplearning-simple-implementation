import numpy as np
from utils import *


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
x = np.random.randn(3,10,4) # (input_size, batch_size, time_steps)
a0 = np.random.randn(5,10) # (rnn_size, batch_size)
Waa = np.random.randn(5,5) # (rnn_size, rnn_size)
Wax = np.random.randn(5,3) # (rnn_size, input_size)
Wya = np.random.randn(2,5) # (num_classes, rnn_size)
ba = np.random.randn(5,1) # (rnn_size, 1)
by = np.random.randn(2,1) # (num_classes, 1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.zeros((n_a + n_x, m))
    concat[:n_a, :] = a_prev
    concat[n_a:, :] = xt

    ft = sigmoid(np.matmul(Wf, concat) + bf)
    it = sigmoid(np.matmul(Wi, concat) + bi)
    cct = np.tanh(np.matmul(Wc, concat) + bc)
    ct = ft * c_prev + it * cct
    ot = sigmoid(np.matmul(Wo, concat) + bo)
    ht = ot * np.tanh(ct)

    a_next = ht
    c_next = ct
    
    yt_pred = softmax(np.matmul(Wy, ht) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt ,parameters)
    

    return a_next, c_next, yt_pred, cache

np.random.seed(1)
xt = np.random.randn(3, 10) # (input_size, batch_size)
a_prev = np.random.randn(5, 10) # (lstm_size, batch_size)
c_prev = np.random.randn(5, 10) # (lstm_size, batch_size)
Wf = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bf = np.random.randn(5, 1) # (lstm_size, 1)
Wi = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bi = np.random.randn(5, 1) # (lstm_size, 1)
Wo = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bo = np.random.randn(5, 1) # (lstm_size, 1)
Wc = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bc = np.random.randn(5, 1) # (lstm_size, 1)
Wy = np.random.randn(2, 5) # (num_classes, lstm_size)
by = np.random.randn(2, 1) # (num_classes, 1)


parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))


def lstm_forward(x, a0, parameters):
    caches = list()
    n_x, m, T_x = x.shape # (input_size, batch_size, time_steps)
    n_y, n_a = parameters['Wy'].shape # (num_classes, rnn_size)

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros(a_next.shape)

    for t in range(T_x):
        #def rnn_cell_forward(xt, a_prev, parameters):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, c, caches
        
np.random.seed(1)
x = np.random.randn(3, 10, 7) # (input_size, batch_size, time_steps)
a0 = np.random.randn(5, 10) # (lstm_size, batch_size)
Wf = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bf = np.random.randn(5, 1) # (lstm_size, 1)
Wi = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bi = np.random.randn(5, 1) # (lstm_size, 1)
Wo = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bo = np.random.randn(5, 1) # (lstm_size, 1)
Wc = np.random.randn(5, 3 + 5) # (lstm_size, input_size + lstm_size)
bc = np.random.randn(5, 1) # (lstm_size, 1)
Wy = np.random.randn(2, 5) # (num_classes, lstm_size)
by = np.random.randn(2, 1) # (num_classes, 1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    
    dtanh = (1 - a_next ** 2) * da_next

    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']


    dxt = np.matmul(Wax.T, dtanh)
    dWax = np.matmul(dtanh, xt.T)

    da_prev = np.matmul(Waa.T, dtanh)
    dWaa = np.matmul(dtanh, a_prev.T)

    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

da_next = np.random.randn(5,10)
gradients = rnn_cell_backward(da_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)



def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        #gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
        gradients = rnn_cell_backward(da[:, :, t] +  da_prevt, caches[t])

        dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']

        dx[:, :, t] = dxt
        
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}

    return gradients

np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)
    dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * it * (1 - it)
    dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)

    dWf = np.matmul(dft, np.hstack([a_prev.T, xt.T]))
    dWi = np.matmul(dit, np.hstack([a_prev.T, xt.T]))
    dWc = np.matmul(dcct, np.hstack([a_prev.T, xt.T]))
    dWo = np.matmul(dot, np.hstack([a_prev.T, xt.T]))
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)


    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wc = parameters['Wc']
    bc = parameters['bc']


    da_prev = np.matmul(Wf[:, :n_a].T, dft) + np.matmul(Wi[:, :n_a].T, dit) + np.matmul(Wo[:, :n_a].T, dot) + np.matmul(Wc[:, :n_a].T, dcct)
    dc_prev = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * ft
    dxt = np.matmul(Wf[:, n_a:].T, dft) + np.matmul(Wi[:, n_a:].T, dit) + np.matmul(Wo[:, n_a:].T, dot) + np.matmul(Wc[:, n_a:].T, dcct)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi, "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients

np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

da_next = np.random.randn(5,10)
dc_next = np.random.randn(5,10)
gradients = lstm_cell_backward(da_next, dc_next, cache)
print()
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)



def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))

    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))

    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        #gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi, "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])

        dx[:, :, t] = gradients['dxt']
        
        #dc_prevt = gradients['dc_prev']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']

    da0 = gradients['da_prev']

    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi, "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients



np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print()
print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)


