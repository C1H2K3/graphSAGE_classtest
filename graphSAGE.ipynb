import numpy as np
import copy

# 随机生成6个节点的特征向量, 这里就设置成256维
X_feature_0 = np.random.standard_normal((6, 256))
X_feature_0

# 6个节点分别是A,B,C,D,E,F

# 这个是个集合X, X的key值为A,B,C,D,E,F5个字母, value值对应为0,1,2,3,4,5
X = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

W_1 = np.random.standard_normal((256, 64))
B_1 = np.random.standard_normal((256, 64))
W_2 = np.random.standard_normal((64, 64))
B_2 = np.random.standard_normal((64, 64))

A_nebor = [X['B'], X['C'], X['D']]
B_nebor = [X['A'], X['C']]
C_nebor = [X['A'], X['B'], X['E'], X['F']]
D_nebor = [X['A']]
E_nebor = [X['C'], X['F']]
F_nebor = [X['C'], X['E']]
neighbor = [A_nebor,B_nebor,C_nebor,D_nebor,E_nebor,F_nebor]

# pooling 参数
W_pool = [np.random.rand(256, 256), np.random.rand(64, 64)]


# LSTM参数
initial_state = [np.zeros(256), np.zeros(64)]
i_w = [np.random.standard_normal((256,256)), np.random.standard_normal((64,64))]
i_b = [np.zeros(256),np.zeros(64)]
f_w = [np.random.standard_normal((256,256)), np.random.standard_normal((64,64))]
f_b = [np.zeros(256),np.zeros(64)]
o_w = [np.random.standard_normal((256,256)), np.random.standard_normal((64,64))]
o_b = [np.zeros(256),np.zeros(64)]
g_w = [np.random.standard_normal((256,256)), np.random.standard_normal((64,64))]
g_b = [np.zeros(256),np.zeros(64)]



# 获取邻居节点的表示向量
def get_nebor_emb(node_index,X_feature):
    nebor_emb = []
    nebor_index = neighbor[node_index]
    for i in range(len(nebor_index)):
        nebor_emb.append(X_feature[nebor_index[i]])
    return np.array(nebor_emb)


# 聚合函数
import math
def sigmoid_function(z):
    ls = []
    for i in range(z.shape[0]):
        ls.append(1/(1+math.exp(-z[i])))
    return np.array(ls)

# AGG函数
def LSTM_AGG(nebor_emb, layer):
    np.random.shuffle(nebor_emb)
    #LSTM AGG
    state = copy.deepcopy(initial_state[layer])
    for i in range(nebor_emb.shape[0]):
        forget_gate = sigmoid_function(nebor_emb[i]@f_w[layer] + f_b[layer])
        input_gate = sigmoid_function(nebor_emb[i]@i_w[layer] + i_b[layer])
        output_gate = sigmoid_function(nebor_emb[i]@o_w[layer] + o_b[layer])
        new_information = nebor_emb[i]@g_w[layer] + g_b[layer]
        state = state*forget_gate + new_information*input_gate
        output = sigmoid_function(state)*output_gate
    return output

def AVG_AGG(nebor_emb):
    sum = 0
    for i in range(nebor_emb.shape[0]):
        sum+=nebor_emb[i]
    return sum/nebor_emb.shape[0]

def pooling(nebor_emb, layer):
    nebor_emb = nebor_emb@W_pool[layer]
    return np.max(nebor_emb_, axis=0)


#用LSTM函数聚合

#A节点
nebor_emb = get_nebor_emb(X['A'],X_feature_0)
X_feature_1A = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1+X_feature_0[X['A']]@B_1)
#B节点
nebor_emb = get_nebor_emb(X['B'],X_feature_0)
X_feature_1B = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1+X_feature_0[X['B']]@B_1)
#C节点
nebor_emb = get_nebor_emb(X['C'],X_feature_0)
X_feature_1C = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1+X_feature_0[X['C']]@B_1)
#D节点
nebor_emb = get_nebor_emb(X['D'],X_feature_0)
X_feature_1D = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1+X_feature_0[X['D']]@B_1)


print("Layer-1 B节点表示向量: \n",X_feature_1B)
print("Layer-1 C节点表示向量: \n",X_feature_1C)
print("Layer-1 D节点表示向量: \n",X_feature_1D)

#Layer2
# A节点的h^2_a
nebor_emb = np.array([X_feature_1B,X_feature_1C,X_feature_1D])
X_feature_2A = sigmoid_function(LSTM_AGG(nebor_emb,1)@W_2 + X_feature_1A@B_2)

print("Layer-2 A节点表示向量h^2(a): \n",X_feature_2A)

# 增加节点G
G_nebor = [X['D']]
neighbor.append(G_nebor)
X['G'] = 6
X_feature_0=np.vstack((X_feature_0,np.random.standard_normal(256)))

# 聚合步骤 A->D->G
# layer1
# D节点
nebor_emb = get_nebor_emb(X['D'],X_feature_0)
X_feature_1D = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1 + X_feature_0[X['D']]@B_1)

# G节点
nebor_emb = get_nebor_emb(X['G'],X_feature_0)
X_feature_1G = sigmoid_function(LSTM_AGG(nebor_emb,0)@W_1 + X_feature_0[X['G']]@B_1)

# layer2
# G节点
nebor_emb = np.array([X_feature_1D])
X_feature_2G = sigmoid_function(LSTM_AGG(nebor_emb,1)@W_2 + X_feature_1G@B_2)

print("Layer-2 G节点表示向量h^2(a): \n",X_feature_2G)
