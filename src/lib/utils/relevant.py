import torch
import numpy
import torch.nn.functional as F

def relevance(X, Y, K=5):
    '''
    X feature map:
    (N, C, H, W)
    Y feature map:
    (N, C, H, W)
    '''
    # X = X.contiguous()
    # Y = Y.contiguous() #TODO 要加么？
    N, C, H, W = X.shape
    rel_shape = (N, K*K, H, W)
    rel = torch.zeros(rel_shape)
    pad_width = K // 2
    Y = F.pad(Y, (pad_width, pad_width, pad_width, pad_width), 'constant', 0)
    print(Y.shape)
    for i in range(H): #TODO 可以转成矩阵运算吗？
        for j in range(W):
            X_ij = X[:, :, i, j].unsqueeze(2).unsqueeze(3) # braod_cast
            X_ij = X_ij.expand(N, C, K, 1).expand(N, C, -1, K) # [4, 64, 5, 5]
            print(X_ij)
            i_index = pad_width + i - pad_width
            j_index = pad_width + j - pad_width
            Y_ij = Y[:, :, i_index:i_index+K, j_index:j_index+K]
            print(Y_ij)
            # print(Y_ij.shape)
            rel_local = (X_ij * Y_ij).sum(1).unsqueeze(1) # (N, 1, K, K)
            print(rel_local)
            # print(rel_local.shape)
            rel[:, :, i, j] = rel_local.view((N, K*K))
    return rel

x = torch.randn((1, 1, 1, 1))
y = torch.randn((1, 1, 1, 1))
rel = relevance(x, y)
print(rel.shape)