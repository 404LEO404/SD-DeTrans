import argparse
import os
import random
import sys
import time
import scipy.sparse as sp
# import scipy as sp
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from load_data_qkl import load_subgraph_data_fixed, load_blockchain_data
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from model import HTKD
from load_data_qkl import load_blockchain_data
from utils.pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('utils/')

def nkd(student_logits, teacher_logits, temperature=1.0):

    student_pred = F.softmax(student_logits / temperature, dim=1)
    teacher_pred = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_pred.log(), teacher_pred, reduction='batchmean')
    return loss

def evaluate_valid( pred, label):
    label = label.cpu().numpy()

    micro = f1_score(label, pred, average='micro')
    macro = f1_score(label, pred, average='macro')
    result = {
        'micro-f1': micro,
        'macro-f1': macro
    }
    return result

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model(args):

    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')


    feats_type = args.feats_type
    if args.dataset == 'qkl':
        adjM, features_list, labels, num_classes, train_idx, val_idx, test_idx, pos = load_blockchain_data(args.dataset)
    else:
        features_list, edge_index_sub, adjM, labels, train_idx, val_idx, test_idx = load_subgraph_data_fixed(args.dataset)

    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # features_list = [mat2tensor(features).to(device)
    #                  for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    print(node_cnt)
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)

    labels = torch.LongTensor(labels).to(device)
    # train_idx = train_val_test_idx['train_idx']
    # train_idx = np.sort(train_idx)
    # val_idx = train_val_test_idx['val_idx']
    # val_idx = np.sort(val_idx)
    # test_idx = train_val_test_idx['test_idx']
    # test_idx = np.sort(test_idx)

    # adjM数据类型 <3999x3999 sparse matrix of type '<class 'numpy.float64'>

    # g = dgl.remove_self_loop(g)
    if args.dataset =='qkl':
        all_nodes = np.arange(0, 3999)
        node_seq = torch.zeros(3999, args.len_seq).long()
        g = dgl.DGLGraph(adjM)
        g = dgl.add_self_loop(g)
    else:
        all_nodes = np.arange(0, 5000)
        node_seq = torch.zeros(5000, args.len_seq).long()
        adjM_sp = sp.coo_matrix(adjM.numpy())
        g = dgl.from_scipy(adjM_sp)
        g = dgl.add_self_loop(g)

    n = 0

    for x in all_nodes:

        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        while (cnt < args.len_seq):
            sample_list = g.successors(start).numpy().tolist()
            nsampled = max(len(sample_list), 1)
            sampled_list = random.sample(sample_list, nsampled)
            for i in range(nsampled):
                node_seq[n, cnt] = sampled_list[i]
                cnt += 1
                if cnt == args.len_seq:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()
        n += 1


    print(node_seq.shape)


    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]

    g = g.to(device)
    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    if args.dataset == 'ACM':
        num_classes = 3
    if args.dataset == 'DBLP':
        num_classes = 4
    if args.dataset == 'YELP':
        num_classes = 4
    if args.dataset == 'IMDB':
        num_classes = 4
    if args.dataset == 'Aminer':
        num_classes = 4
    if args.dataset == 'qkl':
        num_classes = 2
    if args.dataset == 'CLUSTER':
        num_classes = 6
    if args.dataset == 'PATTERN':
        num_classes = 2
    type_emb = torch.eye(len(node_cnt)).to(device)
    node_type = torch.tensor(node_type).to(device)

    for i in range(args.repeat):


        net = HTKD(g, num_classes, in_dims, args.hidden_dim, args.num_layers, args.num_gnns, args.num_heads,
                       args.dropout,
                       temper=args.temperature, num_type=len(node_cnt), beta=args.beta)

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/TKD_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device))
        for i in range(len(features_list)):
            if isinstance(features_list[i], np.ndarray):
                features_list[i] = torch.from_numpy(features_list[i]).to(device)
            else:
                features_list[i] = features_list[i].to(device)
            features_list[i] = features_list[i].float()
        for epoch in range(args.epoch):

            t_start = time.time()
            # training
            net.train()

            teacher_logits = net(features_list, train_seq, type_emb, node_type, teacher_mode=True,
                                 norm=args.l2norm)  # 教师输出

            student_logits = net(features_list, train_seq, type_emb, node_type, teacher_mode=False,
                                 norm=args.l2norm)  # 学生输出
            train_loss = F.nll_loss(F.log_softmax(student_logits, dim=1), labels[train_idx])

            # 蒸馏损失
            nkd_loss = nkd(student_logits, teacher_logits, temperature=1.2)

            t_loss = train_loss + 0.05 * nkd_loss
            # t_loss = train_loss

            # autograd
            optimizer.zero_grad() 
            t_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | t_loss: {:.4f} | Time: {:.4f}'.format(
                epoch, t_loss.item(), t_end-t_start))

            t_start = time.time()

            # validation
            net.eval()
            with torch.no_grad():

                # 使用教师模型进行验证
                teacher_logits = net(features_list, val_seq, type_emb, node_type, teacher_mode=True, norm=args.l2norm)
                logp = F.log_softmax(teacher_logits, dim=1)
                val_loss = F.nll_loss(logp, labels[val_idx])

                # 使用softmax输出进行预测
                pred = teacher_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
    
            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/TKD_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.device)))
        net.eval()

        with torch.no_grad():
            logits = net(features_list, test_seq, type_emb, node_type, args.l2norm)
            h = logits.cpu().numpy()
            test_logits = logits

            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            pred = onehot[pred]
            pred = torch.tensor(pred)
            pred = torch.argmax(pred, dim=1)
            result = evaluate_valid(pred, labels[test_idx])
            print(result)
            # micro_f1[i] = result['micro-f1']
            # macro_f1[i] = result['macro-f1']
    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(h)

    # 获取真实标签
    true_labels = labels[test_idx].cpu().numpy()

    # 计算NMI和ARI
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)

    print(f"聚类评估结果: NMI = {nmi:.4f}, ARI = {ari:.4f}")



    # Y = labels[test_idx].cpu().numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(h)
    # # node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
    # color_idx = {}
    # # for i in range(len(h[test_idx].detach().cpu().numpy())):
    # for i in range(len(h)):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    # plt.legend()
    # plt.show()
    # # 使用 K-Means 聚类（假设有 4 个类，根据你的数据调整 n_clusters）
    # kmeans = KMeans(n_clusters=num_classes, random_state=42)
    # pred_labels = kmeans.fit_predict(h)  # 预测聚类标签
    # # 计算 ARI 和 NMI
    # ari = adjusted_rand_score(Y, pred_labels)
    # nmi = normalized_mutual_info_score(Y, pred_labels)
    # print(f"ARI: {ari:.4f}")
    # print(f"NMI: {nmi:.4f}")

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='HTKDformer')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=128,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--dataset', type=str, default='PATTERN', help='DBLP, IMDB, ACM, qkl, PATTERN')
    ap.add_argument('--num-heads', type=int, default=2,
                    help='Number of the attention heads. Default is 2.')
    ap.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=50, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
    ap.add_argument('--num-gnns', type=int, default=4, help='The number of layers of both structural and heterogeneous encoder')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=3, help='The length of node sequence.')
    ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
    ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
    ap.add_argument('--temperature', type=float, default=1, help='Temperature of attention score')
    ap.add_argument('--beta', type=float, default=1, help='Weight of heterogeneity-level attention score')


    args = ap.parse_args()
    run_model(args)
