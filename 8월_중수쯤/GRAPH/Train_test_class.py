"""
Training
Test
Result Figure
등등 서버에 올려서 받을 Class
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ogb.lsc import MAG240MDataset
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

def confusion(prediction_list, label_list, save_path) : #혼동행렬이랑 기타 성능지표 출력 및 저장
    # 혼동행렬
    my_data = []
    y_pred_list = []
    for data in prediction_list :
        for data2 in data :
            my_data.append(data2.item())
    for data in label_list :
        for data2 in data :
            y_pred_list.append(data2.item())

    confusion_matrix(my_data, y_pred_list)

    confusion_mx = pd.DataFrame(confusion_matrix(y_pred_list, my_data))
    ax =sns.heatmap(confusion_mx, annot=True, fmt='g')
    plt.title('confusion', fontsize=20)
    plt.show()

    print(f"precision : {precision_score(my_data, y_pred_list, average='macro')}")
    print(f"recall : {recall_score(my_data, y_pred_list, average='macro')}")
    print(f"f1 score : {f1_score(my_data, y_pred_list, average='macro')}")
    print(f"accuracy : {accuracy_score(my_data, y_pred_list)}")
    f1_score_detail= classification_report(my_data, y_pred_list,  digits=3)
    print(f1_score_detail)
    plt.savefig(save_path)

"""
tqdm으로 매 iter마다 loss랑 acc 출력 및 진행상황 확인

"""
def train(model, train_loader, evaluator,
          feats, label_feats, labels_cuda, label_emb, loss_fcn = nn.CrossEntropyLoss, mask=None, scalar=None):

    #기본 셋팅
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0)
    start = time.time()

    for epoch in range(10) :
        model.train()
        total_loss = 0
        iter_num = 0
        y_true, y_pred = [], []
        train_loader = tqdm(train_loader, desc='Loading train dataset')

        for i, batch in enumerate(train_loader):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}

            batch_label_emb = label_emb[batch].to(device)
            batch_y = labels_cuda[batch]

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                    if isinstance(loss_fcn, nn.BCELoss): #BCE loss와 같은 type이면 변경
                        output_att = torch.sigmoid(output_att)
                    loss_train = loss_fcn(output_att, batch_y)
                scalar.scale(loss_train).backward()
                scalar.step(optimizer)
                scalar.update()
            else:
                output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                if isinstance(loss_fcn, nn.BCELoss):
                    output_att = torch.sigmoid(output_att)
                L1 = loss_fcn(output_att, batch_y)
                loss_train = L1
                loss_train.backward()
                optimizer.step()

            y_true.append(batch_y.cpu().to(torch.long))
            if isinstance(loss_fcn, nn.BCELoss):
                y_pred.append((output_att.data.cpu() > 0).int())
            else:
                y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            total_loss += loss_train.item()
            iter_num += 1

            train_loader.set_description(
                "Loss %.04f Acc %.04f | step %d Epoch %d" % (loss_train, evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)), i, epoch))

        loss = total_loss / iter_num
        acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        end = time.time()

        print("Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100))


    now = time.localtime()
    print(time.strftime('%Y%m%d %H%M', now))
    confusion(y_pred, y_true, now)
    return loss, acc

def test(model, eval_loader) :
    g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset(args)

    # =======
    # rearange node idx (for feats & labels)
    # =======
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fcn = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        raw_preds = []

        start = time.time()
        for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
            batch_feats = {k: v.to(device) for k, v in batch_feats.items()}
            batch_label_feats = {k: v.to(device) for k, v in batch_label_feats.items()}
            batch_labels_emb = batch_labels_emb.to(device)
            raw_preds.append(model(batch_feats, batch_label_feats, batch_labels_emb).cpu())

        raw_preds = torch.cat(raw_preds, dim=0)

        loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
        loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums + test_node_nums],
                             labels[valtest_point:total_num_nodes]).item()

        preds = raw_preds.argmax(dim=-1)
        val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
        test_acc = evaluator(preds[valid_node_nums:valid_node_nums + test_node_nums],
                             labels[valtest_point:total_num_nodes])

        end = time.time()
        log += f'Time: {end - start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
        log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc * 100, test_acc * 100)

    if val_acc > best_val_acc:
        best_epoch = epoch
        best_val_acc = val_acc
        best_test_acc = test_acc

        torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
        count = 0
    else:
        count = count + args.eval_every
        if count >= args.patience:
            break
    log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc * 100, best_test_acc * 100)
    print(log, flush=True)


    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc * 100, best_test_acc * 100))

    model.load_state_dict(torch.load(checkpt_file + f'_{stage}.pkl'))
    raw_preds = gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)
    torch.save(raw_preds, checkpt_file + f'_{stage}.pt')

def load_dataset():
    g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = load_dataset("ogbn-mag"")

    # =======
    # rearange node idx (for feats & labels)
    # =======

    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

def load_mag(args, symmetric=True):
    dataset = DglNodePropPredDataset(name=args.dataset, root=args.root)
    splitted_idx = dataset.get_idx_split()

    g, init_labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx['train']['paper']
    val_nid = splitted_idx['valid']['paper']
    test_nid = splitted_idx['test']['paper']

    features = g.nodes['paper'].data['feat']
    if len(args.extra_embedding):
        print(f'Use extra embeddings generated with the {args.extra_embedding} method')
        path = os.path.join(args.emb_path, f'{args.extra_embedding}_nars')
        author_emb = torch.load(os.path.join(path, 'author.pt'), map_location=torch.device('cpu')).float()
        topic_emb = torch.load(os.path.join(path, 'field_of_study.pt'), map_location=torch.device('cpu')).float()
        institution_emb = torch.load(os.path.join(path, 'institution.pt'), map_location=torch.device('cpu')).float()
    else:
        author_emb = torch.Tensor(g.num_nodes('author'), args.embed_size).uniform_(-0.5, 0.5)
        topic_emb = torch.Tensor(g.num_nodes('field_of_study'), args.embed_size).uniform_(-0.5, 0.5)
        institution_emb = torch.Tensor(g.num_nodes('institution'), args.embed_size).uniform_(-0.5, 0.5)

    with torch.no_grad():
        if features.size(1) != args.embed_size:
            rand_weight = torch.Tensor(features.size(1), args.embed_size).uniform_(-0.5, 0.5)
            features = features @ rand_weight
        if author_emb.size(1) != args.embed_size:
            rand_weight = torch.Tensor(author_emb.size(1), args.embed_size).uniform_(-0.5, 0.5)
            author_emb = author_emb @ rand_weight
        if topic_emb.size(1) != args.embed_size:
            rand_weight = torch.Tensor(topic_emb.size(1), args.embed_size).uniform_(-0.5, 0.5)
            topic_emb = topic_emb @ rand_weight
        if institution_emb.size(1) != args.embed_size:
            rand_weight = torch.Tensor(institution_emb.size(1), args.embed_size).uniform_(-0.5, 0.5)
            institution_emb = institution_emb @ rand_weight

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb

    init_labels = init_labels['paper'].squeeze()
    n_classes = int(init_labels.max()) + 1
    evaluator = get_ogb_evaluator(args.dataset)

    # for k in g.ntypes:
    #     print(k, g.ndata['feat'][k].shape)
    for k in g.ntypes:
        print(k, g.nodes[k].data['feat'].shape)

    adjs = []
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)

    # F --- *P --- A --- I
    # paper : [736389, 128]
    # author: [1134649, 256]
    # institution [8740, 256]
    # field_of_study [59965, 256]

    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('A', 'A-I', 'I'),
        ('A', 'A-P', 'P'),
        ('P', 'P-P', 'P'),
        ('P', 'P-F', 'F'),
    ]

    if symmetric:
        adjs[2] = adjs[2].to_symmetric()
        assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)

    new_g = dgl.heterograph(new_edges)
    new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
    new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
    new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
    new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']

    IA, PA, PP, FP = adjs

    diag_name = f'{args.dataset}_PFP_diag.pt'
    if not os.path.exists(diag_name):
        PF = FP.t()
        PFP_diag = sparse_tools.spspmm_diag_sym_ABA(PF)
        torch.save(PFP_diag, diag_name)

    if symmetric:
        diag_name = f'{args.dataset}_PPP_diag.pt'
        if not os.path.exists(diag_name):
            # PP = PP.to_symmetric()
            # assert torch.all(PP.get_diag() == 0)
            PPP_diag = sparse_tools.spspmm_diag_sym_AAA(PP)
            torch.save(PPP_diag, diag_name)
    else:
        assert False

    diag_name = f'{args.dataset}_PAP_diag.pt'
    if not os.path.exists(diag_name):
        PAP_diag = sparse_tools.spspmm_diag_sym_ABA(PA)
        torch.save(PAP_diag, diag_name)

    return new_g, init_labels, new_g.num_nodes('P'), n_classes, train_nid, val_nid, test_nid, evaluator
def main() :
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)
