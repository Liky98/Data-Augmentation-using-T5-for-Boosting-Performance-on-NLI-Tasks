import time
import uuid
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import os
import gc
import random
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm
import argparse
import datetime
from model import *
"""
__init__ 부분에서 모델 인자값 추가 및 수정 
그아래부터는 건들거 없음.

Model forward function input값 > (feats_dict, layer_feats_dict, label_emb) 

"""

class kihoon():
    def __init__(self):
        parser = argparse.ArgumentParser(description='모델만 넣으면 뚞딲')

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        For network structure
        모델 구조 관련 인자값. 여기만 수정하여 모델에 넣기
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""
        parser.add_argument("--hidden", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.5,
                            help="dropout on activation")
        parser.add_argument("--n-layers-1", type=int, default=2,
                            help="number of layers of feature projection")
        parser.add_argument("--n-layers-2", type=int, default=2,
                            help="number of layers of the downstream task")
        parser.add_argument("--n-layers-3", type=int, default=4,
                            help="number of layers of residual label connection")
        parser.add_argument("--input-drop", type=float, default=0.1,
                            help="input dropout of input features")
        parser.add_argument("--att-drop", type=float, default=0.,
                            help="attention dropout of model")
        parser.add_argument("--label-drop", type=float, default=0.,
                            help="label feature dropout of model")
        parser.add_argument("--residual", action='store_true', default=False,
                            help="whether to connect the input features")
        parser.add_argument("--act", type=str, default='relu',
                            help="the activation function of the model")
        parser.add_argument("--bns", action='store_true', default=False,
                            help="whether to process the input features")
        # parser.add_argument("--norm", action='store_true', default=False,
        #                     help="whether to norm the input features")
        parser.add_argument("--label-bns", action='store_true', default=False,
                            help="whether to process the input label features")

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        ## For environment costruction
        ## 환경세팅 인자값
        parser.add_argument("--seed", type=int, default=0,
                            help="the seed used in the training")
        parser.add_argument("--dataset", type=str, default="ogbn-mag")
        parser.add_argument("--gpu", type=int, default=0)
        parser.add_argument("--cpu", action='store_true', default=False)
        parser.add_argument("--root", type=str, default='../dataset/')
        parser.add_argument("--emb_path", type=str, default='../dataset/')
        parser.add_argument("--stages", nargs='+', type=int, default=[5, 5],  # default 300,300 으로 되어있었음
                            help="The epoch setting for each stage.")

        ## For pre-processing
        ## 데이터 프로세싱
        parser.add_argument("--extra-embedding", type=str, default='',
                            help="whether to use extra embeddings from RotatE")
        parser.add_argument("--embed-size", type=int, default=256,
                            help="inital embedding size of nodes with no attributes")
        parser.add_argument("--num-hops", type=int, default=2,
                            help="number of hops for propagation of raw labels")
        parser.add_argument("--label-feats", action='store_true', default=False,
                            help="whether to use the label propagated features")
        parser.add_argument("--num-label-hops", type=int, default=2,
                            help="number of hops for propagation of raw features")

        ## for training
        ## 학습 인자값
        parser.add_argument("--amp", action='store_true', default=False,
                            help="whether to amp to accelerate training with float16(half) calculation")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0)
        parser.add_argument("--eval-every", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=10000)
        parser.add_argument("--patience", type=int, default=100,
                            help="early stop of times of the experiment")
        # parser.add_argument("--alpha", type=float, default=0.5,
        #                     help="initial residual parameter for the model")
        parser.add_argument("--threshold", type=float, default=0.75,
                            help="the threshold of multi-stage learning, confident nodes "
                                 + "whose score above this threshold would be added into the training set")
        parser.add_argument("--gama", type=float, default=0.5,
                            help="parameter for the KL loss")
        # parser.add_argument("--use-emb", type=str)
        # parser.add_argument("--use-rlu", action='store_true', default=False,
        #                     help="whether to use the reliable data distillation")
        parser.add_argument("--start-stage", type=int, default=0)
        # parser.add_argument("--pre-dropout", action='store_true', default=False,
        #                     help="whether to process the input features")
        # parser.add_argument("--involve-val-labels", action='store_true', default=False,
        #                     help="whether to process the input features")
        # parser.add_argument("--involve-extra-labels", action='store_true', default=False,
        #                     help="whether to process the input features")
        parser.add_argument("--reload", type=str, default='')
        # parser.add_argument("--split-val-run", type=int, default=-1)
        parser.add_argument("--moving-k", type=int, default=10)
        parser.add_argument("--store-model", action='store_true', default=False,
                            help="whether to save model per epoch per stage. WARNING: it costs lots of disk")
        # parser.add_argument("--focal", action='store_true', default=False,
        #                     help="whether to use FocalLoss")

        self.args = parser.parse_args()
        assert self.args.dataset.startswith('ogbn')
        print(self.args)

    def set_random_seed(self,seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_n_params(self,model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1

            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def get_ogb_evaluator(self,dataset):
        evaluator = Evaluator(name=dataset)
        return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

    def hg_propagate(self, new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
        for hop in range(1, max_hops):
            reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
            for etype in new_g.etypes:
                stype, _, dtype = new_g.to_canonical_etype(etype)

                for k in list(new_g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                                or (hop > num_hops and k not in reserve_heads):
                            continue
                        if echo: print(k, etype, current_dst_name)
                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

            # remove no-use items
            for ntype in new_g.ntypes:
                if ntype == tgt_type: continue
                removes = []
                for k in new_g.nodes[ntype].data.keys():
                    if len(k) <= hop:
                        removes.append(k)
                for k in removes:
                    new_g.nodes[ntype].data.pop(k)
                if echo and len(removes): print('remove', removes)
            gc.collect()

            if echo: print(f'-- hop={hop} ---')
            for ntype in new_g.ntypes:
                for k, v in new_g.nodes[ntype].data.items():
                    if echo: print(f'{ntype} {k} {v.shape}')
            if echo: print(f'------\n')

        return new_g

    def clear_hg(self, new_g, echo=False):
        if echo: print('Remove keys left after propagation')
        for ntype in new_g.ntypes:
            keys = list(new_g.nodes[ntype].data.keys())
            if len(keys):
                if echo: print(ntype, keys)
                for k in keys:
                    new_g.nodes[ntype].data.pop(k)
        return new_g

    def check_acc(self, preds_dict, condition, init_labels, train_nid, val_nid, test_nid):
        mask_train, mask_val, mask_test = [], [], []
        remove_label_keys = []
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)

        for k, v in preds_dict.items():
            pred = v.argmax(1)

            a, b, c = pred[train_nid] == init_labels[train_nid], \
                      pred[val_nid] == init_labels[val_nid], \
                      pred[test_nid] == init_labels[test_nid]
            ra, rb, rc = a.sum() / len(train_nid), b.sum() / len(val_nid), c.sum() / len(test_nid)

            vv = torch.log((v / (v.sum(1, keepdim=True) + 1e-6)).clamp(1e-6, 1 - 1e-6))
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])

            if condition(ra, rb, rc, k):
                mask_train.append(a)
                mask_val.append(b)
                mask_test.append(c)
            else:
                remove_label_keys.append(k)
            print(k, ra, rb, rc, la, lb, lc, (ra / rb - 1) * 100, (ra / rc - 1) * 100, (1 - la / lb) * 100,
                  (1 - la / lc) * 100)

        print(set(list(preds_dict.keys())) - set(remove_label_keys))
        print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / len(train_nid))
        print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / len(val_nid))
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / len(test_nid))
        return remove_label_keys

    def train(self, model, train_loader, loss_fcn, optimizer, evaluator, device,
              feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
        args = self.args

        model.train()
        total_loss = 0
        iter_num = 0
        y_true, y_pred = [], []

        for batch in train_loader:
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
            # if mask is not None:
            #     batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            # else:
            #     batch_mask = None
            batch_label_emb = label_emb[batch].to(device)
            batch_y = labels_cuda[batch]

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                    if isinstance(loss_fcn, nn.BCELoss):
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
        loss = total_loss / iter_num
        acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        return loss, acc

    def train_multi_stage(self,model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
                          feats, label_feats, labels, label_emb, predict_prob, gama, scalar=None):
        model.train()
        loss_fcn = nn.CrossEntropyLoss()
        y_true, y_pred = [], []
        total_loss = 0
        loss_l1, loss_l2 = 0., 0.
        iter_num = 0
        for idx_1, idx_2 in zip(train_loader, enhance_loader):
            idx = torch.cat((idx_1, idx_2), dim=0)
            L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
            L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

            batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
            batch_label_emb = label_emb[idx].to(device)
            y = labels[idx_1].to(torch.long).to(device)
            extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
            extra_weight = extra_weight.to(device)
            extra_y = extra_y.to(device)

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                    L1 = loss_fcn(output_att[:len(idx_1)], y)
                    L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                    L2 = (L2 * extra_weight).sum() / len(idx_2)
                    loss_train = L1_ratio * L1 + gama * L2_ratio * L2
                scalar.scale(loss_train).backward()
                scalar.step(optimizer)
                scalar.update()
            else:
                output_att = model(batch_feats, batch_labels_feats, batch_label_emb)
                L1 = loss_fcn(output_att[:len(idx_1)], y)
                L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                # teacher_soft = predict_prob[idx_2].to(device)
                # teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
                # L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(output_att[len(idx_1):], dim=1)))).sum(1).mean()*(len(idx_2)*1.0/(len(idx_1)+len(idx_2)))
                # loss = L1 + L3*gama
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
                loss_train.backward()
                optimizer.step()

            y_true.append(labels[idx_1].to(torch.long))
            y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
            total_loss += loss_train.item()
            loss_l1 += L1.item()
            loss_l2 += L2.item()
            iter_num += 1

        print(loss_l1 / iter_num, loss_l2 / iter_num)
        loss = total_loss / iter_num
        approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        return loss, approx_acc

    @torch.no_grad()
    def gen_output_torch(self, model, feats, label_feats, label_emb, test_loader, device):
        model.eval()
        preds = []
        for batch in tqdm(test_loader):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
            batch_label_emb = label_emb[batch].to(device)
            preds.append(model(batch_feats, batch_labels_feats, batch_label_emb).cpu())
        preds = torch.cat(preds, dim=0)
        return preds

    def load_mag(self, args, symmetric=True):
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
        evaluator = self.get_ogb_evaluator(args.dataset)

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

        etypes = [  # src->tgt
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

        return new_g, init_labels, new_g.num_nodes('P'), n_classes, train_nid, val_nid, test_nid, evaluator

    def load_dataset(self, args):
        if args.dataset == 'ogbn-mag':
            # train/val/test 629571/64879/41939
            return self.load_mag(args)
        else:
            assert 0, 'Only allowed [ogbn-products, ogbn-proteins, ogbn-arxiv, ogbn-papers100M, ogbn-mag]'

    def load_homo(self, args):
        dataset = DglNodePropPredDataset(name=args.dataset, root=args.root)
        splitted_idx = dataset.get_idx_split()

        g, init_labels = dataset[0]
        splitted_idx = dataset.get_idx_split()
        train_nid = splitted_idx['train']
        val_nid = splitted_idx['valid']
        test_nid = splitted_idx['test']

        # features = g.ndata['feat'].float()
        init_labels = init_labels.squeeze()
        n_classes = dataset.num_classes
        evaluator = self.get_ogb_evaluator(args.dataset)

        diag_name = f'{args.dataset}_diag.pt'
        if not os.path.exists(diag_name):
            src, dst, eid = g._graph.edges(0)
            m = SparseTensor(row=dst, col=src, sparse_sizes=(g.num_nodes(), g.num_nodes()))

        return g, init_labels, g.num_nodes(), n_classes, train_nid, val_nid, test_nid, evaluator

    def forward(self, model):
        args = self.args
        if args.seed > 0:
            self.set_random_seed(args.seed)
        g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator = self.load_dataset(args)

        # =======
        # rearange node idx (for feats & labels)
        # =======
        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(test_nid)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

        if total_num_nodes < num_nodes:
            flag = torch.ones(num_nodes, dtype=bool)
            flag[train_nid] = 0
            flag[val_nid] = 0
            flag[test_nid] = 0
            extra_nid = torch.where(flag)[0]
            print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
        else:
            extra_nid = torch.tensor([], dtype=torch.long)

        init2sort = torch.cat([train_nid, val_nid, test_nid, extra_nid])
        sort2init = torch.argsort(init2sort)
        assert torch.all(init_labels[init2sort][sort2init] == init_labels)
        labels = init_labels[init2sort]

        # embed_size = None
        # embed_train = True

        # =======
        # features propagate alongside the metapath
        # =======
        prop_tic = datetime.datetime.now()

        if args.dataset == 'ogbn-mag': # multi-node-types & multi-edge-types
            tgt_type = 'P'

            extra_metapath = [] # ['AIAP', 'PAIAP']
            extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

            print(f'Current num hops = {args.num_hops}')
            if len(extra_metapath):
                max_hops = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
            else:
                max_hops = args.num_hops + 1

            # compute k-hop feature
            g = self.hg_propagate(g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False)

            feats = {}
            keys = list(g.nodes[tgt_type].data.keys())
            print(f'Involved feat keys {keys}')
            for k in keys:
                feats[k] = g.nodes[tgt_type].data.pop(k)

            g = self.clear_hg(g, echo=False)
        else:
            assert 0

        feats = {k: v[init2sort] for k, v in feats.items()}

        prop_toc = datetime.datetime.now()
        print(f'Time used for feat prop {prop_toc - prop_tic}')
        gc.collect()

        # label_feats = {k: v[init2sort] for k, v in label_feats.items()}
        # label_emb = label_emb[init2sort]

        # if args.embedding:
        #     embed_size = {k: v.size(-1) for k, v in feats.items()}

        # train_loader = torch.utils.data.DataLoader(
        #     torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
        # eval_loader = full_loader = []
        all_loader = torch.utils.data.DataLoader(
            torch.arange(num_nodes), batch_size=args.batch_size, shuffle=False, drop_last=False)

        checkpt_folder = f'./output/{args.dataset}/'
        if not os.path.exists(checkpt_folder):
            os.makedirs(checkpt_folder)

        if args.amp:
            scalar = torch.cuda.amp.GradScaler()
        else:
            scalar = None

        device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
        labels_cuda = labels.long().to(device)
        for run_time in range(10):
            checkpt_file = checkpt_folder + uuid.uuid4().hex
            print(checkpt_file)
            for stage in range(args.start_stage, len(args.stages)):
                epochs = args.stages[stage]

                if len(args.reload):
                    pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
                    assert os.path.exists(pt_path)
                    print(f'Reload raw_preds from {pt_path}', flush=True)
                    raw_preds = torch.load(pt_path, map_location='cpu')

                # =======
                # Expand training set & train loader
                # =======
                if stage > 0:
                    preds = raw_preds.argmax(dim=-1)
                    predict_prob = raw_preds.softmax(dim=1)

                    train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
                    val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
                    test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

                    print(f'Stage {stage-1} history model:\n\t' \
                        + f'Train acc {train_acc*100:.4f} Val acc {val_acc*100:.4f} Test acc {test_acc*100:.4f}')

                    confident_mask = predict_prob.max(1)[0] > args.threshold
                    val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
                    test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
                    val_enhance_nid     = val_enhance_offset + trainval_point
                    test_enhance_nid    = test_enhance_offset + valtest_point
                    enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

                    print(f'Stage: {stage}, threshold {args.threshold}, confident nodes: {len(enhance_nid)} / {total_num_nodes - trainval_point}')
                    val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
                    print(f'\t\t val confident nodes: {len(val_enhance_nid)} / {valid_node_nums},  val confident level: {val_confident_level}')
                    test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
                    print(f'\t\ttest confident nodes: {len(test_enhance_nid)} / {test_node_nums}, test confident_level: {test_confident_level}')

                    del train_loader
                    train_batch_size = int(args.batch_size * len(train_nid) / (len(enhance_nid) + len(train_nid)))
                    train_loader = torch.utils.data.DataLoader(
                        torch.arange(train_node_nums), batch_size=train_batch_size, shuffle=True, drop_last=False)
                    enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_nid)))
                    enhance_loader = torch.utils.data.DataLoader(
                        enhance_nid, batch_size=enhance_batch_size, shuffle=True, drop_last=False)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

                # =======
                # labels propagate alongside the metapath
                # =======
                label_feats = {}
                if args.label_feats:
                    if stage > 0:
                        label_onehot = predict_prob[sort2init].clone()
                    else:
                        label_onehot = torch.zeros((num_nodes, n_classes))
                    label_onehot[train_nid] = F.one_hot(init_labels[train_nid], n_classes).float()

                    if args.dataset == 'ogbn-mag':
                        g.nodes['P'].data['P'] = label_onehot

                        extra_metapath = [] # ['PAIAP']
                        extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]

                        print(f'Current num label hops = {args.num_label_hops}')
                        if len(extra_metapath):
                            max_hops = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
                        else:
                            max_hops = args.num_label_hops + 1

                        g = self.hg_propagate(g, tgt_type, args.num_label_hops, max_hops, extra_metapath, echo=False)

                        keys = list(g.nodes[tgt_type].data.keys())
                        print(f'Involved label keys {keys}')
                        for k in keys:
                            if k == tgt_type: continue
                            label_feats[k] = g.nodes[tgt_type].data.pop(k)
                        g = self.clear_hg(g, echo=False)

                        # label_feats = remove_self_effect_on_label_feats(label_feats, label_onehot)
                        for k in ['PPP', 'PAP', 'PFP', 'PPPP', 'PAPP', 'PPAP', 'PFPP', 'PPFP']:
                            if k in label_feats:
                                diag = torch.load(f'{args.dataset}_{k}_diag.pt')
                                label_feats[k] = label_feats[k] - diag.unsqueeze(-1) * label_onehot
                                assert torch.all(label_feats[k] > -1e-6)
                                print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())

                        condition = lambda ra,rb,rc,k: True
                        self.check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid)

                        label_emb = (label_feats['PPP'] + label_feats['PAP'] + label_feats['PP'] + label_feats['PFP']) / 4
                        self.check_acc({'label_emb': label_emb}, condition, init_labels, train_nid, val_nid, test_nid)
                else:
                    label_emb = torch.zeros((num_nodes, n_classes))

                label_feats = {k: v[init2sort] for k, v in label_feats.items()}
                label_emb = label_emb[init2sort]

                if stage == 0:
                    label_feats = {}

                # =======
                # Eval loader
                # =======
                if stage > 0:
                    del eval_loader
                eval_loader = []
                for batch_idx in range((num_nodes-trainval_point-1) // args.batch_size + 1):
                    batch_start = batch_idx * args.batch_size + trainval_point
                    batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + trainval_point)

                    batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
                    batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
                    batch_labels_emb = label_emb[batch_start:batch_end]
                    eval_loader.append((batch_feats, batch_label_feats, batch_labels_emb))

                # =======
                # Construct network
                # =======
                model = model(args.dataset,
                                   args.embed_size, args.hidden, n_classes,
                                   len(feats), len(label_feats), tgt_type,
                                   dropout=args.dropout,
                                   input_drop=args.input_drop,
                                   att_drop=args.att_drop,
                                   label_drop=args.label_drop,
                                   n_layers_1=args.n_layers_1,
                                   n_layers_2=args.n_layers_2,
                                   n_layers_3=args.n_layers_3,
                                   act=args.act,
                                   residual=args.residual,
                                   bns=args.bns, label_bns=args.label_bns,
                                   # label_residual=stage > 0,
                                   )

                model = model.to(device)
                if stage == args.start_stage:
                    print(model)
                    print("# Params:", self.get_n_params(model))

                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                            weight_decay=args.weight_decay)

                best_epoch = 0
                best_val_acc = 0
                best_test_acc = 0
                count = 0

                for epoch in range(epochs):
                    gc.collect()
                    torch.cuda.empty_cache()
                    start = time.time()
                    if stage == 0:
                        loss, acc = self.train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, scalar=scalar)
                    else:
                        loss, acc = self.train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, predict_prob, args.gama, scalar=scalar)
                    end = time.time()

                    log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100)
                    torch.cuda.empty_cache()

                    if epoch % args.eval_every == 0:
                        with torch.no_grad():
                            model.eval()
                            raw_preds = []

                            start = time.time()
                            for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
                                batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                                batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                                batch_labels_emb = batch_labels_emb.to(device)
                                raw_preds.append(model(batch_feats, batch_label_feats, batch_labels_emb).cpu())
                            raw_preds = torch.cat(raw_preds, dim=0)

                            loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
                            loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

                            preds = raw_preds.argmax(dim=-1)
                            val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
                            test_acc = evaluator(preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

                            end = time.time()
                            log += f'Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
                            log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)

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
                        log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100)
                    print(log, flush=True)

                print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))
                print()

                model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
                raw_preds = self.gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)
                torch.save(raw_preds, checkpt_file+f'_{stage}.pt')

if __name__ == '__main__':

    """아래처럼 선언하고 모델 넣기 """
    trainer = kihoon()
    model = SeHGNN_mag
    trainer.forward(model)


