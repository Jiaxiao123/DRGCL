import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model import Net
from evaluate import evaluate
from data import DrugDataLoader
from utils import *


def train(args, dataset, graph_data, cv):
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_disease = dataset.disease_feature_shape[0]
    args.fdim_drug = dataset.drug_feature_shape[0]

    dis_graph = dataset.disease_graph.to(args.device)
    drug_graph = dataset.drug_graph.to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    args.rating_vals = dataset.possible_rel_values

    # build the model
    model = Net(args=args)
    model = model.to(args.device)
    rel_loss = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)
    print("Loading network finished ...\n")

    # prepare the logger
    test_loss_logger = MetricLogger(['iter', 'loss', 'auroc', 'aupr'], ['%d', '%.4f', '%.4f', '%.4f'],
                                    os.path.join(args.save_dir, 'test_metric%d.csv' % args.save_id))

    # prepare training data
    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature
    print("Start training ...")

    start = time.perf_counter()
    best_iter, best_auroc, best_aupr = 0, 0, 0
    true, score = 0, 0
    for iter_idx in range(1, args.train_max_iter):
        model.train()
        Two_Stage = False
        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out,  = \
            model(train_enc_graph, train_dec_graph,
                  drug_graph, drug_sim_feat, drug_feat,
                  dis_graph, dis_sim_feat, dis_feat,
                  Two_Stage)

        pred_ratings = pred_ratings.squeeze(-1)
        # loss_com_drug = common_loss(drug_out, drug_sim_out)
        # loss_com_dis = common_loss(dis_out, dis_sim_out)
        #
        # loss = F.binary_cross_entropy(pred_ratings, train_gt_ratings) + \
        #        args.beta * loss_com_dis + args.beta * loss_com_drug
        loss_drug = LOSS(args, drug_out, drug_sim_out, batch_size=0, flag=0)
        loss_dis = LOSS(args, dis_out, dis_sim_out, batch_size=0, flag=1)
        loss =args.beta * ( loss_drug +  loss_dis )/2 + rel_loss(pred_ratings, train_gt_ratings)
        # loss = ( loss_drug +  loss_dis )/2
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        auroc, aupr, y_true, y_score = evaluate(args, model, graph_data,
                                                drug_graph, drug_feat, drug_sim_feat,
                                                dis_graph, dis_feat, dis_sim_feat)
        test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)
        logging_str = "Iter={}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f}".format(
            iter_idx, loss.item(), auroc, aupr)
        if auroc > best_auroc:
            best_iter, best_auroc, best_aupr, true, score = iter_idx, auroc, aupr, y_true, y_score
            # path = "../Contrastive_learn/case/model"+''.join(str(cv+1)+'.path')
            # th.save(model, path)
        if iter_idx % args.train_valid_interval == 0:
            print("test-logging_str", logging_str)

    result = {
        "y_score": score,
        "y_true": true
    }
    data_result = pd.DataFrame(result)

    data_result.to_csv(os.path.join(args.save_dir, '%d_result.csv' % int(cv + 1)), index=False)

    end = time.perf_counter()

    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))
    print("Bset_Iter={}, Best_AUROC={:.4f}, Best_AUPR={:.4f}".format(best_iter, best_auroc, best_aupr))
    test_loss_logger.close()
    return best_auroc, best_aupr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRGCL')
    parser.add_argument('--seed', default=125, type=int)
    parser.add_argument('--device', default='1', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--data_name', default='Gdataset', type=str)
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=840)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--train_max_iter', type=int, default=5000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--num_neighbor', type=int, default=12)
    parser.add_argument('--nhid1', type=int, default=500)
    parser.add_argument('--nhid2', type=int, default=75)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--layers', type=int, default=2)

    # parser.add_argument('--tau_drug', type=float, default=0.7)
    # parser.add_argument('--tau_disease', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--intra', type=float, default=0.2)
    parser.add_argument('--inter', type=float, default=0.2)
    parser.add_argument('--num_hidden', type=int, default=75)
    parser.add_argument('--num_proj_hidden1', type=int, default=100)
    parser.add_argument('--num_proj_hidden2', type=int, default=150)
    parser.add_argument('--share_param', default=True, action='store_true')
    args = parser.parse_args()
    print(args)
    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    aucs, auprs = [], []
    for times in range(1, 2):
        print("++++++++++++++++++times", str(times), "++++++++++++++++++++++")
        args.save_dir = args.data_name + "_2layer_" + ''.join(str(times) + 'time')
        args.save_dir = os.path.join("case", args.save_dir)
        # args.save_dir = args.data_name + "beta=" + str(args.beta) + "_" + ''.join(str(times) + 'time')
        # args.save_dir = os.path.join("beta", args.save_dir)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        dataset = DrugDataLoader(args.data_name, args.device,
                                 symm=args.gcn_agg_norm_symm,
                                 k=args.num_neighbor)

        print("Loading dataset finished ...\n")

        auc_list, aupr_list = [], []
        for cv in range(0, 10):
            args.save_id = cv + 1
            print("===============" + str(cv + 1) + "=================")
            graph_data = dataset.data_cv[cv]
            auc, aupr = train(args, dataset, graph_data, cv)
            auc_list.append(round(auc, 4))
            aupr_list.append(round(aupr, 4))

        print("Mean_AUROC{:4f}".format(np.mean(auc_list)), "Mean_AURP{:4f}".format(np.mean(aupr_list)))
        print("auroc_list", auc_list)
        print("aupr_list", aupr_list)
        aucs += auc_list
        auprs += aupr_list
    print("mean times auc{:4f} ".format(np.mean(aucs)),
          "mean times aupr{:4f} ".format(np.mean(auprs)))
    print("aucs", aucs)
    print("auprs", auprs)
