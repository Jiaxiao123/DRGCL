import argparse
import numpy as np
import scipy.io as sio

from sklearn import metrics


def load_data(args):
    data_list = []

    for time in range(1, 313):
        data = sio.loadmat('./BNNR/disease_denovo/'+args.dataset +
                               '/'+ str(time) + '_result.mat')
        data = data['result_data']
        data_list.append(data)
    # break
    auroc_list, aupr_list = [], []
    # y_score_list, y_true_list = [], []
    for result in data_list:
        y_score, y_true = result[:, 0], result[:, 1]
        # auroc = metrics.roc_auc_score(y_true, y_score)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auroc = metrics.auc(fpr, tpr)

        aupr = metrics.average_precision_score(y_true, y_score)
        # precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        # aupr = metrics.auc(recall, precision)
        # print(aupr)

        auroc_list.append(auroc)
        aupr_list.append(aupr)
        print("AUROC {:.4f}".format(auroc), "aupr {:.4f}".format(aupr))
        # y_score_list.extend(y_score)
        # y_true_list.extend(y_true)

    # aurocs = metrics.roc_auc_score(y_true_list, y_score_list)
    # precision, recall, _ = metrics.precision_recall_curve(y_true_list, y_score_list)
    # auprs = metrics.auc(recall, precision)
    # print("aurocs", aurocs, "auprs", auprs)

    print("Average AUROC {:.4f}".format(np.mean(auroc_list)), "std {:.4f}".format(np.std(auroc_list)))
    print("Average AUPR {:.4f}".format(np.mean(aupr_list)), "std {:.4f}".format(np.std(aupr_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='Gdataset')
    args = parser.parse_args()
    print(args)
    load_data(args)
