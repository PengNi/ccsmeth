#! /usr/bin/python
"""
evaluate in read level
"""
import argparse
import os
import random
from collections import namedtuple

import numpy as np
from sklearn.metrics import roc_auc_score


num_sites = [10000, 100000, 200000, 1000000000]
# num_sites = [100000, ]
CallRecord = namedtuple('CallRecord', ['chrom', 'pos', 'strand',
                                       'holeid', 'loc', 'depth',
                                       'prob0', 'prob1',
                                       'predicted_label',
                                       'is_true_methylated'])


def sample_sites(filename, is_methylated, depthcf, probcf, sampleids=None):
    all_crs = list()
    rf = open(filename)
    skip_cnt = 0
    cnt = 0
    repeat_cnt = 0
    read_sampleids = set()
    for line in rf:
        cnt += 1
        words = line.strip().split("\t")
        holeid = words[3].split("/")[1]
        pos = int(words[1]) if words[2] == "+" else int(words[1]) - 1
        sampid = "\t".join([words[0], str(pos), holeid])  # chrom, pos, holeid
        if sampleids is not None and sampid not in sampleids:
            skip_cnt += 1
            continue
        depth = words[5]
        if "," in depth:
            depthW, depthC = float(depth.split(",")[0]), float(depth.split(",")[1])
            if depthW < depthcf or depthC < depthcf:
                skip_cnt += 1
                continue
        elif float(depth) < depthcf:
            skip_cnt += 1
            continue
        prob0, prob1 = float(words[6]), float(words[7])
        if abs(prob0 - prob1) < probcf:
            skip_cnt += 1
            continue
        if sampid in read_sampleids:
            repeat_cnt += 1
            continue
        read_sampleids.add(sampid)
        all_crs.append(CallRecord(words[0], int(words[1]),
                                  words[2], words[3], int(words[4]), depth,
                                  prob0, prob1,
                                  int(words[8]),
                                  is_methylated))
    print('there are {} cpg candidates totally, {} cpgs kept, {} cpgs left, '
          '{} cpgs repeat'.format(cnt, len(all_crs), skip_cnt, repeat_cnt))
    rf.close()
    random.shuffle(all_crs)
    return all_crs


def sample_sites_ont(filename, is_methylated, probcf, sampleids=None):
    all_crs = list()
    skip_cnt = 0
    cnt = 0
    rf = open(filename)
    for line in rf:
        cnt += 1
        words = line.strip().split("\t")
        sampid = "\t".join([words[0], words[1], words[2], words[3], words[4], words[5]])
        if sampleids is not None and sampid not in sampleids:
            skip_cnt += 1
            continue
        prob0, prob1 = float(words[6]), float(words[7])
        if abs(prob0 - prob1) < probcf:
            skip_cnt += 1
            continue
        all_crs.append(CallRecord(words[0], int(words[1]),
                                  words[2], "-", -1, -1,
                                  float(words[6]), float(words[7]),
                                  int(words[8]),
                                  is_methylated))
    rf.close()
    print('there are {} cpg candidates totally, {} cpgs kept, {} cpgs left'.format(cnt,
                                                                                   len(all_crs),
                                                                                   skip_cnt))
    random.shuffle(all_crs)
    return all_crs


def get_sampleids(sampleids_file):
    sampleids = set()
    with open(sampleids_file, "r") as rf:
        for line in rf:
            if not line.startswith("#"):
                sampleids.add(line.strip())
    return sampleids


if __name__ == '__main__':
    random.seed(1234)

    parser = argparse.ArgumentParser(description='Calculate call accuracy stats of hkmodel for cpgs')
    parser.add_argument('--unmethylated', type=str, required=True)
    parser.add_argument('--methylated', type=str, required=True)
    parser.add_argument('--depth_cf', action="append", required=True)
    parser.add_argument('--prob_cf', action="append", required=True,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                             'means use all calls. range [0, 1].')
    parser.add_argument('--round', type=int, default=5, required=False,
                        help="number of repeated tests for random sampling")
    parser.add_argument('--result_file', type=str, required=False,
                        help='the result file are going to save')
    parser.add_argument('--sampleids_file_u', type=str, default=None, required=False,
                        help='the file contains unmethylated ids of sites to be tested')
    parser.add_argument('--sampleids_file_m', type=str, default=None, required=False,
                        help='the file contains methylated ids of sites to be tested')
    parser.add_argument('--ont', action='store_true', default=False, help="is call_mods file from deepsignal series")

    args = parser.parse_args()

    depth_cfs = args.depth_cf
    prob_cfs = args.prob_cf

    sample_ids_u = get_sampleids(args.sampleids_file_u) if args.sampleids_file_u is not None else None
    sample_ids_m = get_sampleids(args.sampleids_file_m) if args.sampleids_file_m is not None else None

    result_file = os.path.abspath(args.result_file)
    pr_writer = open(result_file, 'w')
    pr_writer.write("tested_type\tTP\tFN\tTN\tFP\t"
                    "accuracy\trecall\tspecificity\tprecision\t"
                    "fallout\tmiss_rate\tFDR\tNPV\tAUC\tsamplenum\tdepth_cutoff\tprob_cf\tnum_rounds\n")

    for depth_cf in depth_cfs:
        for prob_cf in prob_cfs:
            if args.ont:
                unmethylated_sites = sample_sites_ont(args.unmethylated, False, float(prob_cf), sample_ids_u)
                methylated_sites = sample_sites_ont(args.methylated, True, float(prob_cf), sample_ids_m)
            else:
                unmethylated_sites = sample_sites(args.unmethylated, False, int(depth_cf), float(prob_cf), sample_ids_u)
                methylated_sites = sample_sites(args.methylated, True, int(depth_cf), float(prob_cf), sample_ids_m)

            for site_num in num_sites:
                num_rounds = args.round
                if site_num >= len(methylated_sites) and site_num >= len(unmethylated_sites):
                    num_rounds = 1
                metrics = []
                for roundidx in range(num_rounds):
                    random.shuffle(methylated_sites)
                    random.shuffle(unmethylated_sites)
                    tested_sites = methylated_sites[:site_num] + unmethylated_sites[:site_num]

                    tp = 0
                    fp = 0
                    tn = 0
                    fn = 0

                    called = 0
                    correct = 0

                    y_truelabel = []
                    y_scores = []

                    for s in tested_sites:
                        tp += s.predicted_label and s.is_true_methylated
                        fp += s.predicted_label and not s.is_true_methylated
                        tn += not s.predicted_label and not s.is_true_methylated
                        fn += not s.predicted_label and s.is_true_methylated

                        y_truelabel.append(s.is_true_methylated)
                        y_scores.append(s.prob1)

                    print(tp, fn, tn, fp)
                    precision, recall, specificity, accuracy = 0, 0, 0, 0
                    fall_out, miss_rate, fdr, npv, = 0, 0, 0, 0
                    auroc = 0
                    if len(tested_sites) > 0:
                        accuracy = float(tp + tn) / len(tested_sites)
                        if tp + fp > 0:
                            precision = float(tp) / (tp + fp)
                            fdr = float(fp) / (tp + fp)  # false discovery rate
                        else:
                            precision = 0
                            fdr = 0
                        if tp + fn > 0:
                            recall = float(tp) / (tp + fn)
                            miss_rate = float(fn) / (tp + fn)  # false negative rate
                        else:
                            recall = 0
                            miss_rate = 0
                        if tn + fp > 0:
                            specificity = float(tn) / (tn + fp)
                            fall_out = float(fp) / (fp + tn)  # false positive rate
                        else:
                            specificity = 0
                            fall_out = 0
                        if tn + fn > 0:
                            npv = float(tn) / (tn + fn)  # negative predictive value
                        else:
                            npv = 0
                        auroc = roc_auc_score(np.array(y_truelabel), np.array(y_scores))
                    metrics.append([tp, fn, tn, fp, accuracy, recall, specificity, precision,
                                    fall_out, miss_rate, fdr, npv, auroc, len(tested_sites)])
                print("")
                # cal mean
                metrics = np.array(metrics, dtype=float)
                metrics_mean = np.mean(metrics, 0)
                mean_tpfntnfp = "\t".join([str(round(x, 1)) for x in metrics_mean[:4]])
                mean_perf = "\t".join([str(round(x, 4)) for x in metrics_mean[4:13]])
                mean_numlen = str(round(metrics_mean[13]))
                pr_writer.write("\t".join([str(site_num), mean_tpfntnfp, mean_perf, mean_numlen,
                                           str(depth_cf), str(prob_cf), str(num_rounds)]) + "\n")

    pr_writer.close()
