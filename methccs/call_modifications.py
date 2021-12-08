"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_strand,
prob_0, prob_1, called_label, seq
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn import metrics

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# from utils.process_utils import Queue
from torch.multiprocessing import Queue
import time
# import random

from models import ModelRNN
from models import ModelAttRNN
from models import ModelAttRNN2s
from models import ModelResNet18
from models import ModelTransEncoder

from utils.process_utils import base2code_dna
from utils.process_utils import code2base_dna
from utils.process_utils import display_args
from utils.process_utils import nproc_to_call_mods_in_cpu_mode
from utils.process_utils import str2bool

from utils.process_utils import get_motif_seqs
from utils.ref_reader import DNAReference

from utils.constants_torch import FloatTensor
from utils.constants_torch import use_cuda

from extract_features import worker_read
from extract_features import handle_one_hole2
from extract_features import _get_holes
from extract_features import _get_holeid

queen_size_border = 2000
time_wait = 3


def _read_features_file_to_str(features_file, featurestrs_batch_q, batch_num=512,
                               holeids_e=None, holeids_ne=None):
    print("read_features process-{} starts".format(os.getpid()))
    b_num = 0
    with open(features_file, "r") as rf:
        featurestrs = []
        for line in rf:
            words = line.strip().split("\t")
            holeid = words[3]
            if holeids_e is not None and holeid not in holeids_e:
                continue
            if holeids_ne is not None and holeid in holeids_ne:
                continue
            featurestrs.append(words)
            if len(featurestrs) == batch_num:
                featurestrs_batch_q.put(featurestrs)
                while featurestrs_batch_q.qsize() > queen_size_border:
                    time.sleep(time_wait)
                featurestrs = []
                b_num += 1
        if len(featurestrs) > 0:
            featurestrs_batch_q.put(featurestrs)
            b_num += 1
    featurestrs_batch_q.put("kill")
    print("read_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _format_features_from_strbatch1(featurestrs_batch_q, features_batch_q):
    print("format_features process-{} starts".format(os.getpid()))
    b_num = 0
    while True:
        if featurestrs_batch_q.empty():
            time.sleep(time_wait)
            continue
        featurestrs = featurestrs_batch_q.get()
        if featurestrs == "kill":
            featurestrs_batch_q.put("kill")
            break
        b_num += 1

        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
        kmers = []
        ipd_means = []
        ipd_stds = []
        pw_means = []
        pw_stds = []
        labels = []

        for words in featurestrs:

            sampleinfo.append("\t".join(words[0:5]))
            kmer = np.array([base2code_dna[x] for x in words[5]])
            kmers.append(kmer)
            ipd_means.append(np.array([float(x) for x in words[7].split(",")]))
            ipd_stds.append(np.array([float(x) for x in words[8].split(",")]))
            pw_means.append(np.array([float(x) for x in words[9].split(",")]))
            pw_stds.append(np.array([float(x) for x in words[10].split(",")]))

            labels.append(int(words[13]))

        features_batch_q.put((sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, labels))
    print("format_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _format_features_from_strbatch2s(featurestrs_batch_q, features_batch_q):
    print("format_features process-{} starts".format(os.getpid()))
    b_num = 0
    while True:
        if featurestrs_batch_q.empty():
            time.sleep(time_wait)
            continue
        featurestrs = featurestrs_batch_q.get()
        if featurestrs == "kill":
            featurestrs_batch_q.put("kill")
            break
        b_num += 1

        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
        kmers = []
        ipd_means = []
        ipd_stds = []
        pw_means = []
        pw_stds = []

        kmers2 = []
        ipd_means2 = []
        ipd_stds2 = []
        pw_means2 = []
        pw_stds2 = []

        labels = []

        for words in featurestrs:

            sampleinfo.append("\t".join(words[0:5]))
            kmer = np.array([base2code_dna[x] for x in words[5]])
            kmers.append(kmer)
            ipd_means.append(np.array([float(x) for x in words[7].split(",")]))
            ipd_stds.append(np.array([float(x) for x in words[8].split(",")]))
            pw_means.append(np.array([float(x) for x in words[9].split(",")]))
            pw_stds.append(np.array([float(x) for x in words[10].split(",")]))

            kmer2 = np.array([base2code_dna[x] for x in words[13]])
            kmers2.append(kmer2)
            ipd_means2.append(np.array([float(x) for x in words[15].split(",")]))
            ipd_stds2.append(np.array([float(x) for x in words[16].split(",")]))
            pw_means2.append(np.array([float(x) for x in words[17].split(",")]))
            pw_stds2.append(np.array([float(x) for x in words[18].split(",")]))

            labels.append(int(words[21]))

        features_batch_q.put((sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds,
                              kmers2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, labels))
    print("format_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _format_features_from_strbatch2(featurestrs_batch_q, features_batch_q):
    print("format_features process-{} starts".format(os.getpid()))
    b_num = 0
    while True:
        if featurestrs_batch_q.empty():
            time.sleep(time_wait)
            continue
        featurestrs = featurestrs_batch_q.get()
        if featurestrs == "kill":
            featurestrs_batch_q.put("kill")
            break
        b_num += 1

        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
        kmers = []
        mats_ccs_mean = []
        mats_ccs_std = []
        labels = []

        for words in featurestrs:
            sampleinfo.append("\t".join(words[0:5]))

            kmer = np.array([base2code_dna[x] for x in words[5]])
            kmers.append(kmer)

            height, width = len(kmer), len(base2code_dna.keys())

            ipd_means = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
            ipd_m_mat = np.zeros((1, height, width), dtype=np.float)
            ipd_m_mat[0, np.arange(len(kmer)), kmer] = ipd_means
            pw_means = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
            pw_m_mat = np.zeros((1, height, width), dtype=np.float)
            pw_m_mat[0, np.arange(len(kmer)), kmer] = pw_means
            mats_ccs_mean.append(np.concatenate((ipd_m_mat, pw_m_mat), axis=0))  # (C=2, H, W)

            ipd_stds = np.array([float(x) for x in words[8].split(",")], dtype=np.float)
            ipd_s_mat = np.zeros((1, height, width), dtype=np.float)
            ipd_s_mat[0, np.arange(len(kmer)), kmer] = ipd_stds
            pw_stds = np.array([float(x) for x in words[10].split(",")], dtype=np.float)
            pw_s_mat = np.zeros((1, height, width), dtype=np.float)
            pw_s_mat[0, np.arange(len(kmer)), kmer] = pw_stds
            mats_ccs_std.append(np.concatenate((ipd_s_mat, pw_s_mat), axis=0))  # (C=2, H, W)

            labels.append(int(words[13]))

        features_batch_q.put((sampleinfo, kmers, mats_ccs_mean, mats_ccs_std, labels))
    print("format_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _call_mods(features_batch, model, batch_size):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples
    # --------------: 2. if from _worker_extract_features(), has uncertain number of samples
    sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, \
        labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]
        b_kmers = kmers[batch_s:batch_e]
        b_ipd_means = ipd_means[batch_s:batch_e]
        b_ipd_stds = ipd_stds[batch_s:batch_e]
        b_pw_means = pw_means[batch_s:batch_e]
        b_pw_stds = pw_stds[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_kmers), FloatTensor(b_ipd_means), FloatTensor(b_ipd_stds),
                                      FloatTensor(b_pw_means), FloatTensor(b_pw_stds))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            acc_batch = metrics.accuracy_score(
                y_true=b_labels, y_pred=predicted)
            accuracys.append(acc_batch)

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, depth, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(prob_1 / (prob_0 + prob_1), 6)
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer)/2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                pred_str.append("\t".join([b_sampleinfo[idx], str(prob_0_norm),
                                           str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1
    accuracy = np.mean(accuracys)

    return pred_str, accuracy, batch_num


def _call_mods2s(features_batch, model, batch_size):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples
    # --------------: 2. if from _worker_extract_features(), has uncertain number of samples
    sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, \
        kmers2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, \
        labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]
        b_kmers = kmers[batch_s:batch_e]
        b_ipd_means = ipd_means[batch_s:batch_e]
        b_ipd_stds = ipd_stds[batch_s:batch_e]
        b_pw_means = pw_means[batch_s:batch_e]
        b_pw_stds = pw_stds[batch_s:batch_e]
        b_kmers2 = kmers2[batch_s:batch_e]
        b_ipd_means2 = ipd_means2[batch_s:batch_e]
        b_ipd_stds2 = ipd_stds2[batch_s:batch_e]
        b_pw_means2 = pw_means2[batch_s:batch_e]
        b_pw_stds2 = pw_stds2[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_kmers), FloatTensor(b_ipd_means), FloatTensor(b_ipd_stds),
                                      FloatTensor(b_pw_means), FloatTensor(b_pw_stds),
                                      FloatTensor(b_kmers2), FloatTensor(b_ipd_means2), FloatTensor(b_ipd_stds2),
                                      FloatTensor(b_pw_means2), FloatTensor(b_pw_stds2))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            acc_batch = metrics.accuracy_score(
                y_true=b_labels, y_pred=predicted)
            accuracys.append(acc_batch)

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, depth, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(prob_1 / (prob_0 + prob_1), 6)
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer)/2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                pred_str.append("\t".join([b_sampleinfo[idx], str(prob_0_norm),
                                           str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1
    accuracy = np.mean(accuracys)

    return pred_str, accuracy, batch_num


def _call_mods2(features_batch, model, batch_size):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples
    # --------------: 2. if from _worker_extract_features(), has uncertain number of samples
    sampleinfo, kmers, mats_ccs_mean, mats_ccs_std, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]
        b_kmers = kmers[batch_s:batch_e]
        b_mats_ccs_mean = mats_ccs_mean[batch_s:batch_e]
        b_mats_ccs_std = mats_ccs_std[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_mats_ccs_mean), FloatTensor(b_mats_ccs_std))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            acc_batch = metrics.accuracy_score(
                y_true=b_labels, y_pred=predicted)
            accuracys.append(acc_batch)

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, depth, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(prob_1 / (prob_0 + prob_1), 6)
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer)/2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                pred_str.append("\t".join([b_sampleinfo[idx], str(prob_0_norm),
                                           str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1
    accuracy = np.mean(accuracys)

    return pred_str, accuracy, batch_num


def _call_mods_q(model_path, features_batch_q, pred_str_q, args):
    print('call_mods process-{} starts'.format(os.getpid()))
    if args.model_type in {"bilstm", "bigru", }:
        model = ModelRNN(args.seq_len, args.layer_rnn, args.class_num,
                         args.dropout_rate, args.hid_rnn,
                         args.n_vocab, args.n_embed,
                         is_stds=str2bool(args.is_stds),
                         model_type=args.model_type)
    elif args.model_type in {"attbilstm", "attbigru", }:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            is_stds=str2bool(args.is_stds),
                            model_type=args.model_type)
    elif args.model_type in {"attbigru2s", }:
        model = ModelAttRNN2s(args.seq_len, args.layer_rnn, args.class_num,
                              args.dropout_rate, args.hid_rnn,
                              args.n_vocab, args.n_embed,
                              is_stds=str2bool(args.is_stds),
                              model_type=args.model_type)
    elif args.model_type in {"transencoder", }:
        model = ModelTransEncoder(args.seq_len, args.layer_tfe, args.class_num,
                                  args.dropout_rate, args.d_model_tfe, args.nhead_tfe, args.nhid_tfe,
                                  args.n_vocab, args.n_embed,
                                  is_stds=str2bool(args.is_stds),
                                  model_type=args.model_type)
    elif args.model_type == "resnet18":
        model = ModelResNet18(args.class_num, args.dropout_rate, str2bool(args.is_stds))
    else:
        raise ValueError("model_type not right!")

    if use_cuda:
        model = model.cuda()
        para_dict = torch.load(model_path)
    else:
        para_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)

    model.eval()

    accuracy_list = []
    batch_num_total = 0
    while True:

        if features_batch_q.empty():
            time.sleep(time_wait)
            continue

        features_batch = features_batch_q.get()
        if features_batch == "kill":
            features_batch_q.put("kill")
            break

        if args.model_type in {"bilstm", "bigru", "attbilstm", "attbigru", "transencoder", }:
            pred_str, accuracy, batch_num = _call_mods(features_batch, model, args.batch_size)
        elif args.model_type in {"resnet18", }:
            pred_str, accuracy, batch_num = _call_mods2(features_batch, model, args.batch_size)
        elif args.model_type in {"attbigru2s", }:
            pred_str, accuracy, batch_num = _call_mods2s(features_batch, model, args.batch_size)
        else:
            raise ValueError("model_type not right!")

        pred_str_q.put(pred_str)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_str_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_str_q.qsize()))
        accuracy_list.append(accuracy)
        batch_num_total += batch_num
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    print('call_mods process-{} ending, proceed {} batches({})'.format(os.getpid(), batch_num_total,
                                                                       args.batch_size))


def _write_predstr_to_file(write_fp, predstr_q):
    print('write_process-{} starts'.format(os.getpid()))
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                print('write_process-{} finished'.format(os.getpid()))
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def _batch_feature_list1(feature_list):
    sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
    kmers = []
    ipd_means = []
    ipd_stds = []
    pw_means = []
    pw_stds = []
    labels = []
    for featureline in feature_list:
        chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
            kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label = featureline
        sampleinfo.append("\t".join(list(map(str, [chrom, abs_loc, strand, holeid, depth_all]))))
        kmers.append(np.array([base2code_dna[x] for x in kmer_seq]))
        ipd_means.append(np.array(kmer_ipdm, dtype=np.float))
        ipd_stds.append(np.array(kmer_ipds, dtype=np.float))
        pw_means.append(np.array(kmer_pwm, dtype=np.float))
        pw_stds.append(np.array(kmer_pws, dtype=np.float))
        labels.append(label)
    return sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, labels


def _batch_feature_list2s(feature_list):
    sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
    kmers = []
    ipd_means = []
    ipd_stds = []
    pw_means = []
    pw_stds = []
    kmers2 = []
    ipd_means2 = []
    ipd_stds2 = []
    pw_means2 = []
    pw_stds2 = []
    labels = []
    for featureline in feature_list:
        chrom, abs_loc, strand, holeid, depth_all, \
            kmer_seq, kmer_depth, kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, \
            kmer_seq2, kmer_depth2, kmer_ipdm2, kmer_ipds2, kmer_pwm2, kmer_pws2, kmer_subr_ipds2, kmer_subr_pws2, \
            label = featureline
        sampleinfo.append("\t".join(list(map(str, [chrom, abs_loc, strand, holeid, depth_all]))))
        kmers.append(np.array([base2code_dna[x] for x in kmer_seq]))
        ipd_means.append(np.array(kmer_ipdm, dtype=np.float))
        ipd_stds.append(np.array(kmer_ipds, dtype=np.float))
        pw_means.append(np.array(kmer_pwm, dtype=np.float))
        pw_stds.append(np.array(kmer_pws, dtype=np.float))
        kmers2.append(np.array([base2code_dna[x] for x in kmer_seq2]))
        ipd_means2.append(np.array(kmer_ipdm2, dtype=np.float))
        ipd_stds2.append(np.array(kmer_ipds2, dtype=np.float))
        pw_means2.append(np.array(kmer_pwm2, dtype=np.float))
        pw_stds2.append(np.array(kmer_pws2, dtype=np.float))
        labels.append(label)
    return sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, \
        kmers2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, labels


def _batch_feature_list2(feature_list):
    sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
    kmers = []
    mats_ccs_mean = []
    mats_ccs_std = []
    labels = []

    for featureline in feature_list:
        chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
            kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label = featureline

        sampleinfo.append("\t".join(list(map(str, [chrom, abs_loc, strand, holeid, depth_all]))))

        kmer = np.array([base2code_dna[x] for x in kmer_seq])
        kmers.append(kmer)

        height, width = len(kmer), len(base2code_dna.keys())

        ipd_means = np.array([float(x) for x in kmer_ipdm.split(",")], dtype=np.float)
        ipd_m_mat = np.zeros((1, height, width), dtype=np.float)
        ipd_m_mat[0, np.arange(len(kmer)), kmer] = ipd_means
        pw_means = np.array([float(x) for x in kmer_pwm.split(",")], dtype=np.float)
        pw_m_mat = np.zeros((1, height, width), dtype=np.float)
        pw_m_mat[0, np.arange(len(kmer)), kmer] = pw_means
        mats_ccs_mean.append(np.concatenate((ipd_m_mat, pw_m_mat), axis=0))  # (C=2, H, W)

        ipd_stds = np.array([float(x) for x in kmer_ipds.split(",")], dtype=np.float)
        ipd_s_mat = np.zeros((1, height, width), dtype=np.float)
        ipd_s_mat[0, np.arange(len(kmer)), kmer] = ipd_stds
        pw_stds = np.array([float(x) for x in kmer_pws.split(",")], dtype=np.float)
        pw_s_mat = np.zeros((1, height, width), dtype=np.float)
        pw_s_mat[0, np.arange(len(kmer)), kmer] = pw_stds
        mats_ccs_std.append(np.concatenate((ipd_s_mat, pw_s_mat), axis=0))  # (C=2, H, W)

        labels.append(int(label))
    return sampleinfo, kmers, mats_ccs_mean, mats_ccs_std, labels


def _worker_extract_features(hole_align_q, features_batch_q, contigs, motifs, args):
    sys.stderr.write("extrac_features process-{} starts\n".format(os.getpid()))
    cnt_holesbatch = 0
    while True:
        # print("hole_align_q size:", hole_align_q.qsize(), "; pid:", os.getpid())
        if hole_align_q.empty():
            time.sleep(time_wait)
            continue
        holes_aligninfo = hole_align_q.get()
        if holes_aligninfo == "kill":
            hole_align_q.put("kill")
            break
        feature_list = []
        for hole_aligninfo in holes_aligninfo:
            feature_list += handle_one_hole2(hole_aligninfo, contigs, motifs, args)
        if len(feature_list) > 0:
            if args.model_type in {"bilstm", "bigru", "attbilstm", "attbigru", "transencoder", }:
                features_batch_q.put(_batch_feature_list1(feature_list))
            elif args.model_type in {"attbigru2s", }:
                features_batch_q.put(_batch_feature_list2s(feature_list))
            elif args.model_type in {"resnet18", }:
                features_batch_q.put(_batch_feature_list2(feature_list))
            else:
                raise ValueError("model_type not right!")

        cnt_holesbatch += 1
        if cnt_holesbatch % 200 == 0:
            sys.stderr.write("extrac_features process-{}, {} hole_batches({}) "
                             "proceed\n".format(os.getpid(), cnt_holesbatch, args.holes_batch))
            sys.stderr.flush()
    sys.stderr.write("extrac_features process-{} ending, proceed {} "
                     "hole_batches({})\n".format(os.getpid(), cnt_holesbatch, args.holes_batch))


def call_mods(args):
    print("[main]call_mods starts..")
    start = time.time()
    torch.manual_seed(args.tseed)
    torch.cuda.manual_seed(args.tseed)

    model_path = os.path.abspath(args.model_file)
    if not os.path.exists(model_path):
        raise ValueError("--model_file is not set right!")
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise ValueError("--input_file does not exist!")

    holeids_e = None if args.holeids_e is None else _get_holes(args.holeids_e)
    holeids_ne = None if args.holeids_ne is None else _get_holes(args.holeids_ne)

    if input_path.endswith(".bam") or input_path.endswith(".sam"):
        reference = os.path.abspath(args.ref)
        if not os.path.exists(reference):
            raise IOError("refernce(--ref) file does not exist!")
        contigs = DNAReference(reference).getcontigs()
        motifs = get_motif_seqs(args.motifs)

        hole_align_q = Queue()
        features_batch_q = Queue()
        pred_str_q = Queue()

        nproc = args.threads
        nproc_dp = args.threads_call
        if use_cuda:
            if nproc_dp < 1:
                nproc_dp = 1
        else:
            if nproc_dp > nproc_to_call_mods_in_cpu_mode:
                nproc_dp = nproc_to_call_mods_in_cpu_mode
        if nproc <= nproc_dp + 2:
            print("--threads must be > nproc_dp + 2!!")
            nproc = nproc_dp + 2 + 1

        p_read = mp.Process(target=worker_read, args=(input_path, hole_align_q, args, holeids_e, holeids_ne))
        p_read.daemon = True
        p_read.start()

        p_w = mp.Process(target=_write_predstr_to_file, args=(args.output, pred_str_q))
        p_w.daemon = True
        p_w.start()

        # TODO: why the processes in ps_extract start so slowly?
        ps_extract = []
        ps_call = []
        nproc_ext = nproc - nproc_dp - 2
        for i in range(max(nproc_ext, nproc_dp)):
            if i < nproc_ext:
                p = mp.Process(target=_worker_extract_features, args=(hole_align_q, features_batch_q,
                                                                      contigs, motifs, args))
                p.daemon = True
                p.start()
                ps_extract.append(p)
            if i < nproc_dp:
                p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q, args))
                p.daemon = True
                p.start()
                ps_call.append(p)

        p_read.join()

        for p in ps_extract:
            p.join()
        features_batch_q.put("kill")

        for p in ps_call:
            p.join()
        pred_str_q.put("kill")

        p_w.join()
    else:
        # features_batch_q = mp.Queue()
        features_batch_q = Queue()
        # pred_str_q = mp.Queue()
        pred_str_q = Queue()
        featurestrs_batch_q = Queue()

        nproc = args.threads
        nproc_dp = args.threads_call
        if use_cuda:
            if nproc_dp < 1:
                nproc_dp = 1
        else:
            if nproc_dp > nproc_to_call_mods_in_cpu_mode:
                nproc_dp = nproc_to_call_mods_in_cpu_mode
        if nproc < nproc_dp + 2:
            print("--threads must be > nproc_dp + 2!!")
            nproc = nproc_dp + 2 + 1
        nproc_cnvt = nproc - nproc_dp - 2

        p_read = mp.Process(target=_read_features_file_to_str, args=(input_path, featurestrs_batch_q,
                                                                     args.batch_size,
                                                                     holeids_e, holeids_ne))
        p_read.daemon = True
        p_read.start()

        ps_str2value = []
        if args.model_type in {"bilstm", "bigru", "attbilstm", "attbigru", "transencoder", }:
            for _ in range(nproc_cnvt):
                p = mp.Process(target=_format_features_from_strbatch1, args=(featurestrs_batch_q,
                                                                             features_batch_q))
                p.daemon = True
                p.start()
                ps_str2value.append(p)
        elif args.model_type in {"resnet18", }:
            for _ in range(nproc_cnvt):
                p = mp.Process(target=_format_features_from_strbatch2, args=(featurestrs_batch_q,
                                                                             features_batch_q))
                p.daemon = True
                p.start()
                ps_str2value.append(p)
        elif args.model_type in {"attbigru2s", }:
            for _ in range(nproc_cnvt):
                p = mp.Process(target=_format_features_from_strbatch2s, args=(featurestrs_batch_q,
                                                                              features_batch_q))
                p.daemon = True
                p.start()
                ps_str2value.append(p)
        else:
            raise ValueError("model_type not right!")

        predstr_procs = []
        for _ in range(nproc_dp):
            p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q, args))
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        # print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(args.output, pred_str_q))
        p_w.daemon = True
        p_w.start()

        p_read.join()

        for p in ps_str2value:
            p.join()
        features_batch_q.put("kill")

        for p in predstr_procs:
            p.join()
        pred_str_q.put("kill")

        p_w.join()

    print("[main]call_mods costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input", "-i", action="store", type=str,
                         required=True,
                         help="input file, can be aligned.bam/sam, or features.tsv generated by "
                              "extract_features.py. If aligned.bam/sam is provided, args in EXTRACTION "
                              "should (reference_path must) be provided.")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_file", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    # model param
    p_call.add_argument('--model_type', type=str, default="attbigru2s",
                        choices=["attbilstm", "attbigru", "bilstm", "bigru",
                                 "transencoder",
                                 "resnet18",
                                 "attbigru2s"],
                        required=False,
                        help="type of model to use, 'attbilstm', 'attbigru', "
                             "'bilstm', 'bigru', 'transencoder', 'resnet18', "
                             "'attbigru2s', "
                             "default: attbigru2s")
    p_call.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")
    p_call.add_argument('--is_stds', type=str, default="yes", required=False,
                        help="if using std features at ccs level, yes or no. default yes.")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")

    # BiRNN/transformerencoder model param
    parser.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    parser.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")

    # BiRNN model param
    parser.add_argument('--layer_rnn', type=int, default=3,
                        required=False, help="BiRNN layer num, default 3")
    parser.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiRNN hidden_size for combined feature")

    # transformerencoder model param
    parser.add_argument('--layer_tfe', type=int, default=6,
                        required=False, help="transformer encoder layer num, default 6")
    parser.add_argument('--d_model_tfe', type=int, default=256,
                        required=False, help="the number of expected features in the "
                                             "transformer encoder/decoder inputs")
    parser.add_argument('--nhead_tfe', type=int, default=4,
                        required=False, help="the number of heads in the multiheadattention models")
    parser.add_argument('--nhid_tfe', type=int, default=512,
                        required=False, help="the dimension of the feedforward network model")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", action="store", type=str, required=True,
                          help="the file path to save the predicted result")

    p_extract = parser.add_argument_group("EXTRACTION")
    p_extract.add_argument("--ref", type=str, required=False,
                           help="path to genome reference to be aligned, in fasta/fa format.")
    p_extract.add_argument("--holeids_e", type=str, default=None, required=False,
                           help="file contains holeids to be extracted, default None")
    p_extract.add_argument("--holeids_ne", type=str, default=None, required=False,
                           help="file contains holeids not to be extracted, default None")
    p_extract.add_argument("--motifs", action="store", type=str,
                           required=False, default='CG',
                           help='motif seq to be extracted, default: CG. '
                                'can be multi motifs splited by comma '
                                '(no space allowed in the input str), '
                                'or use IUPAC alphabet, '
                                'the mod_loc of all motifs must be '
                                'the same')
    p_extract.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                           help='0-based location of the targeted base in the motif, default 0')
    p_extract.add_argument("--methy_label", action="store", type=int,
                           choices=[1, 0], required=False, default=1,
                           help="the label of the interested modified bases, this is for training."
                                " 0 or 1, default 1")
    p_extract.add_argument("--mapq", type=int, default=30, required=False,
                           help="MAPping Quality cutoff for selecting alignment items, default 30")
    p_extract.add_argument("--identity", type=float, default=0.8, required=False,
                           help="identity cutoff for selecting alignment items, default 0.8")
    p_extract.add_argument("--two_strands", action="store_true", default=False, required=False,
                           help="after quality (mapq, identity) control, if then only using CCS reads "
                                "which have subreads in two strands")
    p_extract.add_argument("--comb_strands", action="store_true", default=False, required=False,
                           help="if combining features in two(+/-) strands of one site")
    p_extract.add_argument("--depth", type=int, default=1, required=False,
                           help="(mean) depth (number of subreads) cutoff for "
                                "selecting high-quality aligned reads/kmers "
                                "per strand of a CCS, default 1.")
    p_extract.add_argument("--norm", action="store", type=str, choices=["zscore", "min-mean", "min-max", "mad"],
                           default="zscore", required=False,
                           help="method for normalizing ipd/pw in subread level. "
                                "zscore, min-mean, min-max or mad, default zscore")
    p_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                           help="not use CodecV1 to decode ipd/pw")
    p_extract.add_argument("--num_subreads", type=int, default=0, required=False,
                           help="info of max num of subreads to be extracted to output, default 0")
    p_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
                           help="full path to the executable binary samtools file. "
                                "If not specified, it is assumed that samtools is in "
                                "the PATH.")
    p_extract.add_argument("--holes_batch", type=int, default=50, required=False,
                           help="number of holes in an batch to get/put in queues")
    p_extract.add_argument("--seed", type=int, default=1234, required=False,
                           help="seed for randomly selecting subreads, default 1234")

    parser.add_argument("--threads", "-p", action="store", type=int, default=10,
                        required=False, help="number of threads to be used, default 10.")
    parser.add_argument("--threads_call", action="store", type=int, default=2,
                        required=False, help="number of threads used to call with trained models, "
                                             "no more than threads/4 is suggested. default 2.")
    parser.add_argument('--tseed', type=int, default=1234,
                        help='random seed for torch')

    args = parser.parse_args()
    display_args(args)

    call_mods(args)


if __name__ == '__main__':
    sys.exit(main())
