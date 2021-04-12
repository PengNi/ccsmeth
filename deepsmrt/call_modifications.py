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
from models import ModelResNet18

from utils.process_utils import base2code_dna
from utils.process_utils import code2base_dna
from utils.process_utils import display_args
from utils.process_utils import nproc_to_call_mods_in_cpu_mode
from utils.process_utils import str2bool

from utils.constants_torch import FloatTensor
from utils.constants_torch import use_cuda

queen_size_border = 2000
queen_size_border_f5batch = 100
time_wait = 3


def _read_features_file(features_file, features_batch_q, batch_num=512):
    print("read_features process-{} starts".format(os.getpid()))
    b_num = 0
    with open(features_file, "r") as rf:
        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
        kmers = []
        ipd_means = []
        ipd_stds = []
        pw_means = []
        pw_stds = []
        labels = []

        for line in rf:
            words = line.strip().split("\t")

            sampleinfo.append("\t".join(words[0:5]))
            kmer = np.array([base2code_dna[x] for x in words[5]])
            kmers.append(kmer)
            ipd_means.append(np.array([float(x) for x in words[7].split(",")]))
            ipd_stds.append(np.array([float(x) for x in words[8].split(",")]))
            pw_means.append(np.array([float(x) for x in words[9].split(",")]))
            pw_stds.append(np.array([float(x) for x in words[10].split(",")]))

            labels.append(int(words[13]))

            if len(sampleinfo) == batch_num:
                features_batch_q.put((sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, labels))
                while features_batch_q.qsize() > queen_size_border:
                    time.sleep(time_wait)
                sampleinfo = []
                kmers = []
                ipd_means = []
                ipd_stds = []
                pw_means = []
                pw_stds = []
                labels = []
                b_num += 1
        if len(sampleinfo) > 0:
            features_batch_q.put((sampleinfo, kmers, ipd_means, ipd_stds, pw_means, pw_stds, labels))
    features_batch_q.put("kill")
    print("read_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _read_features_file2(features_file, features_batch_q, batch_num=512):
    print("read_features process-{} starts".format(os.getpid()))
    b_num = 0
    with open(features_file, "r") as rf:
        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, depth_all
        kmers = []
        mats_ccs_mean = []
        mats_ccs_std = []
        labels = []

        for line in rf:
            words = line.strip().split("\t")

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

            if len(sampleinfo) == batch_num:
                features_batch_q.put((sampleinfo, kmers, mats_ccs_mean, mats_ccs_std, labels))
                while features_batch_q.qsize() > queen_size_border:
                    time.sleep(time_wait)
                sampleinfo = []
                kmers = []
                mats_ccs_mean = []
                mats_ccs_std = []
                labels = []
                b_num += 1
        if len(sampleinfo) > 0:
            features_batch_q.put((sampleinfo, kmers, mats_ccs_mean, mats_ccs_std, labels))
    features_batch_q.put("kill")
    print("read_features process-{} ending, read {} batches".format(os.getpid(), b_num))


def _call_mods(features_batch, model, batch_size):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples
    # --------------: 2. if from _read_features_from_fast5s(), has uncertain number of samples
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


def _call_mods2(features_batch, model, batch_size):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples
    # --------------: 2. if from _read_features_from_fast5s(), has uncertain number of samples
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
        model = ModelRNN(args.seq_len, args.layer_num, args.class_num,
                         args.dropout_rate, args.hid_rnn,
                         args.n_vocab, args.n_embed,
                         is_stds=str2bool(args.is_stds),
                         model_type=args.model_type)
    elif args.model_type in {"attbilstm", "attbigru", }:
        model = ModelAttRNN(args.seq_len, args.layer_num, args.class_num,
                            args.dropout_rate, args.hid_rnn,
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

        if args.model_type in {"bilstm", "bigru", "attbilstm", "attbigru", }:
            pred_str, accuracy, batch_num = _call_mods(features_batch, model, args.batch_size)
        elif args.model_type in {"resnet18", }:
            pred_str, accuracy, batch_num = _call_mods2(features_batch, model, args.batch_size)
        else:
            raise ValueError("model_type not right!")

        pred_str_q.put(pred_str)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_str_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_str_q.qsize()))
        accuracy_list.append(accuracy)
        batch_num_total += batch_num
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    print('call_mods process-{} ending, proceed {} batches'.format(os.getpid(), batch_num_total))


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


def call_mods(args):
    print("[main]call_mods starts..")
    start = time.time()

    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        raise ValueError("--model_path is not set right!")
    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        raise ValueError("--input_path does not exist!")

    if os.path.isdir(input_path):
        pass
    else:
        # features_batch_q = mp.Queue()
        features_batch_q = Queue()
        if args.model_type in {"bilstm", "bigru", "attbilstm", "attbigru", }:
            p_rf = mp.Process(target=_read_features_file, args=(input_path, features_batch_q,
                                                                args.batch_size))
        elif args.model_type in {"resnet18", }:
            p_rf = mp.Process(target=_read_features_file2, args=(input_path, features_batch_q,
                                                                 args.batch_size))
        else:
            raise ValueError("model_type not right!")

        p_rf.daemon = True
        p_rf.start()

        # pred_str_q = mp.Queue()
        pred_str_q = Queue()

        predstr_procs = []

        if use_cuda:
            nproc_dp = args.nproc_gpu
            if nproc_dp < 1:
                nproc_dp = 1
        else:
            nproc = args.nproc
            if nproc < 3:
                print("--nproc must be >= 3!!")
                nproc = 3
            nproc_dp = nproc - 2
            if nproc_dp > nproc_to_call_mods_in_cpu_mode:
                nproc_dp = nproc_to_call_mods_in_cpu_mode

        for _ in range(nproc_dp):
            p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q, args))
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        # print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q))
        p_w.daemon = True
        p_w.start()

        for p in predstr_procs:
            p.join()

        # print("finishing the write_process..")
        pred_str_q.put("kill")

        p_rf.join()

        p_w.join()

    print("[main]call_mods costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input_path", "-i", action="store", type=str,
                         required=True,
                         help="the input path, can be a signal_feature file from extract_features.py, "
                              "or a directory of fast5 files. If a directory of fast5 files is provided, "
                              "args in FAST5_EXTRACTION should (reference_path must) be provided.")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    # model param
    p_call.add_argument('--model_type', type=str, default="attbigru",
                        choices=["attbilstm", "attbigru", "bilstm", "bigru",
                                 "resnet18"],
                        required=False,
                        help="type of model to use, 'attbilstm', 'attbigru', "
                             "'bilstm', 'bigru', 'resnet18', default: attbigru")
    p_call.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")
    p_call.add_argument('--is_stds', type=str, default="yes", required=False,
                        help="if using std features at ccs level, yes or no. default yes.")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")

    # BiRNN model param
    p_call.add_argument('--layer_num', type=int, default=3,
                        required=False, help="lstm layer num, default 3")
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiRNN hidden_size for combined feature")
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                          help="the file path to save the predicted result")

    p_extract = parser.add_argument_group("EXTRACTION")

    parser.add_argument("--nproc", "-p", action="store", type=int, default=10,
                        required=False, help="number of processes to be used, default 10.")
    parser.add_argument("--nproc_gpu", action="store", type=int, default=2,
                        required=False, help="number of processes to use gpu (if gpu is available), "
                                             "1 or a number less than nproc-1, no more than "
                                             "nproc/4 is suggested. default 2.")

    args = parser.parse_args()
    display_args(args)

    call_mods(args)


if __name__ == '__main__':
    sys.exit(main())
