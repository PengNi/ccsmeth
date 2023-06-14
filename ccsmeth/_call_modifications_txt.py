"""
call modifications from bam/sam files or extracted features.
output format: chromosome, pos, strand, read_name, read_loc, depth,
prob_0, prob_1, called_label, seq
"""

import os

import numpy as np
import torch
import torch.multiprocessing as mp

import gzip

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
except AttributeError:
    pass

# from utils.process_utils import Queue
from torch.multiprocessing import Queue
import time
# import random

from tqdm import tqdm

from .models import ModelAttRNN

from .utils.process_utils import base2code_dna
from .utils.process_utils import code2base_dna
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode
from .utils.process_utils import str2bool
from .utils.process_utils import max_queue_size

from .utils.constants_torch import FloatTensor
from .utils.constants_torch import use_cuda

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

# add this export temporarily
# https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'

queue_size_border = max_queue_size
time_wait = 1


# read features from feature file ==============================================
def _count_holenum(features_file):
    h_num = 0
    preholeid = None
    if features_file.endswith(".gz"):
        infile = gzip.open(features_file, 'rt')
    else:
        infile = open(features_file, 'r')
    for line in infile:
        words = line.strip().split("\t")
        holeid = words[3]
        if preholeid is None:
            preholeid = holeid
        elif preholeid != holeid:
            preholeid = holeid
            h_num += 1
    h_num += 1
    infile.close()
    return h_num


def _read_features_file_to_str(features_file, featurestrs_batch_q, holes_batch=50):
    LOGGER.info("read_features process-{} starts".format(os.getpid()))
    h_num_total = _count_holenum(features_file)
    hbatch_num = h_num_total // holes_batch
    if h_num_total % holes_batch > 0:
        hbatch_num += 1
    LOGGER.info("read_features process-{} - generate {} hole/read batches({})".format(os.getpid(),
                                                                                      hbatch_num,
                                                                                      holes_batch))

    h_num = 0
    hbatch_num_got = 0
    preholeid = None
    if features_file.endswith(".gz"):
        infile = gzip.open(features_file, 'rt')
    else:
        infile = open(features_file, 'r')
    featurestrs = []
    with tqdm(total=hbatch_num,
              desc="batch_reader") as pbar:
        for line in infile:
            words = line.strip().split("\t")
            holeid = words[3]
            if preholeid is None:
                preholeid = holeid
            elif preholeid != holeid:
                preholeid = holeid
                h_num += 1
                if h_num % holes_batch == 0:
                    featurestrs_batch_q.put(featurestrs)
                    while featurestrs_batch_q.qsize() > queue_size_border:
                        time.sleep(time_wait)
                    featurestrs = []
                    pbar.update(1)
                    hbatch_num_got += 1
            featurestrs.append(words)
        infile.close()
        h_num += 1
        if len(featurestrs) > 0:
            featurestrs_batch_q.put(featurestrs)
            pbar.update(1)
            hbatch_num_got += 1
        assert hbatch_num_got == hbatch_num
    featurestrs_batch_q.put("kill")
    LOGGER.info("read_features process-{} ending, read {} reads/holes batches({})".format(os.getpid(),
                                                                                          hbatch_num_got,
                                                                                          holes_batch))


def _format_features_from_strbatch2s(featurestrs_batch_q, features_batch_q, seq_len,
                                     holeids_e, holeids_ne):
    LOGGER.info("format_features process-{} starts".format(os.getpid()))
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

        sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, loc

        fkmers = []
        fpasss = []
        fipdms = []
        fipdsds = []
        fpwms = []
        fpwsds = []
        fquals = []
        fmaps = []

        rkmers = []
        rpasss = []
        ripdms = []
        ripdsds = []
        rpwms = []
        rpwsds = []
        rquals = []
        rmaps = []

        labels = []

        if len(featurestrs) < 1:
            continue
        oriklen = len(featurestrs[0][5])
        if oriklen == seq_len:
            left_cut = 0
            right_cut = -oriklen
        elif oriklen > seq_len:
            left_cut = right_cut = (oriklen - seq_len) // 2
        else:
            continue
        for words in featurestrs:
            holeid = words[3]
            if holeids_e is not None and holeid not in holeids_e:
                continue
            if holeids_ne is not None and holeid in holeids_ne:
                continue

            sampleinfo.append("\t".join(words[0:5]))

            fkmer = [base2code_dna[x] for x in words[5][left_cut:-right_cut]]
            fkmers.append(fkmer)
            fpasss.append([int(words[6])] * len(fkmer))
            fipdms.append([float(x) for x in words[7].split(",")[left_cut:-right_cut]])
            fipdsds.append([float(x) for x in words[8].split(",")[left_cut:-right_cut]] if words[8] != "." else 0)
            fpwms.append([float(x) for x in words[9].split(",")[left_cut:-right_cut]])
            fpwsds.append([float(x) for x in words[10].split(",")[left_cut:-right_cut]] if words[10] != "." else 0)
            fquals.append([float(x) for x in words[11].split(",")[left_cut:-right_cut]])
            fmaps.append([float(x) for x in words[12].split(",")[left_cut:-right_cut]] if words[12] != "." else 0)

            rkmer = [base2code_dna[x] for x in words[13][left_cut:-right_cut]]
            rkmers.append(rkmer)
            rpasss.append([int(words[14])] * len(rkmer))
            ripdms.append([float(x) for x in words[15].split(",")[left_cut:-right_cut]])
            ripdsds.append([float(x) for x in words[16].split(",")[left_cut:-right_cut]] if words[16] != "." else 0)
            rpwms.append([float(x) for x in words[17].split(",")[left_cut:-right_cut]])
            rpwsds.append([float(x) for x in words[18].split(",")[left_cut:-right_cut]] if words[18] != "." else 0)
            rquals.append([float(x) for x in words[19].split(",")[left_cut:-right_cut]])
            rmaps.append([float(x) for x in words[20].split(",")[left_cut:-right_cut]] if words[20] != "." else 0)

            labels.append(int(words[21]))

        features_batch_q.put((sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fquals, fmaps,
                              rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rquals, rmaps, labels))
        while features_batch_q.qsize() > queue_size_border:
            time.sleep(time_wait)
    LOGGER.info("format_features process-{} ending, read {} batches".format(os.getpid(), b_num))


# call mods =============================================================
def _call_mods2s(features_batch, model, batch_size, device=0):
    sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fquals, fmaps, \
        rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rquals, rmaps, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]

        b_fkmers = np.array(fkmers[batch_s:batch_e])
        b_fpasss = np.array(fpasss[batch_s:batch_e])
        b_fipdms = np.array(fipdms[batch_s:batch_e])
        b_fipdsds = np.array(fipdsds[batch_s:batch_e])
        b_fpwms = np.array(fpwms[batch_s:batch_e])
        b_fpwsds = np.array(fpwsds[batch_s:batch_e])
        b_fquals = np.array(fquals[batch_s:batch_e])
        b_fmaps = np.array(fmaps[batch_s:batch_e])

        b_rkmers = np.array(rkmers[batch_s:batch_e])
        b_rpasss = np.array(rpasss[batch_s:batch_e])
        b_ripdms = np.array(ripdms[batch_s:batch_e])
        b_ripdsds = np.array(ripdsds[batch_s:batch_e])
        b_rpwms = np.array(rpwms[batch_s:batch_e])
        b_rpwsds = np.array(rpwsds[batch_s:batch_e])
        b_rquals = np.array(rquals[batch_s:batch_e])
        b_rmaps = np.array(rmaps[batch_s:batch_e])

        # b_labels = np.array(labels[batch_s:batch_e])
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_fkmers, device), FloatTensor(b_fpasss, device),
                                      FloatTensor(b_fipdms, device), FloatTensor(b_fipdsds, device),
                                      FloatTensor(b_fpwms, device), FloatTensor(b_fpwsds, device),
                                      FloatTensor(b_fquals, device), FloatTensor(b_fmaps, device),
                                      FloatTensor(b_rkmers, device), FloatTensor(b_rpasss, device),
                                      FloatTensor(b_ripdms, device), FloatTensor(b_ripdsds, device),
                                      FloatTensor(b_rpwms, device), FloatTensor(b_rpwsds, device),
                                      FloatTensor(b_rquals, device), FloatTensor(b_rmaps, device))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, loc, depth, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(1 - prob_0_norm, 6)
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_fkmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer)/2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                pred_str.append("\t".join([b_sampleinfo[idx],
                                           str(b_fpasss[idx][0]) + "," + str(b_rpasss[idx][0]),
                                           str(prob_0_norm), str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1

    return pred_str, batch_num


def _call_mods_q(model_path, features_batch_q, pred_str_q, args, device=0):
    LOGGER.info('call_mods process-{} starts'.format(os.getpid()))
    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            is_qual=str2bool(args.is_qual),
                            is_map=str2bool(args.is_map),
                            is_stds=str2bool(args.is_stds),
                            is_npass=str2bool(args.is_npass),
                            model_type=args.model_type,
                            device=device)
    else:
        raise ValueError("--model_type not right!")

    try:
        para_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # para_dict = torch.load(model_path, map_location=torch.device(device))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)
        LOGGER.debug('call_mods process-{} loads model param successfully'.format(os.getpid()))
        del model_dict
    except RuntimeError:
        # for DDP model convertion (key: module.embed.weight -> embed.weight)
        from collections import OrderedDict
        para_dict = torch.load(model_path, map_location=torch.device('cpu'))
        para_dict_new = OrderedDict()
        for param_tensor in para_dict.keys():
            para_dict_new[param_tensor[7:]] = para_dict[param_tensor]
        model.load_state_dict(para_dict_new)
        LOGGER.debug('call_mods process-{} loads model param successfully-1'.format(os.getpid()))
        del para_dict_new
    # sys.stdout.flush()

    if use_cuda:
        model = model.cuda(device)
    model.eval()

    batch_num_total = 0
    while True:

        if features_batch_q.empty():
            time.sleep(time_wait)
            continue

        features_batch = features_batch_q.get()
        if features_batch == "kill":
            features_batch_q.put("kill")
            break

        if args.model_type in {"attbigru2s", "attbilstm2s"}:
            pred_str, batch_num = _call_mods2s(features_batch, model, args.batch_size, device)
        else:
            raise ValueError("--model_type not right!")

        pred_str_q.put(pred_str)
        while pred_str_q.qsize() > queue_size_border:
            time.sleep(time_wait)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_str_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_str_q.qsize()))
        batch_num_total += batch_num
    LOGGER.info('call_mods process-{} ending, proceed {} batches({})'.format(os.getpid(), batch_num_total,
                                                                             args.batch_size))


def _write_predstr_to_file(write_fp, predstr_q, is_gzip):
    LOGGER.info('write_process-{} starts'.format(os.getpid()))
    if is_gzip:
        if not write_fp.endswith(".gz"):
            write_fp += ".gz"
        wf = gzip.open(write_fp, "wt")
    else:
        wf = open(write_fp, 'w')
    while True:
        # during test, it's ok without the sleep()
        if predstr_q.empty():
            time.sleep(time_wait)
            continue
        pred_str = predstr_q.get()
        if pred_str == "kill":
            wf.close()
            LOGGER.info('write_process-{} finished'.format(os.getpid()))
            break
        for one_pred_str in pred_str:
            wf.write(one_pred_str + "\n")
        wf.flush()


def _get_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpulist = list(range(num_gpus))
    else:
        gpulist = [0]
    return gpulist * 1000


def call_mods_txt(input_path, holeids_e, holeids_ne, 
                  out_per_readsite, model_path, args):
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
        LOGGER.warning("--threads must be > --threads_call + 2!!")
        nproc = nproc_dp + 2 + 1  # 1 for reading, 1 for writing, 1 for extracting
    nproc_cnvt = nproc - nproc_dp - 2

    p_read = mp.Process(target=_read_features_file_to_str, args=(input_path, featurestrs_batch_q,
                                                                    args.holes_batch))
    p_read.daemon = True
    p_read.start()

    ps_str2value = []
    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        for _ in range(nproc_cnvt):
            p = mp.Process(target=_format_features_from_strbatch2s, args=(featurestrs_batch_q,
                                                                          features_batch_q,
                                                                          args.seq_len,
                                                                          holeids_e, holeids_ne))
            p.daemon = True
            p.start()
            ps_str2value.append(p)
    else:
        raise ValueError("--model_type not right!")

    predstr_procs = []
    gpulist = _get_gpus()
    gpuindex = 0
    for _ in range(nproc_dp):
        p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                    args, gpulist[gpuindex]))
        gpuindex += 1
        p.daemon = True
        p.start()
        predstr_procs.append(p)

    # print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(out_per_readsite, pred_str_q, args.gzip))
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
