"""
call modifications from bam/sam files or extracted features.
output format: chromosome, pos, strand, read_name, read_loc, depth,
prob_0, prob_1, called_label, seq
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn import metrics

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
from .utils.process_utils import display_args
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode
from .utils.process_utils import str2bool
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import index_bam_if_needed2

from .utils.ref_reader import DNAReference

from .utils.constants_torch import FloatTensor
from .utils.constants_torch import use_cuda

from .extract_features import worker_read_split_holebatches_to_queue
from .extract_features import worker_extract_features_from_holebatches
from .extract_features import _get_holes

queue_size_border = 1000
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
    print("read_features process-{} starts".format(os.getpid()))
    h_num_total = _count_holenum(features_file)
    hbatch_num = h_num_total // holes_batch
    if h_num_total % holes_batch > 0:
        hbatch_num += 1
    print("read_features process-{} - generate {} hole/read batches({})\n".format(os.getpid(),
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
    print("read_features process-{} ending, read {} reads/holes batches({})".format(os.getpid(),
                                                                                    hbatch_num_got,
                                                                                    holes_batch))


def _format_features_from_strbatch2s(featurestrs_batch_q, features_batch_q, seq_len,
                                     holeids_e, holeids_ne):
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

            fkmer = np.array([base2code_dna[x] for x in words[5][left_cut:-right_cut]])
            fkmers.append(fkmer)
            fpasss.append(np.array([int(words[6])] * len(fkmer)))
            fipdms.append(np.array([float(x) for x in words[7].split(",")[left_cut:-right_cut]], dtype=np.float))
            fipdsds.append(np.array([float(x) for x in words[8].split(",")[left_cut:-right_cut]], dtype=np.float)
                           if words[8] != "." else 0)
            fpwms.append(np.array([float(x) for x in words[9].split(",")[left_cut:-right_cut]], dtype=np.float))
            fpwsds.append(np.array([float(x) for x in words[10].split(",")[left_cut:-right_cut]], dtype=np.float)
                          if words[10] != "." else 0)
            fquals.append(np.array([float(x) for x in words[11].split(",")[left_cut:-right_cut]], dtype=np.float))
            fmaps.append(np.array([float(x) for x in words[12].split(",")[left_cut:-right_cut]], dtype=np.float)
                         if words[12] != "." else 0)

            rkmer = np.array([base2code_dna[x] for x in words[13][left_cut:-right_cut]])
            rkmers.append(rkmer)
            rpasss.append(np.array([int(words[14])] * len(rkmer)))
            ripdms.append(np.array([float(x) for x in words[15].split(",")[left_cut:-right_cut]], dtype=np.float))
            ripdsds.append(np.array([float(x) for x in words[16].split(",")[left_cut:-right_cut]], dtype=np.float)
                           if words[16] != "." else 0)
            rpwms.append(np.array([float(x) for x in words[17].split(",")[left_cut:-right_cut]], dtype=np.float))
            rpwsds.append(np.array([float(x) for x in words[18].split(",")[left_cut:-right_cut]], dtype=np.float)
                          if words[18] != "." else 0)
            rquals.append(np.array([float(x) for x in words[19].split(",")[left_cut:-right_cut]], dtype=np.float))
            rmaps.append(np.array([float(x) for x in words[20].split(",")[left_cut:-right_cut]], dtype=np.float)
                         if words[20] != "." else 0)

            labels.append(int(words[21]))

        features_batch_q.put((sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fquals, fmaps,
                              rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rquals, rmaps, labels))
        while features_batch_q.qsize() > queue_size_border:
            time.sleep(time_wait)
    print("format_features process-{} ending, read {} batches".format(os.getpid(), b_num))


# call mods =============================================================
def _call_mods2s(features_batch, model, batch_size):
    sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fquals, fmaps, \
        rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rquals, rmaps, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]

        b_fkmers = fkmers[batch_s:batch_e]
        b_fpasss = fpasss[batch_s:batch_e]
        b_fipdms = fipdms[batch_s:batch_e]
        b_fipdsds = fipdsds[batch_s:batch_e]
        b_fpwms = fpwms[batch_s:batch_e]
        b_fpwsds = fpwsds[batch_s:batch_e]
        b_fquals = fquals[batch_s:batch_e]
        b_fmaps = fmaps[batch_s:batch_e]

        b_rkmers = rkmers[batch_s:batch_e]
        b_rpasss = rpasss[batch_s:batch_e]
        b_ripdms = ripdms[batch_s:batch_e]
        b_ripdsds = ripdsds[batch_s:batch_e]
        b_rpwms = rpwms[batch_s:batch_e]
        b_rpwsds = rpwsds[batch_s:batch_e]
        b_rquals = rquals[batch_s:batch_e]
        b_rmaps = rmaps[batch_s:batch_e]

        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_fkmers), FloatTensor(b_fpasss), FloatTensor(b_fipdms),
                                      FloatTensor(b_fipdsds), FloatTensor(b_fpwms), FloatTensor(b_fpwsds),
                                      FloatTensor(b_fquals), FloatTensor(b_fmaps),
                                      FloatTensor(b_rkmers), FloatTensor(b_rpasss), FloatTensor(b_ripdms),
                                      FloatTensor(b_ripdsds), FloatTensor(b_rpwms), FloatTensor(b_rpwsds),
                                      FloatTensor(b_rquals), FloatTensor(b_rmaps))
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
                # chromosome, pos, strand, holeid, loc, depth, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(prob_1 / (prob_0 + prob_1), 6)
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_fkmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer)/2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                pred_str.append("\t".join([b_sampleinfo[idx],
                                           str(b_fpasss[idx][0]) + "," + str(b_rpasss[idx][0]),
                                           str(prob_0_norm), str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1
    accuracy = np.mean(accuracys) if len(accuracys) > 0 else 0.

    return pred_str, accuracy, batch_num


def _call_mods_q(model_path, features_batch_q, pred_str_q, args):
    print('call_mods process-{} starts'.format(os.getpid()))
    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            is_qual=str2bool(args.is_qual),
                            is_map=str2bool(args.is_map),
                            is_stds=str2bool(args.is_stds),
                            is_npass=str2bool(args.is_npass),
                            model_type=args.model_type)
    else:
        raise ValueError("--model_type not right!")

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

        if args.model_type in {"attbigru2s", "attbilstm2s"}:
            pred_str, accuracy, batch_num = _call_mods2s(features_batch, model, args.batch_size)
        else:
            raise ValueError("--model_type not right!")

        pred_str_q.put(pred_str)
        while pred_str_q.qsize() > queue_size_border:
            time.sleep(time_wait)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_str_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_str_q.qsize()))
        accuracy_list.append(accuracy)
        batch_num_total += batch_num
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    print('call_mods process-{} ending, proceed {} batches({})'.format(os.getpid(), batch_num_total,
                                                                       args.batch_size))


def _write_predstr_to_file(write_fp, predstr_q, is_gzip):
    print('write_process-{} starts'.format(os.getpid()))
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
            print('write_process-{} finished'.format(os.getpid()))
            break
        for one_pred_str in pred_str:
            wf.write(one_pred_str + "\n")
        wf.flush()


def call_mods(args):
    print("[main]call_mods starts..")
    start = time.time()
    print("cuda availability: {}".format(use_cuda))

    torch.manual_seed(args.tseed)
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    model_path = os.path.abspath(args.model_file)
    if not os.path.exists(model_path):
        raise ValueError("--model_file is not set right!")
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise ValueError("--input_file does not exist!")

    out_per_readsite = args.output + ".per_readsite.tsv"

    holeids_e = None if args.holeids_e is None else _get_holes(args.holeids_e)
    holeids_ne = None if args.holeids_ne is None else _get_holes(args.holeids_ne)

    if input_path.endswith(".bam") or input_path.endswith(".sam"):
        if args.seq_len % 2 == 0:
            raise ValueError("--seq_len must be odd")

        if str2bool(args.is_map) and not str2bool(args.is_mapfea):
            print("as --is_map is True, setting --is_mapfea as True")
            args.is_mapfea = "yes"

        index_bam_if_needed2(input_path, args.threads)

        dnacontigs = None
        if args.mode == "reference":
            if args.ref is None:
                raise ValueError("--ref must be provided when using reference mode!")
            reference = os.path.abspath(args.ref)
            if not os.path.exists(reference):
                raise IOError("refernce(--ref) file does not exist!")
            dnacontigs = DNAReference(reference).getcontigs()

        motifs = get_motif_seqs(args.motifs)

        hole_batch_q = Queue()
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

        p_read = mp.Process(target=worker_read_split_holebatches_to_queue,
                            args=(input_path, hole_batch_q, args))
        p_read.daemon = True
        p_read.start()

        p_w = mp.Process(target=_write_predstr_to_file, args=(out_per_readsite, pred_str_q, args.gzip))
        p_w.daemon = True
        p_w.start()

        # TODO: why the processes in ps_extract start so slowly?
        ps_extract = []
        ps_call = []
        nproc_ext = nproc - nproc_dp - 2
        for i in range(max(nproc_ext, nproc_dp)):
            if i < nproc_ext:
                p = mp.Process(target=worker_extract_features_from_holebatches,
                               args=(hole_batch_q, features_batch_q,
                                     motifs, holeids_e, holeids_ne, dnacontigs,
                                     args, False, True))
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

        if str2bool(args.modbam):
            from ._bam2modbam import add_mm_ml_tags_to_bam
            out_modbam = args.output + ".modbam.bam"
            add_mm_ml_tags_to_bam(input_path, out_per_readsite, out_modbam,
                                  rm_pulse=True, threads=args.threads)
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
        for _ in range(nproc_dp):
            p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q, args))
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

    print("[main]call_mods costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input", "-i", action="store", type=str,
                         required=True,
                         help="input file, can be bam/sam, or features.tsv generated by "
                              "extract_features.py.")
    p_input.add_argument("--holes_batch", type=int, default=50, required=False,
                         help="number of holes/hifi-reads in an batch to get/put in queues, default 50. "
                              "only used when --input is bam/sam")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_file", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    # model param
    p_call.add_argument('--model_type', type=str, default="attbigru2s",
                        choices=["attbilstm2s", "attbigru2s"],
                        required=False,
                        help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                             "default: attbigru2s")
    p_call.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")
    p_call.add_argument('--is_npass', type=str, default="yes", required=False,
                        help="if using num_pass features, yes or no, default yes")
    p_call.add_argument('--is_qual', type=str, default="no", required=False,
                        help="if using base_quality features, yes or no, default no")
    p_call.add_argument('--is_map', type=str, default="no", required=False,
                        help="if using mapping features, yes or no, default no")
    p_call.add_argument('--is_stds', type=str, default="no", required=False,
                        help="if using std features, yes or no, default no")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")

    # BiRNN model param
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    p_call.add_argument('--layer_rnn', type=int, default=3,
                        required=False, help="BiRNN layer num, default 3")
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiRNN hidden_size for combined feature")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", action="store", type=str, required=True,
                          help="the prefix of output files to save the predicted results. "
                               "output files will be [--output].per_readsite.tsv/.modbam.bam")
    p_output.add_argument("--gzip", action="store_true", default=False, required=False,
                          help="if compressing .per_readsite.tsv using gzip")
    p_output.add_argument("--modbam", type=str, default="yes", required=False,
                          help="if generating modbam file when --input is in bam/sam format. "
                               "yes or no, default yes")

    p_extract = parser.add_argument_group("EXTRACTION")
    p_extract.add_argument("--mode", type=str, default="align", required=False,
                           choices=["denovo", "align"],
                           help="denovo mode: extract features from unaligned hifi.bam;\n"
                                "align mode: extract features from aligned hifi.bam. default: align")
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
    p_extract.add_argument("--norm", action="store", type=str, choices=["zscore", "min-mean", "min-max", "mad"],
                           default="zscore", required=False,
                           help="method for normalizing ipd/pw in subread level. "
                                "zscore, min-mean, min-max or mad, default zscore")
    p_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                           help="not use CodecV1 to decode ipd/pw")
    # p_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
    #                        help="full path to the executable binary samtools file. "
    #                             "If not specified, it is assumed that samtools is in "
    #                             "the PATH.")
    p_extract.add_argument("--loginfo", type=str, default="no", required=False,
                           help="if printing more info of feature extraction on reads. "
                                "yes or no, default no")

    p_extract_ref = parser.add_argument_group("EXTRACTION ALIGN_MODE")
    p_extract_ref.add_argument("--ref", type=str, required=False,
                               help="path to genome reference to be aligned, in fasta/fa format.")
    p_extract_ref.add_argument("--mapq", type=int, default=10, required=False,
                               help="MAPping Quality cutoff for selecting alignment items, default 10")
    p_extract_ref.add_argument("--identity", type=float, default=0.75, required=False,
                               help="identity cutoff for selecting alignment items, default 0.75")
    p_extract_ref.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                               help="not use supplementary alignment")
    p_extract_ref.add_argument("--is_mapfea", type=str, default="no", required=False,
                               help="if extract mapping features, yes or no, default no")
    p_extract_ref.add_argument("--skip_unmapped", type=str, default="yes", required=False,
                               help="if skipping unmapped sites in reads, yes or no, default yes")

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
