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
# import threading
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
except AttributeError:
    pass

# from utils.process_utils import Queue
from torch.multiprocessing import Queue
import time

import pysam

from collections import OrderedDict

from .models import ModelAttRNN
from .models import ModelTransEnc
from .models import ModelAttRNN2

from .utils.process_utils import base2code_dna
from .utils.process_utils import display_args
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode
from .utils.process_utils import str2bool
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import index_bam_if_needed2
# from .utils.process_utils import max_queue_size
from .utils.process_utils import complement_seq

from .utils.ref_reader import DNAReference

from .utils.constants_torch import FloatTensor
from .utils.constants_torch import use_cuda

from .extract_features import worker_read_split_holebatches_to_queue
from .extract_features import process_one_holebatch
from .extract_features import _get_holes
from .extract_features import _open_inputfile

from ._bam2modbam import _get_necessary_alignment_items
from ._bam2modbam import _convert_locs_to_mmtag
from ._bam2modbam import _convert_probs_to_mltag
from ._bam2modbam import _refill_tags

from ._version import VERSION

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

# add this export temporarily
# https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# queue_size_border = max_queue_size
time_wait = 0.2


# extract features ======================================================
def _batch_feature_list2s(feature_list):
    sampleinfo = []  # contains: chrom, abs_loc, strand, holeid, loc

    fkmers = []
    fpasss = []
    fipdms = []
    fipdsds = []
    fpwms = []
    fpwsds = []
    fsns = []
    fmaps = []

    rkmers = []
    rpasss = []
    ripdms = []
    ripdsds = []
    rpwms = []
    rpwsds = []
    rsns = []
    rmaps = []

    labels = []
    for featureline in feature_list:
        chrom, abs_loc, strand, holeid, loc, \
            kmer_seq, kmer_pass, kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_sn, kmer_map, \
            kmer_seq2, kmer_pass2, kmer_ipdm2, kmer_ipds2, kmer_pwm2, kmer_pws2, kmer_sn2, kmer_map2, \
            label = featureline

        sampleinfo.append("\t".join(list(map(str, [chrom, abs_loc, strand, holeid, loc]))))

        fkmers.append(np.array([base2code_dna[x] for x in kmer_seq]))
        fpasss.append(np.array([kmer_pass] * len(kmer_seq)))
        fipdms.append(np.array(kmer_ipdm, dtype=float))
        fipdsds.append(np.array(kmer_ipds, dtype=float) if type(kmer_ipds) is not str else 0)
        fpwms.append(np.array(kmer_pwm, dtype=float))
        fpwsds.append(np.array(kmer_pws, dtype=float) if type(kmer_pws) is not str else 0)
        fsns.append(np.array(kmer_sn, dtype=float) if type(kmer_sn) is not str else 0)
        fmaps.append(np.array(kmer_map, dtype=float) if type(kmer_map) is not str else 0)

        rkmers.append(np.array([base2code_dna[x] for x in kmer_seq2]))
        rpasss.append(np.array([kmer_pass2] * len(kmer_seq2)))
        ripdms.append(np.array(kmer_ipdm2, dtype=float))
        ripdsds.append(np.array(kmer_ipds2, dtype=float) if type(kmer_ipds2) is not str else 0)
        rpwms.append(np.array(kmer_pwm2, dtype=float))
        rpwsds.append(np.array(kmer_pws2, dtype=float) if type(kmer_pws2) is not str else 0)
        rsns.append(np.array(kmer_sn2, dtype=float) if type(kmer_sn2) is not str else 0)
        rmaps.append(np.array(kmer_map2, dtype=float) if type(kmer_map2) is not str else 0)

        labels.append(label)
    return sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fsns, fmaps, \
        rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rsns, rmaps, labels


def worker_extract_features_with_holeinfo(input_header, holebatch_q, features_q,
                                          motifs, holeids_e, holeids_ne, dnacontigs, args):
    LOGGER.info("extract_features process-{} starts".format(os.getpid()))

    if isinstance(input_header, OrderedDict) or isinstance(input_header, dict):
        input_header2 = pysam.AlignmentHeader.from_dict(input_header)
    else:
        input_header2 = input_header

    cnt_holesbatch = 0
    total_num_batch, failed_num_batch = 0, 0
    while True:
        if holebatch_q.empty():
            time.sleep(time_wait)
            continue
        holebatch = holebatch_q.get()
        if holebatch == "kill":
            holebatch_q.put("kill")
            break
        # handle one holebatch
        holeidxes, feature_list, total_num, failed_num = process_one_holebatch(input_header2, holebatch,
                                                                               motifs, holeids_e, holeids_ne,
                                                                               dnacontigs,
                                                                               args)
        total_num_batch += total_num
        failed_num_batch += failed_num
        if len(feature_list) > 0:
            feature_batch = _batch_feature_list2s(feature_list)
            features_oneholebatch = (holebatch, holeidxes, feature_batch)
            features_q.put(features_oneholebatch)
            # while features_q.qsize() > queue_size_border:
            while features_q.qsize() > (args.threads if args.threads > 1 else 2) * 3:
                time.sleep(time_wait)
        cnt_holesbatch += 1
    LOGGER.info("extract_features process-{} ending, proceed {} "
                "hole_batches({}): {} holes/reads in total, "
                "{} skipped/failed.".format(os.getpid(),
                                            cnt_holesbatch,
                                            args.holes_batch,
                                            total_num_batch,
                                            failed_num_batch))


# call mods =============================================================
def _call_mods2s(features_batch, model, batch_size, device=0):
    sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fsns, fmaps, \
        rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rsns, rmaps, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_info = []
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
        b_fsns = np.array(fsns[batch_s:batch_e])
        b_fmaps = np.array(fmaps[batch_s:batch_e])

        b_rkmers = np.array(rkmers[batch_s:batch_e])
        b_rpasss = np.array(rpasss[batch_s:batch_e])
        b_ripdms = np.array(ripdms[batch_s:batch_e])
        b_ripdsds = np.array(ripdsds[batch_s:batch_e])
        b_rpwms = np.array(rpwms[batch_s:batch_e])
        b_rpwsds = np.array(rpwsds[batch_s:batch_e])
        b_rsns = np.array(rsns[batch_s:batch_e])
        b_rmaps = np.array(rmaps[batch_s:batch_e])

        # b_labels = np.array(labels[batch_s:batch_e])
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_fkmers, device), FloatTensor(b_fpasss, device),
                                      FloatTensor(b_fipdms, device), FloatTensor(b_fipdsds, device),
                                      FloatTensor(b_fpwms, device), FloatTensor(b_fpwsds, device),
                                      FloatTensor(b_fsns, device), FloatTensor(b_fmaps, device),
                                      FloatTensor(b_rkmers, device), FloatTensor(b_rpasss, device),
                                      FloatTensor(b_ripdms, device), FloatTensor(b_ripdsds, device),
                                      FloatTensor(b_rpwms, device), FloatTensor(b_rpwsds, device),
                                      FloatTensor(b_rsns, device), FloatTensor(b_rmaps, device))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            # predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, loc, depth, prob_0, prob_1, called_label, seq
                b_sampleinfo[idx] = b_sampleinfo[idx].split("\t")
                holeid = b_sampleinfo[idx][3]
                loc = int(b_sampleinfo[idx][4])
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_1_norm = round(prob_1 / (prob_0 + prob_1), 6)
                pred_info.append((holeid, loc, prob_1_norm))
            batch_num += 1

    return pred_info, batch_num


def _add_modinfo2alignedseg(read_info, pred_info, input_header, rm_pulse):
    segment_tmp = pysam.AlignedSegment.from_dict(read_info, input_header)

    seq_name, flag, ref_name, ref_start, mapq, cigartuples, rnext, pnext, \
        tlen, seq_seq, seq_qual, all_tags, is_reverse = _get_necessary_alignment_items(segment_tmp)

    # MM: Base modifications / methylation, ML:Base modification probabilities tags
    mm_values = ml_values = None
    mm_flag = 0
    if len(pred_info) == 0:
        new_tags = _refill_tags(all_tags, mm_values, ml_values, rm_pulse)
        return (seq_name, flag, ref_name, ref_start, mapq, cigartuples, rnext, pnext, tlen,
                seq_seq, seq_qual, new_tags, mm_flag)
    
    seq_fwdseq = complement_seq(seq_seq) if is_reverse else seq_seq
    locs_probs = []
    for pred in pred_info:
        holeid_tmp, loc_tmp, prob_tmp = pred
        if holeid_tmp == seq_name:
            locs_probs.append((loc_tmp, prob_tmp))
        else:
            raise ValueError("holeid_tmp != seq_name, {} != {}".format(holeid_tmp, seq_name))
    try:
        locs_probs = sorted(locs_probs, key=lambda x: x[0])
        locs_probs = list(zip(*locs_probs))
        locs = locs_probs[0]
        probs = locs_probs[1]
        mm_values = _convert_locs_to_mmtag(locs, seq_fwdseq)
        ml_values = _convert_probs_to_mltag(probs)
        mm_flag = 1
    except AssertionError:
        # sys.stderr.write("AssertionError, skip this alignment.\n"
        #       "\tDetails: {}, {}, {}\n".format(seq_name, locs, probs))
        LOGGER.info("AssertionError, skip this alignment-{}.".format(seq_name))
    new_tags = _refill_tags(all_tags, mm_values, ml_values, rm_pulse)
    return (seq_name, flag, ref_name, ref_start, mapq, cigartuples, rnext, pnext, tlen,
            seq_seq, seq_qual, new_tags, mm_flag)


def _add_modinfo2alignedseg_batch(holebatch, holeidxes, preds_info, input_header, rm_pulse):
    new_read_infos = []

    def catch_up(holebatch_cnt, cur_holeid):
        while holebatch_cnt < cur_holeid:
            new_read_info = _add_modinfo2alignedseg(holebatch[holebatch_cnt], [], input_header, rm_pulse)
            new_read_infos.append(new_read_info)
            holebatch_cnt += 1
        return holebatch_cnt
    
    if len(holeidxes) == 0:
        for idx, read_info in enumerate(holebatch):
            new_read_info = _add_modinfo2alignedseg(read_info, [], input_header, rm_pulse)
            new_read_infos.append(new_read_info)
        return new_read_infos

    assert len(holeidxes) == len(preds_info)
    holebatch_cnt = 0
    pred_info_tmp = []
    cur_holeid = holeidxes[0]
    for idx, holeid in enumerate(holeidxes):
        if holeid != cur_holeid:
            holebatch_cnt = catch_up(holebatch_cnt, cur_holeid)
            assert holebatch_cnt == cur_holeid
            new_read_info = _add_modinfo2alignedseg(holebatch[holebatch_cnt], pred_info_tmp, input_header, rm_pulse)
            new_read_infos.append(new_read_info)
            pred_info_tmp = []
            holebatch_cnt += 1
            cur_holeid = holeid
        pred_info_tmp.append(preds_info[idx])
    # last pred_info
    holebatch_cnt = catch_up(holebatch_cnt, cur_holeid)
    assert holebatch_cnt == cur_holeid
    new_read_info = _add_modinfo2alignedseg(holebatch[holebatch_cnt], pred_info_tmp, input_header, rm_pulse)
    new_read_infos.append(new_read_info)
    # last holebatches
    if holebatch_cnt < len(holebatch) - 1:
        for idx in range(holebatch_cnt + 1, len(holebatch)):
            new_read_info = _add_modinfo2alignedseg(holebatch[idx], [], input_header, rm_pulse)
            new_read_infos.append(new_read_info)
    assert len(holebatch) == len(new_read_infos)
    return new_read_infos


def _call_mods_q(model_path, features_batch_q, out_info_q, input_header, args, device=0):
    LOGGER.info('call_mods process-{} starts'.format(os.getpid()))
    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            is_sn=str2bool(args.is_sn),
                            is_map=str2bool(args.is_map),
                            is_stds=str2bool(args.is_stds),
                            is_npass=str2bool(args.is_npass),
                            model_type=args.model_type,
                            device=device)
    elif args.model_type in {"attbigru2s2", "attbilstm2s2"}:
        model = ModelAttRNN2(args.seq_len, args.layer_rnn, args.class_num,
                             args.dropout_rate, args.hid_rnn,
                             is_sn=str2bool(args.is_sn),
                             is_map=str2bool(args.is_map),
                             is_stds=str2bool(args.is_stds),
                             is_npass=str2bool(args.is_npass),
                             model_type=args.model_type,
                             device=device)
    elif args.model_type in {"transencoder2s"}:
        model = ModelTransEnc(args.seq_len, args.layer_trans, args.class_num,
                              args.dropout_rate, args.d_model, args.nhead, args.dim_ff,
                              is_npass=str2bool(args.is_npass), is_sn=str2bool(args.is_sn),
                              is_map=str2bool(args.is_map), is_stds=str2bool(args.is_stds), 
                              model_type=args.model_type, device=device)
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
        para_dict = torch.load(model_path, map_location=torch.device('cpu'))
        para_dict_new = OrderedDict()
        for param_tensor in para_dict.keys():
            para_dict_new[param_tensor[7:]] = para_dict[param_tensor]
        model.load_state_dict(para_dict_new)
        LOGGER.debug('call_mods process-{} loads model param successfully-1'.format(os.getpid()))
        del para_dict_new
    # sys.stdout.flush()
    
    if str2bool(args.use_compile):
        try:
            model = torch.compile(model)
        except:
            raise ImportError('torch.compile does not exist in PyTorch<2.0.')

    if use_cuda:
        model = model.cuda(device)
    model.eval()

    if isinstance(input_header, OrderedDict) or isinstance(input_header, dict):
        input_header2 = pysam.AlignmentHeader.from_dict(input_header)
    else:
        input_header2 = input_header
    rm_pulse = not args.keep_pulse
    
    batch_num_total = 0
    while True:

        if features_batch_q.empty():
            time.sleep(time_wait)
            continue

        features_batch = features_batch_q.get()
        if features_batch == "kill":
            features_batch_q.put("kill")
            break

        holebatch, holeidxes, features_oneholebatch = features_batch
        if args.model_type in {"attbigru2s", "attbilstm2s", "transencoder2s", "attbigru2s2", "attbilstm2s2"}:
            pred_info, batch_num = _call_mods2s(features_oneholebatch, model, args.batch_size, device)
            del features_oneholebatch
        else:
            raise ValueError("--model_type not right!")
        
        new_read_infos = _add_modinfo2alignedseg_batch(holebatch, holeidxes, pred_info, input_header2, rm_pulse)
        out_info_q.put(new_read_infos)
        # while out_info_q.qsize() > queue_size_border:
        while out_info_q.qsize() > (args.threads if args.threads > 1 else 2) * 3:
            time.sleep(time_wait)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_info_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_info_q.qsize()))
        batch_num_total += batch_num
    LOGGER.info('call_mods process-{} ending, proceed {} batches({})'.format(os.getpid(), batch_num_total,
                                                                             args.batch_size))


# write modbam =============================================================
def write_alignedsegment(readitem_info, output_bam):
    """
    Writes the readitem_info(tuple) to a bam file
    :param readitem_info:
    :param output_bam:
    :return:
    """
    seq_name, flag, ref_name, ref_start, mapq, cigartuples, \
        rnext, pnext, tlen, seq_seq, seq_qual, all_tags, _ = readitem_info

    out_read = pysam.AlignedSegment(output_bam.header)
    out_read.query_name = seq_name
    out_read.flag = flag
    out_read.reference_name = ref_name
    out_read.reference_start = ref_start
    out_read.mapping_quality = mapq
    out_read.cigar = cigartuples
    out_read.next_reference_name = rnext
    out_read.next_reference_start = pnext
    out_read.template_length = tlen
    out_read.query_sequence = seq_seq
    out_read.query_qualities = seq_qual
    if len(all_tags) >= 1:
        out_read.set_tags(all_tags)
    output_bam.write(out_read)


def _worker_write_modbam(wreads_q, modbamfile, inputheader, threads=1):
    if not (isinstance(inputheader, OrderedDict) or isinstance(inputheader, dict)):
        header2 = inputheader.to_dict()
    else:
        from copy import deepcopy
        header2 = deepcopy(inputheader)
    # try adding PG tag here
    # MUST have the ID entry
    header2["PG"].append({"PN": "ccsmeth", "ID": "ccsmeth", "VN": VERSION, "CL": " ".join(sys.argv)})

    w_bam = pysam.AlignmentFile(modbamfile, "wb", header=header2, threads=threads)
    cnt_w, cnt_mm = 0, 0
    while True:
        if wreads_q.empty():
            time.sleep(time_wait)
            continue
        wreads_batch = wreads_q.get()
        if wreads_batch == "kill":
            w_bam.close()
            LOGGER.info("wrote {} reads, in which {} were added mm tags".format(cnt_w, cnt_mm))
            break
        for walignseg in wreads_batch:
            mm_flag = walignseg[-1]
            write_alignedsegment(walignseg, w_bam)
            cnt_w += 1
            cnt_mm += mm_flag


def _get_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpulist = list(range(num_gpus))
    else:
        gpulist = [0]
    return gpulist * 1000


def call_mods(args):
    LOGGER.info("[main]call_mods starts")
    start = time.time()
    LOGGER.info("cuda availability: {}".format(use_cuda))

    torch.manual_seed(args.tseed)
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    model_path = os.path.abspath(args.model_file)
    if not os.path.exists(model_path):
        raise ValueError("--model_file is not set right!")
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise ValueError("--input_file does not exist!")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_modbam = args.output + ".modbam.bam"

    holeids_e = None if args.holeids_e is None else _get_holes(args.holeids_e)
    holeids_ne = None if args.holeids_ne is None else _get_holes(args.holeids_ne)

    if input_path.endswith(".bam") or input_path.endswith(".sam"):
        if args.seq_len % 2 == 0:
            raise ValueError("--seq_len must be odd")

        if str2bool(args.is_map) and not str2bool(args.is_mapfea):
            LOGGER.info("as --is_map is True, setting --is_mapfea as True")
            args.is_mapfea = "yes"

        index_bam_if_needed2(input_path, args.threads)

        dnacontigs = None
        if args.mode == "align":
            if args.ref is None:
                raise ValueError("--ref must be provided when using align mode!")
            reference = os.path.abspath(args.ref)
            if not os.path.exists(reference):
                raise IOError("refernce(--ref) file does not exist!")
            dnacontigs = DNAReference(reference).getcontigs()

        motifs = get_motif_seqs(args.motifs)

        hole_batch_q = Queue()
        features_batch_q = Queue()
        out_info_q = Queue()

        nproc = args.threads
        nproc_dp = args.threads_call
        if use_cuda:
            if nproc_dp < 1:
                nproc_dp = 1
        else:
            if nproc_dp > nproc_to_call_mods_in_cpu_mode:
                nproc_dp = nproc_to_call_mods_in_cpu_mode
        if nproc <= nproc_dp + 4:
            LOGGER.warning("--threads must be > --threads_call + 4!!")
            nproc = nproc_dp + 4 + 1  # 2 for reading, 2 for writing, 1 for extracting
            threads_r, threads_w = 2, 2
        else:
            threads_r = 2
            if nproc - nproc_dp - threads_r >= 16:
                threads_w = 4
            elif nproc - nproc_dp - threads_r >= 12:
                threads_w = 3
            else:
                threads_w = 2

        inputreads = _open_inputfile(input_path, args.mode, threads=args.threads)
        input_header = inputreads.header.to_dict()
        inputreads.close()

        p_read = mp.Process(target=worker_read_split_holebatches_to_queue,
                            args=(input_path, hole_batch_q, threads_r, args))
        p_read.daemon = True
        p_read.start()

        p_w = mp.Process(target=_worker_write_modbam, args=(out_info_q, out_modbam, input_header, threads_w))
        p_w.daemon = True
        p_w.start()

        # TODO: why the processes in ps_extract start so slowly?
        ps_extract = []
        ps_call = []
        nproc_ext = nproc - nproc_dp - threads_r - threads_w
        gpulist = _get_gpus()
        gpuindex = 0
        for i in range(max(nproc_ext, nproc_dp)):
            if i < nproc_ext:
                p = mp.Process(target=worker_extract_features_with_holeinfo,
                               args=(input_header, hole_batch_q, features_batch_q,
                                     motifs, holeids_e, holeids_ne, dnacontigs, args))
                p.daemon = True
                p.start()
                ps_extract.append(p)
            if i < nproc_dp:
                p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, out_info_q, 
                                                          input_header, args, gpulist[gpuindex]))
                gpuindex += 1
                p.daemon = True
                p.start()
                ps_call.append(p)
        
        p_read.join()

        for p in ps_extract:
            p.join()
        features_batch_q.put("kill")

        for p in ps_call:
            p.join()
        out_info_q.put("kill")

        p_w.join()

        if not args.no_sort:
            post_time_start = time.time()
            LOGGER.info("[post_process] bam_sort_index starts")
            try:
                LOGGER.info("sorting modbam file..")
                modbam_sorted = os.path.splitext(out_modbam)[0] + ".sorted.bam"
                pysam.sort("-o", modbam_sorted, "-@", str(args.threads), out_modbam)
                os.rename(modbam_sorted, out_modbam)
            except Exception:
                LOGGER.warning("failed sorting modbam file..")
            try:
                LOGGER.info("indexing modbam file..")
                pysam.index("-@", str(args.threads), out_modbam)
            except Exception:
                LOGGER.warning("failed indexing modbam file..")
            LOGGER.info("[post_process] bam_sort_index costs %.2f seconds" % (time.time() - post_time_start))
    else:
        from ._call_modifications_txt import call_mods_txt
        out_per_readsite = args.output + ".per_readsite.tsv"
        call_mods_txt(input_path, holeids_e, holeids_ne, out_per_readsite, model_path, args)

    LOGGER.info("[main]call_mods costs %.2f seconds" % (time.time() - start))


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
    
    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", action="store", type=str, required=True,
                          help="the prefix of output files to save the predicted results. "
                               "output files will be [--output].per_readsite.tsv/.modbam.bam")
    p_output.add_argument("--gzip", action="store_true", default=False, required=False,
                          help="if compressing .per_readsite.tsv when --input is not in bam/sam format.")
    p_output.add_argument("--keep_pulse", action="store_true", default=False, required=False,
                          help="if keeping ipd/pw tags in .modbam.bam when --input is in bam/sam format.")
    p_output.add_argument("--no_sort", action="store_true", default=False, required=False,
                          help="don't sort .modbam.bam when --input is in bam/sam format.")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_file", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")
    # model param
    p_call.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s", "transencoder2s", 
                                   "attbilstm2s2", "attbigru2s2",],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "'transencoder2s', 'attbilstm2s2', 'attbigru2s2', "
                               "default: attbigru2s")
    p_call.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")
    p_call.add_argument('--is_npass', type=str, default="yes", required=False,
                        help="if using num_pass features, yes or no, default yes")
    p_call.add_argument('--is_stds', type=str, default="no", required=False,
                        help="if using std features, yes or no, default no")
    
    p_call.add_argument('--is_sn', type=str, default="no", required=False,
                        help="if using signal-to-noise-ratio features, yes or no, default no. "
                             "Effects both MODEL input and feature EXTRACTION")
    p_call.add_argument('--is_map', type=str, default="no", required=False,
                        help="if using mapping features, yes or no, default no. "
                             "Effects both MODEL input and feature EXTRACTION, "
                             "only works in EXTRACTION-ALIGN-MODE")

    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")
    
    p_callb = parser.add_argument_group("CALL MODEL_HYPER RNN")
    # BiRNN model param
    p_callb.add_argument('--layer_rnn', type=int, default=3,
                         required=False, help="BiRNN layer num, default 3")
    p_callb.add_argument('--hid_rnn', type=int, default=256, required=False,
                         help="BiRNN hidden_size, default 256")
    
    p_callt = parser.add_argument_group("CALL MODEL_HYPER TRANSFORMER")
    # Transformer model param
    p_callt.add_argument('--layer_trans', type=int, default=6, required=False,
                         help="TransformerEncoder nlayers, default 6")
    p_callt.add_argument('--nhead', type=int, default=4, required=False,
                         help="TransformerEncoder nhead, default 4")
    p_callt.add_argument('--d_model', type=int, default=256, required=False, 
                         help="TransformerEncoder input feature numbers, default 256")
    p_callt.add_argument('--dim_ff', type=int, default=512, required=False,
                         help="TransformerEncoder dim_feedforward, default 512")

    p_extract = parser.add_argument_group("EXTRACTION")
    p_extract.add_argument("--mode", type=str, default="denovo", required=False,
                           choices=["denovo", "align"],
                           help="denovo mode: extract features from unaligned/aligned hifi.bam without "
                                "reference position info;\n"
                                "align mode: extract features from aligned hifi.bam with "
                                "reference position info. default: denovo")
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
    p_extract.add_argument("--norm", action="store", type=str, 
                           choices=["zscore", "min-mean", "min-max", "mad", "none"],
                           default="zscore", required=False,
                           help="method for normalizing ipd/pw in subread level. "
                                "zscore, min-mean, min-max, mad, or none. default zscore")
    p_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                           help="not use CodecV1 to decode ipd/pw")
    # p_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
    #                        help="full path to the executable binary samtools file. "
    #                             "If not specified, it is assumed that samtools is in "
    #                             "the PATH.")

    p_extract_ref = parser.add_argument_group("EXTRACTION ALIGN_MODE")
    p_extract_ref.add_argument("--ref", type=str, required=False,
                               help="path to genome reference to be aligned, in fasta/fa format.")
    p_extract_ref.add_argument("--mapq", type=int, default=1, required=False,
                               help="MAPping Quality cutoff for selecting alignment items, default 1")
    p_extract_ref.add_argument("--identity", type=float, default=0.0, required=False,
                               help="identity cutoff for selecting alignment items, [0.0, 1.0], default 0.0")
    p_extract_ref.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                               help="not use supplementary alignment")
    p_extract_ref.add_argument("--skip_unmapped", type=str, default="yes", required=False,
                               help="if skipping unmapped sites in reads, yes or no, default yes")

    parser.add_argument("--threads", "-p", action="store", type=int, default=10,
                        required=False, help="number of threads to be used, default 10.")
    parser.add_argument("--threads_call", action="store", type=int, default=3,
                        required=False, help="number of threads used to call modifications with trained "
                                             "models, no more than threads/3 is suggested. default 3.")
    parser.add_argument('--tseed', type=int, default=1234,
                        help='random seed for torch')
    parser.add_argument('--use_compile', type=str, default="no", required=False,
                        help="if using torch.compile, yes or no, default no ('yes' only works in pytorch>=2.0)")

    args = parser.parse_args()
    display_args(args)

    call_mods(args)


if __name__ == '__main__':
    sys.exit(main())
