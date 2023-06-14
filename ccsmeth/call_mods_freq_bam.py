#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

from __future__ import absolute_import

import argparse
import os
import sys
import time
import numpy as np

import multiprocessing as mp
from multiprocessing import Queue
from tqdm import tqdm
import pysam
import pybedtools
import re

from .utils.ref_reader import DNAReference
from .utils.process_utils import compute_pct_identity
from .utils.process_utils import is_file_empty
from .utils.process_utils import index_bam_if_needed2
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import complement_seq
from .utils.process_utils import get_refloc_of_methysite_in_motif

from numpy.lib.stride_tricks import sliding_window_view

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

time_wait = 1
key_sep = "||"


def _check_input_file(inputpath):
    if not inputpath.endswith(".bam"):
        raise ValueError("--input_bam not a bam file!")
    if not os.path.exists(inputpath):
        raise ValueError("--input_bam does not exist!")
    return os.path.abspath(inputpath)


def _read_file_lines(cfile):
    with open(cfile, "r") as rf:
        return rf.read().splitlines()


def _get_reference_chunks(dnacontigs, contig_str, chunk_len=300000, motifs="CG"):
    if contig_str is not None:
        if os.path.isfile(contig_str):
            contigs = sorted(list(set(_read_file_lines(contig_str))))
        else:
            contigs = sorted(list(set(contig_str.strip().split(","))))
    else:
        contigs = sorted(list(dnacontigs.keys()))
    ref_chunks = []
    for contig in contigs:
        contig_len = len(dnacontigs[contig])
        for i in np.arange(0, contig_len, chunk_len):
            if i + chunk_len < contig_len:
                istart, iend = i, i + chunk_len
            else:
                istart, iend = i, contig_len
            ref_chunks.append((contig, istart, iend))
    # adjust start, end if motifs=='CG'
    if motifs == "CG":
        LOGGER.info("adjust regions for CG motif")
        for idx in range(1, len(ref_chunks)):
            pre_ref, pre_s, pre_e = ref_chunks[idx-1]
            cur_ref, cur_s, cur_e = ref_chunks[idx]
            if pre_ref != cur_ref:
                continue
            assert cur_s == pre_e
            if dnacontigs[pre_ref][(pre_e-1):(pre_e+1)] == "CG":
                ref_chunks[idx-1] = (pre_ref, pre_s, pre_e + 1)
                ref_chunks[idx] = (cur_ref, cur_s + 1, cur_e)
                # sys.stderr.write("adjust region {},{} to {},{}\n".format((pre_ref, pre_s, pre_e),
                #                                                          (cur_ref, cur_s, cur_e),
                #                                                          ref_chunks[idx - 1],
                #                                                          ref_chunks[idx]))
    return ref_chunks


def _worker_split_ref_regions(dnacontigs, region_q, args):
    LOGGER.info("worker_split_regions process-{} starts".format(os.getpid()))
    ref_chunks = _get_reference_chunks(dnacontigs, args.contigs, args.chunk_len, args.motifs)
    LOGGER.info("worker_split_regions process-{} generates {} regions".format(os.getpid(),
                                                                              len(ref_chunks)))
    with tqdm(total=len(ref_chunks),
              desc="region_reader") as pbar:
        for ref_chunk in ref_chunks:
            region_q.put(ref_chunk)
            pbar.update(1)
            while region_q.qsize() > (args.threads if args.threads > 1 else 2) * 3:
                time.sleep(time_wait)
    region_q.put("kill")


def _cal_mod_prob(ml_value):
    # WARN: if ori_prob exactly = 0.5, ml_value = floor(0.5*256) = 128
    # WARN: then if we use the following to convert ml_value to prob, the the new_prob = 0.5000001.
    # WARN: if we use the rule: label = 1 if prob>0.5 else 0, then the label of new_prob will be
    # WARN: different with the label of ori_prob
    return round(ml_value / float(256) + 0.000001, 6) if ml_value > 0 else 0


# refer from PacBio https://github.com/PacificBiosciences/pb-CpG-tools under BSD 3-Clause Clear License
def _get_mm_position_iters(mm_idxs):
    base_count = 0
    # base_counts = []
    for idx in mm_idxs:
        base_count += idx + 1
        yield base_count
        # base_counts.append(base_count)
    # return base_counts


# refer from PacBio https://github.com/PacificBiosciences/pb-CpG-tools under BSD 3-Clause Clear License
def _get_all_modbase_positions(fwd_seq, modbase):
    return [i.start() for i in re.finditer(modbase, fwd_seq)]


def _get_moddict_in_tags(readitem, modbase="C", modification="m"):
    mmtag, mltag = None, None
    try:
        mmtag = readitem.get_tag('MM')
        mltag = readitem.get_tag('ML')
    except KeyError:
        pass
    if mmtag is None or mltag is None:
        return {}
    else:
        seq_fwdseq = readitem.get_forward_sequence()
        seq_len = len(seq_fwdseq)
        is_reverse = readitem.is_reverse

        # parse MM/ML tags
        mod_iters = None
        for x in mmtag.split(';'):
            if x.startswith(modbase + "+" + modification):
                start_index = len(modbase) + 1 + len(modification)
                if len(x) > start_index and x[start_index] in "?.":
                    start_index += 1
                if len(x) > start_index and x[start_index] == ",":
                    start_index += 1
                    mod_iters = _get_mm_position_iters([int(y) for y in x[start_index:].split(",")])
                break
        if mod_iters is None:
            return {}
        modbases_all = _get_all_modbase_positions(seq_fwdseq, modbase)
        try:
            modbases = [modbases_all[idx - 1] for idx in mod_iters]
            assert len(modbases) == len(mltag)
            moddict = dict()
            for idx in range(len(modbases)):
                mod_pos = modbases[idx]
                if is_reverse:
                    mod_pos = seq_len - 1 - mod_pos
                moddict[mod_pos] = _cal_mod_prob(mltag[idx])
            return moddict
        except IndexError:
            LOGGER.warning("read {}: MM tag length does not match length of modbases "
                           "in read!".format(readitem.query_name))
            return {}
        except AssertionError:
            LOGGER.warning("read {}: MM tag length != ML tag length!".format(readitem.query_name))
            return {}


def _get_moddict(readitem, modbase="C", modification="m"):
    """

    :param readitem:
    :return: moddict: query_pos(in alignment strand) 2 mod_probs([0,1])
    """

    # first try to use .modified_bases (pysam>=0.19.0) instead of MM/ML tags to get moddict
    modinfo = readitem.modified_bases
    if not (modinfo is None or len(modinfo) == 0):
        modtuple = None
        for modkey in modinfo.keys():
            if modkey[0] == modbase and modkey[2] == modification:
                modtuple = modinfo[modkey]
                break
        if modtuple is None:
            return {}
        moddict = dict(modtuple)
        for modloc in moddict.keys():
            moddict[modloc] = _cal_mod_prob(moddict[modloc])
        return moddict
    else:
        # case 1: the MM tag is "MM:Z:C+m?,..."?
        # case 2: there are no MM/ML tags
        return _get_moddict_in_tags(readitem, modbase, modification)


def _cal_modfreq_in_count_mode(modprobs, prob_cf=0, no_amb_cov=False):
    cnt_all_filtered, cnt_mod = 0, 0
    for modprob in modprobs:
        if abs(modprob - (1 - modprob)) < prob_cf:
            continue
        cnt_all_filtered += 1
        if modprob > 0.5:
            cnt_mod += 1
    modfreq = cnt_mod / float(cnt_all_filtered) if cnt_all_filtered > 0 else 0.
    if no_amb_cov:
        return cnt_all_filtered, cnt_mod, modfreq
    else:
        # WARN -- when prob_cf>0, cnt_mod/modfreq does not always equal len(modprobs)
        # So:
        if cnt_all_filtered != len(modprobs):
            # adjust cnt_mod
            cnt_mod = np.round(len(modprobs) * modfreq, 2)
        return len(modprobs), cnt_mod, modfreq


# from PacBio https://github.com/PacificBiosciences/pb-CpG-tools under BSD 3-Clause Clear License
def _get_normalized_histo(probs, cov_cf=4, binsize=20):
    """
    Create the array data structure needed to apply the model, for a given site.

    :param probs: List of methylation probabilities. (list)
    :param cov_cf:
    :param binsize:
    :return: List with normalized histogram
    """

    cov = len(probs)
    assert cov >= cov_cf

    hist = np.histogram(probs, bins=binsize, range=[0, 1])[0]
    norm = np.linalg.norm(hist)
    # divide hist by norm and add values to arr
    return np.round(hist / norm, 6)


# try discretizing, similar idea from PacBio https://github.com/PacificBiosciences/pb-CpG-tools
# under BSD 3-Clause Clear License
def discretize_score(modprob, coverage):
    """

    """
    # need to round up or round down modified read numbers based on score
    # which allows a push towards 0/100 for adjusted score
    if modprob > 0.66:
        mod_reads = int(np.ceil(modprob * float(coverage)))
    elif modprob <= 0.33:
        mod_reads = int(np.floor(modprob * float(coverage)))
    else:
        mod_reads = round(coverage * modprob, 2)

    unmod_reads = int(coverage) - mod_reads

    if mod_reads == 0:
        adjusted_score = 0.0
    else:
        adjusted_score = float(mod_reads) / (mod_reads + unmod_reads)

    return mod_reads, unmod_reads, adjusted_score


def _cal_modfreq_in_aggregate_mode(refposes, refposes_histos, model, seq_len=11, only_close=False):

    if len(refposes) == 0:
        return None

    from .utils.constants_torch import FloatTensor_cpu

    pad_len = seq_len // 2
    histos_mat = np.pad(np.stack(refposes_histos),
                        pad_width=((pad_len, pad_len), (0, 0)),
                        mode='constant', constant_values=0)
    histos_mat = np.swapaxes(sliding_window_view(histos_mat, seq_len, axis=0), 1, 2)

    if not only_close:
        pos_mat = np.pad(refposes, pad_width=(pad_len, pad_len),
                         mode='constant', constant_values=(refposes[0] - 1000, refposes[-1] + 1000))
        pos_mat = sliding_window_view(pos_mat, seq_len)
        pos_mat_center = np.repeat(refposes, seq_len).reshape((-1, seq_len))
        pos_mat = np.absolute(np.subtract(pos_mat, pos_mat_center))
        # no log2 is a litter better?
        # pos_mat = np.round(1. / (np.log2(np.absolute(pos_mat) + 1) + 1), 6)
        del pos_mat_center
    else:
        pos_mat = np.pad(refposes, pad_width=(pad_len + 1, pad_len),
                         mode='constant', constant_values=(refposes[0] - 1000, refposes[-1] + 1000))
        pos_mat = np.diff(pos_mat)
        pos_mat = (pos_mat == 2).astype(int)
        pos_mat = sliding_window_view(pos_mat, seq_len)

    probs = []
    batch_size = 1024
    for i in np.arange(0, len(histos_mat), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_histos = np.array(histos_mat[batch_s:batch_e].copy())
        b_pos = np.array(pos_mat[batch_s:batch_e])
        if len(b_histos) > 0:
            voutputs = model(FloatTensor_cpu(b_pos), FloatTensor_cpu(b_histos))
            logits = np.round(np.clip(voutputs.data.numpy(), 0, 1), 6)
            for idx in range(len(b_histos)):
                probs.append(logits[idx][0])
    return probs


def _call_modfreq_of_one_region_aggregate_mode(refpos2modinfo, args):
    import torch
    from .models import AggrAttRNN
    # from .utils.constants_torch import use_cuda

    torch.manual_seed(args.tseed)

    # load model
    device = "cpu"
    if args.model_type in {"attbigru", "attbilstm"}:
        model = AggrAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                           0, args.hid_rnn, binsize=args.bin_size,
                           model_type=args.model_type,
                           device=device)
    else:
        raise ValueError("--model_type not right!")
    try:
        para_dict = torch.load(args.aggre_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)
        del model_dict
    except RuntimeError:
        # for DDP model convertion (key: module.embed.weight -> embed.weight)
        from collections import OrderedDict
        para_dict = torch.load(args.aggre_model, map_location=torch.device('cpu'))
        para_dict_new = OrderedDict()
        for param_tensor in para_dict.keys():
            keytmp = param_tensor[7:]
            para_dict_new[keytmp] = para_dict[param_tensor]
        model.load_state_dict(para_dict_new)
        del para_dict_new
    # if use_cuda:
    #     model = model.cuda(device)
    model.eval()

    all_lowcov_pos, all_lowcov_mods, all_highcov_pos, all_highcov_histos = [], [], [], []
    hp1_lowcov_pos, hp1_lowcov_mods, hp1_highcov_pos, hp1_highcov_histos = [], [], [], []
    hp2_lowcov_pos, hp2_lowcov_mods, hp2_highcov_pos, hp2_highcov_histos = [], [], [], []
    all_highcov_covs, hp1_highcov_covs, hp2_highcov_covs = [], [], []
    all_refposes = sorted(refpos2modinfo.keys())
    refpos2result = dict()
    for refpos in all_refposes:
        refpos2result[refpos] = [None, None, None]

        modinfo = refpos2modinfo[refpos]
        total_mods, hp1_mods, hp2_mods = [], [], []
        for modprob, hap in modinfo:
            total_mods.append(modprob)
            if not args.no_hap:
                if hap == 1:
                    hp1_mods.append(modprob)
                elif hap == 2:
                    hp2_mods.append(modprob)

        if len(total_mods) > 0:
            if len(total_mods) >= args.cov_cf:
                all_highcov_pos.append(refpos)
                all_highcov_histos.append(_get_normalized_histo(total_mods, args.cov_cf, args.bin_size))
                all_highcov_covs.append(len(total_mods))
            else:
                all_lowcov_pos.append(refpos)
                all_lowcov_mods.append(_cal_modfreq_in_count_mode(total_mods, args.prob_cf, args.no_amb_cov))
        if len(hp1_mods) > 0:
            if len(hp1_mods) >= args.cov_cf:
                hp1_highcov_pos.append(refpos)
                hp1_highcov_histos.append(_get_normalized_histo(hp1_mods, args.cov_cf, args.bin_size))
                hp1_highcov_covs.append(len(hp1_mods))
            else:
                hp1_lowcov_pos.append(refpos)
                hp1_lowcov_mods.append(_cal_modfreq_in_count_mode(hp1_mods, args.prob_cf, args.no_amb_cov))
        if len(hp2_mods) > 0:
            if len(hp2_mods) >= args.cov_cf:
                hp2_highcov_pos.append(refpos)
                hp2_highcov_histos.append(_get_normalized_histo(hp2_mods, args.cov_cf, args.bin_size))
                hp2_highcov_covs.append(len(hp2_mods))
            else:
                hp2_lowcov_pos.append(refpos)
                hp2_lowcov_mods.append(_cal_modfreq_in_count_mode(hp2_mods, args.prob_cf, args.no_amb_cov))
    for lowcov_idx in range(len(all_lowcov_pos)):
        refpos2result[all_lowcov_pos[lowcov_idx]][0] = all_lowcov_mods[lowcov_idx]
    for lowcov_idx in range(len(hp1_lowcov_pos)):
        refpos2result[hp1_lowcov_pos[lowcov_idx]][1] = hp1_lowcov_mods[lowcov_idx]
    for lowcov_idx in range(len(hp2_lowcov_pos)):
        refpos2result[hp2_lowcov_pos[lowcov_idx]][2] = hp2_lowcov_mods[lowcov_idx]

    probs = _cal_modfreq_in_aggregate_mode(all_highcov_pos, all_highcov_histos, model, args.seq_len, args.only_close)
    for highcov_idx in range(len(all_highcov_pos)):
        modprob_tmp = probs[highcov_idx]
        cov_tmp = all_highcov_covs[highcov_idx]
        if args.discrete:
            d_cnt_mod, _, d_modprob_tmp = discretize_score(modprob_tmp, cov_tmp)
            refpos2result[all_highcov_pos[highcov_idx]][0] = (cov_tmp, d_cnt_mod, d_modprob_tmp)
        else:
            cnt_mod = round(cov_tmp * modprob_tmp, 2)
            refpos2result[all_highcov_pos[highcov_idx]][0] = (cov_tmp, cnt_mod, modprob_tmp)
    probs = _cal_modfreq_in_aggregate_mode(hp1_highcov_pos, hp1_highcov_histos, model, args.seq_len, args.only_close)
    for highcov_idx in range(len(hp1_highcov_pos)):
        modprob_tmp = probs[highcov_idx]
        cov_tmp = hp1_highcov_covs[highcov_idx]
        if args.discrete:
            d_cnt_mod, _, d_modprob_tmp = discretize_score(modprob_tmp, cov_tmp)
            refpos2result[hp1_highcov_pos[highcov_idx]][1] = (cov_tmp, d_cnt_mod, d_modprob_tmp)
        else:
            cnt_mod = round(cov_tmp * modprob_tmp, 2)
            refpos2result[hp1_highcov_pos[highcov_idx]][1] = (cov_tmp, cnt_mod, modprob_tmp)
    probs = _cal_modfreq_in_aggregate_mode(hp2_highcov_pos, hp2_highcov_histos, model, args.seq_len, args.only_close)
    for highcov_idx in range(len(hp2_highcov_pos)):
        modprob_tmp = probs[highcov_idx]
        cov_tmp = hp2_highcov_covs[highcov_idx]
        if args.discrete:
            d_cnt_mod, _, d_modprob_tmp = discretize_score(modprob_tmp, cov_tmp)
            refpos2result[hp2_highcov_pos[highcov_idx]][2] = (cov_tmp, d_cnt_mod, d_modprob_tmp)
        else:
            cnt_mod = round(cov_tmp * modprob_tmp, 2)
            refpos2result[hp2_highcov_pos[highcov_idx]][2] = (cov_tmp, cnt_mod, modprob_tmp)

    refpos_results = []
    for refpos in all_refposes:
        refpos_results.append((refpos, refpos2result[refpos][0], refpos2result[refpos][1],
                               refpos2result[refpos][2]))
    return refpos_results


def _call_modfreq_of_one_region(refpos2modinfo, args):
    if args.call_mode == "count":
        refpos_results = []
        for refpos in sorted(refpos2modinfo.keys()):
            modinfo = refpos2modinfo[refpos]
            total_mods, hp1_mods, hp2_mods = [], [], []
            for modprob, hap in modinfo:
                total_mods.append(modprob)
                if not args.no_hap:
                    if hap == 1:
                        hp1_mods.append(modprob)
                    elif hap == 2:
                        hp2_mods.append(modprob)
            info_all = _cal_modfreq_in_count_mode(total_mods, args.prob_cf, args.no_amb_cov) if len(total_mods) > 0 else None
            info_hp1 = _cal_modfreq_in_count_mode(hp1_mods, args.prob_cf, args.no_amb_cov) if len(hp1_mods) > 0 else None
            info_hp2 = _cal_modfreq_in_count_mode(hp2_mods, args.prob_cf, args.no_amb_cov) if len(hp2_mods) > 0 else None
            refpos_results.append((refpos, info_all, info_hp1, info_hp2))
    elif args.call_mode == "aggregate":
        refpos_results = _call_modfreq_of_one_region_aggregate_mode(refpos2modinfo, args)
    else:
        raise ValueError("wrong --call_mode")

    return refpos_results


def _readmods_to_bed_of_one_region(bam_reader, regioninfo, dnacontigs, motifs_filter, args):
    modbase = "-"
    modification = "-"
    if args.modtype == "5mC":
        modbase = "C"
        modification = "m"

    fwd_s, fwd_e, rev_s, rev_e = None, None, None, None
    if motifs_filter is not None:
        len_motif = len(motifs_filter[0])
        fwd_s = -args.mod_loc
        fwd_e = len_motif - args.mod_loc
        rev_s = -(len_motif - 1 - args.mod_loc)
        rev_e = args.mod_loc + 1
        motifs_filter = set(motifs_filter)

    ref_name, ref_start, ref_end = regioninfo
    if args.refsites_all:
        refmotifsites = get_refloc_of_methysite_in_motif(dnacontigs[ref_name][ref_start:ref_end],
                                                         motifs_filter, args.mod_loc)
        refmotifsites = set([x + ref_start for x in refmotifsites])
        refmotifsites_rev = get_refloc_of_methysite_in_motif(complement_seq(dnacontigs[ref_name][ref_start:ref_end]),
                                                             motifs_filter, args.mod_loc)
        refmotifsites_rev = set([ref_end - 1 - x for x in refmotifsites_rev])
    refposinfo = {}  # {loc: [(prob, hap), ]), }
    refposes = set()
    refposinfo_rev = {}
    refposes_rev = set()
    cnt_all, cnt_used = 0, 0
    # check if (ref_name, ref_start, ref_end) is valid in bam_reader.references
    try:
        for readitem in bam_reader.fetch(contig=ref_name, start=ref_start, stop=ref_end):
            cnt_all += 1
            if readitem.is_unmapped or readitem.is_secondary or readitem.is_duplicate:
                continue
            if args.no_supplementary and readitem.is_supplementary:
                continue
            if readitem.mapping_quality < args.mapq:
                continue
            identity = compute_pct_identity(np.array(readitem.get_cigar_stats()[0]))
            if identity < args.identity:
                continue

            try:
                hap_val = readitem.get_tag(args.hap_tag)
                hap = int(hap_val)
            except ValueError:
                hap = 0
            except KeyError:
                hap = 0
            is_reverse = 1 if readitem.is_reverse else 0
            moddict = _get_moddict(readitem, modbase, modification)
            modlocs = set(moddict.keys())
            matches_only = False if args.refsites_all else True
            aligned_pairs = readitem.get_aligned_pairs(matches_only=matches_only)
            if args.base_clip > 0:
                aligned_pairs = aligned_pairs[args.base_clip:(-args.base_clip)]
            if is_reverse:
                for q_pos, r_pos in aligned_pairs:
                    if r_pos is not None and ref_start <= r_pos < ref_end:
                        if q_pos is not None and q_pos in modlocs:
                            if r_pos not in refposes_rev:
                                refposes_rev.add(r_pos)
                                refposinfo_rev[r_pos] = []
                            refposinfo_rev[r_pos].append((moddict[q_pos], hap))
                        elif args.refsites_all and (r_pos in refmotifsites_rev):
                            if r_pos not in refposes_rev:
                                refposes_rev.add(r_pos)
                                refposinfo_rev[r_pos] = []
                            refposinfo_rev[r_pos].append((0.0, hap))
            else:
                for q_pos, r_pos in aligned_pairs:
                    if r_pos is not None and ref_start <= r_pos < ref_end:
                        if q_pos is not None and q_pos in modlocs:
                            if r_pos not in refposes:
                                refposes.add(r_pos)
                                refposinfo[r_pos] = []
                            refposinfo[r_pos].append((moddict[q_pos], hap))
                        elif args.refsites_all and (r_pos in refmotifsites):
                            if r_pos not in refposes:
                                refposes.add(r_pos)
                                refposinfo[r_pos] = []
                            refposinfo[r_pos].append((0.0, hap))
            cnt_used += 1
    except ValueError:
        LOGGER.warning("worker_gen_bed process-%d: "
                       "region %s:%d-%d is not valid in bam file" % (os.getpid(), ref_name, 
                                                                     ref_start, ref_end))
        return [], [], []
    
    if args.motifs == "CG" and not args.no_comb:
        for rev_pos in refposes_rev:
            if rev_pos == 0:
                continue
            fwd_pos = rev_pos - 1
            if fwd_pos not in refposes:
                refposes.add(fwd_pos)
                refposinfo[fwd_pos] = []
            refposinfo[fwd_pos] += refposinfo_rev[rev_pos]
        del refposinfo_rev
        del refposes_rev

    # gathering modfreq result
    bed_all, bed_hp1, bed_hp2 = [], [], []
    refpos_res = _call_modfreq_of_one_region(refposinfo, args)
    for refpositem in refpos_res:
        refpos, total_info, hp1_info, hp2_info = refpositem

        if motifs_filter is not None:
            motif_seq = dnacontigs[ref_name][(refpos + fwd_s):(refpos + fwd_e)]
            if motif_seq not in motifs_filter:
                continue

        # total_info in (cov, mod_cnt, mod_freq) format, as hp1_info, hp2_info
        if total_info is not None:
            bed_all.append((ref_name, refpos, "+", total_info[0], total_info[1], total_info[2]))
        if hp1_info is not None:
            bed_hp1.append((ref_name, refpos, "+", hp1_info[0], hp1_info[1], hp1_info[2]))
        if hp2_info is not None:
            bed_hp2.append((ref_name, refpos, "+", hp2_info[0], hp2_info[1], hp2_info[2]))
    if not (args.motifs == "CG" and not args.no_comb):
        refposrev_res = _call_modfreq_of_one_region(refposinfo_rev, args)
        for refpositem in refposrev_res:
            refpos, total_info, hp1_info, hp2_info = refpositem

            if motifs_filter is not None:
                motif_seq = complement_seq(dnacontigs[ref_name][(refpos + rev_s):(refpos + rev_e)])
                if motif_seq not in motifs_filter:
                    continue

            # total_info in (cov, mod_cnt, mod_freq) format, as hp1_info, hp2_info
            if total_info is not None:
                bed_all.append((ref_name, refpos, "-", total_info[0], total_info[1], total_info[2]))
            if hp1_info is not None:
                bed_hp1.append((ref_name, refpos, "-", hp1_info[0], hp1_info[1], hp1_info[2]))
            if hp2_info is not None:
                bed_hp2.append((ref_name, refpos, "-", hp2_info[0], hp2_info[1], hp2_info[2]))
    return bed_all, bed_hp1, bed_hp2


def _worker_generate_bed_of_regions(inputbam, region_q, bed_q, dnacontigs, motifs_filter, args):
    LOGGER.info("worker_gen_bed process-{} starts".format(os.getpid()))
    try:
        bam_reader = pysam.AlignmentFile(inputbam, 'rb')
    except ValueError:
        raise ValueError("file has no sequences defined (mode='rb') - pysam - "
                         "Please check and make sure that the reads are aligned to genome referece!")
    cnt_regions = 0
    while True:
        if region_q.empty():
            time.sleep(time_wait)
            continue
        region = region_q.get()
        if region == "kill":
            region_q.put("kill")
            break
        cnt_regions += 1
        bed_all, bed_hp1, bed_hp2 = _readmods_to_bed_of_one_region(bam_reader, region,
                                                                   dnacontigs, motifs_filter, args)
        if len(bed_all) > 0:
            bed_q.put((bed_all, bed_hp1, bed_hp2))
            while bed_q.qsize() > (args.threads if args.threads > 1 else 2) * 3:
                time.sleep(time_wait)

    bam_reader.close()
    LOGGER.info("worker_gen_bed process-{} ending, proceed {} regions".format(os.getpid(),
                                                                              cnt_regions))


def _write_one_line(beditem, wf, is_bed):
    ref_name, refpos, strand, cov, met, metprob = beditem
    if is_bed:
        wf.write("\t".join([ref_name, str(refpos), str(refpos + 1), ".", str(cov),
                            strand, str(refpos), str(refpos + 1),
                            "0,0,0", str(cov), str(int(round(metprob * 100 + 0.001, 0)))]) + "\n")
    else:
        wf.write("\t".join([ref_name, str(refpos), str(refpos + 1), strand, ".", ".", str(met),
                            str(cov-met), str(cov), str(round(metprob + 0.000001, 4)), "."]) + "\n")


def _worker_write_bed_result(output_prefix, bed_q, args):
    LOGGER.info('write_process-{} starts'.format(os.getpid()))

    fext = "bed" if args.bed else "freq.txt"
    op_all = output_prefix + ".{}.all.{}".format(args.call_mode, fext)
    op_hp1 = output_prefix + ".{}.hp1.{}".format(args.call_mode, fext)
    op_hp2 = output_prefix + ".{}.hp2.{}".format(args.call_mode, fext)
    wf_all = open(op_all, "w")
    wf_hp1 = open(op_hp1, "w")
    wf_hp2 = open(op_hp2, "w")
    while True:
        if bed_q.empty():
            time.sleep(time_wait)
            continue
        bed_res = bed_q.get()
        if bed_res == "kill":
            break
        bed_all, bed_hp1, bed_hp2 = bed_res
        for beditem in bed_all:
            _write_one_line(beditem, wf_all, args.bed)
        for beditem in bed_hp1:
            _write_one_line(beditem, wf_hp1, args.bed)
        for beditem in bed_hp2:
            _write_one_line(beditem, wf_hp2, args.bed)
    wf_all.close()
    wf_hp1.close()
    wf_hp2.close()
    for bedfile in (op_all, op_hp1, op_hp2):
        if is_file_empty(bedfile):
            os.remove(bedfile)
            continue
        if args.sort or args.gzip:
            LOGGER.info('write_process-{} sorting results - {}'.format(os.getpid(), bedfile))
            ori_bed = pybedtools.BedTool(bedfile)
            ori_bed.sort().moveto(bedfile)
        if args.gzip:
            LOGGER.info('write_process-{} gzipping results - {}'.format(os.getpid(), bedfile))
            pysam.tabix_index(bedfile, force=True,
                              preset="bed",
                              keep_original=False)
    LOGGER.info('write_process-{} finished'.format(os.getpid()))


def call_mods_frequency_from_bamfile(args):
    LOGGER.info("[main]call_freq_bam starts")
    start = time.time()

    if args.call_mode == "aggregate" and not os.path.exists(args.aggre_model):
        raise ValueError("--aggre_model is not set right!")

    inputpath = _check_input_file(args.input_bam)
    index_bam_if_needed2(inputpath, args.threads)
    if not os.path.exists(args.ref):
        raise ValueError("--ref does not exist!")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dnacontigs = DNAReference(args.ref).getcontigs()
    motifs = get_motif_seqs(args.motifs)

    motifs_filter = None
    if args.refsites_only or args.refsites_all:
        motifs_filter = motifs
        LOGGER.info("[###] --refsites_only (or/and --refsites_all) is set as True, "
                    "gonna keep only motifs({}) sites of genome reference in the "
                    "results".format(motifs_filter))

    nproc = args.threads
    if nproc < 3:
        nproc = 3
    region_q = Queue()
    bed_q = Queue()

    p_read = mp.Process(target=_worker_split_ref_regions,
                        args=(dnacontigs, region_q, args))
    p_read.daemon = True
    p_read.start()

    ps_gen = []
    for _ in range(nproc - 2):
        p = mp.Process(target=_worker_generate_bed_of_regions,
                       args=(inputpath, region_q, bed_q, dnacontigs, motifs_filter, args))
        p.daemon = True
        p.start()
        ps_gen.append(p)

    p_w = mp.Process(target=_worker_write_bed_result,
                     args=(args.output, bed_q, args))
    p_w.daemon = True
    p_w.start()

    for p in ps_gen:
        p.join()
    p_read.join()
    bed_q.put("kill")
    p_w.join()

    LOGGER.info("[main]call_freq_bam costs %.1f seconds" % (time.time() - start))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level '
                                                 'from aligned.sorted.bam')

    parser.add_argument('--threads', action="store", type=int, required=False, default=5,
                        help="number of subprocesses used. default 5")

    scfb_input = parser.add_argument_group("INPUT")
    scfb_input.add_argument('--input_bam', action="store", type=str, required=True,
                            help='input bam, should be aligned and sorted')
    scfb_input.add_argument("--ref", type=str, required=True,
                            help="path to genome reference, in fasta/fa format.")
    scfb_input.add_argument('--contigs', action="store", type=str, required=False, default=None,
                            help="path of a file containing chromosome/contig names, one name each line; "
                                 "or a string contains multiple chromosome names splited by comma."
                                 "default None, which means all chromosomes will be processed.")
    scfb_input.add_argument('--chunk_len', type=int, required=False, default=500000,
                            help="chunk length, default 500000")

    scfb_output = parser.add_argument_group("OUTPUT")
    scfb_output.add_argument('--output', '-o', action="store", type=str, required=True,
                             help='prefix of output file to save the results')
    scfb_output.add_argument('--bed', action='store_true', default=False,
                             help="save the result in bedMethyl format")
    scfb_output.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    scfb_output.add_argument("--gzip", action="store_true", default=False, required=False,
                             help="if compressing the output using gzip")

    scfb_callfreq = parser.add_argument_group("CALL_FREQ")
    scfb_callfreq.add_argument('--modtype', type=str, action="store", required=False, default="5mC",
                               choices=["5mC", ],
                               help='modification type, default 5mC.')
    scfb_callfreq.add_argument('--call_mode', type=str, action="store", required=False, default="count",
                               choices=["count", "aggregate"],
                               help='call mode: count, aggregate. default count.')
    scfb_callfreq.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                               help='this is to remove ambiguous calls (only for count-mode now). '
                               'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                               'means use all calls. range [0, 1], default 0.0.')
    scfb_callfreq.add_argument('--no_amb_cov', action="store_true", required=False, default=False,
                               help='when using prob_cf>0, DO NOT count ambiguous calls '
                                    'for calculating reads coverage')
    scfb_callfreq.add_argument("--hap_tag", type=str, action="store", required=False, default="HP",
                               help="haplotype tag, default HP")
    scfb_callfreq.add_argument("--mapq", type=int, default=1, required=False,
                               help="MAPping Quality cutoff for selecting alignment items, default 1")
    scfb_callfreq.add_argument("--identity", type=float, default=0.0, required=False,
                               help="identity cutoff for selecting alignment items, [0.0, 1.0], default 0.0")
    scfb_callfreq.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                               help="not use supplementary alignment")
    scfb_callfreq.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    scfb_callfreq.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    scfb_callfreq.add_argument("--no_comb", action="store_true", default=False, required=False,
                               help="don't combine fwd/rev reads of one CG. [Only works when motifs is CG]")
    scfb_callfreq.add_argument('--refsites_only', action='store_true', default=False,
                               help="only keep sites which are target motifs in both reference and reads")
    scfb_callfreq.add_argument('--refsites_all', action='store_true', default=False,
                               help="output all covered sites which are target motifs in reference. "
                                    "--refsites_all is True, also means we do not output sites which "
                                    "are target motifs only in reads.")
    scfb_callfreq.add_argument("--no_hap", action="store_true", default=False, required=False,
                               help="don't call_freq on haplotypes")
    scfb_callfreq.add_argument("--base_clip", action="store", type=int, required=False, default=0,
                               help='number of base clipped in each read, default 0')

    scfb_aggre = parser.add_argument_group("AGGREGATE_MODE")
    scfb_aggre.add_argument("--aggre_model", "-m", action="store", type=str, required=False,
                            help="file path of the aggregate model (.ckpt)")
    scfb_aggre.add_argument('--model_type', type=str, default="attbigru",
                            choices=["attbilstm", "attbigru"],
                            required=False,
                            help="type of model to use, 'attbigru', 'attbilstm', "
                                 "default: attbigru")
    scfb_aggre.add_argument('--seq_len', type=int, default=11, required=False,
                            help="len of sites used. default 11")
    scfb_aggre.add_argument('--class_num', type=int, default=1, required=False)
    scfb_aggre.add_argument('--layer_rnn', type=int, default=1,
                            required=False, help="BiRNN layer num, default 1")
    scfb_aggre.add_argument('--hid_rnn', type=int, default=32, required=False,
                            help="BiRNN hidden_size, default 32")
    scfb_aggre.add_argument('--bin_size', type=int, action="store", required=False, default=20,
                            help="histogram bin size, default 20")
    scfb_aggre.add_argument('--cov_cf', action="store", type=int, required=False,
                            default=4, help="coverage cutoff, to consider if use aggregate model to "
                                            "re-predict the modstate of the site")
    scfb_aggre.add_argument('--only_close', action="store_true", default=False, required=False,
                            help="[EXPERIMENTAL]")
    scfb_aggre.add_argument('--discrete', action="store_true", default=False, required=False,
                            help="[EXPERIMENTAL]")
    scfb_aggre.add_argument('--tseed', type=int, default=1234,
                            help='random seed for torch')

    args = parser.parse_args()

    call_mods_frequency_from_bamfile(args)


if __name__ == '__main__':
    sys.exit(main())
