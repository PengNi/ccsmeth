import os
import argparse
import sys
import time
import numpy as np
from statsmodels import robust
from subprocess import Popen, PIPE
import multiprocessing as mp
from multiprocessing import Queue
import re
import random
# from collections import Counter

from utils.process_utils import display_args
from utils.process_utils import codecv1_to_frame
from utils.process_utils import generate_samtools_view_cmd
from utils.process_utils import get_refloc_of_methysite_in_motif
from utils.process_utils import get_motif_seqs
from utils.ref_reader import DNAReference
from utils.process_utils import complement_seq

code2frames = codecv1_to_frame()
queen_size_border = 1000
time_wait = 3

exceptval = 1000
subreads_value_default = "-"


def check_input_file(inputfile):
    if not (inputfile.endswith(".bam") or inputfile.endswith(".sam")):
        raise ValueError("--input/-i must be in bam/sam format!")
    inputpath = os.path.abspath(inputfile)
    return inputpath


def check_output_file(outputfile, inputfile):
    if outputfile is None:
        fname, fext = os.path.splitext(inputfile)
        output_path = fname + ".features.tsv"
    else:
        output_path = os.path.abspath(outputfile)
    return output_path


def cmd_get_stdout_of_input(inputpath, path_to_samtools):
    if inputpath.endswith(".bam"):
        samtools_view = generate_samtools_view_cmd(path_to_samtools)
        cmd = samtools_view + " " + inputpath
    elif inputpath.endswith(".sam"):
        cmd = "cat " + inputpath
    else:
        raise ValueError()
    return cmd


def _get_holeid(subread_id):
    words = subread_id.strip().split("/")
    # assume movie_id is the same in one bam
    # holeid = words[0] + "/" + words[1]
    holeid = words[1]
    return holeid


def worker_read(inputfile, hole_align_q, args, holeids_e=None, holeids_ne=None):
    sys.stderr.write("read_input process-{} starts\n".format(os.getpid()))
    cmd_view_input = cmd_get_stdout_of_input(inputfile, args.path_to_samtools)
    sys.stderr.write("cmd to view input: {}\n".format(cmd_view_input))
    proc_read = Popen(cmd_view_input, shell=True, stdout=PIPE)

    holes_align_tmp = []
    holeid_curr = ""
    hole_align_tmp = []
    cnt_holes = 0
    while True:
        output = str(proc_read.stdout.readline(), 'utf-8')
        if output != "":
            try:
                if output.startswith("#") or output.startswith("@"):
                    continue
                words = output.strip().split("\t")
                holeid = _get_holeid(words[0])
                if holeids_e is not None and holeid not in holeids_e:
                    continue
                if holeids_ne is not None and holeid in holeids_ne:
                    continue

                flag = int(words[1])
                mapq = int(words[4])
                if not (flag == 0 or flag == 16):  # skip segment alignment
                    continue
                if mapq < args.mapq:  # skip low mapq alignment
                    continue
                if holeid != holeid_curr:
                    if len(hole_align_tmp) > 0:
                        cnt_holes += 1
                        holes_align_tmp.append((holeid_curr, hole_align_tmp))
                        if len(holes_align_tmp) >= args.holes_batch:
                            hole_align_q.put(holes_align_tmp)
                            holes_align_tmp = []
                            while hole_align_q.qsize() > queen_size_border:
                                time.sleep(time_wait)
                    hole_align_tmp = []
                    holeid_curr = holeid
                hole_align_tmp.append(words)
            except Exception:
                # raise ValueError("error in parsing lines of input!")
                continue
        elif proc_read.poll() is not None:
            if len(hole_align_tmp) > 0:
                cnt_holes += 1
                holes_align_tmp.append((holeid_curr, hole_align_tmp))
            if len(holes_align_tmp) > 0:
                hole_align_q.put(holes_align_tmp)
            break
        else:
            # print("output:", output)
            continue
    hole_align_q.put("kill")
    rc_read = proc_read.poll()
    sys.stderr.write("read_input process-{} ending, read {} holes, with return_code-{}\n".format(os.getpid(),
                                                                                                 cnt_holes,
                                                                                                 rc_read))


def _normalize_signals(signals, normalize_method="zscore"):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), np.std(signals)
    elif normalize_method == 'min-max':
        sshift, sscale = np.min(signals), np.max(signals) - np.min(signals)
    elif normalize_method == 'min-mean':
        sshift, sscale = np.min(signals), np.mean(signals)
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), np.float(robust.mad(signals))
    else:
        raise ValueError("")
    if sscale == 0.0:
        norm_signals = signals
    else:
        norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def _parse_cigar(cigarseq):
    pattern = re.compile(r'((\d)+(S|H|X|=|M|I|D|N|P))')
    it = pattern.findall(cigarseq)
    # q_adjseq = ""
    queryseq_poses = []
    refpos2querypos = {}
    cnt_s, cnt_m, cnt_i, cnt_d = 0, 0, 0, 0
    cidx_q, cidx_t = 0, 0
    for match in it:
        num = int(match[0][:-1])
        if match[0].endswith('S'):
            # https://support.bioconductor.org/p/128310/
            # sys.stderr.write("warning: got {} soft clipping in cigar!\n".format(match[0][:-1]))
            cidx_q += num
            cnt_s += num
        elif match[0].endswith('X') or match[0].endswith('=') or match[0].endswith('M'):
            # q_adjseq += queryseq[cidx_q:(cidx_q + num)]
            queryseq_poses += [idx for idx in range(cidx_q, (cidx_q + num))]
            for i in range(0, num):
                refpos2querypos[cidx_t + i] = cidx_q + i
            cidx_q += num
            cidx_t += num
            cnt_m += num
        elif match[0].endswith('I'):
            cidx_q += num
            cnt_i += num
        elif match[0].endswith('D'):
            # q_adjseq += "-" * num
            queryseq_poses += [-1] * num  # use -1 for missing values
            cidx_t += num
            cnt_d += num
        elif match[0].endswith('N') or match[0].endswith('P') or match[0].endswith('H'):
            sys.stderr.write("warning: got {} in cigar!\n".format(match[0][-1]))
    identity = float(cnt_m)/(cnt_s + cnt_m + cnt_i + cnt_d)
    # assert (q_adjseq[0] != "-")
    return identity, queryseq_poses, refpos2querypos


def _cal_mean_n_std(mylist):
    return round(np.mean(mylist), 6), round(np.std(mylist), 6)


def check_excpval(myarray):
    if exceptval in set(myarray):
        return True
    return False


def _extract_kmer_features(holeid, chrom, pos_min, pos_max, strand, ipd_mean, ipd_std, pw_mean, pw_std,
                           ipd_depth, depth_all, subreads_info, motifs, mod_loc, seq_len, label, depth,
                           num_subreads, seed, contigs):
    align_seq = contigs[chrom][pos_min:(pos_max+1)]
    if strand == "-":
        align_seq = complement_seq(align_seq)
        ipd_mean = ipd_mean[::-1]
        ipd_std = ipd_std[::-1]
        pw_mean = pw_mean[::-1]
        pw_std = pw_std[::-1]
        ipd_depth = ipd_depth[::-1]
    chromlen = len(contigs[chrom])
    if strand == "+":
        abs_start = pos_min
    else:
        abs_start = chromlen - (pos_min + len(align_seq))
    tsite_locs = get_refloc_of_methysite_in_motif(align_seq, set(motifs), mod_loc)
    num_bases = (seq_len - 1) // 2
    feature_list = []
    for offset_loc in tsite_locs:
        if num_bases <= offset_loc < len(align_seq) - num_bases:
            if strand == '-':
                abs_loc = chromlen - 1 - (abs_start + offset_loc)
            else:
                abs_loc = abs_start + offset_loc

            kmer_seq = align_seq[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            kmer_ipdm = ipd_mean[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            if check_excpval(kmer_ipdm):
                continue
            kmer_ipds = ipd_std[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            kmer_pwm = pw_mean[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            kmer_pws = pw_std[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            kmer_depth = ipd_depth[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            if np.mean(kmer_depth) < depth:
                continue

            feature = (chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth,
                       kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws)

            if num_subreads > 0:
                kmer_subr_ipds, kmer_subr_pws = [], []
                excep_ipd, excep_pw = None, None
                for subreadinfo in subreads_info:
                    subr_ipd, subr_pw = subreadinfo
                    if strand == "-":
                        subr_ipd = subr_ipd[::-1]
                        subr_pw = subr_pw[::-1]
                    kmer_subr_ipd = subr_ipd[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
                    kmer_subr_pw = subr_pw[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
                    if check_excpval(kmer_subr_ipd):
                        if excep_ipd is None:
                            excep_ipd = kmer_subr_ipd
                            excep_pw = kmer_subr_pw
                        continue
                    kmer_subr_ipds.append(kmer_subr_ipd)
                    kmer_subr_pws.append(kmer_subr_pw)

                if len(kmer_subr_ipds) > 0:
                    if len(kmer_subr_ipds) > num_subreads:
                        random.seed(seed)
                        seled_idxs = sorted(random.sample(range(len(kmer_subr_ipds)), num_subreads))
                        # print(holeid, offset_loc, len(kmer_subr_ipds), seled_idxs)
                        kmer_subr_ipds = [kmer_subr_ipds[idx] for idx in seled_idxs]
                        kmer_subr_pws = [kmer_subr_pws[idx] for idx in seled_idxs]
                    feature = feature + (kmer_subr_ipds, kmer_subr_pws)
                else:
                    excep_ipd = [x if x != exceptval else 0.0 for x in excep_ipd]
                    excep_pw = [x if x != exceptval else 0.0 for x in excep_pw]
                    feature = feature + ([excep_ipd, ], [excep_pw, ])
            else:
                feature = feature + (subreads_value_default, subreads_value_default)
            feature = feature + (label, )
            feature_list.append(feature)
    return feature_list


# # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
# # data.most_common(1)[0][0] always return one same item in the lst, no matter lst is sorted or not, even shuffled.
# # but item returned by data.most_common(1)[0][0] can't be certained
# # use max(lst, key=data.get) can return the element that occurs first in the list in case of ties:
# def _most_common(lst):
#     data = Counter(lst)
#     return max(lst, key=data.get)


def _handle_one_strand_of_hole2(holeid, holechrom, ccs_strand, subreads_lines, contigs, motifs, args):
    refpos2ipd, refpos2pw = {}, {}
    refposes = set()
    subreads_info = []
    depth_all = len(subreads_lines)
    for subread_info in subreads_lines:
        chrom, start, strand, ipd, pw, qlocs_to_ref, refpos2querypos = subread_info
        # assert (strand == ccs_strand)  # duplicate

        if strand == "-":
            ipd = ipd[::-1]
            pw = pw[::-1]

        for rpos in refpos2querypos.keys():
            qpos = refpos2querypos[rpos]
            if (start+rpos) not in refposes:
                refposes.add((start+rpos))
                refpos2ipd[(start+rpos)] = []
                refpos2pw[(start+rpos)] = []
            refpos2ipd[(start+rpos)].append(ipd[qpos])
            refpos2pw[(start+rpos)].append(pw[qpos])

        # TODO: disable subreads_features for now
        # # to handle missing values (deletion in cigar, -1),
        # # append 1000 in the ipd/pw for index -1
        # subread_ipd = [np.insert(ipd, len(ipd), exceptval)[idx] for idx in qlocs_to_ref]
        # subread_pw = [np.insert(pw, len(pw), exceptval)[idx] for idx in qlocs_to_ref]
        # subreads_info.append((start, subread_ipd, subread_pw))

    # calculate mean/std of ipd/pw
    if len(refposes) == 0:
        return []

    refposes_list = list(refposes)
    refpos_max = np.max(refposes_list)
    refpos_min = np.min(refposes_list)
    ref_len = refpos_max - refpos_min + 1
    ipd_mean, ipd_std, pw_mean, pw_std = [exceptval] * ref_len, [exceptval] * ref_len, \
                                         [exceptval] * ref_len, [exceptval] * ref_len
    ipd_depth = [0] * ref_len
    for idx in range(0, ref_len):
        if (idx + refpos_min) in refposes:
            ipd_m, ipd_s = _cal_mean_n_std(refpos2ipd[idx + refpos_min])
            pw_m, pw_s = _cal_mean_n_std(refpos2pw[idx + refpos_min])
            ipd_mean[idx] = ipd_m
            ipd_std[idx] = ipd_s
            pw_mean[idx] = pw_m
            pw_std[idx] = pw_s
            ipd_depth[idx] = len(refpos2ipd[idx + refpos_min])
    del refpos2ipd
    del refpos2pw
    del refposes
    del refposes_list

    # TODO: disable subreads_features for now
    # # paddle subreads ipd/pw list to align ref
    # for idx in range(0, len(subreads_info)):
    #     start, subread_ipd, subread_pw = subreads_info[idx]
    #     pad_left = start - refpos_min
    #     pad_right = refpos_max + 1 - (start + len(subread_ipd))
    #     subread_ipd = [exceptval] * pad_left + subread_ipd + [exceptval] * pad_right
    #     subread_pw = [exceptval] * pad_left + subread_pw + [exceptval] * pad_right
    #     subreads_info[idx] = (subread_ipd, subread_pw)

    feature_list = _extract_kmer_features(holeid, holechrom, refpos_min, refpos_max, ccs_strand,
                                          ipd_mean, ipd_std, pw_mean, pw_std, ipd_depth, depth_all,
                                          subreads_info, motifs, args.mod_loc, args.seq_len,
                                          args.methy_label, args.depth, args.num_subreads, args.seed,
                                          contigs)

    return feature_list


def _comb_fb_features(fwd_feas, bwd_feas):
    if len(fwd_feas) <= 0 or len(bwd_feas) <= 0:
        return []
    fwd_feas = sorted(fwd_feas, key=lambda x: x[1])
    bwd_feas = sorted(bwd_feas[::-1], key=lambda x: x[1])
    comb_feas = []
    idx_f, idx_b = 0, 0
    while idx_f < len(fwd_feas) and idx_b < len(bwd_feas):
        ffea = fwd_feas[idx_f]
        bfea = bwd_feas[idx_b]
        fpos = ffea[1]
        bpos = bfea[1]
        if fpos == bpos - 1:
            ffea[4] = max(ffea[4], bfea[4])
            comb_feas.append(ffea[:13] + bfea[5:])
            idx_f += 1
            idx_b += 1
        elif fpos < bpos - 1:
            idx_f += 1
        else:
            idx_b += 1
    return comb_feas


def handle_one_hole2(hole_aligninfo, contigs, motifs, args):
    two_strands = args.two_strands
    comb_strands = args.comb_strands
    two_strands = True if comb_strands else two_strands

    holeid, hole_aligns = hole_aligninfo

    chrom2lines = {}
    chrom2starts = {}
    for sridx in range(len(hole_aligns)):
        words = hole_aligns[sridx]
        chrom = words[2]
        start = int(words[3]) - 1
        if chrom not in chrom2lines.keys():
            chrom2lines[chrom] = []
            chrom2starts[chrom] = []
        chrom2lines[chrom].append(sridx)
        chrom2starts[chrom].append(start)

    feature_list = []
    for holechrom in chrom2lines.keys():
        chromlineidxs = chrom2lines[holechrom]
        start_median = np.median(chrom2starts[holechrom])
        subreads_fwd, subreads_bwd = [], []

        for clidx in chromlineidxs:
            words = hole_aligns[clidx]
            flag, start = int(words[1]), int(words[3]) - 1
            chrom = words[2]
            if abs(start - start_median) > 100e3:  # filter reads aligned too far away from main alignments
                # print(holeid, holechrom, flag, start, start_median, "start - start_median too far")
                continue
            cigar = words[5]
            identity, qlocs_to_ref, refpos2querypos = _parse_cigar(cigar)
            if identity < args.identity:  # skip reads with low identity
                # print(holeid, holechrom, identity, "identity too low")
                continue

            # _, flag, chrom, start = words[0], int(words[1]), words[2], int(words[3]) - 1
            # assert (chrom == holechrom)  # duplicate
            strand = "+" if flag == 0 else "-"
            ipd, pw = [], []
            try:
                for i in range(11, len(words)):
                    if words[i].startswith("ip:B:C,"):
                        ipd = [int(ipdval) for ipdval in words[i].split(",")[1:]]
                    elif words[i].startswith("pw:B:C,"):
                        pw = [int(pwval) for pwval in words[i].split(",")[1:]]
                if len(ipd) == 0 or len(pw) == 0:
                    # print(holeid, "no ipd")
                    continue
                if len(ipd) != len(pw):
                    # print(holeid, "len ipd!=pw")
                    continue
            except ValueError:
                continue
            if not args.no_decode:
                ipd = [code2frames[ipdval] for ipdval in ipd]
                pw = [code2frames[pwval] for pwval in pw]
            ipd = _normalize_signals(ipd, args.norm)
            pw = _normalize_signals(pw, args.norm)

            # assert (flag == 0 or flag == 16)  # duplicate
            if flag == 0:
                subreads_fwd.append((chrom, start, strand, ipd, pw, qlocs_to_ref, refpos2querypos))
            else:
                subreads_bwd.append((chrom, start, strand, ipd, pw, qlocs_to_ref, refpos2querypos))

        # skip read which only have subreads in one strand
        if two_strands and (len(subreads_fwd) < 1 or len(subreads_bwd) < 1):
            continue

        fwd_features, bwd_features = [], []
        if len(subreads_fwd) >= args.depth:
            fwd_features = _handle_one_strand_of_hole2(holeid, holechrom, "+", subreads_fwd, contigs, motifs, args)
        if len(subreads_bwd) >= args.depth:
            bwd_features = _handle_one_strand_of_hole2(holeid, holechrom, "-", subreads_bwd, contigs, motifs, args)
        if comb_strands:
            feature_list += _comb_fb_features(fwd_features, bwd_features)
            del fwd_features
            del bwd_features
        else:
            feature_list += fwd_features + bwd_features
    return feature_list


def _features_to_str(features):
    """

    :param features: a tuple
    :return:
    """
    chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
        kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label = features

    kmer_depth_str = ",".join([str(x) for x in kmer_depth])
    kmer_ipdm_str = ",".join([str(x) for x in kmer_ipdm])
    kmer_ipds_str = ",".join([str(x) for x in kmer_ipds])
    kmer_pwm_str = ",".join([str(x) for x in kmer_pwm])
    kmer_pws_str = ",".join([str(x) for x in kmer_pws])

    if kmer_subr_ipds != subreads_value_default:
        kmer_subr_ipds_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_ipds])
        kmer_subr_pws_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_pws])
    else:
        kmer_subr_ipds_str = kmer_subr_ipds
        kmer_subr_pws_str = kmer_subr_pws

    return "\t".join([chrom, str(abs_loc), strand, str(holeid), str(depth_all), kmer_seq,
                      kmer_depth_str, kmer_ipdm_str, kmer_ipds_str, kmer_pwm_str, kmer_pws_str,
                      kmer_subr_ipds_str, kmer_subr_pws_str, str(label)])


def _features_to_str_combedfeatures(features):
    """

    :param features: a tuple
    :return:
    """
    chrom, abs_loc, strand, holeid, depth_all, \
        kmer_seq, kmer_depth, kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, \
        kmer_seq2, kmer_depth2, kmer_ipdm2, kmer_ipds2, kmer_pwm2, kmer_pws2, kmer_subr_ipds2, kmer_subr_pws2, \
        label = features

    kmer_depth_str = ",".join([str(x) for x in kmer_depth])
    kmer_ipdm_str = ",".join([str(x) for x in kmer_ipdm])
    kmer_ipds_str = ",".join([str(x) for x in kmer_ipds])
    kmer_pwm_str = ",".join([str(x) for x in kmer_pwm])
    kmer_pws_str = ",".join([str(x) for x in kmer_pws])
    if kmer_subr_ipds != subreads_value_default:
        kmer_subr_ipds_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_ipds])
        kmer_subr_pws_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_pws])
    else:
        kmer_subr_ipds_str = kmer_subr_ipds
        kmer_subr_pws_str = kmer_subr_pws

    kmer_depth_str2 = ",".join([str(x) for x in kmer_depth2])
    kmer_ipdm_str2 = ",".join([str(x) for x in kmer_ipdm2])
    kmer_ipds_str2 = ",".join([str(x) for x in kmer_ipds2])
    kmer_pwm_str2 = ",".join([str(x) for x in kmer_pwm2])
    kmer_pws_str2 = ",".join([str(x) for x in kmer_pws2])
    if kmer_subr_ipds2 != subreads_value_default:
        kmer_subr_ipds_str2 = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_ipds2])
        kmer_subr_pws_str2 = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_pws2])
    else:
        kmer_subr_ipds_str2 = kmer_subr_ipds2
        kmer_subr_pws_str2 = kmer_subr_pws2

    return "\t".join([chrom, str(abs_loc), strand, str(holeid), str(depth_all),
                      kmer_seq, kmer_depth_str, kmer_ipdm_str, kmer_ipds_str, kmer_pwm_str, kmer_pws_str,
                      kmer_subr_ipds_str, kmer_subr_pws_str,
                      kmer_seq2, kmer_depth_str2, kmer_ipdm_str2, kmer_ipds_str2, kmer_pwm_str2, kmer_pws_str2,
                      kmer_subr_ipds_str2, kmer_subr_pws_str2,
                      str(label)])


def _worker_extract(hole_align_q, featurestr_q, contigs, motifs, args):
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
        feature_strs = []
        if args.comb_strands:
            for feature in feature_list:
                feature_strs.append(_features_to_str_combedfeatures(feature))
        else:
            for feature in feature_list:
                feature_strs.append(_features_to_str(feature))
        featurestr_q.put(feature_strs)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)
        cnt_holesbatch += 1
        if cnt_holesbatch % 200 == 0:
            sys.stderr.write("extrac_features process-{}, {} hole_batches({}) "
                             "proceed\n".format(os.getpid(), cnt_holesbatch, args.holes_batch))
            sys.stderr.flush()
    sys.stderr.write("extrac_features process-{} ending, proceed {} "
                     "hole_batches({})\n".format(os.getpid(), cnt_holesbatch, args.holes_batch))


def _write_featurestr_to_file(write_fp, featurestr_q):
    sys.stderr.write('write_process-{} started\n'.format(os.getpid()))
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep(time_wait)
            if featurestr_q.empty():
                time.sleep(time_wait)
                continue
            features_str = featurestr_q.get()
            if features_str == "kill":
                sys.stderr.write('write_process-{} finished\n'.format(os.getpid()))
                break
            for one_features_str in features_str:
                wf.write(one_features_str + "\n")
            wf.flush()


def _get_holes(holeidfile):
    holes = set()
    with open(holeidfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            holeid = words[0]
            holes.add(holeid)
    sys.stderr.write("get {} holeids from {}\n".format(len(holes), holeidfile))
    return holes


def extract_subreads_features(args):
    sys.stderr.write("[extract_features]start..\n")
    start = time.time()

    inputpath = check_input_file(args.input)
    outputpath = check_output_file(args.output, inputpath)
    reference = os.path.abspath(args.ref)

    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")
    if not os.path.exists(reference):
        raise IOError("refernce(--ref) file does not exist!")

    if args.seq_len % 2 == 0:
        raise ValueError("seq_len must be odd")

    holeids_e = None if args.holeids_e is None else _get_holes(args.holeids_e)
    holeids_ne = None if args.holeids_ne is None else _get_holes(args.holeids_ne)

    contigs = DNAReference(reference).getcontigs()
    motifs = get_motif_seqs(args.motifs)

    hole_align_q = Queue()
    featurestr_q = Queue()

    p_read = mp.Process(target=worker_read, args=(inputpath, hole_align_q, args, holeids_e, holeids_ne))
    p_read.daemon = True
    p_read.start()

    ps_extract = []
    nproc = args.threads
    if nproc == 2:
        nproc -= 1
    if nproc > 2:
        nproc -= 2
    for _ in range(nproc):
        p = mp.Process(target=_worker_extract, args=(hole_align_q, featurestr_q, contigs, motifs, args))
        p.daemon = True
        p.start()
        ps_extract.append(p)

    # print("write_process started..")
    p_w = mp.Process(target=_write_featurestr_to_file, args=(outputpath, featurestr_q))
    p_w.daemon = True
    p_w.start()

    while True:
        # print("killing _worker_extract process")
        running = any(p.is_alive() for p in ps_extract)
        if not running:
            break

    for p in ps_extract:
        p.join()
    p_read.join()

    # sys.stderr.write("finishing the write_process..\n")
    featurestr_q.put("kill")
    p_w.join()

    endtime = time.time()
    sys.stderr.write("[extract_features]costs {:.1f} seconds\n".format(endtime - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=5, required=False,
                        help="number of threads, default 5")
    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input", "-i", type=str, required=True,
                         help="alignment results in bam/sam format. "
                              "We assume that all items/reads are sorted by hole_ids "
                              "in aligned.bam, which generated by align_subreads.py from subreads.bam.")
    p_input.add_argument("--ref", type=str, required=True,
                         help="path to genome reference to be aligned, in fasta/fa format.")
    p_input.add_argument("--holeids_e", type=str, default=None, required=False,
                         help="file contains holeids to be extracted, default None")
    p_input.add_argument("--holeids_ne", type=str, default=None, required=False,
                         help="file contains holeids not to be extracted, default None")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", type=str, required=False,
                          help="output file path to save the extracted features. "
                               "If not specified, use input_prefix.tsv as default.")

    p_extract = parser.add_argument_group("EXTRACT")
    p_extract.add_argument("--seq_len", type=int, default=21, required=False,
                           help="len of kmer. default 21")
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

    args = parser.parse_args()

    display_args(args, True)
    extract_subreads_features(args)


if __name__ == '__main__':
    main()
