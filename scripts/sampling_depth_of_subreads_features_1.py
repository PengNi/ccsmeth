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


basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
                 'Z': 'Z'}
iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}
samtools_exec = "samtools"


def _alphabet(letter, dbasepairs):
    if letter in dbasepairs.keys():
        return dbasepairs[letter]
    return 'N'


def complement_seq(base_seq, seq_type="DNA"):
    rbase_seq = base_seq[::-1]
    comseq = ''
    try:
        if seq_type == "DNA":
            comseq = ''.join([_alphabet(x, basepairs) for x in rbase_seq])
        elif seq_type == "RNA":
            comseq = ''.join([_alphabet(x, basepairs_rna) for x in rbase_seq])
        else:
            raise ValueError("the seq_type must be DNA or RNA")
    except Exception:
        print('something wrong in the dna/rna sequence.')
    return comseq


class DNAReference:
    def __init__(self, reffile):
        self._contignames = []
        self._contigs = {}  # contigname 2 contigseq
        with open(reffile, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        self._contigs[contigname] = contigseq
                        self._contignames.append(contigname)
                    contigname = line.strip()[1:].split(' ')[0]
                    contigseq = ''
                else:
                    # turn to upper case
                    contigseq += line.strip().upper()
            self._contigs[contigname] = contigseq
            self._contignames.append(contigname)

    def getcontigs(self):
        return self._contigs

    def getcontignames(self):
        return self._contignames


def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0):
    """

    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            sites.append(i+methyloc_in_motif)
    return sites


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)
    return recursive_permute(outbases)


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(',')

    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)
    return motif_seqs


def generate_samtools_view_cmd(path_to_samtools):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " view -@ 5"


def display_args(args, is_stderr=False):
    def print_outputstr(outstr):
        if is_stderr:
            sys.stderr.write(outstr + "\n")
        else:
            print(outstr)
    arg_vars = vars(args)
    outputstr = "# ===============================================\n## parameters: "
    print_outputstr(outputstr)
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            outputstr = "{}:\n\t{}".format(arg_key, arg_vars[arg_key])
            print_outputstr(outputstr)
    print_outputstr("# ===============================================")


def codecv1_to_frame():
    _code2frames = dict()
    for i in range(0, 64):
        _code2frames[i] = i

    frames = []
    for frame in range(64, 190 + 1, 2):
        frames.append(frame)
    for i in range(64, 128):
        _code2frames[i] = frames[i - 64]

    frames = []
    for frame in range(192, 444 + 1, 4):
        frames.append(frame)
    for i in range(128, 192):
        _code2frames[i] = frames[i - 128]

    frames = []
    for frame in range(448, 952 + 1, 8):
        frames.append(frame)
    for i in range(192, 256):
        _code2frames[i] = frames[i - 192]

    return _code2frames


code2frames = codecv1_to_frame()
queen_size_border = 5000
time_wait = 3

exceptval = 1000


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


def _worker_read(inputfile, hole_align_q, args):
    sys.stderr.write("read_input process-{} starts\n".format(os.getpid()))
    cmd_view_input = cmd_get_stdout_of_input(inputfile, args.path_to_samtools)
    sys.stderr.write("cmd to view input: {}\n".format(cmd_view_input))
    proc_read = Popen(cmd_view_input, shell=True, stdout=PIPE)
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
                flag = int(words[1])
                mapq = int(words[4])
                if not (flag == 0 or flag == 16):  # skip segment alignment
                    continue
                if mapq < args.mapq:  # skip low mapq alignment
                    continue
                if holeid != holeid_curr:
                    if len(hole_align_tmp) > 0:
                        cnt_holes += 1
                        hole_align_q.put((holeid_curr, hole_align_tmp))
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
                hole_align_q.put((holeid_curr, hole_align_tmp))
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
            sys.stderr.write("warning: got {} in cigar!".format(match[0][-1]))
    identity = float(cnt_m)/(cnt_s + cnt_m + cnt_i + cnt_d)
    # assert (q_adjseq[0] != "-")
    return identity, queryseq_poses, refpos2querypos


def _cal_mean_n_std(mylist):
    return round(np.mean(mylist), 6), round(np.std(mylist), 6)


def check_excpval(myarray):
    if exceptval in myarray:
        return True
    return False


def _extract_kmer_features(holeid, chrom, pos_min, pos_max, strand, ipd_mean, ipd_std, pw_mean, pw_std,
                           ipd_depth, depth_all, subreads_info, motifs, mod_loc, seq_len, label, depth,
                           num_subreads, seed, contigs):
    align_seq = contigs[chrom][pos_min:(pos_max+1)]
    if strand == "-":
        align_seq = complement_seq(align_seq)
        ipd_depth = ipd_depth[::-1]
    chromlen = len(contigs[chrom])
    if strand == "+":
        abs_start = pos_min
    else:
        abs_start = chromlen - (pos_min + len(align_seq))
    tsite_locs = get_refloc_of_methysite_in_motif(align_seq, set(motifs), mod_loc)
    if seq_len % 2 == 0:
        raise ValueError("seq_len must be odd")
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
            # kmer_ipds = ipd_std[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            # kmer_pwm = pw_mean[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            # kmer_pws = pw_std[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            kmer_depth = ipd_depth[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
            if np.mean(kmer_depth) < depth:
                continue

            feature = (chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth)

            kmer_subr_ipds, kmer_subr_pws = [], []
            for subreadinfo in subreads_info:
                subr_ipd, subr_pw = subreadinfo
                if strand == "-":
                    subr_ipd = subr_ipd[::-1]
                    subr_pw = subr_pw[::-1]
                kmer_subr_ipd = subr_ipd[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
                kmer_subr_pw = subr_pw[(offset_loc - num_bases):(offset_loc + num_bases + 1)]
                if check_excpval(kmer_subr_ipd):
                    continue
                kmer_subr_ipds.append(kmer_subr_ipd)
                kmer_subr_pws.append(kmer_subr_pw)

            if len(kmer_subr_ipds) < depth:
                continue
            feature = feature + (kmer_subr_ipds, kmer_subr_pws)
            feature = feature + (label, )
            feature_list.append(feature)
    return feature_list


def _handle_one_strand_of_hole2(holeid, holechrom, ccs_strand, subreads_lines, contigs, motifs, args):
    refpos2ipd, refpos2pw = {}, {}
    refposes = set()
    subreads_info = []
    depth_all = len(subreads_lines)
    for subread_info in subreads_lines:
        words, qlocs_to_ref, refpos2querypos = subread_info
        # _, flag, chrom, start, cigar, _ = words[0], int(words[1]), words[2], int(words[3]) - 1, words[5], words[9]
        _, flag, chrom, start = words[0], int(words[1]), words[2], int(words[3]) - 1
        assert (chrom == holechrom)
        strand = "+" if flag == 0 else "-"
        assert (strand == ccs_strand)

        ipd, pw = [], []
        for i in range(11, len(words)):
            if words[i].startswith("ip:B:C,"):
                ipd = [int(ipdval) for ipdval in words[i].split(",")[1:]]
            elif words[i].startswith("pw:B:C,"):
                pw = [int(pwval) for pwval in words[i].split(",")[1:]]
        if len(ipd) == 0 or len(pw) == 0:
            # print(holeid, subread_id, "no ipd")
            continue
        if not args.no_decode:
            ipd = [code2frames[ipdval] for ipdval in ipd]
            pw = [code2frames[pwval] for pwval in pw]
        ipd = _normalize_signals(ipd, args.norm)
        pw = _normalize_signals(pw, args.norm)
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

        # to handle missing values (deletion in cigar, -1),
        # append 1000 in the ipd/pw for index -1
        subread_ipd = [np.insert(ipd, len(ipd), exceptval)[idx] for idx in qlocs_to_ref]
        subread_pw = [np.insert(pw, len(pw), exceptval)[idx] for idx in qlocs_to_ref]
        subreads_info.append((start, subread_ipd, subread_pw))

    # calculate mean/std of ipd/pw
    if len(refposes) == 0:
        return []

    refpos_max = np.max(list(refposes))
    refpos_min = np.min(list(refposes))
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

    # paddle subreads ipd/pw list to align ref
    for idx in range(0, len(subreads_info)):
        start, subread_ipd, subread_pw = subreads_info[idx]
        pad_left = start - refpos_min
        pad_right = refpos_max + 1 - (start + len(subread_ipd))
        subread_ipd = [exceptval] * pad_left + subread_ipd + [exceptval] * pad_right
        subread_pw = [exceptval] * pad_left + subread_pw + [exceptval] * pad_right
        subreads_info[idx] = (subread_ipd, subread_pw)

    feature_list = _extract_kmer_features(holeid, holechrom, refpos_min, refpos_max, ccs_strand,
                                          ipd_mean, ipd_std, pw_mean, pw_std, ipd_depth, depth_all,
                                          subreads_info, motifs, args.mod_loc, args.seq_len,
                                          args.methy_label, args.depth, None, args.seed,
                                          contigs)

    return feature_list


def handle_one_hole2(hole_aligninfo, contigs, motifs, args):
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
            if abs(start - start_median) > 100e3:  # filter reads aligned too far away from main alignments
                # print(holeid, holechrom, flag, start, start_median, "start - start_median too far")
                continue
            cigar = words[5]
            identity, qlocs_to_ref, refpos2querypos = _parse_cigar(cigar)
            if identity < args.identity:  # skip reads with low identity
                # print(holeid, holechrom, subread_id, identity, "identity too low")
                continue
            assert (flag == 0 or flag == 16)
            if flag == 0:
                subreads_fwd.append((words, qlocs_to_ref, refpos2querypos))
            else:
                subreads_bwd.append((words, qlocs_to_ref, refpos2querypos))

        if args.two_strands and len(subreads_fwd) < args.depth and len(subreads_bwd) < args.depth:
            continue

        feature_list += _handle_one_strand_of_hole2(holeid, holechrom, "+", subreads_fwd, contigs, motifs, args)
        feature_list += _handle_one_strand_of_hole2(holeid, holechrom, "-", subreads_bwd, contigs, motifs, args)
    return feature_list


def _features_to_str(features):
    """

    :param features: a tuple
    :return:
    """
    chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
        kmer_subr_ipds, kmer_subr_pws, label = features

    kmer_depth_str = ",".join([str(x) for x in kmer_depth])

    if kmer_subr_ipds != "-":
        kmer_subr_ipds_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_ipds])
        kmer_subr_pws_str = ";".join([",".join([str(x) for x in y]) for y in kmer_subr_pws])
    else:
        kmer_subr_ipds_str = kmer_subr_ipds
        kmer_subr_pws_str = kmer_subr_pws

    return "\t".join([chrom, str(abs_loc), strand, str(holeid), str(depth_all), kmer_seq,
                      kmer_depth_str, kmer_subr_ipds_str, kmer_subr_pws_str, str(label)])


def _worker_extract(hole_align_q, featurestr_q, contigs, motifs, args):
    sys.stderr.write("extrac_features process-{} starts\n".format(os.getpid()))
    cnt_holes = 0
    while True:
        # print("hole_align_q size:", hole_align_q.qsize(), "; pid:", os.getpid())
        if hole_align_q.empty():
            time.sleep(time_wait)
        hole_aligninfo = hole_align_q.get()
        if hole_aligninfo == "kill":
            hole_align_q.put("kill")
            break
        # feature_list = handle_one_hole(hole_aligninfo, contigs, motifs, args)
        feature_list = handle_one_hole2(hole_aligninfo, contigs, motifs, args)
        feature_strs = []
        for feature in feature_list:
            feature_strs.append(_features_to_str(feature))
        featurestr_q.put(feature_strs)
        while featurestr_q.qsize() > queen_size_border:
            time.sleep(time_wait)
        cnt_holes += 1
        if cnt_holes % 1000 == 0:
            sys.stderr.write("extrac_features process-{}, {} holes proceed\n".format(os.getpid(),
                                                                                     cnt_holes))
            sys.stderr.flush()
    sys.stderr.write("extrac_features process-{} ending, proceed {} holes\n".format(os.getpid(),
                                                                                    cnt_holes))


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

    contigs = DNAReference(reference).getcontigs()
    motifs = get_motif_seqs(args.motifs)

    hole_align_q = Queue()
    featurestr_q = Queue()

    p_read = mp.Process(target=_worker_read, args=(inputpath, hole_align_q, args))
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
    parser = argparse.ArgumentParser("extract features with info of all subreads, and only keep reads with "
                                     "subread_depth >= 30, step1")
    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input", "-i", type=str, required=True,
                         help="alignment results in bam/sam format. "
                              "We assume that all items/reads are sorted by hole_ids "
                              "in aligned.bam/subreads.bam, which generated by align_subreads.py.")
    p_input.add_argument("--ref", type=str, required=True,
                         help="path to genome reference to be aligned, in fasta/fa format.")

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
    p_extract.add_argument("--threads", type=int, default=5, required=False,
                           help="number of threads, default 5")
    p_extract.add_argument("--mapq", type=int, default=30, required=False,
                           help="MAPping Quality cutoff for selecting alignment items, default 30")
    p_extract.add_argument("--identity", type=float, default=0.8, required=False,
                           help="identity cutoff for selecting alignment items, default 0.8")
    p_extract.add_argument("--two_strands", action="store_true", default=False, required=False,
                           help="after quality (mapq, identity) control, if then only using CCS reads "
                                "which have subreads in two strands")
    p_extract.add_argument("--depth", type=int, default=30, required=False,
                           help="(mean) depth (number of subreads) cutoff for "
                                "selecting high-quality aligned reads/kmers "
                                "per strand of a CCS, default 30.")
    p_extract.add_argument("--norm", action="store", type=str, choices=["zscore", "min-mean", "min-max", "mad"],
                           default="zscore", required=False,
                           help="method for normalizing ipd/pw in subread level. "
                                "zscore, min-mean, min-max or mad, default zscore")
    p_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                           help="not use CodecV1 to decode ipd/pw")
    p_extract.add_argument("--seed", type=int, default=111, required=False,
                           help="seed for randomly selecting subreads, default 111")
    p_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
                           help="full path to the executable binary samtools file. "
                                "If not specified, it is assumed that samtools is in "
                                "the PATH.")

    args = parser.parse_args()

    display_args(args, True)
    extract_subreads_features(args)


if __name__ == '__main__':
    main()
