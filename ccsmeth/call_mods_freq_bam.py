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

from .utils.ref_reader import DNAReference
from .utils.process_utils import compute_pct_identity
from .utils.process_utils import is_file_empty
from .utils.process_utils import index_bam_if_needed2
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import complement_seq
from .utils.process_utils import get_refloc_of_methysite_in_motif

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
        sys.stderr.write("adjust regions for CG motif\n")
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
    sys.stderr.write("worker_split_regions process-{} starts\n".format(os.getpid()))
    ref_chunks = _get_reference_chunks(dnacontigs, args.contigs, args.chunk_len, args.motifs)
    sys.stderr.write("worker_split_regions process-{} generates {} regions\n".format(os.getpid(),
                                                                                     len(ref_chunks)))
    with tqdm(total=len(ref_chunks),
              desc="region_reader") as pbar:
        for ref_chunk in ref_chunks:
            region_q.put(ref_chunk)
            pbar.update(1)
            while region_q.qsize() > args.threads * 3:
                time.sleep(time_wait)
    region_q.put("kill")


def _cal_mod_prob(ml_value):
    # WARN: if ori_prob exactly = 0.5, ml_value = floor(0.5*256) = 128
    # WARN: then if we use the following to convert ml_value to prob, the the new_prob = 0.5000001.
    # WARN: if we use the rule: label = 1 if prob>0.5 else 0, then the label of new_prob will be
    # WARN: different with the label of ori_prob
    return round(ml_value / float(256) + 0.000001, 6)


def _get_moddict(readitem, modbase="C", modification="m"):
    """

    :param readitem:
    :return: moddict: query_pos(in alignment strand) 2 mod_probs([0,1])
    """
    # mmtag, mltag = None, None
    # try:
    #     mmtag = readitem.get_tag('MM')
    #     mltag = readitem.get_tag('ML')
    #     seq_fwdseq = readitem.get_forward_sequence()
    #     is_reverse = readitem.is_reverse
    # except KeyError:
    #     pass
    # if mmtag is None or mltag is None:
    #     return {}

    # use .modified_bases instead of MM/ML tags to get moddict
    modinfo = readitem.modified_bases
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


def _cal_modprob_in_count_mode(modprobs, prob_cf=0):
    cnt_all, cnt_mod = 0, 0
    for modprob in modprobs:
        if abs(modprob - (1 - modprob)) < prob_cf:
            continue
        cnt_all += 1
        if modprob > 0.5:
            cnt_mod += 1
    modfreq = cnt_mod / float(cnt_all) if cnt_all > 0 else 0.
    return cnt_all, cnt_mod, modfreq


def _call_modfreq_of_one_region(refpos2modinfo, args):
    refpos_results = []
    if args.call_mode == "count":
        for refpos in sorted(refpos2modinfo.keys()):
            modinfo = refpos2modinfo[refpos]
            total_mods, hp1_mods, hp2_mods = [], [], []
            for modprob, hap in modinfo:
                total_mods.append(modprob)
                if hap == 1:
                    hp1_mods.append(modprob)
                elif hap == 2:
                    hp2_mods.append(modprob)
            info_all = _cal_modprob_in_count_mode(total_mods, args.prob_cf) if len(total_mods) > 0 else None
            info_hp1 = _cal_modprob_in_count_mode(hp1_mods, args.prob_cf) if len(hp1_mods) > 0 else None
            info_hp2 = _cal_modprob_in_count_mode(hp2_mods, args.prob_cf) if len(hp2_mods) > 0 else None
            refpos_results.append((refpos, info_all, info_hp1, info_hp2))
    elif args.call_mode == "aggregate":
        pass

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
    if args.motifs == "CG" and not args.no_comb:
        for rev_pos in refposes_rev:
            fwd_pos = rev_pos - 1
            if fwd_pos not in refposes:
                refposes.add(fwd_pos)
                refposinfo[fwd_pos] = []
            refposinfo[fwd_pos] += refposinfo_rev[rev_pos]
        del refposinfo_rev
        del refposes_rev
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
    sys.stderr.write("worker_gen_bed process-{} starts\n".format(os.getpid()))
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
            while bed_q.qsize() > args.threads * 3:
                time.sleep(time_wait)

    bam_reader.close()
    sys.stderr.write("worker_gen_bed process-{} ending, proceed {} regions\n".format(os.getpid(),
                                                                                     cnt_regions))


def _write_one_line(beditem, wf, is_bed):
    ref_name, refpos, strand, cov, met, metprob = beditem
    if is_bed:
        wf.write("\t".join([ref_name, str(refpos), str(refpos + 1), ".", str(cov),
                            strand, str(refpos), str(refpos + 1),
                            "0,0,0", str(cov), str(int(round(metprob * 100 + 0.001, 0)))]) + "\n")
    else:
        wf.write("\t".join([ref_name, str(refpos), str(refpos + 1), strand, "-", "-", str(met),
                            str(cov-met), str(cov), str(round(metprob + 0.000001, 4))]) + "\n")


def _worker_write_bed_result(output_prefix, bed_q, args):
    sys.stderr.write('write_process-{} starts\n'.format(os.getpid()))

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
            sys.stderr.write('write_process-{} sorting results\n'.format(os.getpid()))
            ori_bed = pybedtools.BedTool(bedfile)
            ori_bed.sort().moveto(bedfile)
        if args.gzip:
            sys.stderr.write('write_process-{} gzipping results\n'.format(os.getpid()))
            pysam.tabix_index(bedfile, force=True,
                              preset="bed",
                              keep_original=False)
    sys.stderr.write('write_process-{} finished\n'.format(os.getpid()))


def call_mods_frequency_from_bamfile(args):
    print("[main]call_freq_bam starts..")
    start = time.time()

    inputpath = _check_input_file(args.input_bam)
    index_bam_if_needed2(inputpath, args.threads)
    if not os.path.exists(args.ref):
        raise ValueError("--ref does not exist!")
    dnacontigs = DNAReference(args.ref).getcontigs()
    motifs = get_motif_seqs(args.motifs)

    motifs_filter = None
    if args.refsites_only or args.refsites_all:
        motifs_filter = motifs
        print("[###] --refsites_only (or/and --refsites_all) is set as True, gonna keep only motifs({}) sites "
              "of genome reference in the results".format(motifs_filter))

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

    print("[main]call_freq_bam costs %.1f seconds.." % (time.time() - start))


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
                               help='this is to remove ambiguous calls. '
                               'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                               'means use all calls. range [0, 1], default 0.0.')
    scfb_callfreq.add_argument("--hap_tag", type=str, action="store", required=False, default="HP",
                               help="haplotype tag, default HP")
    scfb_callfreq.add_argument("--mapq", type=int, default=10, required=False,
                               help="MAPping Quality cutoff for selecting alignment items, default 10")
    scfb_callfreq.add_argument("--identity", type=float, default=0.75, required=False,
                               help="identity cutoff for selecting alignment items, default 0.75")
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
                               help="dont combine fwd/rev reads of one CG. [Only works when motifs is CG]")
    scfb_callfreq.add_argument('--refsites_only', action='store_true', default=False,
                               help="only keep sites which are target motifs in both reference and reads")
    scfb_callfreq.add_argument('--refsites_all', action='store_true', default=False,
                               help="output all covered sites which are target motifs in reference. "
                                    "--refsites_all is True, also means we do not output sites which "
                                    "are target motifs only in reads.")

    args = parser.parse_args()

    call_mods_frequency_from_bamfile(args)


if __name__ == '__main__':
    sys.exit(main())
