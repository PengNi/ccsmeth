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

time_wait = 1
key_sep = "||"


# imports ========================================================================
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


CODE2CIGAR = "MIDNSHP=XB"


def compute_pct_identity(cigarnum_array):
    try:
        nalign, nmatch = 0, 0
        for idx in range(len(CODE2CIGAR)):
            if idx not in {4, 5}:
                nalign += cigarnum_array[idx]
            if idx in {0, 7}:
                nmatch += cigarnum_array[idx]
        return nmatch / float(nalign)
    except IndexError:
        return 0.
    except ZeroDivisionError:
        return 0.


def index_bam_if_needed2(inputfile, threads):
    if str(inputfile).endswith(".bam") and not os.path.exists(inputfile + ".bai"):
        sys.stderr.write("indexing bam file-{}\n".format(inputfile))
        pysam.index("-@", str(threads), inputfile)


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


basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D',
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D',
                 'Z': 'Z'}


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
# =================================================================================


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
        if is_reverse:
            for q_pos, r_pos in readitem.get_aligned_pairs(matches_only=True):
                if ref_start <= r_pos < ref_end:
                    if q_pos in modlocs:
                        if r_pos not in refposes_rev:
                            refposes_rev.add(r_pos)
                            refposinfo_rev[r_pos] = []
                        refposinfo_rev[r_pos].append((moddict[q_pos], hap, readitem.query_name))
        else:
            for q_pos, r_pos in readitem.get_aligned_pairs(matches_only=True):
                if ref_start <= r_pos < ref_end:
                    if q_pos in modlocs:
                        if r_pos not in refposes:
                            refposes.add(r_pos)
                            refposinfo[r_pos] = []
                        refposinfo[r_pos].append((moddict[q_pos], hap, readitem.query_name))
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
    perreadsite_res = []
    for refpos in refposinfo:
        if motifs_filter is not None:
            motif_seq = dnacontigs[ref_name][(refpos + fwd_s):(refpos + fwd_e)]
            if motif_seq not in motifs_filter:
                continue
        for refpositem in refposinfo[refpos]:
            mprob, hap, read_name = refpositem
            label = 1 if mprob > 0.5 else 0
            perreadsite_res.append((ref_name, refpos, "+", read_name, "-1", "1,1",
                                    1 - mprob, mprob, label, "-"))
    if not (args.motifs == "CG" and not args.no_comb):
        for refpos in refposinfo_rev:
            if motifs_filter is not None:
                motif_seq = complement_seq(dnacontigs[ref_name][(refpos + rev_s):(refpos + rev_e)])
                if motif_seq not in motifs_filter:
                    continue
            for refpositem in refposinfo_rev[refpos]:
                mprob, hap, read_name = refpositem
                label = 1 if mprob > 0.5 else 0
                perreadsite_res.append((ref_name, refpos, "-", read_name, "-1", "1,1",
                                        1 - mprob, mprob, label, "-"))
    return perreadsite_res


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
        bed_all = _readmods_to_bed_of_one_region(bam_reader, region,
                                                 dnacontigs, motifs_filter, args)
        if len(bed_all) > 0:
            bed_q.put(bed_all)
            while bed_q.qsize() > args.threads * 3:
                time.sleep(time_wait)

    bam_reader.close()
    sys.stderr.write("worker_gen_bed process-{} ending, proceed {} regions\n".format(os.getpid(),
                                                                                     cnt_regions))


def _worker_write_bed_result(output_prefix, bed_q):
    sys.stderr.write('write_process-{} starts\n'.format(os.getpid()))

    op_all = output_prefix + ".per_readsite.tsv"
    wf_all = open(op_all, "w")
    while True:
        if bed_q.empty():
            time.sleep(time_wait)
            continue
        bed_res = bed_q.get()
        if bed_res == "kill":
            break
        bed_all = bed_res
        for beditem in bed_all:
            wf_all.write("\t".join(list(map(str, beditem))) + "\n")
    wf_all.flush()
    wf_all.close()
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
    if args.refsites_only:
        motifs_filter = motifs
        print("[###] --refsites_only is set as True, gonna keep only motifs({}) sites of genome reference "
              "in the results".format(motifs_filter))

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
                     args=(args.output, bed_q))
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

    scfb_callfreq = parser.add_argument_group("CALL_FREQ")
    scfb_callfreq.add_argument('--modtype', type=str, action="store", required=False, default="5mC",
                               choices=["5mC", ],
                               help='modification type, default 5mC.')
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
                               help="only keep sites which is a target motif in reference")

    args = parser.parse_args()

    call_mods_frequency_from_bamfile(args)


if __name__ == '__main__':
    sys.exit(main())
