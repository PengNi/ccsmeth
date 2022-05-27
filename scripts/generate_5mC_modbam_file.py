import os
import argparse
import pysam
import re
import math
import sys
import time
import tabix

import multiprocessing as mp
from multiprocessing import Queue

from generate_per_read_modscall import _generate_sorted_per_read_calls

base = "C"
pred_base = "CG"

queue_size_border = 1000
time_wait = 1

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


def open_input_bamfile(bamfile, threads=1):
    if bamfile.endswith(".bam"):
        try:
            ori_bam = pysam.AlignmentFile(bamfile, 'rb', threads=threads)
        except ValueError:
            ori_bam = pysam.AlignmentFile(bamfile, 'rb', check_sq=False, threads=threads)
    else:
        ori_bam = pysam.AlignmentFile(bamfile, 'r', threads=threads)
    return ori_bam


def _get_necessary_alignment_items(readitem):
    seq_name = readitem.query_name
    flag = readitem.flag
    ref_name = readitem.reference_name
    ref_start = readitem.reference_start
    mapq = readitem.mapping_quality
    cigartuples = readitem.cigartuples
    rnext = readitem.next_reference_name
    pnext = readitem.next_reference_start
    tlen = readitem.template_length
    seq_seq = readitem.query_sequence
    seq_qual = readitem.query_qualities
    all_tags = readitem.get_tags(with_value_type=True)
    is_reverse = readitem.is_reverse
    return (seq_name, flag, ref_name, ref_start, mapq, cigartuples,
            rnext, pnext, tlen, seq_seq, seq_qual, all_tags, is_reverse)


def _worker_reader(bamfile, batch_size, rreads_q, threads=1):
    ori_bam = open_input_bamfile(bamfile, threads=threads)
    cnt_all = 0
    reads_batch = []
    for readitem in ori_bam.fetch(until_eof=True):
        readitem_info = _get_necessary_alignment_items(readitem)
        reads_batch.append(readitem_info)
        cnt_all += 1
        if cnt_all % batch_size == 0:
            rreads_q.put(reads_batch)
            reads_batch = []
            while rreads_q.qsize() > queue_size_border:
                time.sleep(time_wait)
    ori_bam.close()
    if len(reads_batch) > 0:
        rreads_q.put(reads_batch)
    rreads_q.put("kill")
    sys.stderr.write("read {} reads from input file\n".format(cnt_all))


def _convert_locstr(locstr):
    return [int(x) for x in locstr.split(",")]


def _convert_probstr(probstr):
    return [float(x) for x in probstr.split(",")]


def _fetch_locprobs_of_a_read_from_tabixobj2(readname, tabixobj):
    # pytabix is faster
    try:
        rows = tabixobj.query(readname, 0, 5000000)
        row_list = []
        for row in rows:
            row_list.append(row)
        if len(row_list) == 1:
            return _convert_locstr(row_list[0][4]), _convert_probstr(row_list[0][5])
        else:
            locs_0, probs_0 = _convert_locstr(row_list[0][4]), _convert_probstr(row_list[0][5])
            loc_probs = list(zip(locs_0, probs_0))
            locs_set = set(locs_0)
            for ridx in range(1, len(row_list)):
                locs_tmp, probs_tmp = _convert_locstr(row_list[ridx][4]), _convert_probstr(row_list[ridx][5])
                for lidx in range(len(locs_tmp)):
                    if locs_tmp[lidx] not in locs_set:
                        locs_set.add(locs_tmp[lidx])
                        loc_probs.append((locs_tmp[lidx], probs_tmp[lidx]))
            loc_probs = sorted(loc_probs, key=lambda x: x[0])
            loc_probs = list(zip(*loc_probs))
            return loc_probs[0], loc_probs[1]
    except tabix.TabixError:
        return None


def query_locs_probs_of_a_read(readname, tabixobj):
    loc_prob = _fetch_locprobs_of_a_read_from_tabixobj2(readname, tabixobj)
    if loc_prob is not None:
        return loc_prob[0], loc_prob[1]
    return None, None


def _convert_locs_to_mmtag(locs, seq_fwseq):
    assert len(locs) > 0
    base_alllocs = [i.start() for i in re.finditer(base, seq_fwseq)]
    base_orders = [-1] * len(locs)
    order_idx = 0
    for base_idx in range(0, len(base_alllocs)):
        try:
            if base_alllocs[base_idx] == locs[order_idx]:
                base_orders[order_idx] = base_idx
                order_idx += 1
        except IndexError:
            break
    assert base_orders[-1] != -1
    mm_idxes = [base_orders[0]]
    for i in range(1, len(base_orders)):
        mm_idxes.append(base_orders[i] - 1 - base_orders[i-1])
    return mm_idxes


def _convert_probs_to_mltag(probs):
    # force returned values in [0, 255]
    return [math.floor(prob * 256) if prob < 1 else 255 for prob in probs]


def _refill_tags(all_tags, mm_values, ml_values, rm_pulse=True):
    new_tags = []
    # TODO: if with_value_type, pysam has a bug (0.19.0, pysam/libcalignedsegment.pyx line 396)
    for tagtuple in all_tags:
        if tagtuple[0] in {"MM", "ML"}:
            continue
        if rm_pulse and tagtuple[0] in {"fi", "fp", "ri", "rp"}:
            continue
        # new_tags.append(tagtuple)
        new_tags.append((tagtuple[0], tagtuple[1]))
    if mm_values is not None:
        # new_tags.append(('MM', 'C+m,' + ",".join(list(map(str, mm_values))), 'Z'))
        new_tags.append(('MM', 'C+m,' + ",".join(list(map(str, mm_values))) + ";"))
        # new_tags.append(('ML', ml_values, 'B'))
        new_tags.append(('ML', ml_values))
    return new_tags


def _worker_process_reads_batch(rreads_q, wreads_q, tabix_file, rm_pulse=True):
    # perread_tbx = pysam.TabixFile(tabix_file)
    perread_tbx = tabix.open(tabix_file)
    while True:
        if rreads_q.empty():
            time.sleep(time_wait)
            continue
        reads_batch = rreads_q.get()
        if reads_batch == "kill":
            rreads_q.put("kill")
            break
        wreads_tmp = []
        for rread in reads_batch:
            seq_name, flag, ref_name, ref_start, mapq, cigartuples, rnext, pnext, tlen, \
                seq_seq, seq_qual, all_tags, is_reverse = rread
            seq_fwdseq = complement_seq(seq_seq) if is_reverse else seq_seq

            # MM: Base modifications / methylation, ML:Base modification probabilities tags
            mm_values = ml_values = None
            mm_flag = 0
            # TODO: there are chances that supplementary alignments cannot get corresponding mm/ml values
            locs, probs = query_locs_probs_of_a_read(seq_name, perread_tbx)
            if locs is not None:
                try:
                    mm_values = _convert_locs_to_mmtag(locs, seq_fwdseq)
                    ml_values = _convert_probs_to_mltag(probs)
                    mm_flag = 1
                except AssertionError:
                    # sys.stderr.write("AssertionError, skip this alignment.\n"
                    #       "\tDetails: {}, {}, {}\n".format(seq_name, locs, probs))
                    sys.stderr.write("AssertionError, skip this alignment-{}.\n".format(seq_name))
                    continue
            new_tags = _refill_tags(all_tags, mm_values, ml_values, rm_pulse)
            wreads_tmp.append((seq_name, flag, ref_name, ref_start, mapq, cigartuples, rnext, pnext, tlen,
                               seq_seq, seq_qual, new_tags, mm_flag))
        if len(wreads_tmp) > 0:
            wreads_q.put(wreads_tmp)
            while wreads_q.qsize() > queue_size_border:
                time.sleep(time_wait)


def write_alignedsegment(readitem_info, output_bam):
    """
    Writes the readitem_info(tuple) to a bam file
    :param readitem_info:
    :param output_bam:
    :return:
    """
    seq_name, flag, ref_name, ref_start, mapq, cigartuples, \
        rnext, pnext, tlen, seq_seq, seq_qual, all_tags, mm_flag = readitem_info

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


def _worker_write_modbam(wreads_q, modbamfile, inputbamfile, threads=1):
    ori_bam = open_input_bamfile(inputbamfile)
    w_bam = pysam.AlignmentFile(modbamfile, "wb", template=ori_bam, threads=threads)
    ori_bam.close()
    cnt_w, cnt_mm = 0, 0
    while True:
        if wreads_q.empty():
            time.sleep(time_wait)
            continue
        wreads_batch = wreads_q.get()
        if wreads_batch == "kill":
            w_bam.close()
            sys.stderr.write("write {} reads, in which {} were added mm tags\n".format(cnt_w,
                                                                                       cnt_mm))
            break
        for walignseg in wreads_batch:
            mm_flag = walignseg[-1]
            write_alignedsegment(walignseg, w_bam)
            cnt_w += 1
            cnt_mm += mm_flag


def add_mm_ml_tags_to_bam(bamfile, per_readsite, modbamfile,
                          rm_pulse=True, threads=3,
                          reads_batch=100, mode="align"):
    sys.stderr.write("[generate_modbam_file]starts\n")
    start = time.time()

    sys.stderr.write("generating per_read mod_calls..\n")
    per_read_file = _generate_sorted_per_read_calls(per_readsite, None)

    sys.stderr.write("add per_read mod_calls to bam file..\n")
    rreads_q = Queue()
    wreads_q = Queue()

    nproc = threads
    if nproc < 5:
        nproc = 5
    if nproc > 8:
        threads_r, threads_w = 4, 4
    elif nproc > 6:
        threads_r, threads_w = 3, 3
    else:
        threads_r, threads_w = 2, 2

    p_read = mp.Process(target=_worker_reader,
                        args=(bamfile, reads_batch, rreads_q, threads_r))
    p_read.daemon = True
    p_read.start()

    ps_gen = []
    for _ in range(nproc - threads_r - threads_w):
        p_gen = mp.Process(target=_worker_process_reads_batch,
                           args=(rreads_q, wreads_q, per_read_file, rm_pulse))
        p_gen.daemon = True
        p_gen.start()
        ps_gen.append(p_gen)

    fname, fext = os.path.splitext(bamfile)
    if modbamfile is None:
        modbamfile = fname + ".modbam.bam"
    p_w = mp.Process(target=_worker_write_modbam,
                     args=(wreads_q, modbamfile, bamfile, threads_w))
    p_w.daemon = True
    p_w.start()

    for p in ps_gen:
        p.join()
    p_read.join()
    wreads_q.put("kill")
    p_w.join()

    if modbamfile.endswith(".bam") and mode == "align":
        sys.stderr.write("sorting and indexing new bam file..\n")
        modbam_sorted = modbamfile + ".sorted.bam"
        pysam.sort("-o", modbam_sorted, "-@", str(threads), modbamfile)
        os.rename(modbam_sorted, modbamfile)
        pysam.index("-@", str(threads), modbamfile)

    if os.path.exists(per_read_file):
        os.remove(per_read_file)
    if os.path.exists(per_read_file + ".tbi"):
        os.remove(per_read_file + ".tbi")

    endtime = time.time()
    sys.stderr.write("[generate_modbam_file]costs {:.1f} seconds\n".format(endtime - start))


def main():
    parser = argparse.ArgumentParser("add MM/ML tags to bam/sam")
    parser.add_argument("--per_readsite", type=str, required=True, help="from call_mods module")
    parser.add_argument("--bam", type=str, required=True, help="input bam file")
    parser.add_argument("--mode", type=str, default="align", required=False,
                        choices=["denovo", "align"],
                        help="denovo mode: add tags to unaligned hifi.bam;\n"
                             "align mode: add tags to aligned hifi.bam. default: align")

    parser.add_argument("--modbam", type=str, required=False, help="output modbam file")
    parser.add_argument("--rm_pulse", action="store_true", default=False, required=False,
                        help="if remove ipd/pw tags in the bam file")

    parser.add_argument("--threads", "-p", action="store", type=int, default=10,
                        required=False, help="number of threads to be used, default 10.")
    parser.add_argument("--batch_size", type=int, required=False, default=100,
                        help="batch size of reads to be processed at one time, default 100")

    args = parser.parse_args()

    add_mm_ml_tags_to_bam(args.bam, args.per_readsite, args.modbam,
                          args.rm_pulse, args.threads, args.batch_size,
                          args.mode)


if __name__ == '__main__':
    main()


