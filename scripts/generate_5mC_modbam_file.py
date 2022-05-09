import os
import argparse
import pysam
import re
import math
import sys
import time
import tabix
import numpy as np

from generate_per_read_modscall import _generate_sorted_per_read_calls

base = "C"
pred_base = "CG"


# def _fetch_a_read_from_tabixobj(readname, tabixobj):
#     # pysam
#     try:
#         rows = tabixobj.fetch(readname, parser=pysam.asTuple())
#         for row in rows:
#             return row
#     except ValueError:
#         return None


def _fetch_a_read_from_tabixobj2(readname, tabixobj):
    # pytabix is faster
    try:
        rows = tabixobj.query(readname, 0, 5000000)
        row_list = []
        row_lens = []
        for row in rows:
            row_list.append(row)
            row_lens.append(int(row[3]))
        if len(row_list) == 1:
            return row_list[0]
        else:
            max_idx = np.argmax(np.array(row_lens))
            return row_list[max_idx]
    except tabix.TabixError:
        return None


def _convert_locstr(locstr):
    return [int(x) for x in locstr.split(",")]


def _convert_probstr(probstr):
    return [float(x) for x in probstr.split(",")]


def query_locs_probs_of_a_read(readname, tabixobj):
    # row = _fetch_a_read_from_tabixobj(readname, tabixobj)
    row = _fetch_a_read_from_tabixobj2(readname, tabixobj)
    if row is not None:
        return _convert_locstr(row[4]), _convert_probstr(row[5])
    return None, None


def _convert_locs_to_mmtag(locs, seq_fwseq):
    assert len(locs) > 0
    base_locs = [i.start() for i in re.finditer(base, seq_fwseq)]
    base_orders = [-1] * len(locs)
    locs_idx = 0
    for base_idx in range(0, len(base_locs)):
        try:
            if base_locs[base_idx] == locs[locs_idx]:
                base_orders[locs_idx] = base_idx
                locs_idx += 1
        except IndexError:
            break
    assert base_orders[-1] != -1
    mm_idxes = [base_orders[0]]
    for i in range(1, len(base_orders)):
        mm_idxes.append(base_orders[i] - 1 - base_orders[i-1])
    return mm_idxes


def _convert_probs_to_mltag(probs):
    return [math.floor(prob * 256) for prob in probs]


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
    return (seq_name, flag, ref_name, ref_start, mapq, cigartuples,
            rnext, pnext, tlen, seq_seq, seq_qual, all_tags)


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
        new_tags.append(('MM', 'C+m,' + ",".join(list(map(str, mm_values)))))
        # new_tags.append(('ML', ml_values, 'B'))
        new_tags.append(('ML', ml_values))
    return new_tags


def write_alignedsegment(readitem_info, output_bam, mm_values, ml_values, rm_pulse=True):
    """
    Writes the readitem_info(tuple) to a bam file
    :param readitem_info:
    :param output_bam:
    :param mm_values:
    :param ml_values:
    :param rm_pulse:
    :return:
    """
    seq_name, flag, ref_name, ref_start, mapq, cigartuples, \
        rnext, pnext, tlen, seq_seq, seq_qual, all_tags = readitem_info
    new_tags = _refill_tags(all_tags, mm_values, ml_values, rm_pulse)

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
    if len(new_tags) >= 1:
        out_read.set_tags(new_tags)
    output_bam.write(out_read)


def add_mm_ml_tags_to_bam(bamfile, per_readsite, modbamfile,
                          rm_pulse=True, skip_unmapped=False, threads=3):
    sys.stderr.write("[generate_modbam_file]starts\n")
    start = time.time()

    sys.stderr.write("generating per_read mod_calls..\n")
    per_read_file = _generate_sorted_per_read_calls(per_readsite, None, is_gzip=True,
                                                    skip_unmapped=skip_unmapped)
    # perread_tbx = pysam.TabixFile(per_read_file)
    perread_tbx = tabix.open(per_read_file)

    sys.stderr.write("add per_read mod_calls to bam file..\n")
    if bamfile.endswith(".bam"):
        try:
            ori_bam = pysam.AlignmentFile(bamfile, 'rb')
        except ValueError:
            ori_bam = pysam.AlignmentFile(bamfile, 'rb', check_sq=False)
    else:
        ori_bam = pysam.AlignmentFile(bamfile, 'r')
    fname, fext = os.path.splitext(bamfile)
    if modbamfile is None:
        modbamfile = fname + ".modbam.bam"
    w_bam = pysam.AlignmentFile(modbamfile, "wb", template=ori_bam)

    cnt_all = cnt_write = cnt_mm_added = 0
    for readitem in ori_bam.fetch(until_eof=True):
        cnt_all += 1
        seq_name = readitem.query_name
        seq_fwseq = readitem.get_forward_sequence()

        # MM: Base modifications / methylation, ML:Base modification probabilities tags
        mm_values = ml_values = None
        # TODO: there are chances that supplementary alignments cannot get corresponding mm/ml values
        locs, probs = query_locs_probs_of_a_read(seq_name, perread_tbx)
        if locs is not None:
            try:
                mm_values = _convert_locs_to_mmtag(locs, seq_fwseq)
                ml_values = _convert_probs_to_mltag(probs)
                cnt_mm_added += 1
            except AssertionError:
                # print("AssertionError, skip this alignment.\n"
                #       "\tDetails: {}, {}, {}".format(seq_name, locs, probs))
                print("AssertionError, skip this alignment-{}.".format(seq_name))
                continue

        readitem_info = _get_necessary_alignment_items(readitem)
        write_alignedsegment(readitem_info, w_bam, mm_values, ml_values, rm_pulse)
        cnt_write += 1

    # perread_tbx.close()
    ori_bam.close()
    w_bam.close()

    if modbamfile.endswith(".bam"):
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
    sys.stderr.write("[generate_modbam_file]costs {:.1f} seconds, {} reads in total, "
                     "{} reads written, {} reads added mm tags\n".format(endtime - start,
                                                                         cnt_all,
                                                                         cnt_write,
                                                                         cnt_mm_added))


def main():
    parser = argparse.ArgumentParser("add MM/ML tags to bam/sam")
    parser.add_argument("--per_readsite", type=str, required=True, help="from call_mods module")
    parser.add_argument("--bam", type=str, required=True, help="input bam file")
    parser.add_argument("--modbam", type=str, required=False, help="output modbam file")
    parser.add_argument("--rm_pulse", action="store_true", default=False, required=False,
                        help="if remove ipd/pw tags in the bam file")
    parser.add_argument("--skip_unmapped", action="store_true", default=False, required=False,
                        help="skip unmapped sites")
    parser.add_argument("--threads", "-p", action="store", type=int, default=10,
                        required=False, help="number of threads to be used, default 10.")

    args = parser.parse_args()

    add_mm_ml_tags_to_bam(args.bam, args.per_readsite, args.modbam, args.rm_pulse, args.skip_unmapped)


if __name__ == '__main__':
    main()


