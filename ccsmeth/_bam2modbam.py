import os
import argparse
import pysam
import re
import math
import time
import tabix
import pybedtools
import gzip

import multiprocessing as mp
from multiprocessing import Queue

from .utils.process_utils import complement_seq
from .utils.process_utils import max_queue_size

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

base = "C"
pred_base = "CG"

queue_size_border = max_queue_size
time_wait = 1


# generate per_read bed file ==============================
def _generate_per_read_calls(per_readsite, output):
    # chromosome, pos, strand, read_name, read_loc, depth, prob_0, prob_1, called_label, seq
    wf = open(output, "w")
    if per_readsite.endswith(".gz"):
        rf = gzip.open(per_readsite, "rt")
    else:
        rf = open(per_readsite, "r")
    holeid_curr = ""
    holeid_info = []
    cur_locs = set()
    # TODO: multi-process? - read 1 -> queue -> merge multi -> queue -> write
    for line in rf:
        words = line.strip().split("\t")
        holeid, loc, prob_1 = words[3], int(words[4]), float(words[7])
        if holeid != holeid_curr:
            if len(holeid_info) > 0:
                holeid_info = sorted(holeid_info, key=lambda x: x[0])
                holeid_info = list(zip(*holeid_info))
                locs = holeid_info[0]
                prob_1s = holeid_info[1]
                # format: holeid, start, end, num_locs, locs_list, probs_list
                wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                                    str(len(locs)),
                                    ",".join(list(map(str, locs))),
                                    ",".join(list(map(str, prob_1s)))]) + "\n")
            holeid_info = []
            cur_locs = set()
            holeid_curr = holeid
        if loc not in cur_locs:
            cur_locs.add(loc)
            holeid_info.append((loc, prob_1))
    if len(holeid_info) > 0:
        holeid_info = sorted(holeid_info, key=lambda x: x[0])
        holeid_info = list(zip(*holeid_info))
        locs = holeid_info[0]
        prob_1s = holeid_info[1]
        wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                            str(len(locs)),
                            ",".join(list(map(str, locs))),
                            ",".join(list(map(str, prob_1s)))]) + "\n")
    rf.close()
    wf.close()


def _sort_and_index_bedfile(bedfile):
    ori_bed = pybedtools.BedTool(bedfile)
    ori_bed.sort().moveto(bedfile)
    # TODO: check if zerobased=True need to be setted
    pysam.tabix_index(bedfile, force=True,
                      preset="bed",
                      keep_original=False)
    return bedfile if bedfile.endswith(".gz") else bedfile + ".gz"


def _generate_sorted_per_read_calls(per_readsite, output):
    fname, fext = os.path.splitext(per_readsite)
    if output is None:
        wfile = fname + ".per_read.bed"
    else:
        wfile = output
        if wfile.endswith(".gz"):
            wfile = wfile[:-3]
    _generate_per_read_calls(per_readsite, wfile)
    # sort+index
    bedfile = _sort_and_index_bedfile(wfile)
    return bedfile


# generate modbam files ============
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
    LOGGER.info("read {} reads from input file".format(cnt_all))


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
            locs, probs = query_locs_probs_of_a_read(seq_name, perread_tbx)
            if locs is not None:
                try:
                    mm_values = _convert_locs_to_mmtag(locs, seq_fwdseq)
                    ml_values = _convert_probs_to_mltag(probs)
                    mm_flag = 1
                except AssertionError:
                    # sys.stderr.write("AssertionError, skip this alignment.\n"
                    #       "\tDetails: {}, {}, {}\n".format(seq_name, locs, probs))
                    LOGGER.warning("AssertionError, skip this alignment-{}.".format(seq_name))
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
            LOGGER.info("write {} reads, in which {} were added mm tags".format(cnt_w, cnt_mm))
            break
        for walignseg in wreads_batch:
            mm_flag = walignseg[-1]
            write_alignedsegment(walignseg, w_bam)
            cnt_w += 1
            cnt_mm += mm_flag


def add_mm_ml_tags_to_bam(bamfile, per_readsite, modbamfile,
                          rm_pulse=True, threads=3,
                          reads_batch=100, mode="align"):
    LOGGER.info("[generate_modbam_file]starts")
    # sys.stderr.flush()
    start = time.time()

    LOGGER.info("generating per_read mod_calls..")
    per_read_file = _generate_sorted_per_read_calls(per_readsite, None)

    LOGGER.info("add per_read mod_calls to bam file..")
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

    if modbamfile.endswith(".bam"):
        try:
            LOGGER.info("sorting modbam file..")
            modbam_sorted = os.path.splitext(modbamfile)[0] + ".sorted.bam"
            pysam.sort("-o", modbam_sorted, "-@", str(threads), modbamfile)
            os.rename(modbam_sorted, modbamfile)
        except Exception:
            LOGGER.warning("failed sorting modbam file..")
        try:
            LOGGER.info("indexing modbam file..")
            pysam.index("-@", str(threads), modbamfile)
        except Exception:
            LOGGER.warning("failed indexing modbam file..")

    if os.path.exists(per_read_file):
        os.remove(per_read_file)
    if os.path.exists(per_read_file + ".tbi"):
        os.remove(per_read_file + ".tbi")

    endtime = time.time()
    LOGGER.info("[generate_modbam_file]costs {:.1f} seconds".format(endtime - start))


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
    parser.add_argument("--batch_size", type=int, required=False, default=100,
                        help="batch size of reads to be processed at one time, default 100")

    args = parser.parse_args()

    add_mm_ml_tags_to_bam(args.bam, args.per_readsite, args.modbam,
                          args.rm_pulse, args.threads,
                          args.batch_size)


if __name__ == '__main__':
    main()


