#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

from __future__ import absolute_import

import argparse
import os
import sys
import gzip
import time

import multiprocessing as mp
from multiprocessing import Queue
from .utils.process_utils import is_file_empty
import uuid

from .utils.process_utils import default_ref_loc
from .utils.process_utils import complement_seq

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

time_wait = 1
key_sep = "||"


class ModRecord:
    def __init__(self, fields):
        self._chromosome = fields[0]
        self._pos = int(fields[1])
        self._strand = fields[2]
        self._site_key = key_sep.join([self._chromosome, str(self._pos), self._strand])

        self._holeid = fields[3]

        self._loc = fields[4]
        self._depthstr = fields[5]
        self._depth = max(list(map(int, self._depthstr.split(",")))) if "," in self._depthstr else int(self._depthstr)
        self._prob_0 = float(fields[6])
        self._prob_1 = float(fields[7])
        self._called_label = int(fields[8])
        self._kmer = fields[9] if len(fields) > 9 else "-"

    def is_record_callable(self, prob_threshold):
        if abs(self._prob_0 - self._prob_1) < prob_threshold:
            return False
        return True


def split_key(key):
    words = key.split(key_sep)
    return words[0], int(words[1]), words[2]


class SiteStats:
    def __init__(self, kmer):

        self._kmer = kmer

        self._prob_0 = 0.0
        self._prob_1 = 0.0
        self._met = 0
        self._unmet = 0
        self._coverage = 0
        # self._rmet = -1.0


def calculate_mods_frequency(mods_files, prob_cf, rm_1strand=False, contig_name=None):
    """
    call mod_freq from call_mods files
    :param mods_files: a list of call_mods files
    :param prob_cf:
    :param rm_1strand:
    :param contig_name:
    :return: key2value obj
    """
    sitekeys = set()
    sitekey2stats = dict()

    if type(mods_files) is str:
        mods_files = [mods_files, ]

    count, used = 0, 0
    for mods_file in mods_files:
        if mods_file.endswith(".gz"):
            infile = gzip.open(mods_file, 'rt')
        else:
            infile = open(mods_file, 'r')
        for line in infile:
            words = line.strip().split("\t")
            mod_record = ModRecord(words)
            if mod_record._pos == default_ref_loc:
                continue
            if contig_name is not None and mod_record._chromosome != contig_name:
                continue
            count += 1
            if rm_1strand and "," not in mod_record._depthstr:
                continue
            if not mod_record.is_record_callable(prob_cf):
                continue
            if mod_record._site_key not in sitekeys:
                sitekeys.add(mod_record._site_key)
                sitekey2stats[mod_record._site_key] = SiteStats(mod_record._kmer)
            sitekey2stats[mod_record._site_key]._prob_0 += mod_record._prob_0
            sitekey2stats[mod_record._site_key]._prob_1 += mod_record._prob_1
            sitekey2stats[mod_record._site_key]._coverage += 1
            if mod_record._called_label == 1:
                sitekey2stats[mod_record._site_key]._met += 1
            else:
                sitekey2stats[mod_record._site_key]._unmet += 1
            used += 1
        infile.close()
    if count == 0:
        raise ValueError("No modification calls found in {}..".format(mods_files))
    if contig_name is None:
        LOGGER.info("{:.2f}% ({} of {}) calls used..".format(used/float(count) * 100, used, count))
    else:
        LOGGER.info("{:.2f}% ({} of {}) calls used for {}..".format(used / float(count) * 100, used, count, contig_name))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file, is_sort, is_bed, is_gzip,
                        motifs=None, mod_loc=None, dnacontigs=None):
    """
    write methylfreq of sites into files
    :param sitekey2stats:
    :param result_file:
    :param is_sort: sorted by poses
    :param is_bed: in bed format or not
    :param is_gzip:
    :param motifs:
    :param mod_loc:
    :param dnacontigs:
    :return:
    """
    fwd_s, fwd_e, rev_s, rev_e = None, None, None, None
    if motifs is not None:
        len_motif = len(motifs[0])
        fwd_s = -mod_loc
        fwd_e = len_motif - mod_loc
        rev_s = -(len_motif - 1 - mod_loc)
        rev_e = mod_loc + 1
        motifs = set(motifs)

    if is_sort:
        keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))
    else:
        keys = list(sitekey2stats.keys())

    if is_gzip:
        if not result_file.endswith(".gz"):
            result_file += ".gz"
        wf = gzip.open(result_file, "wt")
    else:
        wf = open(result_file, 'w')
    # wf.write('\t'.join(['chromosome', 'pos', 'pos+1', 'strand', 'prob0', 'prob1',
    #                     'met', 'unmet', 'coverage', 'rmet', 'kmer']) + '\n')
    for key in keys:
        chrom, pos, strand = split_key(key)
        if motifs is not None:
            motif_seq = dnacontigs[chrom][(pos+fwd_s):(pos+fwd_e)] if strand == "+" else \
                complement_seq(dnacontigs[chrom][(pos+rev_s):(pos+rev_e)])
            if motif_seq not in motifs:
                continue
        sitestats = sitekey2stats[key]
        assert(sitestats._coverage == (sitestats._met + sitestats._unmet))
        if sitestats._coverage > 0:
            rmet = float(sitestats._met) / sitestats._coverage
            if is_bed:
                wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(sitestats._coverage),
                                    strand,
                                    str(pos), str(pos + 1), "0,0,0", str(sitestats._coverage),
                                    str(int(round(rmet * 100 + 0.001, 0)))]) + "\n")
            else:
                wf.write("%s\t%d\t%d\t%s\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (chrom, pos, pos+1,
                                                                                 strand,
                                                                                 sitestats._prob_0,
                                                                                 sitestats._prob_1,
                                                                                 sitestats._met,
                                                                                 sitestats._unmet,
                                                                                 sitestats._coverage,
                                                                                 rmet + 0.000001,
                                                                                 sitestats._kmer))
        else:
            LOGGER.info("{} {} has no coverage..".format(chrom, pos))
    wf.flush()
    wf.close()


def _read_file_lines(cfile):
    with open(cfile, "r") as rf:
        return rf.read().splitlines()


def _get_contignams_from_genome_fasta(genomefa):
    contigs = []
    with open(genomefa, "r") as rf:
        for line in rf:
            if line.startswith(">"):
                contigname = line.strip()[1:].split(' ')[0]
                contigs.append(contigname)
    return contigs


def _is_file_a_genome_fasta(contigfile):
    with open(contigfile, "r") as rf:
        for line in rf:
            if line.startswith("#"):
                continue
            elif line.startswith(">"):
                return True
    return False


def _get_contigfile_name(wprefix, contig):
    return wprefix + "." + contig + ".txt"


def _split_file_by_contignames(mods_files, wprefix, contigs):
    contigs = set(contigs)
    wfs = {}
    for contig in contigs:
        wfs[contig] = open(_get_contigfile_name(wprefix, contig), "w")
    for input_file in mods_files:
        if input_file.endswith(".gz"):
            infile = gzip.open(input_file, 'rt')
        else:
            infile = open(input_file, 'r')
        for line in infile:
            chrom = line.strip().split("\t")[0]
            if chrom not in contigs:
                continue
            wfs[chrom].write(line)
        infile.close()
    for contig in contigs:
        wfs[contig].flush()
        wfs[contig].close()


def _call_and_write_modsfreq_process(wprefix, prob_cf, result_file, issort, isbed, rm_1strand, isgzip,
                                     motifs, modloc, dnacontigs,
                                     contigs_q, resfiles_q):
    LOGGER.info("process-{} -- starts".format(os.getpid()))
    while True:
        if contigs_q.empty():
            time.sleep(time_wait)
        contig_name = contigs_q.get()
        if contig_name == "kill":
            contigs_q.put("kill")
            break
        LOGGER.info("process-{} for contig-{} -- reading the input files..".format(os.getpid(), contig_name))
        input_file = _get_contigfile_name(wprefix, contig_name)
        if not os.path.isfile(input_file):
            LOGGER.warning("process-{} for contig-{} -- the input file does not exist..".format(os.getpid(), contig_name))
            continue
        if is_file_empty(input_file):
            LOGGER.warning("process-{} for contig-{} -- the input file is empty..".format(os.getpid(), contig_name))
        else:
            sites_stats = calculate_mods_frequency(input_file, prob_cf, rm_1strand, contig_name)
            LOGGER.info("process-{} for contig-{} -- writing the result..".format(os.getpid(), contig_name))
            fname, fext = os.path.splitext(result_file)
            c_result_file = fname + "." + contig_name + "." + str(uuid.uuid1()) + fext
            write_sitekey2stats(sites_stats, c_result_file, issort, isbed, isgzip,
                                motifs, modloc, dnacontigs)
            resfiles_q.put(c_result_file)
        os.remove(input_file)
    LOGGER.info("process-{} -- ends".format(os.getpid()))


def _concat_contig_results(contig_files, result_file, is_gzip=False):
    if is_gzip:
        if not result_file.endswith(".gz"):
            result_file += ".gz"
        wf = gzip.open(result_file, "wt")
    else:
        wf = open(result_file, 'w')
    for cfile in sorted(contig_files):
        with open(cfile, 'r') as rf:
            for line in rf:
                wf.write(line)
        os.remove(cfile)
    wf.close()


def call_mods_frequency_to_file(args):
    LOGGER.info("[main]call_freq starts")
    start = time.time()

    input_paths = args.input_path
    result_file = args.result_file
    prob_cf = args.prob_cf
    file_uid = args.file_uid
    issort = args.sort
    isbed = args.bed
    rm_1strand = args.rm_1strand
    is_gzip = args.gzip

    # check if need to read genome reference
    dnacontigs = None
    motifs = None
    modloc = None
    if args.refsites_only:
        from .utils.ref_reader import DNAReference
        from .utils.process_utils import get_motif_seqs
        if args.ref is None:
            raise ValueError("--ref must be set when --refsites_only is True!")
        if not os.path.exists(args.ref):
            raise ValueError("--ref doesn't exist!")
        dnacontigs = DNAReference(args.ref).getcontigs()
        motifs = get_motif_seqs(args.motifs)
        modloc = args.mod_loc
        LOGGER.info("[###] --refsites_only is set as True, gonna keep only motifs({}) sites of genome reference "
                    "in the results".format(motifs))

    mods_files = []
    for ipath in input_paths:
        input_path = os.path.abspath(ipath)
        if os.path.isdir(input_path):
            for ifile in os.listdir(input_path):
                if file_uid is None:
                    mods_files.append('/'.join([input_path, ifile]))
                elif ifile.find(file_uid) != -1:
                    mods_files.append('/'.join([input_path, ifile]))
        elif os.path.isfile(input_path):
            mods_files.append(input_path)
        else:
            raise ValueError("--input_path is not a file or a directory!")
    LOGGER.info("get {} input file(s)..".format(len(mods_files)))

    contigs = None
    if args.contigs is not None:
        if os.path.isfile(args.contigs):
            if args.contigs.endswith(".fa") or args.contigs.endswith(".fasta") or args.contigs.endswith(".fna"):
                contigs = _get_contignams_from_genome_fasta(args.contigs)
            elif _is_file_a_genome_fasta(args.contigs):
                contigs = _get_contignams_from_genome_fasta(args.contigs)
            else:
                contigs = sorted(list(set(_read_file_lines(args.contigs))))
        else:
            contigs = sorted(list(set(args.contigs.strip().split(","))))

    if contigs is None:
        LOGGER.info("read the input files..")
        sites_stats = calculate_mods_frequency(mods_files, prob_cf, rm_1strand)
        LOGGER.info("write the result..")
        write_sitekey2stats(sites_stats, result_file, issort, isbed, is_gzip,
                            motifs, modloc, dnacontigs)
    else:
        LOGGER.info("start processing {} contigs..".format(len(contigs)))
        wprefix = os.path.dirname(os.path.abspath(result_file)) + "/tmp." + str(uuid.uuid1())
        LOGGER.info("generate input files for each contig..")
        _split_file_by_contignames(mods_files, wprefix, contigs)
        LOGGER.info("read the input files of each contig..")
        contigs_q = Queue()
        for contig in contigs:
            contigs_q.put(contig)
        contigs_q.put("kill")
        resfiles_q = Queue()
        procs_contig = []
        for _ in range(args.threads):
            p_contig = mp.Process(target=_call_and_write_modsfreq_process,
                                  args=(wprefix, prob_cf, result_file, issort, isbed, rm_1strand, False,
                                        motifs, modloc, dnacontigs,
                                        contigs_q, resfiles_q))
            p_contig.daemon = True
            p_contig.start()
            procs_contig.append(p_contig)
        resfiles_cs = []
        while True:
            running = any(p.is_alive() for p in procs_contig)
            while not resfiles_q.empty():
                resfiles_cs.append(resfiles_q.get())
            if not running:
                break
        for p in procs_contig:
            p.join()
        try:
            assert len(contigs) == len(resfiles_cs)
        except AssertionError:
            LOGGER.warning("!!!Please check the result files -- seems not all inputed contigs have result!!!")
        LOGGER.info("combine results of {} contigs..".format(len(resfiles_cs)))
        _concat_contig_results(resfiles_cs, result_file, is_gzip)
    LOGGER.info("[main]call_freq costs %.1f seconds" % (time.time() - start))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level')
    parser.add_argument('--input_path', '-i', action="append", type=str, required=True,
                        help='an output file from call_mods/call_modifications.py, or a directory contains '
                             'a bunch of output files. this arg is in "append" mode, can be used multiple times')
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for finding all input files '
                             'and ignoring the not-input-files in a input directory. if input_path is a file, '
                             'ignore this arg.')

    parser.add_argument('--result_file', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')

    parser.add_argument('--contigs', action="store", type=str, required=False, default=None,
                        help="a reference genome file (.fa/.fasta/.fna), used for extracting all "
                             "contig names for parallel; "
                             "or path of a file containing chromosome/contig names, one name each line; "
                             "or a string contains multiple chromosome names splited by comma."
                             "default None, which means all chromosomes will be processed at one time. "
                             "If not None, one chromosome will be processed by one subprocess.")
    parser.add_argument('--threads', action="store", type=int, required=False, default=1,
                        help="number of subprocesses used when --contigs is set. i.e., number of contigs processed "
                             "in parallel. default 1")

    parser.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    parser.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                             'means use all calls. range [0, 1], default 0.0.')
    parser.add_argument('--rm_1strand', action='store_true', default=False,
                        help="abandon ccs reads with only 1 strand subreads [DEPRECATED]")
    parser.add_argument("--gzip", action="store_true", default=False, required=False,
                        help="if compressing the output using gzip")

    parser.add_argument('--refsites_only', action='store_true', default=False,
                        help="only keep sites which are target motifs in both reference and reads")
    parser.add_argument("--motifs", action="store", type=str,
                        required=False, default='CG',
                        help='motif seq to be extracted, default: CG. '
                             'can be multi motifs splited by comma '
                             '(no space allowed in the input str), '
                             'or use IUPAC alphabet, '
                             'the mod_loc of all motifs must be '
                             'the same. [Only useful when --refsites_only is True]')
    parser.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                        help='0-based location of the targeted base in the motif, default 0. '
                             '[Only useful when --refsites_only is True]')
    parser.add_argument("--ref", type=str, required=False,
                        help="path to genome reference, in fasta/fa format. "
                             "[Only useful when --refsites_only is True]")

    args = parser.parse_args()

    call_mods_frequency_to_file(args)


if __name__ == '__main__':
    sys.exit(main())
