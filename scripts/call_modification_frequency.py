#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

import argparse
import os
import sys
import gzip

key_sep = "||"


class ModRecord:
    def __init__(self, fields):
        self._chromosome = fields[0]
        self._pos = int(fields[1])
        self._site_key = key_sep.join([self._chromosome, str(self._pos)])

        self._strand = fields[2]
        self.holeid = fields[3]

        self._depthstr = fields[4]
        self._depth = sum(list(map(int, self._depthstr.split(",")))) if "," in self._depthstr else int(self._depthstr)
        self._prob_0 = float(fields[5])
        self._prob_1 = float(fields[6])
        self._called_label = int(fields[7])
        self._kmer = fields[8]

    def is_record_callable(self, prob_threshold):
        if abs(self._prob_0 - self._prob_1) < prob_threshold:
            return False
        return True


def split_key(key):
    words = key.split(key_sep)
    return words[0], int(words[1])


class SiteStats:
    def __init__(self, strand, kmer):

        self._strand = strand
        self._kmer = kmer

        self._prob_0 = 0.0
        self._prob_1 = 0.0
        self._met = 0
        self._unmet = 0
        self._coverage = 0
        # self._rmet = -1.0


def calculate_mods_frequency(mods_files, prob_cf, rm_1strand=False):
    sitekeys = set()
    sitekey2stats = dict()

    count, used = 0, 0
    for mods_file in mods_files:
        if mods_file.endswith(".gz"):
            infile = gzip.open(mods_file, 'rt')
        else:
            infile = open(mods_file, 'r')
        for line in infile:
            count += 1
            words = line.strip().split("\t")
            mod_record = ModRecord(words)
            if rm_1strand and "," not in mod_record._depthstr:
                continue
            if not mod_record.is_record_callable(prob_cf):
                continue
            if mod_record._site_key not in sitekeys:
                sitekeys.add(mod_record._site_key)
                sitekey2stats[mod_record._site_key] = SiteStats(mod_record._strand,
                                                                mod_record._kmer)
            sitekey2stats[mod_record._site_key]._prob_0 += mod_record._prob_0
            sitekey2stats[mod_record._site_key]._prob_1 += mod_record._prob_1
            sitekey2stats[mod_record._site_key]._coverage += 1
            if mod_record._called_label == 1:
                sitekey2stats[mod_record._site_key]._met += 1
            else:
                sitekey2stats[mod_record._site_key]._unmet += 1
            used += 1
        infile.close()
    print("{:.2f}% ({} of {}) calls used..".format(used/float(count) * 100, used, count))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file, is_sort, is_bed):
    if is_sort:
        keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))
    else:
        keys = list(sitekey2stats.keys())

    with open(result_file, 'w') as wf:
        # wf.write('\t'.join(['chromosome', 'pos', 'strand', 'prob0', 'prob1',
        #                     'met', 'unmet', 'coverage', 'rmet', 'kmer']) + '\n')
        for key in keys:
            chrom, pos = split_key(key)
            sitestats = sitekey2stats[key]
            assert(sitestats._coverage == (sitestats._met + sitestats._unmet))
            if sitestats._coverage > 0:
                rmet = float(sitestats._met) / sitestats._coverage
                if is_bed:
                    wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(sitestats._coverage),
                                        sitestats._strand,
                                        str(pos), str(pos + 1), "0,0,0", str(sitestats._coverage),
                                        str(int(round(rmet * 100, 0)))]) + "\n")
                else:
                    wf.write("%s\t%d\t%s\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (chrom, pos, sitestats._strand,
                                                                                 sitestats._prob_0,
                                                                                 sitestats._prob_1,
                                                                                 sitestats._met, sitestats._unmet,
                                                                                 sitestats._coverage, rmet,
                                                                                 sitestats._kmer))
            else:
                print("{} {} has no coverage..".format(chrom, pos))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level')
    parser.add_argument('--input_path', '-i', action="append", type=str, required=True,
                        help='a result file from call_modifications.py, or a directory contains a bunch of '
                             'result files.')
    parser.add_argument('--result_file', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')
    parser.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    parser.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                             'means use all calls. range [0, 1], default 0.0.')
    parser.add_argument('--rm_1strand', action='store_true', default=False,
                        help="abandon ccs reads with only 1 strand subreads")
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for finding all input files and ignoring '
                             'the un-input-files in a input directory. if input_path is a file, ignore this arg.')

    args = parser.parse_args()

    input_paths = args.input_path
    result_file = args.result_file
    prob_cf = args.prob_cf
    file_uid = args.file_uid
    issort = args.sort
    isbed = args.bed
    rm_1strand = args.rm_1strand

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
            raise ValueError()
    print("get {} input file(s)..".format(len(mods_files)))

    print("reading the input files..")
    sites_stats = calculate_mods_frequency(mods_files, prob_cf, rm_1strand)
    print("writing the result..")
    write_sitekey2stats(sites_stats, result_file, issort, isbed)


if __name__ == '__main__':
    sys.exit(main())
