#! /usr/bin/python
import argparse
import numpy as np
import scipy.stats
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import math
import gzip


def _read_one_mod_bed_file(freqfile, covcf, args):
    # methylbed format
    # "chromosome", "pos", "end", "na1", "na2", "strand", "na3", "na4", "na5", "coverage", "rpercent"
    contigset = set(args.contig_names.strip().split(",")) if args.contig_names is not None else None
    freqinfo = {}
    if freqfile.endswith(".gz"):
        infile = gzip.open(freqfile, 'rt')
    else:
        infile = open(freqfile, 'r')
    for line in infile:
        words = line.strip().split("\t")
        chrom = words[0]
        m_key = "\t".join([words[0], words[1], words[5]])
        cov = float(words[9])
        rmet = float(words[10]) / 100
        # methy_cov = rmet * cov

        cnt_flag = 0
        if args.contig_prefix is not None:
            if str(chrom).startswith(args.contig_prefix):
                cnt_flag = 1
        elif args.contig_names is not None:
            if chrom in contigset:
                cnt_flag = 1
        else:
            cnt_flag = 1

        if cnt_flag == 1:
            if cov >= covcf:
                freqinfo[m_key] = rmet
    infile.close()
    return freqinfo


def _read_one_mod_freq_file(freqfile, covcf, args):
    contigset = set(args.contig_names.strip().split(",")) if args.contig_names is not None else None
    freqinfo = {}
    if freqfile.endswith(".gz"):
        infile = gzip.open(freqfile, 'rt')
    else:
        infile = open(freqfile, 'r')
    for line in infile:
        words = line.strip().split("\t")
        chrom = words[0]
        m_key = "\t".join([words[0], words[1], words[2]])
        if len(words) == 11:
            cov = int(words[8])
            rmet = float(words[9])
        elif len(words) == 10:
            cov = int(words[7])
            rmet = float(words[8])
        else:
            raise ValueError("freq wrong!")

        cnt_flag = 0
        if args.contig_prefix is not None:
            if str(chrom).startswith(args.contig_prefix):
                cnt_flag = 1
        elif args.contig_names is not None:
            if chrom in contigset:
                cnt_flag = 1
        else:
            cnt_flag = 1
        if cnt_flag == 1:
            if cov >= covcf:
                freqinfo[m_key] = rmet
    infile.close()
    return freqinfo


def _cal_corr_of_rmet1_and_rmet2(freqinfo1, freqinfo2):
    keys_inter = set(freqinfo1.keys()).intersection(set(freqinfo2.keys()))
    keys_inter = sorted(list(keys_inter))
    rmet1, rmet2 = [], []
    for ktmp in keys_inter:
        rmet1.append(freqinfo1[ktmp])
        rmet2.append(freqinfo2[ktmp])
    rmet1 = np.array(rmet1)
    rmet2 = np.array(rmet2)
    if len(rmet1) > 1 and len(rmet2) > 1:
        corr, pvalue = scipy.stats.pearsonr(rmet1, rmet2)
        scorr, _ = scipy.stats.spearmanr(rmet1, rmet2)  # spearman
        _, _, r_value, _, _ = scipy.stats.linregress(rmet1, rmet2)
        r_square = r_value ** 2  # coefficient of determination
        rmse = math.sqrt(mean_squared_error(rmet1, rmet2))  # RMSE
    else:
        corr, r_square, scorr, rmse = 0, 0, 0, 0

    return rmet1, rmet2, len(freqinfo1.keys()), len(freqinfo2.keys()), len(keys_inter), \
        corr, r_square, scorr, rmse


def read_sites(site_file, covcf):
    site_keys = set()
    with open(site_file, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            # ONLY support bedmethyl file now
            if float(words[9]) >= covcf:
                m_key = "\t".join([words[0], words[1], words[5]])
                site_keys.add(m_key)
    return site_keys


def cmp_sitesrmet_of_tgs_and_bs(args):
    print("==coverage cutoff for query file: {}".format(args.covcf))
    print("==coverage cutoff for target file: {}".format(args.covcf_t))
    print()
    print("====query: {}".format(args.queryfile))
    if str(args.queryfile).endswith(".bed") or str(args.queryfile).endswith(".bed.gz"):
        freqinfo_nano = _read_one_mod_bed_file(args.queryfile, args.covcf, args)
    else:
        freqinfo_nano = _read_one_mod_freq_file(args.queryfile, args.covcf, args)

    if args.filter is not None:
        print()
        print("filter query file by positions in {}".format(args.filter))
        f_count = 0
        site_filter = read_sites(args.filter, args.covcf)
        for mkey in list(freqinfo_nano.keys()):
            if mkey not in site_filter:
                del freqinfo_nano[mkey]
                f_count += 1
        print("filter {} sites".format(f_count))
        print()
    freqinfo_t = []
    for targetfile in args.targetfile:
        if str(targetfile).endswith(".bed") or str(targetfile).endswith(".bed.gz"):
            freqtmp = _read_one_mod_bed_file(targetfile, args.covcf_t, args)
        else:
            freqtmp = _read_one_mod_freq_file(targetfile, args.covcf_t, args)
        _, _, qnum, tnum, sitesnum, corr, r_square, scorr, rmse = _cal_corr_of_rmet1_and_rmet2(freqinfo_nano, 
                                                                                               freqtmp)
        print("==target: {}\n"
              "\t\tsites: inter-{}/query-{}/target-{}\n"
              "\t\tpearson: {:.4f}, r_square: {:.4f}, "
              "spearman: {:.4f}, RMSE: {:.4f}".format(targetfile, sitesnum, qnum, tnum, 
                                                      corr, r_square, scorr, rmse))
        freqinfo_t.append(freqtmp)
    freqinfo_t_comb = dict()
    keyset = set()
    for freqinfotmp in freqinfo_t:
        for fkey in freqinfotmp.keys():
            if fkey not in keyset:
                keyset.add(fkey)
                freqinfo_t_comb[fkey] = []
            freqinfo_t_comb[fkey].append(freqinfotmp[fkey])
    for fkey in freqinfo_t_comb.keys():
        freqinfo_t_comb[fkey] = sum(freqinfo_t_comb[fkey]) / len(freqinfo_t_comb[fkey])
    rmet_q, rmet_t, qnum, tnum, sitesnum, corr, r_square, scorr, rmse = _cal_corr_of_rmet1_and_rmet2(freqinfo_nano,
                                                                                                     freqinfo_t_comb)
    print("==target: combined\n"
          "\t\tsites: inter-{}/query-{}/target-{}\n"
          "\t\tpearson: {:.4f}, r_square: {:.4f}, "
          "spearman: {:.4f}, RMSE: {:.4f}".format(sitesnum, qnum, tnum,
                                                  corr, r_square, scorr, rmse))
    wf = open(args.wfile, "w")
    wf.write("\t".join(["rmet_target", "rmet_query"]) + "\n")
    for ridx in range(0, len(rmet_t)):
        wf.write("\t".join([str(rmet_t[ridx]), str(rmet_q[ridx])]) + "\n")
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queryfile", action="store", type=str, required=True,
                        help="call_mods_freq_tgs/bed file")
    parser.add_argument("--targetfile", type=str, action="append", required=True,
                        help="rmet file")
    parser.add_argument("--covcf", type=int, required=False, default=5,
                        help="query covcf")
    parser.add_argument("--covcf_t", type=int, required=False, default=5,
                        help="target covcf")
    parser.add_argument("--contig_prefix", type=str, required=False, default=None)  # NC_003,
    parser.add_argument("--contig_names", type=str, required=False, default=None)  # 1,2,3,4,5,6,7,8,9,10,11,12
    parser.add_argument("--wfile", type=str, required=True)

    parser.add_argument("--filter", type=str, required=False, default=None)

    args = parser.parse_args()
    cmp_sitesrmet_of_tgs_and_bs(args)


if __name__ == '__main__':
    main()
