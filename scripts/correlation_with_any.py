#! /usr/bin/python
import argparse
import os
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import gzip

sep = "||"


def get_intersected_keys(tgs_files, cov_cf=5):
    keys = set()
    for tgs_file in tgs_files:
        keystmp = set()
        with open(tgs_file, "r") as rf:
            for line in rf:
                words = line.strip().split("\t")
                chrom, pos = words[0], words[1]
                if str(tgs_file).endswith(".bed"):
                    strand = words[5]
                    cov = int(words[9])
                else:
                    strand = words[2]
                    if len(words) == 11:
                        cov = int(words[8])
                    elif len(words) == 10:
                        cov = int(words[7])
                    else:
                        raise ValueError("freq wrong!")
                if cov >= cov_cf:
                    keystmp.add(sep.join([chrom, pos, strand]))
        if len(keys) == 0:
            keys.update(keystmp)
        else:
            keys = keys.intersection(keystmp)
    return keys


def read_methylbed(bed_file, contig_prefix, contig_names, cov_cf, keys=None):
    # methylbed format
    # "chromosome", "pos", "end", "na1", "na2", "strand", "na3", "na4", "na5", "coverage", "rpercent"
    contigset = set(contig_names.strip().split(",")) if contig_names is not None else None
    freqinfo = {}
    covs = []
    if bed_file.endswith(".gz"):
        infile = gzip.open(bed_file, 'rt')
    else:
        infile = open(bed_file, 'r')
    for line in infile:
        words = line.strip().split("\t")
        chrom = words[0]
        m_key = sep.join([words[0], words[1], words[5]])
        cov = float(words[9])
        rmet = float(words[10]) / 100
        # methy_cov = rmet * cov

        cnt_flag = 0
        if contig_prefix is not None:
            if str(chrom).startswith(contig_prefix):
                cnt_flag = 1
        elif contig_names is not None:
            if chrom in contigset:
                cnt_flag = 1
        else:
            cnt_flag = 1

        if cnt_flag == 1:
            covs.append(cov)
            if cov >= cov_cf:
                freqinfo[m_key] = rmet
    infile.close()
    if keys is not None:
        for m_key in list(freqinfo.keys()):
            if m_key not in keys:
                del freqinfo[m_key]
    return np.mean(covs) if len(covs) > 0 else 0, freqinfo


def read_rmetfile_of_tgs(tgs_file, contig_prefix, contig_names, cov_cf, keys=None):
    contigset = set(contig_names.strip().split(",")) if contig_names is not None else None
    freqinfo = {}
    covs = []
    if tgs_file.endswith(".gz"):
        infile = gzip.open(tgs_file, 'rt')
    else:
        infile = open(tgs_file, 'r')
    for line in infile:
        words = line.strip().split("\t")
        chrom = words[0]
        m_key = sep.join([words[0], words[1], words[2]])
        if len(words) == 11:
            cov = int(words[8])
            rmet = float(words[9])
        elif len(words) == 10:
            cov = int(words[7])
            rmet = float(words[8])
        else:
            raise ValueError("freq wrong!")

        cnt_flag = 0
        if contig_prefix is not None:
            if str(chrom).startswith(contig_prefix):
                cnt_flag = 1
        elif contig_names is not None:
            if chrom in contigset:
                cnt_flag = 1
        else:
            cnt_flag = 1
        if cnt_flag == 1:
            covs.append(cov)
            if cov >= cov_cf:
                freqinfo[m_key] = rmet
    infile.close()
    if keys is not None:
        for m_key in list(freqinfo.keys()):
            if m_key not in keys:
                del freqinfo[m_key]
    return np.mean(covs) if len(covs) > 0 else 0, freqinfo


def read_rmetfile_of_bed_or_bsformat(bs_file, contig_prefix, contig_names, cov_cf):
    if bs_file.endswith(".bed"):
        return read_methylbed(bs_file, contig_prefix, contig_names, cov_cf)
    else:
        # CpG_report.txt
        contigset = set(contig_names.strip().split(",")) if contig_names is not None else None
        freqinfo = {}
        covs = []
        if bs_file.endswith(".gz"):
            infile = gzip.open(bs_file, 'rt')
        else:
            infile = open(bs_file, 'r')
        for line in infile:
            words = line.strip().split("\t")
            chrom = words[0]
            m_key = sep.join([words[0], words[1], words[2]])
            cov = int(words[3]) + int(words[4])
            rmet = float(words[3]) / cov if cov > 0 else 0

            cnt_flag = 0
            if contig_prefix is not None:
                if str(chrom).startswith(contig_prefix):
                    cnt_flag = 1
            elif contig_names is not None:
                if chrom in contigset:
                    cnt_flag = 1
            else:
                cnt_flag = 1
            if cnt_flag == 1:
                covs.append(cov)
                if cov >= cov_cf:
                    freqinfo[m_key] = rmet
        infile.close()

        return np.mean(covs) if len(covs) > 0 else 0, freqinfo


def cal_corr_df1_vs_df2(freqinfo1, freqinfo2):
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

    return len(freqinfo1.keys()), len(freqinfo2.keys()), len(keys_inter), \
        corr, scorr, r_square, rmse


def correlation_with_any_rmets(args):
    tgs_files = args.tgs_file
    cmp_files = args.cmp_file

    print("==nanofile coverage cutoff: {}\n".format(args.cov_cf))
    print("==cmpfile coverage cutoff: {}\n\n".format(args.cov_cf_cmp))

    bs_fname2rmetinfo = dict()
    for cmp_file in cmp_files:
        if not args.tgs:
            bsmean_cov, bs_fname2rmetinfo[os.path.basename(cmp_file)] = read_rmetfile_of_bed_or_bsformat(cmp_file,
                                                                                                args.contig_prefix,
                                                                                                args.contig_names,
                                                                                                args.cov_cf_cmp)
        else:
            bsmean_cov, bs_fname2rmetinfo[os.path.basename(cmp_file)] = read_rmetfile_of_tgs(cmp_file,
                                                                                             args.contig_prefix,
                                                                                             args.contig_names,
                                                                                             args.cov_cf_cmp)
        print("cmpfile: {}, mean_covarge: {}".format(cmp_file, bsmean_cov))
    keys = None
    if args.inter:
        keys = get_intersected_keys(tgs_files, args.cov_cf)
    for tgs_file in tgs_files:
        print("====== {}".format(tgs_file))
        if str(tgs_file).endswith(".bed"):
            mean_cov, dp2rmetinfo = read_methylbed(tgs_file, args.contig_prefix, args.contig_names, args.cov_cf,
                                                   keys)
        else:
            mean_cov, dp2rmetinfo = read_rmetfile_of_tgs(tgs_file, args.contig_prefix,
                                                         args.contig_names, args.cov_cf,
                                                         keys)
        print("mean_covarge: {}".format(mean_cov))
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("cmp_file", "cmpnum", "tgsnum", "internum", "pearson",
                                                      "rsquare", "spearman", "RMSE"))
        sitenums_inter = []
        sitenums_smrt = []
        sitenums_bs = []
        corrs_pearson = []
        corrs_spearman = []
        corrs_rsquare = []
        corrs_rmse = []
        for bs_fname in sorted(list(bs_fname2rmetinfo.keys())):
            smrtnum, bsnum, internum, \
                pcorr, scorr, r_square, rmse = cal_corr_df1_vs_df2(dp2rmetinfo, bs_fname2rmetinfo[bs_fname])

            sitenums_inter.append(internum)
            sitenums_smrt.append(smrtnum)
            sitenums_bs.append(bsnum)
            corrs_pearson.append(pcorr)
            corrs_spearman.append(scorr)
            corrs_rsquare.append(r_square)
            corrs_rmse.append(rmse)
            print("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(bs_fname, bsnum, smrtnum,
                                                                          internum, pcorr, r_square,
                                                                          scorr, rmse))
        print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format("average",
                                                                                  sum(sitenums_bs)/len(sitenums_bs),
                                                                                  sum(sitenums_smrt) / len(
                                                                                      sitenums_smrt),
                                                                                  sum(sitenums_inter) / len(
                                                                                      sitenums_inter),
                                                                                  sum(corrs_pearson) / len(
                                                                                      corrs_pearson),
                                                                                  sum(corrs_rsquare) / len(
                                                                                      corrs_rsquare),
                                                                                  sum(corrs_spearman) / len(
                                                                                      corrs_spearman),
                                                                                  sum(corrs_rmse) / len(
                                                                                      corrs_rmse)
                                                                                  ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgs_file", type=str, action="append", required=True, help="smrt/nanopore freq file, "
                                                                                     ".freq.tsv/.bed "
                                                                                     "(self-defined format)")
    parser.add_argument("--cmp_file", type=str, action="append",
                        required=True, help="bs(.bed/CpG.report.txt) or tgs file")
    parser.add_argument('--tgs', action='store_true', default=False, help="cmp file is in tgs format")
    parser.add_argument("--contig_prefix", type=str, required=False, default=None)
    parser.add_argument("--contig_names", type=str, required=False, default=None)
    parser.add_argument("--cov_cf", type=int, required=False, default=5, help="")
    parser.add_argument("--cov_cf_cmp", type=int, required=False, default=5, help="")
    parser.add_argument("--inter", action="store_true", required=False, default=False,
                        help="if using intersected keys of tgs_file")

    args = parser.parse_args()
    correlation_with_any_rmets(args)


if __name__ == '__main__':
    main()
