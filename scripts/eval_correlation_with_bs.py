#! /usr/bin/python
import argparse
import os
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error
import math

sep = "||"


def read_methylbed(bed_file, contig_prefix, contig_names, cov_cf):
    rmet_bed = pd.read_csv(bed_file, sep="\t", header=None,
                           names=["chromosome", "pos", "end", "na1", "na2", "strand",
                                  "na3", "na4", "na5", "coverage", "rpercent"],
                           dtype={"chromosome": str})
    rmet_bed["Rmet"] = rmet_bed.apply(lambda row: row["rpercent"] / 100.0, axis=1)
    if contig_prefix is not None:
        rmet_bed = rmet_bed[rmet_bed.apply(lambda row: row["chromosome"].startswith(contig_prefix), axis=1)]
    elif contig_names is not None:
        contigset = pd.Series(contig_names.split(","))
        rmet_bed = rmet_bed[rmet_bed.chromosome.isin(contigset)]
    else:
        pass
    rmet_bed['key'] = rmet_bed.apply(lambda row: row["chromosome"] + sep + str(row["pos"]), axis=1)
    rmet_bed = rmet_bed[["chromosome", "pos", "coverage", "Rmet", "key"]]

    meancov = rmet_bed["coverage"].mean()
    rmet_bed = rmet_bed[rmet_bed["coverage"] >= cov_cf]
    return meancov, rmet_bed.sort_values(by=['chromosome', 'pos'])


def read_rmetfile_of_tgs(tgs_file, contig_prefix, contig_names, cov_cf):
    rftmp = open(tgs_file, 'r')
    words = next(rftmp).strip().split("\t")
    rftmp.close()
    if len(words) == 11:
        # nanopore deepsignal
        rmet_dp2 = pd.read_csv(tgs_file, sep="\t", header=None,
                               names=["chromosome", "pos", "strand", "pos_in_strand", "prob0", "prob1", "met", "unmet",
                                      "coverage", "Rmet", "kmer"],
                               dtype={"chromosome": str})
    elif len(words) == 10:
        # smrt methccs
        rmet_dp2 = pd.read_csv(tgs_file, sep="\t", header=None,
                               names=["chromosome", "pos", "strand", "prob0", "prob1", "met", "unmet",
                                      "coverage", "Rmet", "kmer"],
                               dtype={"chromosome": str})
    else:
        raise ValueError("tgs_file wrong!")

    if contig_prefix is not None:
        rmet_dp2 = rmet_dp2[rmet_dp2.apply(lambda row: row["chromosome"].startswith(contig_prefix), axis=1)]
    elif contig_names is not None:
        contigset = pd.Series(contig_names.split(","))
        rmet_dp2 = rmet_dp2[rmet_dp2.chromosome.isin(contigset)]
    else:
        pass
    rmet_dp2['key'] = rmet_dp2.apply(lambda row: row["chromosome"] + sep + str(row["pos"]), axis=1)
    rmet_dp2 = rmet_dp2[["chromosome", "pos", "coverage", "Rmet", "key"]]

    meancov = rmet_dp2["coverage"].mean()
    rmet_dp2 = rmet_dp2[rmet_dp2["coverage"] >= cov_cf]

    return meancov, rmet_dp2.sort_values(by=['chromosome', 'pos'])


def read_rmetfile_of_bs(bs_file, contig_prefix, contig_names, cov_cf):
    def _cal_ratio(row):
        cov = row["met"] + row["unmet"]
        return float(row["met"]) / cov if cov > 0 else 0
    if bs_file.endswith(".bed"):
        rmet_bs = pd.read_csv(bs_file, sep="\t", header=None,
                              names=["chromosome", "pos", "end", "na1", "na2", "strand",
                                     "na3", "na4", "na5", "coverage", "rpercent"],
                              dtype={"chromosome": str})
        rmet_bs["Rmet"] = rmet_bs.apply(lambda row: row["rpercent"] / 100.0, axis=1)
    else:
        # CpG_report.txt
        rmet_bs = pd.read_csv(bs_file, sep="\t", header=None,
                              names=["chromosome", "pos", "strand", "met", "unmet",
                                     "motif", "kmer"],
                              dtype={"chromosome": str, "motif": str, "kmer": str})
        rmet_bs["coverage"] = rmet_bs.apply(lambda row: row["met"] + row["unmet"], axis=1)
        rmet_bs["Rmet"] = rmet_bs.apply(lambda row: _cal_ratio(row), axis=1)

    if contig_prefix is not None:
        rmet_bs = rmet_bs[rmet_bs.apply(lambda row: row["chromosome"].startswith(contig_prefix), axis=1)]
    elif contig_names is not None:
        contigset = pd.Series(contig_names.split(","))
        rmet_bs = rmet_bs[rmet_bs.chromosome.isin(contigset)]
    else:
        pass

    rmet_bs['key'] = rmet_bs.apply(lambda row: row["chromosome"] + sep + str(row["pos"]), axis=1)
    rmet_bs = rmet_bs[["chromosome", "pos", "coverage", "Rmet", "key"]]

    meancov = rmet_bs["coverage"].mean()
    rmet_bs = rmet_bs[rmet_bs["coverage"] >= cov_cf]

    return meancov, rmet_bs.sort_values(by=['chromosome', 'pos'])


def cal_corr_df1_vs_df2(df1, df2):
    df1_inter = df1[df1.key.isin(df2.key)].sort_values(by=['chromosome', 'pos'])
    df2_inter = df2[df2.key.isin(df1.key)].sort_values(by=['chromosome', 'pos'])
    # df1_inter["Rmet"].corr(df2_inter['Rmet'], method='pearson'), wrong? 0.2660 vs scipy 0.9459
    df1_array, df2_array = np.array(list(df1_inter["Rmet"])), np.array(list(df2_inter["Rmet"]))
    pcorr, _ = scipy.stats.pearsonr(df1_array, df2_array)  # pearson
    scorr, _ = scipy.stats.spearmanr(df1_array, df2_array)  # spearman
    _, _, r_value, _, _ = scipy.stats.linregress(df1_array, df2_array)
    r_square = r_value ** 2  # coefficient of determination
    rmse = math.sqrt(mean_squared_error(df2_array, df1_array))  # RMSE

    return len(df1.index), len(df2.index), len(df1_inter.index), pcorr, scorr, r_square, rmse


def correlation_with_bs_rmets(args):
    tgs_files = args.tgs_file
    bs_files = args.bs_file

    print("==nanofile coverage cutoff: {}\n".format(args.cov_cf))
    print("==bsfile coverage cutoff: {}\n\n".format(args.cov_cf_bs))

    bs_fname2rmetinfo = dict()
    for bs_file in bs_files:
        bsmean_cov, bs_fname2rmetinfo[os.path.basename(bs_file)] = read_rmetfile_of_bs(bs_file, args.contig_prefix,
                                                                                       args.contig_names,
                                                                                       args.cov_cf_bs)
        print("bsfile: {}, mean_covarge: {}".format(bs_file, bsmean_cov))
    for tgs_file in tgs_files:
        print("====== {}".format(tgs_file))
        if str(tgs_file).endswith(".bed"):
            mean_cov, dp2rmetinfo = read_methylbed(tgs_file, args.contig_prefix, args.contig_names, args.cov_cf)
        else:
            mean_cov, dp2rmetinfo = read_rmetfile_of_tgs(tgs_file, args.contig_prefix,
                                                         args.contig_names, args.cov_cf)
        print("mean_covarge: {}".format(mean_cov))
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format("bs_file", "bsnum", "smrtnum", "internum", "pearson",
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
    parser.add_argument("--bs_file", type=str, action="append",
                        required=True, help=".bed/CpG.report.txt")
    parser.add_argument("--contig_prefix", type=str, required=False, default=None)
    parser.add_argument("--contig_names", type=str, required=False, default=None)
    parser.add_argument("--cov_cf", type=int, required=False, default=5, help="")
    parser.add_argument("--cov_cf_bs", type=int, required=False, default=5, help="")

    args = parser.parse_args()
    correlation_with_bs_rmets(args)


if __name__ == '__main__':
    main()
