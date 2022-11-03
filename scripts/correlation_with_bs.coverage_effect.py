#! /usr/bin/python
import argparse
import os
import random
import uuid
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error
import math

sep = "||"
tmp_dir = "/home/nipeng"


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def _read_one_mod_freq_file(freqfile):
    freqinfo = {}
    if str(freqfile).endswith(".bed"):
        # methylbed format
        # "chromosome", "pos", "end", "na1", "na2", "strand", "na3", "na4", "na5", "coverage", "rpercent"
        with open(freqfile, "r") as rf:
            for line in rf:
                words = line.strip().split("\t")
                m_key = "\t".join([words[0], words[1], words[5]])
                cov = float(words[9])
                rmet = float(words[10]) / 100
                methy_cov = rmet * cov
                freqinfo[m_key] = [methy_cov, cov, rmet]
    else:
        with open(freqfile, "r") as rf:
            for line in rf:
                words = line.strip().split("\t")
                m_key = "\t".join([words[0], words[1], words[2]])
                if len(words) == 11:
                    methy_cov = int(words[6])
                    cov = int(words[8])
                    rmet = float(words[9])
                elif len(words) == 10:
                    methy_cov = int(words[5])
                    cov = int(words[7])
                    rmet = float(words[8])
                else:
                    raise ValueError("wrong file format of freq file")
                freqinfo[m_key] = [methy_cov, cov, rmet]
    return freqinfo


def _get_combined_freq_info(freqinfos):
    freqinfo = {}
    freqkeys = set()
    for finfo in freqinfos:
        for fkey in finfo.keys():
            if fkey not in freqkeys:
                freqkeys.add(fkey)
                freqinfo[fkey] = [0.0, 0.0, 0.0]
            freqinfo[fkey][0] += finfo[fkey][0]
            freqinfo[fkey][1] += finfo[fkey][1]
            if freqinfo[fkey][1] > 0:
                freqinfo[fkey][2] = freqinfo[fkey][0] / float(freqinfo[fkey][1])
    return freqinfo


def _write_freqinfo(freqinfo, wfile):
    wf = open(wfile, "w")
    for fkey in freqinfo.keys():
        wstr = "\t".join([fkey, ] + list(map(str, freqinfo[fkey]))) + "\n"
        wf.write(wstr)
    wf.close()


def _get_one_corrinfo(freqfile, bsfile, args):

    def read_methylbed_6cols(bed_file, contig_prefix, contig_names, cov_cf):
        rmet_bed = pd.read_csv(bed_file, sep="\t", header=None,
                               names=["chromosome", "pos", "strand", "nmet", "coverage", "Rmet"],
                               dtype={"chromosome": str})
        if contig_prefix is not None:
            rmet_bed = rmet_bed[rmet_bed.apply(lambda row: row["chromosome"].startswith(contig_prefix), axis=1)]
        elif contig_names is not None:
            contigset = pd.Series(contig_names.split(","))
            rmet_bed = rmet_bed[rmet_bed.chromosome.isin(contigset)]
        else:
            pass
        rmet_bed['key'] = rmet_bed.apply(lambda row: row["chromosome"] + sep + str(row["pos"]), axis=1)
        rmet_bed = rmet_bed[["chromosome", "pos", "coverage", "Rmet", "key"]]

        rmet_bed = rmet_bed[rmet_bed["coverage"] >= cov_cf]
        return rmet_bed.sort_values(by=['chromosome', 'pos'])

    def read_rmetfile_of_bs(bs_file, contig_prefix, contig_names, cov_cf):
        if bs_file.endswith(".bed"):
            rmet_bs = pd.read_csv(bs_file, sep="\t", header=None,
                                  names=["chromosome", "pos", "end", "na1", "na2", "strand",
                                         "na3", "na4", "na5", "coverage", "rpercent"],
                                  dtype={"chromosome": str})
            rmet_bs["Rmet"] = rmet_bs.apply(lambda row: row["rpercent"] / 100.0, axis=1)
        else:
            rmet_bs = pd.read_csv(bs_file, sep="\t", header=0, dtype={"chromosome": str})

        if contig_prefix is not None:
            rmet_bs = rmet_bs[rmet_bs.apply(lambda row: row["chromosome"].startswith(contig_prefix), axis=1)]
        elif contig_names is not None:
            contigset = pd.Series(contig_names.split(","))
            rmet_bs = rmet_bs[rmet_bs.chromosome.isin(contigset)]
        else:
            pass

        rmet_bs['key'] = rmet_bs.apply(lambda row: row["chromosome"] + sep + str(row["pos"]), axis=1)
        rmet_bs = rmet_bs[["chromosome", "pos", "coverage", "Rmet", "key"]]

        rmet_bs = rmet_bs[rmet_bs["coverage"] >= cov_cf]

        return rmet_bs.sort_values(by=['chromosome', 'pos'])

    def cal_corr_df1_vs_df2(df1, df2):
        df1_inter = df1[df1.key.isin(df2.key)].sort_values(by=['chromosome', 'pos'])
        df2_inter = df2[df2.key.isin(df1.key)].sort_values(by=['chromosome', 'pos'])

        # df1_inter["Rmet"].corr(df2_inter['Rmet'], method='pearson'), wrong? 0.2660 vs scipy 0.9459
        df1_array, df2_array = np.array(list(df1_inter["Rmet"])), np.array(list(df2_inter["Rmet"]))
        pcorr, _ = scipy.stats.pearsonr(df1_array, df2_array)  # pearson
        scorr, _ = scipy.stats.spearmanr(df1_array, df2_array)  # spearman
        _, _, r_value, _, _ = scipy.stats.linregress(df1_array, df2_array)
        r_square = r_value**2  # coefficient of determination
        rmse = math.sqrt(mean_squared_error(df2_array, df1_array))  # RMSE

        # spearman, after limit digits to 2
        scorr2, _ = scipy.stats.spearmanr(np.around(df1_array, decimals=2), df2_array)

        return len(df1.index), len(df2.index), len(df1_inter.index), pcorr, scorr, r_square, rmse, scorr2

    freqinfo = read_methylbed_6cols(freqfile, args.contig_prefix, args.contig_names, args.cov_cf)
    bsinfo = read_rmetfile_of_bs(bsfile, args.contig_prefix, args.contig_names, args.cov_cf_bs)
    len_nano, len_bs, len_inter, cor_pear, cor_spear, cor_rsquare, \
        cor_rmse, cor_spear2 = cal_corr_df1_vs_df2(freqinfo, bsinfo)

    return len_nano, len_bs, len_inter, cor_pear, cor_spear, cor_rsquare, cor_rmse, cor_spear2


def findsubsets(s, n):
    import itertools
    return list(itertools.combinations(s, n))


def eval_coverage_effect(args):
    wf = open(args.wfile, "w")
    wf.write("====\n")
    wf.write("==nanofile coverage cutoff: {}\n".format(args.cov_cf))
    wf.write("==bsfile coverage cutoff: {}\n\n".format(args.cov_cf_bs))

    modsfiles = args.tgsfile
    modsfile_len = len(modsfiles)
    for modfile_idx in range(0, modsfile_len):
        wf.write("nano=={}, {}\n".format(modfile_idx, modsfiles[modfile_idx]))
    wf.write("\n")
    modsfreqinfos = []
    for modfile_idx in range(0, modsfile_len):
        modsfreqinfos.append(_read_one_mod_freq_file(modsfiles[modfile_idx]))

    bsfiles = args.bsfile
    bsfile_len = len(bsfiles)
    for bs_idx in range(0, bsfile_len):
        wf.write("bs=={}, {}\n".format(bs_idx, bsfiles[bs_idx]))
    wf.write("\n")
    wf.flush()

    # coverage
    for coverage_idx in range(1, modsfile_len + 1):
        coverage_name = "{}0x".format(coverage_idx)
        combines = findsubsets([i for i in range(0, modsfile_len)], coverage_idx)
        random.shuffle(combines)
        iterations = args.repeat if args.repeat < len(combines) else len(combines)
        wf.write("================={}\n".format(coverage_name))
        # iteration
        stats_cove = []
        for iter_idx in range(0, iterations):
            # wf.write("=====iteration{}\n".format(iter_idx+1))
            iter_name = "_".join(list(map(str, list(combines[iter_idx]))))

            tmp_freqfile = tmp_dir + "/mod_freq_" + iter_name + "." + str(uuid.uuid1()) + ".tsv"
            tmp_modsinfos = [modsfreqinfos[i] for i in combines[iter_idx]]
            tmp_freqinfo = _get_combined_freq_info(tmp_modsinfos)
            _write_freqinfo(tmp_freqinfo, tmp_freqfile)

            curid_corr_stats = []
            for bs_idx in range(0, len(bsfiles)):
                len_nano, len_bs, len_inter, cor_pear, cor_spear, \
                    cor_rsquare, cor_rmse, cor_spear2 = _get_one_corrinfo(tmp_freqfile, bsfiles[bs_idx], args)
                curid_corr_stats.append([len_nano, len_bs, len_inter, cor_pear, cor_spear, cor_rsquare, cor_rmse,
                                         cor_spear2])
            os.remove(tmp_freqfile)
            if iter_idx == 0:
                wf.write('\t'.join(['iter', 'num_nano>=cov', 'num_bs>=cov', "num_inter>=cov",
                                    "pearson", "spearman", "r_square", "RMSE", "spearman2"]) + '\n')
            for bs_idx in range(0, len(bsfiles)):
                curid_corr_tmp = curid_corr_stats[bs_idx]
                wf.write('\t'.join([str(iter_name) + "_vs_" + str(bs_idx), ] + list(map(str, curid_corr_tmp))) + '\n')
            curid_corr_stats = np.array(curid_corr_stats, dtype=np.float)
            curid_corr_stats_mean = np.mean(curid_corr_stats, 0)
            wf.write('\t'.join([str(iter_name) + "_bsmean", ] + list(map(str, curid_corr_stats_mean))) + '\n')
            stats_cove.append(curid_corr_stats_mean)
        stats_cove = np.array(stats_cove, dtype=np.float)
        stats_cove_mean = np.mean(stats_cove, 0)
        stats_cove_std = np.std(stats_cove, 0)
        wf.write('mean\t' + '\t'.join(list(map(str, stats_cove_mean))) + '\n')
        wf.write('std\t' + '\t'.join(list(map(str, stats_cove_std))) + '\n')
        wf.write("\n")
        wf.flush()
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgsfile", action="append", type=str, required=True,
                        help="10x methylbed/deepsignal_freq file")
    parser.add_argument("--bsfile", type=str, action="append", required=True,
                        help="bs rmet file")
    parser.add_argument("--bs_id", type=str, required=False, default="rep",
                        help="replicate num, e.g.: rep1")
    parser.add_argument("--repeat", type=int, required=False, default=5,
                        help="random repeat times, default 5")
    parser.add_argument("--wfile", type=str, required=True,
                        help="")

    parser.add_argument("--contig_prefix", type=str, required=False, default=None)  # NC_003,
    parser.add_argument("--contig_names", type=str, required=False, default=None)  # 1,2,3,4,5,6,7,8,9,10,11,12
    parser.add_argument("--cov_cf", type=int, required=False, default=5, help="")
    parser.add_argument("--cov_cf_bs", type=int, required=False, default=5, help="")

    args = parser.parse_args()
    eval_coverage_effect(args)


if __name__ == '__main__':
    main()
