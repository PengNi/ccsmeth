#! /usr/bin/python
import argparse
import os
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error
import math

sep = "||"


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


def _generate_regions(ref, resolution, contig_prefix, contig_names):
    contigs = set(contig_names.split(",")) if contig_names is not None else None
    dnaref = DNAReference(ref)
    contig2len = []
    contig2seq = dnaref.getcontigs()
    for contig in contig2seq.keys():
        if contigs is not None and contig not in contigs:
            continue
        if contig_prefix is not None and (not contig.startswith(contig_prefix)):
            continue
        contig2len.append((contig, len(contig2seq[contig])))
    contig2len = sorted(contig2len, key=lambda key: key[0])
    del dnaref

    print("==contig num: {}".format(len(contig2len)))

    regionkeys = set()
    for contig_len in contig2len:
        chrom, clen = contig_len
        for i in range(0, clen, resolution):
            start = i
            # end = i + resolution if i+resolution <= clen else clen
            regionkeys.add((chrom, start))
    print("==region num: {}\n".format(len(regionkeys)))
    return regionkeys


def read_methylbed(bed_file, contig_prefix, contig_names, resolution, regionkeys):
    region2info = dict()
    for region in regionkeys:
        region2info[region] = [0., 0.]  # cov, met
    contigs = set(contig_names.split(",")) if contig_names is not None else None
    with open(bed_file, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom, pos, cov, rmet = words[0], int(words[1]), int(words[9]), float(words[10])/100
            if contigs is not None and chrom not in contigs:
                continue
            if contig_prefix is not None and (not chrom.startswith(contig_prefix)):
                continue
            pos_fold, posmod = pos // resolution, pos % resolution
            pos_start = pos_fold * resolution
            region2info[(chrom, pos_start)][0] += cov
            region2info[(chrom, pos_start)][1] += rmet * cov
    for region in region2info.keys():
        cov, met = region2info[region][0], region2info[region][1]
        rmet = met / float(cov) if cov > 0 else 0
        region2info[region].append(rmet)
    return region2info


def read_rmetfile_of_deepsmrt(smrt_file, contig_prefix, contig_names, resolution, regionkeys):
    # "chromosome", "pos", "strand", "prob0", "prob1", "met", "unmet", "coverage", "Rmet", "kmer"
    region2info = dict()
    for region in regionkeys:
        region2info[region] = [0., 0.]  # cov, met
    contigs = set(contig_names.split(",")) if contig_names is not None else None
    with open(smrt_file, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom, pos, cov, met = words[0], int(words[1]), int(words[7]), int(words[5])
            if contigs is not None and chrom not in contigs:
                continue
            if contig_prefix is not None and (not chrom.startswith(contig_prefix)):
                continue
            pos_fold, posmod = pos // resolution, pos % resolution
            pos_start = pos_fold * resolution
            region2info[(chrom, pos_start)][0] += cov
            region2info[(chrom, pos_start)][1] += met
    for region in region2info.keys():
        cov, met = region2info[region][0], region2info[region][1]
        rmet = met / float(cov) if cov > 0 else 0
        region2info[region].append(rmet)
    return region2info


def read_rmetfile_of_bs(bs_file, contig_prefix, contig_names, resolution, regionkeys):
    if bs_file.endswith(".bed"):
        region2info = read_methylbed(bs_file, contig_prefix, contig_names, resolution, regionkeys)
    else:
        # CpG_report.txt
        # "chromosome", "pos", "strand", "met", "unmet", "motif", "kmer"
        region2info = dict()
        for region in regionkeys:
            region2info[region] = [0., 0.]  # cov, met
        contigs = set(contig_names.split(",")) if contig_names is not None else None
        with open(bs_file, "r") as rf:
            for line in rf:
                words = line.strip().split("\t")
                chrom, pos, met, unmet = words[0], int(words[1]), int(words[3]), int(words[4])
                cov = met + unmet
                if contigs is not None and chrom not in contigs:
                    continue
                if contig_prefix is not None and (not chrom.startswith(contig_prefix)):
                    continue
                pos_fold, posmod = pos // resolution, pos % resolution
                pos_start = pos_fold * resolution
                region2info[(chrom, pos_start)][0] += cov
                region2info[(chrom, pos_start)][1] += met
        for region in region2info.keys():
            cov, met = region2info[region][0], region2info[region][1]
            rmet = met / float(cov) if cov > 0 else 0
            region2info[region].append(rmet)
    return region2info


def cal_corr_df1_vs_df2(region2info1, region2info2):
    rmet1, rmet2 = [], []
    zeros1, zeros2 = 0, 0
    for region in region2info1.keys():
        rcov1, rrmet1 = region2info1[region][0], region2info1[region][2]
        rcov2, rrmet2 = region2info2[region][0], region2info2[region][2]
        if rcov1 == 0 or rcov2 == 0:
            if rcov1 == 0:
                zeros1 += 1
            if rcov2 == 0:
                zeros2 += 1
            continue
        rmet1.append(rrmet1)
        rmet2.append(rrmet2)
    rmet1 = np.array(rmet1)
    rmet2 = np.array(rmet2)
    pcorr, _ = scipy.stats.pearsonr(rmet1, rmet2)  # pearson
    scorr, _ = scipy.stats.spearmanr(rmet1, rmet2)  # spearman
    _, _, r_value, _, _ = scipy.stats.linregress(rmet1, rmet2)
    r_square = r_value ** 2  # coefficient of determination
    rmse = math.sqrt(mean_squared_error(rmet1, rmet2))  # RMSE

    # =====print rmet1 vs rmet2 to file
    # todo
    # =========================
    return len(region2info1.keys()) - zeros1, len(region2info2.keys()) - zeros2, \
        len(rmet1), pcorr, scorr, r_square, rmse


def correlation_with_bs_rmets(args):
    smrt_files = args.smrt_file
    bs_files = args.bs_file
    ref = args.ref
    print("==resolution: {}\n".format(args.resolution))
    print("==contig_prefix: {}\n".format(args.contig_prefix))
    print("==contig_names: {}\n".format(args.contig_names))
    regionkeys = _generate_regions(ref, args.resolution, args.contig_prefix, args.contig_names)

    bs_fname2rmetinfo = dict()
    for bs_file in bs_files:
        bs_fname2rmetinfo[os.path.basename(bs_file)] = read_rmetfile_of_bs(bs_file, args.contig_prefix,
                                                                           args.contig_names,
                                                                           args.resolution,
                                                                           regionkeys)
        print("bsfile: {}".format(bs_file))
    for smrt_file in smrt_files:
        print("====== {}".format(smrt_file))
        if str(smrt_file).endswith(".bed"):
            dp2rmetinfo = read_methylbed(smrt_file, args.contig_prefix, args.contig_names,
                                         args.resolution, regionkeys)
        elif str(smrt_file).endswith("freq.tsv") or str(smrt_file).endswith("freq.fb_comb.tsv"):
            dp2rmetinfo = read_rmetfile_of_deepsmrt(smrt_file, args.contig_prefix, args.contig_names,
                                                    args.resolution, regionkeys)
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
                                                                                  sum(sitenums_bs) / len(
                                                                                      sitenums_bs),
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
    parser.add_argument("--smrt_file", type=str, action="append", required=True, help="smrt freq file, .freq.tsv/.bed")
    parser.add_argument("--bs_file", type=str, action="append",
                        required=True, help=".bed/CpG.report.txt")
    parser.add_argument("--contig_prefix", type=str, required=False, default=None)
    parser.add_argument("--contig_names", type=str, required=False, default=None)
    parser.add_argument("--ref", type=str, required=True, help="reference path")
    parser.add_argument("--resolution", type=int, required=False, default=1000000,
                        help="len of one region unit to cal correlation, default 1M")

    args = parser.parse_args()
    correlation_with_bs_rmets(args)


if __name__ == '__main__':
    main()
