import os
import argparse
from subprocess import Popen, PIPE
from uuid import uuid1
import sys
import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error
import math

sep = "||"
seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1111]


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def _get_fasta_from_bam(bamfile):
    fname, fext = os.path.splitext(bamfile)
    bam_collate = fname + ".collate.bam"
    fafile = fname + ".fa"
    bam2fa_cmd = "samtools collate -@40 -o {} {} {}_prefix && " \
                 "samtools fasta -F 4 -@40 {} > {}".format(bam_collate, bamfile, bamfile,
                                                           bam_collate, fafile)
    sys.stderr.write("bam2fa cmd: {}\n".format(bam2fa_cmd))
    sys.stderr.flush()
    stdinfo, returncode = run_cmd(bam2fa_cmd)
    stdout, stderr = stdinfo
    sys.stderr.write("stdout:\n{}\n,stderr:\n{}\n".format(stdout, stderr))
    if os.path.exists(bam_collate):
        os.remove(bam_collate)
    if returncode:
        raise ValueError("bam2fa_cmd wrong!")
    else:
        return fafile


def _extract_readids_from_fa(fafile):
    fname, fext = os.path.splitext(fafile)
    fa_readids = fname + ".readids.txt"
    wf = open(fa_readids, "w")
    with open(fafile, "r") as rf:
        for line in rf:
            if line.startswith(">"):
                wf.write(line.strip()[1:] + "\n")
    wf.flush()
    wf.close()
    return fa_readids


def _sample_bam(bamfile, fafile, genome_size, coverage, seed):
    uuidstr = str(uuid1())
    fname, fext = os.path.splitext(fafile)
    sub_fafile = fname + ".sampled_cov_{}.{}.fa".format(coverage, uuidstr)
    sampfa_cmd = "rasusa --input {} --coverage {} --genome-size {} " \
                 "-o {} --seed {}".format(fafile, coverage, genome_size, sub_fafile, seed)
    sys.stderr.write("sampfa cmd: {}\n".format(sampfa_cmd))
    sys.stderr.flush()
    stdinfo, returncode = run_cmd(sampfa_cmd)
    stdout, stderr = stdinfo
    sys.stderr.write("stdout:\n{}\n,stderr:\n{}\n".format(stdout, stderr))
    if returncode:
        raise ValueError("sampfa_cmd wrong!")
    else:
        subfa_readids = _extract_readids_from_fa(sub_fafile)
        fname, fext = os.path.splitext(bamfile)
        sub_bamfile = fname + ".sampled_cov_{}.{}.bam".format(coverage, uuidstr)
        sampbam_cmd = "samtools view -@40 -bh -N {} {} > {} && " \
                      "samtools index -@40 {}".format(subfa_readids, bamfile, sub_bamfile, sub_bamfile)
        sys.stderr.write("sampbam cmd: {}\n".format(sampbam_cmd))
        sys.stderr.flush()
        stdinfo, returncode = run_cmd(sampbam_cmd)
        stdout, stderr = stdinfo
        sys.stderr.write("stdout:\n{}\n,stderr:\n{}\n".format(stdout, stderr))
        if returncode:
            raise ValueError("sampbam wrong!")
        else:
            os.remove(sub_fafile)
            os.remove(subfa_readids)
            return sub_bamfile


def run_ccsmeth_call_freqb(bamfile, genomefa, aggre_model, out_dir):
    fname, fext = os.path.splitext(os.path.basename(bamfile))
    freq_prefix = out_dir + "/" + fname + ".freq"
    ccsmeth_count_cmd = "ccsmeth call_freqb --input_bam {} --ref {} " \
                        "--output {} --bed --sort --threads 40 --refsites_all --identity 0 --mapq 0 " \
                        "--call_mode count".format(bamfile, genomefa, freq_prefix)
    sys.stderr.write("ccsmeth_count cmd: {}\n".format(ccsmeth_count_cmd))
    sys.stderr.flush()
    stdinfo, returncode = run_cmd(ccsmeth_count_cmd)
    stdout, stderr = stdinfo
    sys.stderr.write("stdout:\n{}\n,stderr:\n{}\n".format(stdout, stderr))
    if returncode:
        raise ValueError("ccsmeth_count wrong!")
    else:
        count_bed = freq_prefix + ".count.all.bed"
        # if os.path.exists(freq_prefix + ".count.hp1.bed"):
        #     os.remove(freq_prefix + ".count.hp1.bed")
        # if os.path.exists(freq_prefix + ".count.hp2.bed"):
        #     os.remove(freq_prefix + ".count.hp2.bed")
    ccsmeth_aggre_cmd = "ccsmeth call_freqb --input_bam {} --ref {} " \
                        "--output {} --bed --sort --threads 40 --refsites_all --identity 0 --mapq 0 " \
                        "--call_mode aggregate --aggre_model {}".format(bamfile, genomefa, freq_prefix, aggre_model)
    sys.stderr.write("ccsmeth_aggre cmd: {}\n".format(ccsmeth_aggre_cmd))
    sys.stderr.flush()
    stdinfo, returncode = run_cmd(ccsmeth_aggre_cmd)
    stdout, stderr = stdinfo
    sys.stderr.write("stdout:\n{}\n,stderr:\n{}\n".format(stdout, stderr))
    if returncode:
        raise ValueError("ccsmeth_aggre wrong!")
    else:
        aggre_bed = freq_prefix + ".aggregate.all.bed"
        # if os.path.exists(freq_prefix + ".aggregate.hp1.bed"):
        #     os.remove(freq_prefix + ".aggregate.hp1.bed")
        # if os.path.exists(freq_prefix + ".aggregate.hp2.bed"):
        #     os.remove(freq_prefix + ".aggregate.hp2.bed")

    return count_bed, aggre_bed


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


def display_args(args, is_stderr=False):
    def print_outputstr(outstr):
        if is_stderr:
            sys.stderr.write(outstr + "\n")
        else:
            sys.stdout.write(outstr + "\n")
    arg_vars = vars(args)
    outputstr = "# ===============================================\n## parameters: "
    print_outputstr(outputstr)
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            outputstr = "{}:\n\t{}".format(arg_key, arg_vars[arg_key])
            print_outputstr(outputstr)
    print_outputstr("# ===============================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=str, required=True, help="bam file")
    parser.add_argument("--covs", type=str, required=False, default="5,10,15,20,25",
                        help="")
    parser.add_argument("--genome_size", type=str, required=False, default="3.05gb",
                        help="default 3.05gb")
    parser.add_argument("--repeat", type=int, required=False, default=5, help="repeat times for a coverage, "
                                                                              "default 5")
    parser.add_argument("--total", action="store_true", default=False, help="if eval all reads in bam")

    parser.add_argument("--cmp_bed", action="append", required=True, help="cmp bed")
    parser.add_argument("--contig_names", type=str, required=False, default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,"
                                                                            "chr8,chr9,chr10,chr11,chr12,chr13,"
                                                                            "chr14,chr15,chr16,chr17,chr18,chr19,"
                                                                            "chr20,chr21,chr22,chrX,chrY")
    parser.add_argument("--cov_cf", type=int, required=False, default=5, help="default 5")

    parser.add_argument("--genomefa", type=str, required=True, help="genome reference")
    parser.add_argument("--aggre_model", type=str, required=True, help="aggre model/pileup model")
    parser.add_argument("--out_dir", type=str, required=False, default="results",
                        help="tmp/out dir")

    args = parser.parse_args()

    sys.stdout.write("eval bamfile - {} ====\n".format(args.bam))
    display_args(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    sys.stdout.write("\ncmp files:\n")
    cmpfname2rmetinfo = dict()
    for cmp_file in args.cmp_bed:
        cmpmean_cov, cmpfname2rmetinfo[os.path.basename(cmp_file)] = read_methylbed(cmp_file,
                                                                                    None,
                                                                                    args.contig_names,
                                                                                    args.cov_cf)
        sys.stdout.write("cmpfile: {}, mean_covarge: {}\n".format(cmp_file, cmpmean_cov))
    sys.stdout.flush()

    fafile = _get_fasta_from_bam(args.bam)
    covs = [int(x) for x in args.covs.strip().split(",")]
    for cov in covs:
        sys.stdout.write("\nfor cov-{} ====\n".format(cov))
        count_beds, aggre_beds = [], []
        for ridx in range(args.repeat):
            seedtmp = seeds[ridx]
            sampledbam = _sample_bam(args.bam, fafile, args.genome_size,
                                     cov, seedtmp)
            count_bed, aggre_bed = run_ccsmeth_call_freqb(sampledbam, args.genomefa, args.aggre_model, args.out_dir)
            count_beds.append(count_bed)
            aggre_beds.append(aggre_bed)
            os.remove(sampledbam)
            os.remove(sampledbam + ".bai")
        sys.stdout.write("count-mode ==\n")
        c_covs, c_rmets = [], []
        for cbedtmp in count_beds:
            mean_cov, rmetinfo = read_methylbed(cbedtmp, None, args.contig_names, args.cov_cf)
            c_covs.append(mean_cov)
            c_rmets.append(rmetinfo)
        sys.stdout.write("means: {}\n".format(c_covs))
        for cmp_file in cmpfname2rmetinfo.keys():
            sys.stdout.write("== cmp_file: {}\n".format(cmp_file))
            sys.stdout.write("\t".join(["cmp_file", "cmpnum", "tgsnum", "internum",
                                        "pearson", "rsquare", "spearman", "RMSE"]) + "\n")
            cmp_df = cmpfname2rmetinfo[cmp_file]
            sitenums_inter = []
            sitenums_pb = []
            sitenums_cmp = []
            corrs_pearson = []
            corrs_spearman = []
            corrs_rsquare = []
            corrs_rmse = []
            for c_rmet_tmp in c_rmets:
                smrtnum, cmpnum, internum, \
                    pcorr, scorr, r_square, rmse = cal_corr_df1_vs_df2(c_rmet_tmp, cmp_df)
                sitenums_inter.append(internum)
                sitenums_pb.append(smrtnum)
                sitenums_cmp.append(cmpnum)
                corrs_pearson.append(pcorr)
                corrs_spearman.append(scorr)
                corrs_rsquare.append(r_square)
                corrs_rmse.append(rmse)
                sys.stdout.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(cmp_file, cmpnum, smrtnum,
                                                                              internum, pcorr, r_square,
                                                                              scorr, rmse))
            sys.stdout.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("mean",
                                                                                      np.mean(sitenums_cmp),
                                                                                      np.mean(sitenums_pb),
                                                                                      np.mean(sitenums_inter),
                                                                                      np.mean(corrs_pearson),
                                                                                      np.mean(corrs_rsquare),
                                                                                      np.mean(corrs_spearman),
                                                                                      np.mean(corrs_rmse)
                                                                                      ))
            sys.stdout.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("SD",
                                                                                                   np.std(
                                                                                                       sitenums_cmp),
                                                                                                   np.std(sitenums_pb),
                                                                                                   np.std(
                                                                                                       sitenums_inter),
                                                                                                   np.std(
                                                                                                       corrs_pearson),
                                                                                                   np.std(
                                                                                                       corrs_rsquare),
                                                                                                   np.std(
                                                                                                       corrs_spearman),
                                                                                                   np.std(corrs_rmse)
                                                                                                   ))
        sys.stdout.flush()
        del c_rmets

        sys.stdout.write("aggregate/pileup-mode ==\n")
        a_covs, a_rmets = [], []
        for abedtmp in aggre_beds:
            mean_cov, rmetinfo = read_methylbed(abedtmp, None, args.contig_names, args.cov_cf)
            a_covs.append(mean_cov)
            a_rmets.append(rmetinfo)
        sys.stdout.write("means: {}\n".format(a_covs))
        for cmp_file in cmpfname2rmetinfo.keys():
            sys.stdout.write("== cmp_file: {}\n".format(cmp_file))
            sys.stdout.write("\t".join(["cmp_file", "cmpnum", "tgsnum", "internum",
                                        "pearson", "rsquare", "spearman", "RMSE"]) + "\n")
            cmp_df = cmpfname2rmetinfo[cmp_file]
            sitenums_inter = []
            sitenums_pb = []
            sitenums_cmp = []
            corrs_pearson = []
            corrs_spearman = []
            corrs_rsquare = []
            corrs_rmse = []
            for a_rmet_tmp in a_rmets:
                smrtnum, cmpnum, internum, \
                    pcorr, scorr, r_square, rmse = cal_corr_df1_vs_df2(a_rmet_tmp, cmp_df)
                sitenums_inter.append(internum)
                sitenums_pb.append(smrtnum)
                sitenums_cmp.append(cmpnum)
                corrs_pearson.append(pcorr)
                corrs_spearman.append(scorr)
                corrs_rsquare.append(r_square)
                corrs_rmse.append(rmse)
                sys.stdout.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(cmp_file, cmpnum, smrtnum,
                                                                                           internum, pcorr, r_square,
                                                                                           scorr, rmse))
            sys.stdout.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("mean",
                                                                                                   np.mean(
                                                                                                       sitenums_cmp),
                                                                                                   np.mean(sitenums_pb),
                                                                                                   np.mean(
                                                                                                       sitenums_inter),
                                                                                                   np.mean(
                                                                                                       corrs_pearson),
                                                                                                   np.mean(
                                                                                                       corrs_rsquare),
                                                                                                   np.mean(
                                                                                                       corrs_spearman),
                                                                                                   np.mean(corrs_rmse)
                                                                                                   ))
            sys.stdout.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format("SD",
                                                                                                   np.std(
                                                                                                       sitenums_cmp),
                                                                                                   np.std(sitenums_pb),
                                                                                                   np.std(
                                                                                                       sitenums_inter),
                                                                                                   np.std(
                                                                                                       corrs_pearson),
                                                                                                   np.std(
                                                                                                       corrs_rsquare),
                                                                                                   np.std(
                                                                                                       corrs_spearman),
                                                                                                   np.std(corrs_rmse)
                                                                                                   ))
        sys.stdout.flush()
        del a_rmets

        # for bedfiletmp in count_beds:
        #     os.remove(bedfiletmp)
        # for bedfiletmp in aggre_beds:
        #     os.remove(bedfiletmp)
    sys.stdout.flush()
    os.remove(fafile)

    if args.total:
        sys.stdout.write("\nfor total reads ====\n")
        count_bed, aggre_bed = run_ccsmeth_call_freqb(args.bam, args.genomefa, args.aggre_model, args.out_dir)
        sys.stdout.write("count-mode ==\n")
        mean_cov, rmetinfo = read_methylbed(count_bed, None, args.contig_names, args.cov_cf)
        sys.stdout.write("means: {}\n".format(mean_cov))
        for cmp_file in cmpfname2rmetinfo.keys():
            sys.stdout.write("== cmp_file: {}\n".format(cmp_file))
            sys.stdout.write("\t".join(["cmp_file", "cmpnum", "tgsnum", "internum",
                                        "pearson", "rsquare", "spearman", "RMSE"]) + "\n")
            cmp_df = cmpfname2rmetinfo[cmp_file]
            smrtnum, cmpnum, internum, \
                pcorr, scorr, r_square, rmse = cal_corr_df1_vs_df2(rmetinfo, cmp_df)
            sys.stdout.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(cmp_file, cmpnum, smrtnum,
                                                                                       internum, pcorr, r_square,
                                                                                       scorr, rmse))
        sys.stdout.write("aggre/pileup-mode ==\n")
        mean_cov, rmetinfo = read_methylbed(aggre_bed, None, args.contig_names, args.cov_cf)
        sys.stdout.write("means: {}\n".format(mean_cov))
        for cmp_file in cmpfname2rmetinfo.keys():
            sys.stdout.write("== cmp_file: {}\n".format(cmp_file))
            sys.stdout.write("\t".join(["cmp_file", "cmpnum", "tgsnum", "internum",
                                        "pearson", "rsquare", "spearman", "RMSE"]) + "\n")
            cmp_df = cmpfname2rmetinfo[cmp_file]
            smrtnum, cmpnum, internum, \
                pcorr, scorr, r_square, rmse = cal_corr_df1_vs_df2(rmetinfo, cmp_df)
            sys.stdout.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(cmp_file, cmpnum, smrtnum,
                                                                                       internum, pcorr, r_square,
                                                                                       scorr, rmse))


if __name__ == '__main__':
    main()
