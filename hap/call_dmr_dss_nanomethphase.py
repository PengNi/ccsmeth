#! /usr/bin/env python3
"""
this script is a modified version of dma module of NanoMethPhase written by Vahid Akbari, two minor changes:
    1. remove header in input files
    2. accept bedmethyl files
"""

import os
import glob
import gzip
import bz2
import argparse
import subprocess
from collections import defaultdict
import time


def openfile(file):
    '''
    Opens a file
    '''
    if file.endswith('.gz'):
        opened_file = gzip.open(file,'rt')
    elif file.endswith('bz') or file.endswith('bz2'):
        opened_file = bz2.open(file,'rt')
    else:
        opened_file = open(file,'rt')
    return opened_file


def main_dma(args):
    """
    This is the DMA module which does differential methylation analysis
    using DSS R package to find differentially methylated regions.
    """
    t_start = time.time()
    if os.path.isdir(os.path.abspath(args.case)):
        cases = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(args.case)):
            for filename in filenames:
                cases.append(dirpath + '/' + filename)
    else:
        cases = [os.path.abspath(args.case)]

    if os.path.isdir(os.path.abspath(args.control)):
        controls = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(args.control)):
            for filename in filenames:
                controls.append(dirpath + '/' + filename)
    else:
        controls = [os.path.abspath(args.control)]

    out_dir = os.path.abspath(args.out_dir)
    out_prefix = out_dir + '/' + (args.out_prefix)
    coverage = args.coverage
    Rscript = args.Rscript  # os.path.abspath(args.Rscript)
    script = os.path.abspath(args.script_file)
    dis_merge = args.dis_merge
    minlen = args.minlen
    minCG = args.minCG
    smoothing_span = args.smoothing_span
    smoothing_flag = args.smoothing_flag.upper()
    equal_disp = args.equal_disp.upper()
    pval_cutoff = args.pval_cutoff
    delta_cutoff = args.delta_cutoff
    pct_sig = args.pct_sig
    # check if outputs exist
    if args.columns:
        check_outs = [x for x in glob.glob("{}*".format(out_prefix))]
        if check_outs and not args.overwrite:
            raise FileExistsError("The selected output files {} already "
                                  "exist. Select --overwrite option if you "
                                  "want to overwrite them or use a different "
                                  "prefix".format(check_outs))
    else:
        check_outs = [x for x in glob.glob("{}*DM*.txt".format(out_prefix))]
        if check_outs and not args.overwrite:
            raise FileExistsError("The selected output files {} already "
                                  "exist. Select --overwrite option if you "
                                  "want to overwrite them or use a different "
                                  "prefix".format(check_outs))
    if args.columns:
        ready_cases = []
        ready_controls = []
        columns = args.columns
        columns = list(map(int, columns.split(',')))
        if len(columns) == 4:
            col1, col2, col3, col4 = columns
            out_putNumber = 0
            for case in cases:
                out_putNumber += 1
                case_out = open("{}_ReadyForDSS_case{}.tsv".format(
                    out_prefix, out_putNumber), "w")
                with openfile(case) as case_file:
                    # next(case_file)  # Exclude header
                    for line in case_file:
                        try:
                            line = line.rstrip().split('\t')
                            cov = int(line[col3 - 1])
                            if cov >= coverage:
                                chrom = line[col1 - 1]
                                start = line[col2 - 1]
                                methyfreq = float(line[col4 - 1])
                                if str(case).endswith(".bed") or args.is_bed:
                                    methyfreq = methyfreq / 100
                                mod_sites = round(cov * methyfreq)
                                case_out.write('\t'.join([chrom,
                                                          start,
                                                          str(cov),
                                                          str(mod_sites) + '\n']))
                        except ValueError:
                            continue
                case_out.close()
                ready_cases.append("{}_ReadyForDSS_case{}.tsv".format(
                    out_prefix, out_putNumber))
            out_putNumber = 0
            for control in controls:
                out_putNumber += 1
                control_out = open("{}_ReadyForDSS_control{}.tsv".format(
                    out_prefix, out_putNumber), "w")
                with openfile(control) as control_file:
                    # next(control_file)
                    for line in control_file:
                        try:
                            line = line.rstrip().split('\t')
                            cov = int(line[col3 - 1])
                            if cov >= coverage:
                                chrom = line[col1 - 1]
                                start = line[col2 - 1]
                                methyfreq = float(line[col4 - 1])
                                if str(control).endswith(".bed") or args.is_bed:
                                    methyfreq = methyfreq / 100
                                mod_sites = round(cov * methyfreq)
                                control_out.write('\t'.join([chrom,
                                                             start,
                                                             str(cov),
                                                             str(mod_sites) + '\n']))
                        except ValueError:
                            continue
                control_out.close()
                ready_controls.append("{}_ReadyForDSS_control{}.tsv".format(
                    out_prefix, out_putNumber))
        elif len(columns) == 5:
            col1, col2, col3, col4, col5 = columns
            out_putNumber = 0
            for case in cases:
                out_putNumber += 1
                case_out = open("{}_ReadyForDSS_case{}.tsv".format(
                    out_prefix, out_putNumber), "w")
                cov_dict = defaultdict(int)
                mod_sites_dict = defaultdict(int)
                with openfile(case) as case_file:
                    # next(case_file)
                    for line in case_file:
                        try:
                            line = line.rstrip().split('\t')
                            cov = int(line[col4 - 1])
                            if cov >= coverage:
                                strand = line[col3 - 1]
                                start = int(line[col2 - 1])
                                chrom = line[col1 - 1]
                                if strand == "-":
                                    start = start - 1
                                key = (chrom, start)
                                cov_dict[key] += cov

                                methyfreq = float(line[col5 - 1])
                                if str(case).endswith(".bed") or args.is_bed:
                                    methyfreq = methyfreq / 100
                                mod_sites_dict[key] += round(cov * methyfreq)
                        except ValueError:
                            continue
                    for key in sorted(mod_sites_dict.keys()):
                        cov = cov_dict[key]
                        chrom = key[0]
                        start = key[1]
                        mod_sites = mod_sites_dict[key]
                        case_out.write('\t'.join([chrom,
                                                  str(start),
                                                  str(cov),
                                                  str(mod_sites) + '\n']))
                    case_out.close()
                ready_cases.append("{}_ReadyForDSS_case{}.tsv".format(
                    out_prefix, out_putNumber))
            out_putNumber = 0
            for control in controls:
                out_putNumber += 1
                control_out = open("{}_ReadyForDSS_control{}.tsv".format(
                    out_prefix, out_putNumber), "w")
                cov_dict = defaultdict(int)
                mod_sites_dict = defaultdict(int)
                with openfile(control) as control_file:
                    # next(control_file)
                    for line in control_file:
                        try:
                            line = line.rstrip().split('\t')
                            cov = int(line[col4 - 1])
                            if cov >= coverage:
                                strand = line[col3 - 1]
                                start = int(line[col2 - 1])
                                chrom = line[col1 - 1]
                                if strand == "-":
                                    start = start - 1
                                key = (chrom, start)
                                cov_dict[key] += cov

                                methyfreq = float(line[col5 - 1])
                                if str(control).endswith(".bed") or args.is_bed:
                                    methyfreq = methyfreq / 100
                                mod_sites_dict[key] += round(cov * methyfreq)
                        except ValueError:
                            continue
                    for key in sorted(mod_sites_dict.keys()):
                        cov = cov_dict[key]
                        chrom = key[0]
                        start = key[1]
                        mod_sites = mod_sites_dict[key]
                        control_out.write('\t'.join([chrom,
                                                     str(start),
                                                     str(cov),
                                                     str(mod_sites) + '\n']))
                    control_out.close()
                ready_controls.append("{}_ReadyForDSS_control{}.tsv".format(
                    out_prefix,
                    out_putNumber))
        else:
            raise TypeError("Please enter columns correctly")
        ready_cases = ','.join(ready_cases)
        ready_controls = ','.join(ready_controls)
    else:
        ready_cases = ','.join(cases)
        ready_controls = ','.join(controls)
    t_end1 = time.time()
    print("===data preparation costs {:.1f} seconds".format(t_end1 - t_start))
    subprocess.call(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(Rscript,
                                                           script,
                                                           ready_cases,
                                                           ready_controls,
                                                           out_prefix,
                                                           dis_merge,
                                                           minlen,
                                                           minCG,
                                                           smoothing_span,
                                                           smoothing_flag,
                                                           pval_cutoff,
                                                           delta_cutoff,
                                                           pct_sig,
                                                           equal_disp),
        shell=True)
    t_end2 = time.time()
    print("===DSS call DMR costs {:.1f} seconds".format(t_end2 - t_end1))


def main():
    parser = argparse.ArgumentParser()
    dma_input = parser.add_argument_group("required arguments")
    dma_input.add_argument("--case", "-ca",
                           action="store",
                           type=str,
                           required=True,
                           help=("The path to the tab delimited input "
                                 "methylation frequency or ready input case "
                                 "file(s). If multiple files, files must be "
                                 "in the same directory and give the path to the directory."))
    dma_input.add_argument("--control", "-co",
                           action="store",
                           type=str,
                           required=True,
                           help=("The path to the tab delimited input "
                                 "methylation frequency or ready input "
                                 "control file(s). If multiple files, files must be "
                                 "in the same directory and give the path to the directory."))
    dma_input.add_argument("--out_dir", "-o",
                           action="store",
                           type=str,
                           required=True,
                           help="The path to the output directory")
    dma_input.add_argument("--out_prefix", "-op",
                           action="store",
                           type=str,
                           required=True,
                           help="The prefix for the output files")
    dma_opt = parser.add_argument_group("optional arguments")
    dma_opt.add_argument("--columns", "-c",
                         action="store",
                         type=str,
                         required=False,
                         help=("Comma seperated Columns in the methylation "
                               "frequency files that include the following "
                               "information, respectively:\n"
                               "chromosome\tstart\tstrand\tcoverage\t"
                               "methylation_frequency.\n"
                               "If the methylation frequency file does not "
                               "have strand level information then just "
                               "enter columns number for\n"
                               "chromosome\tstart\tcoverage\t"
                               "methylation_frequency.\n"
                               "Default is that your input files are "
                               "already in a format required by DSS so you "
                               "do not need to select any column.\n"
                               "If you giving as input NanoMethPhase "
                               "frequency files select this:"
                               "--columns 1,2,4,5,7; for bedmethyl: 1,2,6,10,11\n"))
    dma_opt.add_argument("--Rscript", "-rs",
                         action="store",
                         type=str,
                         required=False,
                         default="Rscript",
                         help="The path to a particular instance of "
                              "Rscript to use")
    dma_opt.add_argument("--script_file", "-sf",
                         action="store",
                         type=str,
                         required=False,
                         default=os.path.join(os.path.dirname(
                             os.path.realpath(__file__)
                         ),
                             "DSS_DMA.R"),
                         help="The path to the DSS_DMA.R script file")
    dma_opt.add_argument("--coverage", "-cov",
                         action="store",
                         type=int,
                         default=1,
                         required=False,
                         help=("Coverage cutoff. Default is >=1. It is "
                               "recommended that do not filter for "
                               "coverage as DSS R package will take care "
                               "of it."))
    dma_opt.add_argument("--dis_merge", "-dm",
                         action="store",
                         type=int,
                         default=1500,
                         required=False,
                         help=("When two DMRs are very close to each other "
                               "and the distance (in bps) is less than "
                               "this number, they will be merged into one. "
                               "Default is 1500 bps."))
    dma_opt.add_argument("--minlen", "-ml",
                         action="store",
                         type=int,
                         default=100,
                         required=False,
                         help=("Minimum length (in basepairs) required for "
                               "DMR. Default is 100 bps."))
    dma_opt.add_argument("--minCG", "-mcg",
                         action="store",
                         type=int,
                         default=15,
                         required=False,
                         help=("Minimum number of CpG sites required for "
                               "DMR. Default is 15."))
    dma_opt.add_argument("--smoothing_span", "-sms",
                         action="store",
                         type=int,
                         default=500,
                         required=False,
                         help=("The size of smoothing window, in "
                               "basepairs. Default is 500."))
    dma_opt.add_argument("--smoothing_flag", "-smf",
                         action="store",
                         type=str,
                         default="TRUE",
                         required=False,
                         help=("TRUE/FALSE. The size of smoothing window, "
                               "in basepairs. Default is TRUE. We "
                               "recommend to use smoothing=TRUE for "
                               "whole-genome BS-seq data, and "
                               "smoothing=FALSE for sparser data such "
                               "like from RRBS or hydroxyl-methylation "
                               "data (TAB-seq). If there is not biological "
                               "replicate, smoothing=TRUE is required. "
                               "Default is TRUE"))
    dma_opt.add_argument("--equal_disp", "-ed",
                         action="store",
                         type=str,
                         default="FALSE",
                         required=False,
                         help=("TRUE/FALSE. When there is no biological "
                               "replicate in one or both treatment groups, "
                               "users can either (1) specify "
                               "equal.disp=TRUE, which assumes both groups "
                               "have the same dispersion, then the data "
                               "from two groups are combined and used as "
                               "replicates to estimate dispersion; or (2) "
                               "specify smoothing=TRUE, which uses the "
                               "smoothed means (methylation levels) to "
                               "estimate dispersions via a shrinkage "
                               "estimator. This smoothing procedure uses "
                               "data from neighboring CpG sites as "
                               "\"pseudo-replicate\" for estimating "
                               "biological variance. Default is FALSE"))
    dma_opt.add_argument("--pval_cutoff", "-pvc",
                         action="store",
                         type=float,
                         default=0.001,
                         required=False,
                         help=("A threshold of p-values for calling DMR. "
                               "Loci with p-values less than this "
                               "threshold will be picked and joint to form "
                               "the DMRs. See 'details' for more "
                               "information. Default is 0.001"))
    dma_opt.add_argument("--delta_cutoff", "-dc",
                         action="store",
                         type=float,
                         default=0,
                         required=False,
                         help=("A threshold for defining DMR. In DML "
                               "detection procedure, a hypothesis test "
                               "that the two groups means are equal is "
                               "conducted at each CpG site. Here if "
                               "'delta' is specified, the function will "
                               "compute the posterior probability that the "
                               "difference of the means are greater than "
                               "delta, and then construct DMR based on "
                               "that. This only works when the test "
                               "results are from 'DMLtest', which is for "
                               "two-group comparison. Default is 0"))
    dma_opt.add_argument("--pct_sig", "-pct",
                         action="store",
                         type=float,
                         default=0.5,
                         required=False,
                         help=("In all DMRs, the percentage of CG sites "
                               "with significant p-values (less than "
                               "p.threshold) must be greater than this "
                               "threshold. Default is 0.5. This is mainly "
                               "used for correcting the effects of merging "
                               "of nearby DMRs."))
    dma_opt.add_argument("--overwrite", "-ow",
                         action="store_true",
                         required=False,
                         help="If output files exist overwrite them")
    dma_opt.add_argument("--is_bed",
                         action="store_true",
                         required=False,
                         help="If input files are in bedmethyl format")
    args = parser.parse_args()

    main_dma(args)


if __name__ == '__main__':
    main()


