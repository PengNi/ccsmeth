import os
import argparse
import time

from .utils.process_utils import run_cmd
from .utils.process_utils import display_args
from .utils.process_utils import ccs_exec
from .utils.process_utils import generate_samtools_view_cmd
from .utils.process_utils import generate_samtools_index_cmd

from .utils.logging import mylogger
LOGGER = mylogger(__name__)


def check_input_file(inputfile):
    if not inputfile.endswith(".bam"):
        raise ValueError("--subreads/-i must be in bam format!")
    inputpath = os.path.abspath(inputfile)
    return inputpath


def check_output_file(outputfile, inputfile):
    if outputfile is None:
        fname, fext = os.path.splitext(inputfile)
        midfix = "hifi"
        output_path = fname.rstrip(".subreads") + "." + midfix + ".bam"
    else:
        if not (outputfile.endswith(".sam") or outputfile.endswith(".bam")):
            raise ValueError("--output/-o must be in bam/sam format!")
        output_path = os.path.abspath(outputfile)
    return output_path


def generate_ccscmd_with_options(args):
    ccs_path = ccs_exec
    if args.path_to_ccs is not None:
        ccs_path = os.path.abspath(args.path_to_ccs)
    ccs_cmd = ccs_path + " --hifi-kinetics"
    ccs_cmd += " --log-level {}".format(args.log_level)
    if args.min_passes is not None:
        ccs_cmd += " --min-passes {}".format(args.min_passes)
    if args.threads is not None:
        ccs_cmd += " --num-threads {}".format(args.threads)
    if args.by_strand:
        ccs_cmd += " --by-strand"
    if args.hd_finder:
        ccs_cmd += " --hd-finder"
    return ccs_cmd


def ccs_call_hifi_reads(args):
    LOGGER.info("[main]call_hifi_reads starts")
    start = time.time()
    inputpath = check_input_file(args.subreads)
    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")
    outputpath = check_output_file(args.output, inputpath)

    ccscaller = generate_ccscmd_with_options(args)
    samtools_view = generate_samtools_view_cmd(args.path_to_samtools, args.threads)
    samtools_index = generate_samtools_index_cmd(args.path_to_samtools, args.threads)

    if outputpath.endswith(".bam"):
        ccs_cmds = " ".join([ccscaller, inputpath, outputpath])
        post_ccs_cmd = " ".join([samtools_index, outputpath])
        ccs_cmds = " && ".join([ccs_cmds, post_ccs_cmd])
    elif outputpath.endswith(".sam"):
        tmp_bam = outputpath + ".tmp.bam"
        ccs_cmds = " ".join([ccscaller, inputpath, tmp_bam])
        post_ccs_cmd = "echo '@SQ\tSN:chr\tLN:1' > {} && ".format(outputpath)
        post_ccs_cmd += " ".join([samtools_view, tmp_bam, " >>", outputpath])
        clean_cmd = " ".join(["rm", tmp_bam + "*"])
        ccs_cmds = " && ".join([ccs_cmds, post_ccs_cmd, clean_cmd])
    else:
        raise ValueError("--output/-o must be in bam/sam format!")

    LOGGER.info("cmds: {}".format(ccs_cmds))
    stdinfo, returncode = run_cmd(ccs_cmds)
    stdout, stderr = stdinfo
    if returncode:
        LOGGER.warning("failed")
    else:
        LOGGER.info("succeeded")
    LOGGER.info("stdout:\n{}".format(str(stdout, 'utf-8')))
    LOGGER.info("stderr:\n{}".format(str(stderr, 'utf-8')))

    endtime = time.time()
    LOGGER.info("[main]call_hifi_reads costs {:.1f} seconds".format(endtime - start))


def main():
    parser = argparse.ArgumentParser("call hifi reads with kinetics from subreads.bam using CCS, "
                                     "save in bam/sam format.\n"
                                     "  cmd: ccsmeth call_hifi -i input.subreads.bam")
    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--subreads", "-i", type=str, required=True,
                         help="path to subreads.bam file as input")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", type=str, required=False,
                          help="output file path for alignment results, bam/sam supported. "
                               "If not specified, the results will be saved in "
                               "input_file_prefix.hifi.bam by default.")

    p_ccs = parser.add_argument_group("CCS ARG")
    p_ccs.add_argument("--path_to_ccs", type=str, default=None, required=False,
                       help="full path to the executable binary ccs(PBCCS) file. "
                            "If not specified, it is assumed that ccs is "
                            "in the PATH.")
    p_ccs.add_argument("--threads", "-t", type=int, default=None, required=False,
                       help="number of threads to call hifi reads, "
                            "default None -> means using all available processors")
    p_ccs.add_argument("--min-passes", type=int, default=None, required=False,
                       help="CCS: Minimum number of full-length subreads required to generate CCS for a ZMW. "
                            "default None -> means using a default value set by CCS")
    p_ccs.add_argument("--by-strand", action="store_true", default=False, required=False,
                       help="CCS: Generate a consensus for each strand.")
    p_ccs.add_argument("--hd-finder", action="store_true", default=False, required=False,
                       help="CCS: Enable heteroduplex finder and splitting.")
    p_ccs.add_argument("--log-level", action="store", default="WARN", required=False,
                       help="CCS: Set log level. "
                            "Valid choices: (TRACE, DEBUG, INFO, WARN, FATAL). [WARN]")

    parser.add_argument("--path_to_samtools", type=str, default=None, required=False,
                        help="full path to the executable binary samtools file. "
                             "If not specified, it is assumed that samtools is in "
                             "the PATH.")

    args = parser.parse_args()

    display_args(args)
    ccs_call_hifi_reads(args)


if __name__ == '__main__':
    main()