import os
import argparse
import sys
import time

from utils.process_utils import run_cmd
from utils.process_utils import display_args
from utils.process_utils import minimap2_exec
from utils.process_utils import bwa_exec
from utils.process_utils import generate_samtools_view_cmd


here = os.path.abspath(os.path.dirname(__file__))
sam2fq_exec = "python " + here + "/utils/subreads_sam2fastq_std.py"


def check_input_file(inputfile):
    if not (inputfile.endswith(".bam") or inputfile.endswith(".sam") or inputfile.endswith(".fq")
            or inputfile.endswith(".fastq")):
        raise ValueError("--subreads/-i must be in bam/sam/fastq format!")
    inputpath = os.path.abspath(inputfile)
    return inputpath


def check_output_file(outputfile, inputfile, is_bwa):
    if outputfile is None:
        fname, fext = os.path.splitext(inputfile)
        aligner = "bwa" if is_bwa else "minimap2"
        output_path = fname + "." + aligner + ".bam"
    else:
        if not (outputfile.endswith(".sam") or outputfile.endswith(".bam")):
            raise ValueError("--output/-o must be in bam/sam format!")
        output_path = os.path.abspath(outputfile)
    return output_path


def generate_aligner_with_options(is_bwa, path_to_bwa, path_to_minimap2, bestn, threads):
    aligner = minimap2_exec
    if is_bwa:
        aligner = bwa_exec
        if path_to_bwa is not None:
            aligner = os.path.abspath(path_to_bwa)
        aligner += " mem -x pacbio -C -V -t {t}".format(t=threads)
    else:
        if path_to_minimap2 is not None:
            aligner = os.path.abspath(path_to_minimap2)
        num_sencondary = bestn - 1
        if num_sencondary == 0:
            aligner += " -ax map-pb -y -t {t} --secondary=no".format(t=threads)
        else:
            aligner += " -ax map-pb -y -t {t} -N {N}".format(t=threads,
                                                             N=num_sencondary)
    return aligner


def align_subreads_to_genome(args):
    sys.stderr.write("[align_subreads]start..\n")
    start = time.time()
    inputpath = check_input_file(args.subreads)
    outputpath = check_output_file(args.output, inputpath, args.bwa)
    reference = os.path.abspath(args.ref)

    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")
    if not os.path.exists(reference):
        raise IOError("refernce(--ref) file does not exist!")

    aligner = generate_aligner_with_options(args.bwa,
                                            args.path_to_bwa,
                                            args.path_to_minimap2,
                                            args.bestn,
                                            args.threads)
    align_cmds = " ".join([aligner, reference, "-"])
    if inputpath.endswith(".fq") or inputpath.endswith(".fastq"):
        align_cmds = " ".join([aligner, reference, inputpath])
    if outputpath.endswith(".sam"):
        align_cmds += " > {}".format(outputpath)

    samtools_view = generate_samtools_view_cmd(args.path_to_samtools)

    pre_align_cmds = ""
    if inputpath.endswith(".fq") or inputpath.endswith(".fastq"):
        # pre_align_cmds += " ".join(["cat", inputpath])
        pass
    else:
        if inputpath.endswith(".bam"):
            pre_align_cmds += " ".join([samtools_view, "-h", inputpath])
            pre_align_cmds += " | " + sam2fq_exec
        elif inputpath.endswith(".sam"):
            # pre_align_cmds += " ".join(["cat", inputpath])
            pre_align_cmds += sam2fq_exec + " < " + inputpath
        else:
            raise ValueError()

    post_align_cmds = ""
    if outputpath.endswith(".bam"):
        post_align_cmds = " ".join([samtools_view, "-b - >", outputpath])
    elif outputpath.endswith(".sam"):
        pass

    if pre_align_cmds != "":
        align_cmds = " | ".join([pre_align_cmds, align_cmds])
    if post_align_cmds != "":
        align_cmds = " | ".join([align_cmds, post_align_cmds])

    sys.stderr.write("cmds: {}\n".format(align_cmds))
    stdinfo, returncode = run_cmd(align_cmds)
    stdout, stderr = stdinfo
    if returncode:
        sys.stderr.write("failed..\n")
    else:
        sys.stderr.write("succeeded..\n")
    sys.stderr.write("==stdout:\n{}\n".format(str(stdout, 'utf-8')))
    sys.stderr.write("==stderr:\n{}\n".format(str(stderr, 'utf-8')))

    endtime = time.time()
    sys.stderr.write("[align_subreads]costs {:.1f} seconds\n".format(endtime - start))


def main():
    parser = argparse.ArgumentParser()
    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--subreads", "-i", type=str, required=True,
                         help="path to subreads.bam/sam/fastq file as input")
    p_input.add_argument("--ref", type=str, required=True,
                         help="path to genome reference to be aligned, in fasta/fa format. "
                              "If using bwa, the reference must have already been indexed.")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--output", "-o", type=str, required=False,
                          help="output file path for alignment results, bam/sam supported. "
                               "If not specified, the results will be saved in "
                               "input_file_prefix.bam by default.")
    p_output.add_argument("--header", action="store_true", default=False, required=False,
                          help="save header annotations from bam/sam. DEPRECATED")

    p_align = parser.add_argument_group("ALIGN")
    p_align.add_argument("--bestn", "-n", type=int, default=2, required=False,
                         help="retain at most n alignments in minimap2. "
                              "default 2, which means 1 secondary alignments are retained. "
                              "[This arg is for further extension, for now it is no use cause "
                              "we use only primary alignment.]")
    p_align.add_argument("--bwa", action="store_true", default=False, required=False,
                         help="use bwa instead of minimap2 for alignment")
    p_align.add_argument("--path_to_minimap2", type=str, default=None, required=False,
                         help="full path to the executable binary minimap2 file. "
                              "If not specified, it is assumed that minimap2 is "
                              "in the PATH.")
    p_align.add_argument("--path_to_bwa", type=str, default=None, required=False,
                         help="full path to the executable binary bwa file. If not "
                              "specified, it is assumed that bwa is in the PATH.")
    p_align.add_argument("--path_to_samtools", type=str, default=None, required=False,
                         help="full path to the executable binary samtools file. "
                              "If not specified, it is assumed that samtools is in "
                              "the PATH.")
    p_align.add_argument("--threads", "-t", type=int, default=5, required=False,
                         help="number of threads, default 5")

    args = parser.parse_args()

    display_args(args, True)
    align_subreads_to_genome(args)


if __name__ == '__main__':
    main()
