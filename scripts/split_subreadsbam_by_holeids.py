import os
import argparse
import time
import random
import sys
from subprocess import Popen, PIPE


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def generate_samtools_view_cmd(path_to_samtools):
    samtools = "samtools"
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " view -@ 5 -h"


def cmd_get_stdout_of_input(inputpath, path_to_samtools):
    if inputpath.endswith(".bam"):
        samtools_view = generate_samtools_view_cmd(path_to_samtools)
        cmd = samtools_view + " " + inputpath
    elif inputpath.endswith(".sam"):
        cmd = "cat " + inputpath
    else:
        raise ValueError()
    return cmd


def _get_holeid(subread_id):
    words = subread_id.strip().split("/")
    # assume movie_id is the same in one bam
    # holeid = words[0] + "/" + words[1]
    holeid = words[1]
    return holeid


def get_holeids_from_subreadsbam(inputfile, args):
    holeids = set()
    sys.stderr.write("read_input process-{} starts\n".format(os.getpid()))
    cmd_view_input = cmd_get_stdout_of_input(inputfile, args.path_to_samtools)
    sys.stderr.write("cmd to view input: {}\n".format(cmd_view_input))
    proc_read = Popen(cmd_view_input, shell=True, stdout=PIPE)

    while True:
        output = str(proc_read.stdout.readline(), 'utf-8')
        if output != "":
            try:
                if output.startswith("#") or output.startswith("@"):
                    continue
                words = output.strip().split("\t")
                holeid = _get_holeid(words[0])
                holeids.add(holeid)
            except Exception:
                # raise ValueError("error in parsing lines of input!")
                continue
        elif proc_read.poll() is not None:
            break
        else:
            # print("output:", output)
            continue
    return holeids


def split_holeids(holeids, ratio, seed):
    assert (0 <= ratio <= 1)
    holeids = list(holeids)
    random.seed(seed)
    random.shuffle(holeids)

    numholes = len(holeids)
    num_first = round(numholes * ratio)

    return holeids[:num_first], holeids[num_first:]


def split_fastq(args):
    print("[split_subreadsbam]starts..")
    start = time.time()
    inputpath = os.path.abspath(args.bam)
    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")
    fname, fext = os.path.splitext(inputpath)
    output_prefix = os.path.abspath(args.out_prefix) if args.out_prefix is not None else fname
    output1 = output_prefix + ".part1_" + str(args.ratio) + ".sam"
    output2 = output_prefix + ".part2_" + str(1 - args.ratio) + ".sam"

    holeids = get_holeids_from_subreadsbam(inputpath, args)
    holeids1, holeids2 = split_holeids(holeids, args.ratio, args.seed)
    holeids1, holeids2 = set(holeids1), set(holeids2)
    wf1 = open(output1, "w")
    wf2 = open(output2, "w")

    sys.stderr.write("read_input process-{} starts\n".format(os.getpid()))
    cmd_view_input = cmd_get_stdout_of_input(inputpath, args.path_to_samtools)
    sys.stderr.write("cmd to view input: {}\n".format(cmd_view_input))
    proc_read = Popen(cmd_view_input, shell=True, stdout=PIPE)
    while True:
        output = str(proc_read.stdout.readline(), 'utf-8')
        if output != "":
            try:
                if output.startswith("#") or output.startswith("@"):
                    wf1.write(output.strip() + '\n')
                    wf2.write(output.strip() + '\n')
                    continue
                words = output.strip().split("\t")
                holeid = _get_holeid(words[0])
                if holeid in holeids1:
                    wf1.write(output.strip() + '\n')
                else:
                    wf2.write(output.strip() + '\n')
            except Exception:
                # raise ValueError("error in parsing lines of input!")
                continue
        elif proc_read.poll() is not None:
            break
        else:
            # print("output:", output)
            continue
    wf1.flush()
    wf1.close()
    wf2.flush()
    wf2.close()

    fname, fext = os.path.splitext(output1)
    run_cmd(" ".join(["samtools view -@ 5 -bSh", output1, ">", fname + ".bam"]))
    os.remove(output1)
    fname, fext = os.path.splitext(output2)
    run_cmd(" ".join(["samtools view -@ 5 -bSh", output2, ">", fname + ".bam"]))
    os.remove(output2)

    endtime = time.time()
    print("[split_subreadsbam]costs {:.1f} seconds, read {} holes, {} to part1, "
          "{} to part2".format(endtime - start, len(holeids), len(holeids1), len(holeids2)))


def main():
    parser = argparse.ArgumentParser("split fastq files into two parts by hole_ids")
    parser.add_argument("--bam", "-i", type=str, required=True,
                        help="path to subreads.bam file as input")
    parser.add_argument("--out_prefix", "-o", type=str, required=False,
                        help="prefix of output file path. If not specified, the prefix will be "
                             "input_file_prefix by default.")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="ratio for the first part, ratio of the second part will be 1-ratio. "
                             "default 0.5")
    parser.add_argument("--seed", type=int, default=111, required=False,
                        help="seed for randomly selecting subreads, default 111")
    parser.add_argument("--path_to_samtools", type=str, default=None, required=False,
                        help="full path to the executable binary samtools file. "
                             "If not specified, it is assumed that samtools is in "
                             "the PATH.")

    args = parser.parse_args()
    split_fastq(args)


if __name__ == '__main__':
    main()
