import os
import argparse
import time
import random


def _get_holeid(subread_id):
    words = subread_id.strip().split("/")
    # assume movie_id is the same in one bam
    # holeid = words[0] + "/" + words[1]
    holeid = words[1]
    return holeid


def get_holeids_from_fastq(fastqfile):
    n = 4
    holeids = set()
    with open(fastqfile, "r") as rf:
        lidx = 0
        for line in rf:
            if lidx % n == 0:
                assert (line.startswith("@"))
                readid = line.strip().split()[0][1:]
                holeid = _get_holeid(readid)
                holeids.add(holeid)
            lidx += 1
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
    print("[split_fastq]start..")
    start = time.time()
    inputpath = os.path.abspath(args.fastq)
    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")
    fname, fext = os.path.splitext(inputpath)
    output_prefix = os.path.abspath(args.out_prefix) if args.out_prefix is not None else fname
    output1 = output_prefix + ".part1_" + str(args.ratio) + ".fq"
    output2 = output_prefix + ".part2_" + str(1 - args.ratio) + ".fq"

    holeids = get_holeids_from_fastq(inputpath)
    holeids1, holeids2 = split_holeids(holeids, args.ratio, args.seed)
    holeids1, holeids2 = set(holeids1), set(holeids2)
    wf1 = open(output1, "w")
    wf2 = open(output2, "w")
    n = 4
    with open(inputpath, "r") as rf:
        fqitem = []
        for line in rf:
            fqitem.append(line.rstrip())
            if len(fqitem) == n:
                holeid = _get_holeid(fqitem[0].split()[0][1:])
                # write fqitems
                if holeid in holeids1:
                    for fqstr in fqitem:
                        wf1.write(str(fqstr) + "\n")
                else:  # holeids2 not used actually
                    for fqstr in fqitem:
                        wf2.write(str(fqstr) + "\n")
                fqitem = []
    wf1.flush()
    wf1.close()
    wf2.flush()
    wf2.close()

    endtime = time.time()
    print("[split_fastq]costs {:.1f} seconds, read {} holes, {} to part1, "
          "{} to part2".format(endtime - start, len(holeids), len(holeids1), len(holeids2)))


def main():
    parser = argparse.ArgumentParser("split fastq files into two parts by hole_ids")
    parser.add_argument("--fastq", "-i", type=str, required=True,
                        help="path to subreads.fastq file as input")
    parser.add_argument("--out_prefix", "-o", type=str, required=False,
                        help="prefix of output file path. If not specified, the prefix will be "
                             "input_file_prefix by default.")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="ratio for the first part, ratio of the second part will be 1-ratio. "
                             "default 0.5")
    parser.add_argument("--seed", type=int, default=111, required=False,
                        help="seed for randomly selecting subreads, default 111")

    args = parser.parse_args()
    split_fastq(args)


if __name__ == '__main__':
    main()
