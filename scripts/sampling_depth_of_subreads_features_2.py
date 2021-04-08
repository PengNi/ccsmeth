import os
import argparse
import sys
import numpy as np
import random

exceptval = 1000


def check_excpval(myarray):
    if exceptval in myarray:
        return True
    return False


def _sampling_subreads(infile, outfile, depth=1):
    wf = open(outfile, "w")
    with open(infile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            sampleinfo = "\t".join(words[:7])
            ipd_mat = [[float(y) for y in str(x).split(",")] for x in words[7].split(";")]
            pw_mat = [[float(y) for y in str(x).split(",")] for x in words[8].split(";")]
            label = words[9]

            if depth < len(ipd_mat):
                seled_idx = sorted(random.sample(range(len(ipd_mat)), depth))
                ipd_mat = [ipd_mat[idx] for idx in seled_idx]
                pw_mat = [pw_mat[idx] for idx in seled_idx]

            ipd_mat = np.array(ipd_mat, dtype=np.float)
            ipd_mean = np.round(np.mean(ipd_mat, 0), decimals=6)
            ipd_std = np.round(np.std(ipd_mat, 0), decimals=6)

            pw_mat = np.array(pw_mat, dtype=np.float)
            pw_mean = np.round(np.mean(pw_mat, 0), decimals=6)
            pw_std = np.round(np.std(pw_mat, 0), decimals=6)

            wf.write("\t".join([sampleinfo, ",".join([str(x) for x in ipd_mean]),
                                ",".join([str(x) for x in ipd_std]),
                                ",".join([str(x) for x in pw_mean]),
                                ",".join([str(x) for x in pw_std]), "-", "-", label]) + "\n")
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser("extract features with info of all subreads, and only keep reads with "
                                     "subread_depth >= 30, step2")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="")
    parser.add_argument("--depth", type=int, required=False, default=1,
                        help="depth to sampling, default 1")

    args = parser.parse_args()
    _sampling_subreads(args.input, args.output, args.depth)


if __name__ == '__main__':
    main()
