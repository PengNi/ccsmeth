import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idsrc", type=str, required=True)
    # default false, which means add pos in rc strand too, for CG sites
    parser.add_argument("--one_strand", action="store_true", default=False)
    # parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    result_file = args.idsrc

    fname, fext = os.path.splitext(result_file)
    sampleids_file = fname + ".sampleids.txt"
    wf = open(sampleids_file, "w")
    with open(result_file, "r") as rf:
        for line in rf:
            if line.startswith("#"):
                continue
            words = line.strip().split("\t")
            # depths = words[5].split(",")
            # depthW = float(depths[0])
            # depthC = float(depths[1])
            # if depthW < args.depth or depthC < args.depth:
            #     continue
            chrom, pos, strand, holeid = words[0], int(words[1]), words[2], words[3]
            holeid = holeid.split("/")[1]
            if strand == "-":
                pos = pos - 1
            sampid = "\t".join([chrom, str(pos), holeid])  # chrom, pos, holeid
            wf.write(sampid + "\n")
            if not args.one_strand:
                sampid = "\t".join([chrom, str(int(pos) + 1), holeid])
                wf.write(sampid + "\n")
    wf.flush()
    wf.close()


if __name__ == '__main__':
    main()
