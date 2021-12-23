#! /usr/bin/env python
import argparse
import os


sep_key = "||"


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def read_position_file(positionfp, col_chrom=0, col_pos=1, header=False):
    posstrs = set()
    with open(positionfp, 'r') as rf:
        if header:
            next(rf)
        for line in rf:
            words = line.strip().split("\t")
            posstrs.add(sep_key.join([words[col_chrom], words[col_pos]]))
    return posstrs


def main():
    parser = argparse.ArgumentParser(description='extract samples with interested ref_positions '
                                                 'from signal feature file')
    parser.add_argument('--cm_path', type=str, required=True, action="append",
                        help='the call_mods file path needed to be filtered')
    parser.add_argument('-p', "--pos_fp",
                        help="the directory of position file, per line: chromosome\tpos_in_forward_strand",
                        type=str, required=True)
    parser.add_argument('--wfile', type=str, required=True, help="write path")

    args = parser.parse_args()
    cm_fps = args.cm_path
    positionfp = args.pos_fp  # position file
    wfile = args.wfile

    positions = read_position_file(positionfp)
    print('there are {} positions to be chosen'.format(len(positions)))
    wf = open(wfile, "w")
    for cmfp in cm_fps:
        with open(cmfp, "r") as rf:
            for line in rf:
                words = line.strip().split("\t")
                postmp = sep_key.join([words[0], words[1]])
                if postmp in positions:
                    wf.write(line)
    wf.flush()
    wf.close()


if __name__ == '__main__':
    main()
