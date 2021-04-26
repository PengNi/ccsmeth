import argparse
import numpy
import os

split_key = "||"


def _read_poses(posfile):
    poses = set()
    with open(posfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            poses.add(split_key.join(words[:2]))
    return poses


def _filter_freqbed_by_poses(infile, posfile):
    poses = _read_poses(posfile)
    fname, fext = os.path.splitext(infile)
    outfile = fname + ".pos_f" + fext
    wf = open(outfile, "w")
    with open(infile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            pos_key = split_key.join(words[:2])
            if pos_key in poses:
                wf.write(line)
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser("filter call_mods_freq file by poes")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="")
    parser.add_argument("--posfile", "-o", type=str, required=True,
                        help="")

    args = parser.parse_args()
    _filter_freqbed_by_poses(args.input, args.posfile)


if __name__ == '__main__':
    main()
