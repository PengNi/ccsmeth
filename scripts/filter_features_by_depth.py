import argparse
import numpy


def _filter_features_by_depth(infile, outfile, depth=1):
    wf = open(outfile, "w")
    with open(infile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            depths = [int(x) for x in words[6].split(",")]
            if depth <= numpy.mean(depths):
                wf.write(line)
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser("filter features by depth")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="")
    parser.add_argument("--depth", type=int, required=False, default=1,
                        help="depth cutoff, default 1")

    args = parser.parse_args()
    _filter_features_by_depth(args.input, args.output, args.depth)


if __name__ == '__main__':
    main()
