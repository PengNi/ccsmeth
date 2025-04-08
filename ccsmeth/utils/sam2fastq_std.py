import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header_file", type=str, default=None,
                        help="file to save header in sam")

    args = parser.parse_args()

    for line in sys.stdin:
        headers = ""
        if str(line).startswith("@"):
            headers += line.strip() + "\n"
            continue
        if args.header_file is not None:
            with open(args.header_file, "w") as wf:
                wf.write(headers)
                wf.flush()
        words = line.strip().split("\t")
        readid = words[0]
        comments = "\t".join(words[11:])
        bases = words[9]
        qualities = words[10]

        fq_read = "\n".join(["\t".join(["@" + readid, comments]),
                             bases,
                             "+",
                             qualities])
        sys.stdout.write(fq_read + "\n")


if __name__ == '__main__':
    main()
