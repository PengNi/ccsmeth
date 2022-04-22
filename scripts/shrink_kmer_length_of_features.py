import argparse
import numpy
import os

split_key = "||"


def _shrink_kmer_len_of_features(infile, klen, output):
    if klen % 2 == 0:
        raise ValueError("--klen is not odd")
    fname, fext = os.path.splitext(infile)
    wfile = output if output is not None else fname + ".{}".format(klen) + fext
    wf = open(wfile, "w")
    with open(infile, "r") as rf:
        first_line = next(rf)
        words = first_line.strip().split("\t")
        oriklen = len(words[5])
        if oriklen <= klen:
            raise ValueError("--klen equals/greats kmer-lenth in the file, ori-klen:{}".format(oriklen))
        left_cut = (oriklen - klen) // 2
        right_cut = oriklen - klen - left_cut
        for widx in range(5, len(words)-1):
            tempstr = words[widx].split(",")
            if len(tempstr) == oriklen:
                words[widx] = ",".join(tempstr[left_cut:-right_cut])
            elif len(words[widx]) == oriklen:
                words[widx] = words[widx][left_cut:-right_cut]
            else:
                pass
        wf.write("\t".join(words) + "\n")
        for line in rf:
            words = line.strip().split("\t")
            for widx in range(5, len(words) - 1):
                tempstr = words[widx].split(",")
                if len(tempstr) == oriklen:
                    words[widx] = ",".join(tempstr[left_cut:-right_cut])
                elif len(words[widx]) == oriklen:
                    words[widx] = words[widx][left_cut:-right_cut]
                else:
                    pass
            wf.write("\t".join(words) + "\n")
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser("filter call_mods_freq file by poes")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="")
    parser.add_argument("--output", type=str, required=False,
                        help="")
    parser.add_argument("--klen", type=int, required=False, default=17,
                        help="default 17")

    args = parser.parse_args()
    _shrink_kmer_len_of_features(args.input, args.klen, args.output)


if __name__ == '__main__':
    main()
