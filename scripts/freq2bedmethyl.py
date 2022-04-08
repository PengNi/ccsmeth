#! /usr/bin/python
import argparse
import os


def convert_ccsmeth_rmet_file2bedmethyl(args):
    freqinfo = dict()
    with open(args.freqfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom = words[0]
            pos = int(words[1])
            strand = words[2]
            methy_prob = float(words[3])
            unmethy_prob = float(words[4])
            methy_cov = int(words[5])
            unmethy_cov = int(words[6])
            cov = int(words[7])
            rmet = float(words[8])
            kmer = words[9]

            mkey = (chrom, pos, strand)
            if cov >= args.covcf:
                freqinfo[mkey] = (cov, rmet)
    fkeys = freqinfo.keys()
    if args.sort:
        fkeys = sorted(list(fkeys))
    fname, fext = os.path.splitext(args.freqfile)
    wfile = args.wfile if args.wfile is not None else fname + ".bed"
    wf = open(wfile, "w")
    for fkey in fkeys:
        chrom, pos, strand = fkey
        cov, rmet = freqinfo[fkey]
        wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(cov), strand,
                            str(pos), str(pos + 1), "0,0,0", str(cov),
                            str(int(round(rmet * 100, 0)))]) + "\n")
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqfile", type=str, action="store", required=True,
                        help="ccsmeth freq file")
    parser.add_argument("--covcf", type=int, required=False, default=1,
                        help="")
    parser.add_argument("--wfile", type=str, required=False, default=None)
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")

    args = parser.parse_args()
    convert_ccsmeth_rmet_file2bedmethyl(args)


if __name__ == '__main__':
    main()
