import os
import argparse
import gzip
import pysam
import pybedtools


def _generate_per_read_calls(per_readsite, output):
    # chromosome, pos, strand, read_name, read_loc, depth, prob_0, prob_1, called_label, seq
    wf = open(output, "w")
    if per_readsite.endswith(".gz"):
        rf = gzip.open(per_readsite, "rt")
    else:
        rf = open(per_readsite, "r")
    holeid_curr = ""
    holeid_info = []
    for line in rf:
        words = line.strip().split("\t")
        holeid, loc, prob_1 = words[3], int(words[4]), float(words[7])
        if holeid != holeid_curr:
            if len(holeid_info) > 0:
                holeid_info = sorted(holeid_info, key=lambda x: x[0])
                holeid_info = list(zip(*holeid_info))
                locs = holeid_info[0]
                prob_1s = holeid_info[1]
                wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                                    ",".join(list(map(str, locs))),
                                    ",".join(list(map(str, prob_1s)))]) + "\n")
            holeid_info = []
            holeid_curr = holeid
        holeid_info.append((loc, prob_1))
    if len(holeid_info) > 0:
        holeid_info = sorted(holeid_info, key=lambda x: x[0])
        holeid_info = list(zip(*holeid_info))
        locs = holeid_info[0]
        prob_1s = holeid_info[1]
        wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                            ",".join(list(map(str, locs))),
                            ",".join(list(map(str, prob_1s)))]) + "\n")
    rf.close()
    wf.close()


def _sort_and_index_bedfile(bedfile):
    ori_bed = pybedtools.BedTool(bedfile)
    ori_bed.sort().moveto(bedfile)
    # TODO: check if zerobased=True need to be setted
    pysam.tabix_index(bedfile, force=True,
                      preset="bed",
                      keep_original=False)


def _generate_sorted_per_read_calls(per_readsite, output, is_gzip):
    fname, fext = os.path.splitext(per_readsite)
    if output is None:
        wfile = fname + ".per_read.bed"
    else:
        wfile = output
        if wfile.endswith(".gz"):
            wfile = wfile[:-3]
    _generate_per_read_calls(per_readsite, wfile)
    if (output is not None and output.endswith(".gz")) or is_gzip:
        _sort_and_index_bedfile(wfile)
        return wfile + ".gz"
    else:
        return wfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_readsite", type=str, required=True, help="from call_mods module")
    parser.add_argument("--output", type=str, required=False, help="per_read.bed/per_read.bed.gz")
    parser.add_argument("--gzip", action="store_true", default=False, required=False,
                        help="if compressing result file using gzip")

    args = parser.parse_args()

    _generate_sorted_per_read_calls(args.per_readsite, args.output, args.gzip)


if __name__ == '__main__':
    main()
