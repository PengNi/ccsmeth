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
    cur_locs = set()
    for line in rf:
        words = line.strip().split("\t")
        ref_loc = int(words[1])
        holeid, loc, prob_1 = words[3], int(words[4]), float(words[7])
        if holeid != holeid_curr:
            if len(holeid_info) > 0:
                holeid_info = sorted(holeid_info, key=lambda x: x[0])
                holeid_info = list(zip(*holeid_info))
                locs = holeid_info[0]
                prob_1s = holeid_info[1]
                # format: holeid, start, end, num_locs, locs_list, probs_list
                wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                                    str(len(locs)),
                                    ",".join(list(map(str, locs))),
                                    ",".join(list(map(str, prob_1s)))]) + "\n")
            holeid_info = []
            cur_locs = set()
            holeid_curr = holeid
        if loc not in cur_locs:
            cur_locs.add(loc)
            holeid_info.append((loc, prob_1))
    if len(holeid_info) > 0:
        holeid_info = sorted(holeid_info, key=lambda x: x[0])
        holeid_info = list(zip(*holeid_info))
        locs = holeid_info[0]
        prob_1s = holeid_info[1]
        wf.write("\t".join([holeid_curr, str(locs[0]), str(locs[-1]+1),
                            str(len(locs)),
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
    return bedfile if bedfile.endswith(".gz") else bedfile + ".gz"


def _generate_sorted_per_read_calls(per_readsite, output):
    fname, fext = os.path.splitext(per_readsite)
    if output is None:
        wfile = fname + ".per_read.bed"
    else:
        wfile = output
        if wfile.endswith(".gz"):
            wfile = wfile[:-3]
    _generate_per_read_calls(per_readsite, wfile)
    # sort+index
    bedfile = _sort_and_index_bedfile(wfile)
    return bedfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_readsite", type=str, required=True, help="from call_mods module")
    parser.add_argument("--output", type=str, required=False, help="per_read.bed.gz")

    args = parser.parse_args()

    _generate_sorted_per_read_calls(args.per_readsite, args.output)


if __name__ == '__main__':
    main()
