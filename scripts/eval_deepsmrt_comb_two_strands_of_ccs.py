#! /usr/bin/python
import argparse
import os

split_key = "||"


def combine_fb_of_ccs_results(result_fp):
    pos2info = {}
    with open(result_fp, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom, pos, strand, holeid, depth, prob0, prob1, kmer = words[0], words[1], words[2], words[3], \
                                                                    words[4], float(words[5]), float(words[6]), \
                                                                    words[8]
            if strand == "-":
                pos = str(int(pos) - 1)
            poskey = split_key.join([chrom, pos, holeid])
            if poskey not in pos2info.keys():
                pos2info[poskey] = []
            pos2info[poskey].append((prob0, prob1, strand, depth, kmer))
    combedinfo = []
    for pos in pos2info.keys():
        infotmp = pos2info[pos]
        chrom, pos, holeid = pos.split(split_key)
        if len(infotmp) == 1:
            prob0, prob1, strand, depth, kmer = infotmp[0]
            label = 0 if prob0 > prob1 else 1
        elif len(infotmp) == 2:
            prob0_1, prob1_1, strand_1, depth_1, kmer_1 = infotmp[0]
            prob0_2, prob1_2, strand_2, depth_2, kmer_2 = infotmp[1]
            prob0 = round((prob0_1 + prob0_2)/2, 6)
            prob1 = round((prob1_1 + prob1_2)/2, 6)
            label = 0 if prob0 > prob1 else 1
            strand = "+"
            depth = depth_1 + "," + depth_2 if strand_1 == "+" else depth_2 + "," + depth_1
            kmer = kmer_1 if strand_1 == "+" else kmer_2
        else:
            continue
        combedinfo.append((chrom, pos, strand, holeid, depth, prob0, prob1, label, kmer))
    return combedinfo


def write_combedinfo(combedinfo, wfile):
    with open(wfile, 'w') as wf:
        # wf.write('\t'.join(['chromosome', 'pos', 'strand', 'holeid', 'depthW,depthC', 'prob0', 'prob1',
        #                     label, 'kmer']) + '\n')
        for cinfotmp in combedinfo:
            wf.write('\t'.join(list(map(str, list(cinfotmp)))) + '\n')


def main():
    parser = argparse.ArgumentParser("combine modification_frequency of CG in forward and backward strand of CCS")
    parser.add_argument("--result_fp", help="the call_modification.py file path",
                        type=str, required=True)
    argv = parser.parse_args()

    result_fp = argv.result_fp

    print('start to combine forward backward strands..')
    fname, fext = os.path.splitext(result_fp)
    wfp = fname + '.fb_comb' + fext
    combedinfo = combine_fb_of_ccs_results(result_fp)
    write_combedinfo(combedinfo, wfp)


if __name__ == '__main__':
    main()
