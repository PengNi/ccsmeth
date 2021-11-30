#! /usr/bin/python
import argparse
import os

split_key = "||"


def _write_combedinfo_2strands(pair1info, pair2info, poskey, is_weight, wf):
    chrom, pos, holeid = poskey.split(split_key)
    prob0_1, prob1_1, strand_1, depth_1, kmer_1 = pair1info
    prob0_2, prob1_2, strand_2, depth_2, kmer_2 = pair2info
    if is_weight:
        depth_12 = depth_1 + depth_2
        prob0 = round((prob0_1 * depth_1 + prob0_2 * depth_2) / depth_12, 6)
        prob1 = round((prob1_1 * depth_1 + prob1_2 * depth_2) / depth_12, 6)
    else:
        prob0 = round((prob0_1 + prob0_2) / 2, 6)
        prob1 = round((prob1_1 + prob1_2) / 2, 6)
    label = 0 if prob0 > prob1 else 1
    strand = "+"
    depth = str(depth_1) + "," + str(depth_2) if strand_1 == "+" else str(depth_2) + "," + str(depth_1)
    kmer = kmer_1 if strand_1 == "+" else kmer_2
    cinfotmp = (chrom, pos, strand, holeid, depth, prob0, prob1, label, kmer)
    wf.write('\t'.join(list(map(str, list(cinfotmp)))) + '\n')


def _write_combedinfo_1strand(pos2info, wf):
    for poskey in pos2info.keys():
        chrom, pos, holeid = poskey.split(split_key)
        prob0, prob1, strand, depth, kmer = pos2info[poskey]
        label = 0 if prob0 > prob1 else 1
        strand = "+"
        cinfotmp = (chrom, pos, strand, holeid, depth, prob0, prob1, label, kmer)
        wf.write('\t'.join(list(map(str, list(cinfotmp)))) + '\n')


def combine_fb_of_ccs_results(result_fp, is_weight, wfile):
    pos2info = {}
    poskeys = set()
    wf = open(wfile, "w")
    cnt_line = 0
    cnt_paired = 0
    with open(result_fp, "r") as rf:
        for line in rf:
            cnt_line += 1
            words = line.strip().split("\t")
            chrom, pos, strand, holeid, depth, prob0, prob1, kmer = words[0], words[1], words[2], words[3], \
                int(words[4]), float(words[5]), float(words[6]), \
                words[8]
            if strand == "-":
                pos = str(int(pos) - 1)
            poskey = split_key.join([chrom, pos, holeid])
            if poskey not in poskeys:
                poskeys.add(poskey)
                pos2info[poskey] = (prob0, prob1, strand, depth, kmer)
            else:
                pair2info = (prob0, prob1, strand, depth, kmer)
                _write_combedinfo_2strands(pos2info[poskey], pair2info, poskey, is_weight, wf)
                del pos2info[poskey]
                poskeys.remove(poskey)
                cnt_paired += 1
    cnt_unpaired = len(poskeys)
    if len(pos2info.keys()) > 0:
        del poskeys
        _write_combedinfo_1strand(pos2info, wf)
    wf.flush()
    wf.close()
    print("read {} lines, get {} 2strands sites, {} 1strand sites".format(cnt_line,
                                                                          cnt_paired,
                                                                          cnt_unpaired))


def main():
    parser = argparse.ArgumentParser("combine modification_frequency of CG in forward and backward strand of CCS")
    parser.add_argument("--result_fp", help="the call_modification.py file path",
                        type=str, required=True)
    parser.add_argument("--weight", action='store_true', default=False,
                        help="use depth to calculated weighted combined prob")
    argv = parser.parse_args()

    result_fp = argv.result_fp

    print('start to combine forward backward strands..')
    fname, fext = os.path.splitext(result_fp)
    if argv.weight:
        wfp = fname + '.fb_comb_w' + fext
    else:
        wfp = fname + '.fb_comb' + fext
    combine_fb_of_ccs_results(result_fp, argv.weight, wfp)


if __name__ == '__main__':
    main()
