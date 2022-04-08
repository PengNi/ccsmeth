#! /usr/bin/python
import argparse
import os


def combine_fb_of_ccsmeth(report_fp):
    pos2info = {}
    poses = set()
    with open(report_fp, "r") as rf:
        for line in rf:
            words = line.strip().split('\t')
            keytmp = (words[0], int(words[1]))
            if words[2] == '-':
                keytmp = (words[0], int(words[1]) - 1)
            prob0, prob1, met, unmet, coverage, rmet, kmer = float(words[3]), \
                int(words[4]), int(words[5]), int(words[6]), int(words[7]), float(words[8]), words[9]
            if keytmp not in poses:
                poses.add(keytmp)
                pos2info[keytmp] = [0., 0., 0, 0, 0, 0., ""]
            pos2info[keytmp][0] += prob0
            pos2info[keytmp][1] += prob1
            pos2info[keytmp][2] += met
            pos2info[keytmp][3] += unmet
            pos2info[keytmp][4] += coverage
            if words[2] == "+":
                pos2info[keytmp[6]] = kmer
    for cgpos in list(pos2info.keys()):
        if pos2info[cgpos][4] == 0:
            del pos2info[cgpos]
        else:
            pos2info[cgpos][5] = round(float(pos2info[cgpos][2]) / pos2info[cgpos][4], 4)
    mposinfo = []
    for cgpos in pos2info.keys():
        mposinfo.append("\t".join(list(map(str, list(cgpos) + ['+', ] + pos2info[cgpos]))))
    return mposinfo


def combine_fb_of_bs_bismark_CpG_report(report_fp):  # bismark CpG_report.txt
    pos2info = {}
    poses = set()
    with open(report_fp, "r") as rf:
        for line in rf:
            words = line.strip().split('\t')
            chrom, pos, strand, cnt_m, cnt_um, motif, context = words[0], int(words[1]), words[2], \
                int(words[3]), int(words[4]), words[5], words[6]
            if strand == '-':
                pos -= 1
            keytmp = "\t".join([chrom, str(pos)])
            if keytmp not in poses:
                poses.add(keytmp)
                pos2info[keytmp] = []
            pos2info[keytmp].append((strand, cnt_m, cnt_um, motif, context))
    mposinfo = []
    for cgpos in pos2info.keys():
        if len(pos2info[cgpos]) == 1:
            strand, cnt_m, cnt_um, motif, context = pos2info[cgpos]
        elif len(pos2info[cgpos]) == 2:
            strand, cnt_m, cnt_um, motif, context = "+", 0, 0, "", ""
            for posinfo in pos2info[cgpos]:
                strandtmp, cnt_mtmp, cnt_umtmp, motiftmp, contexttmp = posinfo
                if strandtmp == "+":
                    motif = motiftmp
                    context = contexttmp
                cnt_m += cnt_mtmp
                cnt_um += cnt_umtmp
        else:
            continue
        mposinfo.append("\t".join([cgpos, strand, str(cnt_m), str(cnt_um), motif, context]))
    return mposinfo


def combine_fb_of_bedmethyl(report_fp):
    pos2info = {}
    poses = set()
    with open(report_fp, "r") as rf:
        # next(rf)
        for line in rf:
            words = line.strip().split('\t')
            keytmp = (words[0], int(words[1]))
            if words[5] == '-':
                keytmp = (words[0], int(words[1]) - 1)
            coverage, met = int(words[9]), float(words[10]) / 100 * int(words[9])
            if keytmp not in poses:
                poses.add(keytmp)
                pos2info[keytmp] = [0, 0.0]
            pos2info[keytmp][0] += coverage
            pos2info[keytmp][1] += met
    for cgpos in list(pos2info.keys()):
        if pos2info[cgpos][0] == 0:
            del pos2info[cgpos]
        else:
            pos2info[cgpos][1] = round((pos2info[cgpos][1] / pos2info[cgpos][0]) * 100)
    mposinfo = []
    for cgpos in pos2info.keys():
        chrom, pos = cgpos
        coverage, rmet = pos2info[cgpos][0], pos2info[cgpos][1]
        mposinfo.append("\t".join([chrom, str(pos), str(pos+1), ".", str(coverage), "+",
                                   str(pos), str(pos + 1), "0,0,0", str(coverage),
                                   str(rmet)]))
    return mposinfo


def write_mpos2covinfo(mclist, mcfile):
    with open(mcfile, 'w') as wf:
        for mctmp in mclist:
            wf.write(mctmp + '\n')
    return mclist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_fp", help="the pred_in_ref report file path",
                        type=str, required=True)
    parser.add_argument('-t', "--rtype", help="bismark, bedmethyl, or ccsmeth",
                        type=str, default='bedmethyl')
    parser.add_argument("--out", "-o", help="wfile",
                        type=str, default=None)
    argv = parser.parse_args()

    report_fp = argv.report_fp
    rtype = argv.rtype

    print('start to combine forward backward strands..')
    if rtype == 'bismark':
        # ===
        mposinfo = combine_fb_of_bs_bismark_CpG_report(report_fp)
    elif rtype == 'ccsmeth':
        mposinfo = combine_fb_of_ccsmeth(report_fp)
    elif rtype == 'bedmethyl':
        mposinfo = combine_fb_of_bedmethyl(report_fp)
    else:
        raise ValueError()
    fname, fext = os.path.splitext(report_fp)
    wfp = argv.out if argv.out is not None else fname + '.fb_comb' + fext
    write_mpos2covinfo(mposinfo, wfp)


if __name__ == '__main__':
    main()
