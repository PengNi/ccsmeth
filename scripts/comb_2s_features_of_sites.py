#! /usr/bin/python
import argparse
import os


def _comb_fb_features(fwd_feas, bwd_feas):
    if len(fwd_feas) <= 0 or len(bwd_feas) <= 0:
        return []
    fwd_feas = sorted(fwd_feas, key=lambda x: x[1])
    bwd_feas = sorted(bwd_feas[::-1], key=lambda x: x[1])
    comb_feas = []
    idx_f, idx_b = 0, 0
    while idx_f < len(fwd_feas) and idx_b < len(bwd_feas):
        ffea = fwd_feas[idx_f]
        bfea = bwd_feas[idx_b]
        fpos = ffea[1]
        bpos = bfea[1]
        if fpos == bpos - 1:
            ffea[4] = max(ffea[4], bfea[4])
            comb_feas.append(ffea[:13] + bfea[5:])
            idx_f += 1
            idx_b += 1
        elif fpos < bpos - 1:
            idx_f += 1
        else:
            idx_b += 1
    return comb_feas


def _comb_grouped_features(fea, wf):
    feas_f = []
    feas_b = []
    curr_holeid = ""
    cnt_line = 0
    cnt_holes = 0
    with open(fea, "r") as rf:
        for line in rf:
            cnt_line += 1
            words = line.strip().split("\t")
            holeid = words[3]
            if holeid != curr_holeid:
                if curr_holeid != "":
                    cnt_holes += 1
                    feas_fb = _comb_fb_features(feas_f, feas_b)
                    for ffbtmp in feas_fb:
                        wf.write("\t".join(list(map(str, ffbtmp))) + "\n")
                curr_holeid = holeid
                feas_f = []
                feas_b = []
            strand = words[2]
            words[1] = int(words[1])
            if strand == "+":
                feas_f.append(words)
            else:
                feas_b.append(words)
        cnt_holes += 1
        feas_fb = _comb_fb_features(feas_f, feas_b)
        for ffbtmp in feas_fb:
            wf.write("\t".join(list(map(str, ffbtmp))) + "\n")
    print("proceed {} lines, comb {} holes".format(cnt_line, cnt_holes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fea", help="feature file",
                        type=str, required=True)
    parser.add_argument("--not_grouped", action="store_true", default=False, required=False,
                        help="the --fea file not grouped by holeids")
    argv = parser.parse_args()

    print('start to combine features of forward backward strands..')
    fname, fext = os.path.splitext(argv.fea)
    wfile = fname + ".fb" + fext
    wf = open(wfile, "w")
    if argv.not_grouped:
        pass
    else:
        _comb_grouped_features(argv.fea, wf)
    wf.flush()
    wf.close()


if __name__ == '__main__':
    main()