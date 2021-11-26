#! /usr/bin/python
import argparse
import os


def _read_one_mod_freq_file(freqfile):
    freqinfo = {}
    with open(freqfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            m_key = (words[0], int(words[1]), words[2])
            if len(words) == 11:
                pos_in_strand = int(words[3])
                methy_prob = float(words[4])
                unmethy_prob = float(words[5])
                methy_cov = int(words[6])
                unmethy_cov = int(words[7])
                cov = int(words[8])
                rmet = float(words[9])
                kmer = words[10]
                freqinfo[m_key] = [pos_in_strand, methy_prob, unmethy_prob, methy_cov, unmethy_cov,
                                   cov, rmet, kmer]
            elif len(words) == 10:
                methy_prob = float(words[3])
                unmethy_prob = float(words[4])
                methy_cov = int(words[5])
                unmethy_cov = int(words[6])
                cov = int(words[7])
                rmet = float(words[8])
                kmer = words[9]
                freqinfo[m_key] = [methy_prob, unmethy_prob, methy_cov, unmethy_cov,
                                   cov, rmet, kmer]
            else:
                raise ValueError("input format wrong!")
    return freqinfo


def _get_combined_freq_file(freqfiles):
    freqinfo = {}
    freqkeys = set()
    for ffile in freqfiles:
        finfo_tmp = _read_one_mod_freq_file(ffile)
        for fkey in finfo_tmp.keys():
            if len(finfo_tmp[fkey]) == 8:
                if fkey not in freqkeys:
                    freqkeys.add(fkey)
                    freqinfo[fkey] = [-1, 0.0, 0.0, 0, 0, 0, 0.0, "-"]
                freqinfo[fkey][0] = finfo_tmp[fkey][0]
                freqinfo[fkey][1] += finfo_tmp[fkey][1]
                freqinfo[fkey][2] += finfo_tmp[fkey][2]
                freqinfo[fkey][3] += finfo_tmp[fkey][3]
                freqinfo[fkey][4] += finfo_tmp[fkey][4]
                freqinfo[fkey][5] += finfo_tmp[fkey][5]
                freqinfo[fkey][6] = freqinfo[fkey][3] / float(freqinfo[fkey][5])
                freqinfo[fkey][7] = finfo_tmp[fkey][7]
            elif len(finfo_tmp[fkey]) == 7:
                if fkey not in freqkeys:
                    freqkeys.add(fkey)
                    freqinfo[fkey] = [0.0, 0.0, 0, 0, 0, 0.0, "-"]
                freqinfo[fkey][0] += finfo_tmp[fkey][0]
                freqinfo[fkey][1] += finfo_tmp[fkey][1]
                freqinfo[fkey][2] += finfo_tmp[fkey][2]
                freqinfo[fkey][3] += finfo_tmp[fkey][3]
                freqinfo[fkey][4] += finfo_tmp[fkey][4]
                freqinfo[fkey][5] = freqinfo[fkey][2] / float(freqinfo[fkey][4])
                freqinfo[fkey][6] = finfo_tmp[fkey][6]
    return freqinfo


def _write_freqinfo(freqinfo, wfile, is_sort, is_bed):
    wf = open(wfile, "w")
    if is_sort:
        fkeys = sorted(list(freqinfo.keys()))
    else:
        fkeys = list(freqinfo.keys())
    for fkey in fkeys:
        tmpinfo = list(fkey) + freqinfo[fkey]
        if len(tmpinfo) == 11:
            if is_bed:
                wf.write("\t".join([tmpinfo[0], str(tmpinfo[1]), str(tmpinfo[1] + 1), ".", str(tmpinfo[8]),
                                    tmpinfo[2],
                                    str(tmpinfo[1]), str(tmpinfo[1] + 1), "0,0,0", str(tmpinfo[8]),
                                    str(int(round(tmpinfo[9] * 100, 0)))]) + "\n")
            else:
                wf.write("%s\t%d\t%s\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (tmpinfo[0], tmpinfo[1], tmpinfo[2],
                                                                                 tmpinfo[3],
                                                                                 tmpinfo[4],
                                                                                 tmpinfo[5],
                                                                                 tmpinfo[6], tmpinfo[7],
                                                                                 tmpinfo[8], tmpinfo[9],
                                                                                 tmpinfo[10]))
        elif len(tmpinfo) == 10:
            if is_bed:
                wf.write("\t".join([tmpinfo[0], str(tmpinfo[1]), str(tmpinfo[1] + 1), ".", str(tmpinfo[7]),
                                    tmpinfo[2],
                                    str(tmpinfo[1]), str(tmpinfo[1] + 1), "0,0,0", str(tmpinfo[7]),
                                    str(int(round(tmpinfo[8] * 100, 0)))]) + "\n")
            else:
                wf.write("%s\t%d\t%s\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (tmpinfo[0], tmpinfo[1], tmpinfo[2],
                                                                             tmpinfo[3],
                                                                             tmpinfo[4],
                                                                             tmpinfo[5], tmpinfo[6],
                                                                             tmpinfo[7], tmpinfo[8],
                                                                             tmpinfo[9]))
    wf.close()


def combine_freq_files(args):
    modspaths = args.modspath
    file_uid = args.file_uid
    modsfiles = []
    for ipath in modspaths:
        input_path = os.path.abspath(ipath)
        if os.path.isdir(input_path):
            for ifile in os.listdir(input_path):
                if file_uid is None:
                    modsfiles.append('/'.join([input_path, ifile]))
                elif ifile.find(file_uid) != -1:
                    modsfiles.append('/'.join([input_path, ifile]))
        elif os.path.isfile(input_path):
            modsfiles.append(input_path)
        else:
            raise ValueError()
    print("get {} input file(s)..".format(len(modsfiles)))
    freqinfo = _get_combined_freq_file(modsfiles)
    _write_freqinfo(freqinfo, args.wfile, args.sort, args.bed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modspath", action="append", type=str, required=True,
                        help="call_mods_freq file or dir, files all in .freq.txt format")
    parser.add_argument("--wfile", type=str, required=True,
                        help=".freq.txt or .bed format")
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for finding all input files and ignoring '
                             'the un-input-files in a input directory. if input_path is a file, ignore this arg.')
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    parser.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")

    args = parser.parse_args()
    combine_freq_files(args)


if __name__ == '__main__':
    main()
