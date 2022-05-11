import os
import argparse
from subprocess import Popen, PIPE
import sys
import time
import gzip

contigs = set(["chr" + str(idx) for idx in range(1, 23)] + ["chrX", "chrY"])
samtools_exec = "samtools"


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def generate_samtools_view_cmd(path_to_samtools):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " view -@ 3 -h"


def cmd_get_stdout_of_input(inputpath, path_to_samtools):
    if inputpath.endswith(".bam"):
        samtools_view = generate_samtools_view_cmd(path_to_samtools)
        cmd = samtools_view + " " + inputpath
    elif inputpath.endswith(".sam"):
        cmd = "cat " + inputpath
    else:
        raise ValueError()
    return cmd


def get_haplotag_of_reads(bam, mapq, readlist):
    readinfo = []
    cnt_all, cnt_ana, cnt_haped = 0, 0, 0
    if bam is not None:
        if readlist is not None:
            sys.stderr.write("--readlist is set, however we still will use --bam to extract haplotaged reads info\n")
        view_cmd = cmd_get_stdout_of_input(bam, None)
        sys.stderr.write("cmd to view input: {}\n".format(view_cmd))
        proc_read = Popen(view_cmd, shell=True, stdout=PIPE)
        while True:
            output = str(proc_read.stdout.readline(), 'utf-8')
            if output != "":
                try:
                    if output.startswith("#") or output.startswith("@"):
                        continue
                    cnt_all += 1
                    words = output.strip().split("\t")
                    readid = words[0]
                    if readid.endswith("/ccs"):
                        readid = readid[:-4]
                    flag = int(words[1])
                    chromname = words[2]
                    mapqtmp = int(words[4])
                    if chromname not in contigs:
                        continue
                    if not (flag == 0 or flag == 16):  # skip segment alignment
                        continue
                    if mapqtmp < mapq:  # skip low mapq alignment
                        continue
                    # TODO: filter use PC (phase confident)?
                    cnt_ana += 1
                    hptag = None
                    try:
                        for i in range(11, len(words)):
                            if words[i].startswith("HP:i"):
                                hptag = int(words[i].split(":")[2])
                                break
                    except ValueError:
                        continue
                    if hptag is None:
                        hptagstr = "unknown"
                    else:
                        hptagstr = "HP" + str(hptag)
                        cnt_haped += 1
                    readinfo.append((readid, hptagstr))
                except Exception:
                    # raise ValueError("error in parsing lines of input!")
                    continue
            elif proc_read.poll() is not None:
                sys.stderr.write("\n")
                sys.stderr.flush()
                break
            else:
                # print("output:", output)
                continue
    elif readlist is not None:
        with open(readlist, "r") as rf:
            next(rf)
            for line in rf:
                cnt_all += 1
                words = line.strip().split("\t")
                readid = words[0]
                if readid.endswith("/ccs"):
                    readid = readid[:-4]
                haplotype = words[1]
                chromname = words[3]
                if chromname not in contigs:
                    continue
                cnt_ana += 1
                if haplotype == "H1":
                    readinfo.append((readid, "HP1"))
                    cnt_haped += 1
                elif haplotype == "H2":
                    readinfo.append((readid, "HP2"))
                    cnt_haped += 1
                elif haplotype == "none":
                    readinfo.append((readid, "unknown"))
                else:
                    sys.stderr.write("{}, {}\n".format(readid, haplotype))
    else:
        raise ValueError("--bam or --readlist must be provided")
    sys.stderr.write("parse whatshap haplotag results: \n"
                     "\tall_reads: {}, kept_reads: {}, haped_reads: {}\n".format(cnt_all, cnt_ana, cnt_haped))
    sys.stderr.flush()
    return readinfo


def _convert_readlist_to_dict(reads):
    haptag2readids = dict()
    haptag2readids["HP1"] = set()
    haptag2readids["HP2"] = set()
    haptag2readids["unknown"] = set()
    for readinfo in reads:
        haptag2readids[readinfo[1]].add(readinfo[0])
    return haptag2readids


def _isright_haped_reads(haptag2readids):
    reads_hp1 = haptag2readids["HP1"]
    reads_hp2 = haptag2readids["HP2"]
    if len(reads_hp1.intersection(reads_hp2)) > 0:
        return False
    return True


def _phase_call_mods(mcallfile, haptag2readids, str_id=None):
    fname, fext = os.path.splitext(mcallfile)
    if fext == ".gz":
        fname, fext = os.path.splitext(fname)
    if str_id is None:
        mcall_hp1 = fname + ".whatshap_HP1" + fext
        mcall_hp2 = fname + ".whatshap_HP2" + fext
    else:
        mcall_hp1 = fname + ".{}.whatshap_HP1".format(str_id) + fext
        mcall_hp2 = fname + ".{}.whatshap_HP2".format(str_id) + fext

    cnt_all, cnt_kept, cnt_hp1, cnt_hp2 = 0, 0, 0, 0
    reads_skipped, reads_unknown, reads_hp1, reads_hp2 = set(), set(), set(), set()
    wf_hp1 = open(mcall_hp1, "w")
    wf_hp2 = open(mcall_hp2, "w")
    if str(mcallfile).endswith(".gz"):
        rf = gzip.open(mcallfile, "rt")
    else:
        rf = open(mcallfile, "r")
    for line in rf:
        words = line.strip().split("\t")
        cnt_all += 1
        chromname = words[0]
        readid = words[len(words) - 6]  # 10:4, 9:3
        if chromname not in contigs:
            reads_skipped.add(readid)
            continue
        cnt_kept += 1
        if readid in haptag2readids["HP1"]:
            wf_hp1.write(line)
            reads_hp1.add(readid)
            cnt_hp1 += 1
        elif readid in haptag2readids["HP2"]:
            wf_hp2.write(line)
            reads_hp2.add(readid)
            cnt_hp2 += 1
        else:
            reads_unknown.add(readid)
    rf.close()
    wf_hp1.close()
    wf_hp2.close()
    cnt_rhp1 = len(reads_hp1)
    cnt_rhp2 = len(reads_hp2)
    cnt_rkept = len(reads_unknown) + cnt_rhp1 + cnt_rhp2
    cnt_rall = len(reads_skipped) + cnt_rkept
    sys.stderr.write("phase methyl calls: \n"
                     "\tall_calls: {}, kept_calls: {}, hp1_calls: {}, hp2_calls: {}\n"
                     "\tall_reads: {}, kept_reads: {}, hp1_reads: {}, hp2_reads: {}\n".format(cnt_all, cnt_kept,
                                                                                              cnt_hp1, cnt_hp2,
                                                                                              cnt_rall, cnt_rkept,
                                                                                              cnt_rhp1, cnt_rhp2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", "-b", type=str, required=False, help=".haplotagged.bam, from whatshap haplotag")
    parser.add_argument("--mapq", type=int, required=False, default=0, help="mapQ cutoff of aligned reads, "
                                                                            "default 0, only being used when "
                                                                            "--bam is provided")
    parser.add_argument("--readlist", "-r", type=str, required=False, help="readlist file from whatshap haplotag, "
                                                                           "not required, "
                                                                           "only required when --bam is not provided")
    parser.add_argument("--methylcall", "-m", type=str, required=False, help="call_mods.tsv from methccs/deepsignal2")
    parser.add_argument("--id", type=str, required=False, default=None, help="id")

    args = parser.parse_args()

    sys.stderr.write("[phase methyl calls]starts..\n")
    start = time.time()

    readsinfo = get_haplotag_of_reads(args.bam, args.mapq, args.readlist)
    haptag2readids = _convert_readlist_to_dict(readsinfo)
    if not _isright_haped_reads(haptag2readids):
        raise ValueError("haptag reads error!")
    _phase_call_mods(args.methylcall, haptag2readids, args.id)

    endtime = time.time()
    sys.stderr.write("[phase methyl calls]costs {:.1f} seconds\n".format(endtime - start))


if __name__ == '__main__':
    main()
