import os
import argparse
import sys
import time
from subprocess import Popen, PIPE

samtools_exec = "samtools"


def generate_samtools_view_cmd(path_to_samtools):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " view -@ 5"


def check_input_file(inputfile):
    if not (inputfile.endswith(".bam") or inputfile.endswith(".sam")):
        raise ValueError("--subreads/-i must be in bam/sam format!")
    inputpath = os.path.abspath(inputfile)
    return inputpath


def check_output_file(outputfile, inputfile):
    if outputfile is None:
        fname, fext = os.path.splitext(inputfile)
        output_path = fname + ".fq"
    else:
        if not (outputfile.endswith(".fq") or outputfile.endswith(".fastq")):
            raise ValueError("--output/-o must be in fastq format!")
        output_path = os.path.abspath(outputfile)
    return output_path


def _get_holeid(subread_id):
    words = subread_id.strip().split("/")
    # assume movie_id is the same in one bam
    # holeid = words[0] + "/" + words[1]
    holeid = words[1]
    return holeid


def get_subreads_fastq(args):
    sys.stderr.write("[bam2fq]start..\n")
    start = time.time()
    inputpath = check_input_file(args.subreads)
    outputpath = check_output_file(args.output, inputpath)

    if not os.path.exists(inputpath):
        raise IOError("input file does not exist!")

    samtools_view = generate_samtools_view_cmd(args.path_to_samtools)

    view_cmd = ""
    if inputpath.endswith(".bam"):
        view_cmd += " ".join([samtools_view, "-h", inputpath])
    elif inputpath.endswith(".sam"):
        view_cmd += " ".join(["cat", inputpath])
    else:
        raise ValueError()

    sys.stderr.write("cmd to view input: {}\n".format(view_cmd))
    proc_read = Popen(view_cmd, shell=True, stdout=PIPE)
    cnt_holes = 0
    cnt_lines = 0
    holeid_curr = ""
    wf = open(outputpath, "w")
    while True:
        output = str(proc_read.stdout.readline(), 'utf-8')
        if output != "":
            try:
                if output.startswith("#") or output.startswith("@"):
                    continue
                words = output.strip().split("\t")
                holeid = _get_holeid(words[0])

                readid = words[0]
                comments = "\t".join(words[11:])
                bases = words[9]
                qualities = words[10]

                fq_read = "\n".join(["\t".join(["@" + readid, comments]),
                                     bases,
                                     "+",
                                     qualities])
                wf.write(fq_read + "\n")

                if holeid != holeid_curr:
                    cnt_holes += 1
                    holeid_curr = holeid
                cnt_lines += 1
            except Exception:
                # raise ValueError("error in parsing lines of input!")
                continue
        elif proc_read.poll() is not None:
            break
        else:
            # print("output:", output)
            continue
    rc = proc_read.poll()
    wf.flush()
    wf.close()
    endtime = time.time()
    sys.stderr.write("[bam2fq]costs {:.1f} seconds, {} holes ({} lines) proceed, "
                     "with returncode-{}\n".format(endtime - start,
                                                   cnt_holes,
                                                   cnt_lines,
                                                   rc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreads", "-i", type=str, required=True,
                        help="path to subreads.bam/sam file as input")

    parser.add_argument("--output", "-o", type=str, required=False,
                        help="output file path. If not specified, the results will be saved in "
                             "input_file_prefix.fq by default.")

    parser.add_argument("--path_to_samtools", type=str, default=None, required=False,
                        help="full path to the executable binary samtools file. "
                             "If not specified, it is assumed that samtools is in "
                             "the PATH.")

    args = parser.parse_args()

    get_subreads_fastq(args)


if __name__ == '__main__':
    main()
