import numpy as np
import gc
from subprocess import Popen, PIPE
import os
import sys
import re
import pysam

from .logging import mylogger
LOGGER = mylogger(__name__)

basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D',
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D',
                 'Z': 'Z'}

# base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
#                   'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
#                   'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
#                   'Z': 15}
# code2base_dna = dict((v, k) for k, v in base2code_dna.items())
base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                 'W': 4, 'S': 4, 'M': 4, 'K': 4, 'R': 4,
                 'Y': 4, 'B': 4, 'V': 4, 'D': 4, 'H': 4,
                 'Z': 4}
code2base_dna = dict((v, k) for k, v in base2code_dna.items() if k in {'A', 'C', 'G', 'T', 'N'})

# base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4,
#                  'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
#                  'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
#                  'Z': 15}
# code2base_rna = dict((v, k) for k, v in base2code_rna.items())
base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4,
                 'W': 4, 'S': 4, 'M': 4, 'K': 4, 'R': 4,
                 'Y': 4, 'B': 4, 'V': 4, 'D': 4, 'H': 4,
                 'Z': 4}
code2base_rna = dict((v, k) for k, v in base2code_rna.items() if k in {'A', 'C', 'G', 'U', 'N'})

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}

CODE2CIGAR = "MIDNSHP=XB"
CIGAR_REGEX = re.compile("(\d+)([MIDNSHP=XB])")
CIGAR2CODE = dict([y, x] for x, y in enumerate(CODE2CIGAR))

SEQ_ORDER = "ACGT"
SEQ_ENCODE = dict([y, x] for x, y in enumerate(SEQ_ORDER))

# for feature extraction/embedding
N_VOCAB = 5  # related to base2code_dna/rna
MAX_KINETICS = 952  # 952 or 255?
MAX_PASSES = 30
MAX_MAP = 8
NEMBED_BASE = 8
NEMBED_KINETICS = 8
NEMBED_PASSES = 4
NEMBED_MAP = 4
NEMBED_SN = 4  # TODO: need to see if this is suitable
NEMBED_KINETICS_STD = 8  # placeholder

max_queue_size = 600

nproc_to_call_mods_in_cpu_mode = 2

pbmm2_exec = "pbmm2"
minimap2_exec = "minimap2"
bwa_exec = "bwa"
samtools_exec = "samtools"
ccs_exec = "ccs"

default_ref_loc = -1


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


# https://thispointer.com/python-three-ways-to-check-if-a-file-is-empty/
def is_file_empty(file_name):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.isfile(file_name) and os.path.getsize(file_name) == 0


def _alphabet(letter, dbasepairs):
    if letter in dbasepairs.keys():
        return dbasepairs[letter]
    return 'N'


def complement_seq(base_seq, seq_type="DNA") -> str:
    rbase_seq = base_seq[::-1]
    comseq = ''
    try:
        if seq_type == "DNA":
            comseq = ''.join([_alphabet(x, basepairs) for x in rbase_seq])
        elif seq_type == "RNA":
            comseq = ''.join([_alphabet(x, basepairs_rna) for x in rbase_seq])
        else:
            raise ValueError("the seq_type must be DNA or RNA")
    except Exception:
        LOGGER.warning('something wrong in the dna/rna sequence.')
    return comseq


# motifs ======================================================================
def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0) -> list:
    """

    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            sites.append(i+methyloc_in_motif)
    return sites


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)
    return recursive_permute(outbases)


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(',')

    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)
    return motif_seqs


# CIGAR =============================================================================
def compute_pct_identity(cigarnum_array):
    try:
        nalign, nmatch = 0, 0
        for idx in range(len(CODE2CIGAR)):
            if idx not in {4, 5}:
                nalign += cigarnum_array[idx]
            if idx in {0, 7}:
                nmatch += cigarnum_array[idx]
        return nmatch / float(nalign)
    except IndexError:
        return 0.
    except ZeroDivisionError:
        return 0.


# modified version of parse_cigar() in megalodon.mapping
def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    insertion: -1, deletion: -2, mismatch: -3
    :param r_cigar_tuple: pysam.alignmentSegment.cigartuples
    :param strand: 1/-1 for fwd/rev
    :param seq_len: read alignment length
    :return: query pos to ref pos
    """
    fill_invalid = -2
    # get each base calls genomic position
    q_to_r_poss = np.full(seq_len + 1, fill_invalid, dtype=np.int32)
    # process cigar ops in read direction
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar_tuple if strand == 1 else r_cigar_tuple[::-1]
    for op, op_len in cigar_ops:
        if op == 1:
            # inserted bases into ref
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = -1
            curr_q_pos += op_len
        elif op in (2, 3):
            # deleted ref bases
            curr_r_pos += op_len
        elif op in (0, 7, 8):
            # aligned bases
            for op_offset in range(op_len):
                q_to_r_poss[curr_q_pos + op_offset] = curr_r_pos + op_offset
            curr_q_pos += op_len
            curr_r_pos += op_len
        elif op == 6:
            # padding (shouldn't happen in mappy)
            pass
    q_to_r_poss[curr_q_pos] = curr_r_pos
    if q_to_r_poss[-1] == fill_invalid:
        raise ValueError(('Invalid cigar string encountered. Reference length: {}  Cigar ' +
                          'implied reference length: {}').format(seq_len, curr_r_pos))
    return q_to_r_poss


# arg display ========================================================================
def display_args(args, is_stderr=True):
    def print_outputstr(outstr):
        if is_stderr:
            sys.stderr.write(outstr + "\n")
            sys.stderr.flush()
        else:
            print(outstr)
            sys.stdout.flush()
    arg_vars = vars(args)
    outputstr = "# ===============================================\n## parameters: "
    print_outputstr(outputstr)
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            outputstr = "{}:\n\t{}".format(arg_key, arg_vars[arg_key])
            print_outputstr(outputstr)
    print_outputstr("# ===============================================")


# subprocess ========================================================================
def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def run_cmd_live_stdout(args_list):
    # https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
    proc = Popen(args_list, shell=True, stdout=PIPE)
    while True:
        output = str(proc.stdout.readline(), 'utf-8')
        if output == "" and proc.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = proc.poll()
    return rc


def generate_samtools_view_cmd(path_to_samtools, threads=3):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " view -@ {} -h".format(threads if threads is not None else 3)


def generate_samtools_index_cmd(path_to_samtools, threads=10):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " index -@ {}".format(threads if threads is not None else 3)


def generate_samtools_sort_cmd(path_to_samtools, outputfile, threads=10):
    samtools = samtools_exec
    if path_to_samtools is not None:
        samtools = os.path.abspath(path_to_samtools)
    return samtools + " sort -@ {} -o {}".format(threads if threads is not None else 3,
                                                 outputfile)


# def index_bam_if_needed(inputfile, args):
#     if str(inputfile).endswith(".bam") and not os.path.exists(inputfile + ".bai"):
#         samtools_index = generate_samtools_index_cmd(args.path_to_samtools, args.threads)
#         index_cmd = " ".join([samtools_index, inputfile])
#         sys.stderr.write("indexing bam file: {}\n".format(index_cmd))
#         stdinfo, returncode = run_cmd(index_cmd)
#         if returncode:
#             raise ValueError("indexing bam file failed, please try indexing it mannually using samtools")
#         else:
#             sys.stderr.write("indexing bam file succeeded..\n")


def index_bam_if_needed2(inputfile, threads):
    if str(inputfile).endswith(".bam") \
            and not os.path.exists(inputfile + ".bai") \
            and not os.path.exists(inputfile + ".csi"):
        LOGGER.info("indexing bam file-{}\n".format(inputfile))
        try:
            pysam.index("-@", str(threads), inputfile)
        except pysam.utils.SamtoolsError:
            pysam.index("-@", str(threads), inputfile, "-c")


# =================================================================
def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    # print('done count the lines of file {}'.format(sl_filepath))
    return count


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        LOGGER.info('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    LOGGER.info('done writing features info to {}'.format(wfilepath))


def concat_two_files(file1, file2, concated_fp, shuffle_lines_num=2000000,
                     lines_num=1000000000000, isheader=False):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, False)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, False)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()
    LOGGER.info('done concating files to: {}'.format(concated_fp))
# =================================================================


def codecv1_to_frame():
    code2frames = dict()
    for i in range(0, 64):
        code2frames[i] = i

    frames = []
    for frame in range(64, 190 + 1, 2):
        frames.append(frame)
    for i in range(64, 128):
        code2frames[i] = frames[i - 64]

    frames = []
    for frame in range(192, 444 + 1, 4):
        frames.append(frame)
    for i in range(128, 192):
        code2frames[i] = frames[i - 128]

    frames = []
    for frame in range(448, 952 + 1, 8):
        frames.append(frame)
    for i in range(192, 256):
        code2frames[i] = frames[i - 192]

    return code2frames


def codecv1_to_frame2():
    code2frames = [0] * 256
    for i in range(0, 64):
        code2frames[i] = i

    frames = []
    for frame in range(64, 190 + 1, 2):
        frames.append(frame)
    for i in range(64, 128):
        code2frames[i] = frames[i - 64]

    frames = []
    for frame in range(192, 444 + 1, 4):
        frames.append(frame)
    for i in range(128, 192):
        code2frames[i] = frames[i - 128]

    frames = []
    for frame in range(448, 952 + 1, 8):
        frames.append(frame)
    for i in range(192, 256):
        code2frames[i] = frames[i - 192]

    return code2frames
