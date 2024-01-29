#!/usr/bin/python
from __future__ import absolute_import

import sys
import argparse

from .utils.process_utils import display_args

from ._version import VERSION


def main_call_hifi(args):
    from .call_hifi_reads import ccs_call_hifi_reads

    display_args(args)
    ccs_call_hifi_reads(args)


def main_align_hifi(args):
    from .align_hifi_reads import align_hifi_reads_to_genome

    display_args(args)
    align_hifi_reads_to_genome(args)


def main_call_mods(args):
    from .call_modifications import call_mods

    display_args(args)
    call_mods(args)


def main_call_freqt(args):
    from .call_mods_freq_txt import call_mods_frequency_to_file

    display_args(args)
    call_mods_frequency_to_file(args)


def main_call_freqb(args):
    from .call_mods_freq_bam import call_mods_frequency_from_bamfile

    display_args(args)
    call_mods_frequency_from_bamfile(args)


def main_extract(args):
    from .extract_features import extract_hifireads_features

    display_args(args)
    extract_hifireads_features(args)


def main_train(args):
    from .train import train

    display_args(args)
    train(args)


def main_trainm(args):
    from .train_multigpu import train

    display_args(args)
    train(args)


def main():
    parser = argparse.ArgumentParser(prog='ccsmeth',
                                     description="ccsmeth detects methylation from PacBio CCS reads, "
                                                 "which contains 8 modules:\n"
                                                 "\t%(prog)s call_hifi: call hifi reads from subreads "
                                                 "using CCS (PBCCS)\n"
                                                 "\t%(prog)s call_mods: call modifications\n"
                                                 "\t%(prog)s align_hifi: align hifi reads to reference\n"
                                                 "\t%(prog)s call_freqt: call modification frequencies from "
                                                 "per_readsite text files\n"
                                                 "\t%(prog)s call_freqb: call modification frequencies from "
                                                 "modbam.bam file\n"
                                                 "\t%(prog)s extract: extract features from hifi reads "
                                                 "for training or testing\n"
                                                 "\t%(prog)s train: train a model, need two independent "
                                                 "datasets for training and validating\n"
                                                 "\t%(prog)s trainm: [EXPERIMENTAL]train a model using multi "
                                                 "gpus",
                                                 # "\t%(prog)s denoise: [EXPERIMENTAL]denoise",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-v', '--version', action='version',
        version='ccsmeth version: {}'.format(VERSION),
        help='show ccsmeth version and exit.')

    subparsers = parser.add_subparsers(title="modules", help='ccsmeth modules, use -h/--help for help')
    sub_call_hifi = subparsers.add_parser("call_hifi", description="call hifi reads with kinetics from subreads.bam "
                                                                   "using CCS, save in bam/sam format.\n"
                                                                   "  cmd: ccsmeth call_hifi -i input.subreads.bam")
    sub_call_mods = subparsers.add_parser("call_mods", description="call modifications")
    sub_align_hifi = subparsers.add_parser("align_hifi", description="align hifi reads using pbmm2/minimap2/bwa, "
                                                                     "default pbmm2")
    sub_call_freqt = subparsers.add_parser("call_freqt",
                                           description="call frequency of modifications at genome level from "
                                                       "per_readsite text files")
    sub_call_freqb = subparsers.add_parser("call_freqb",
                                           description="call frequency of modifications at genome level from "
                                                       "modbam.bam file")
    sub_extract = subparsers.add_parser("extract", description="extract features from hifi reads.")
    sub_train = subparsers.add_parser("train", description="train a model, need two independent datasets for training "
                                                           "and validating")
    sub_trainm = subparsers.add_parser("trainm", description="[EXPERIMENTAL]train a model using multi gpus")
    # sub_denoise = subparsers.add_parser("denoise", description="[EXPERIMENTAL]denoise")

    # sub_call_hifi ============================================================================
    sch_input = sub_call_hifi.add_argument_group("INPUT")
    sch_input.add_argument("--subreads", "-i", type=str, required=True,
                           help="path to subreads.bam file as input")

    sch_output = sub_call_hifi.add_argument_group("OUTPUT")
    sch_output.add_argument("--output", "-o", type=str, required=False,
                            help="output file path for alignment results, bam/sam supported. "
                                 "If not specified, the results will be saved in "
                                 "input_file_prefix.hifi.bam by default.")

    sch_ccs = sub_call_hifi.add_argument_group("CCS ARG")
    sch_ccs.add_argument("--path_to_ccs", type=str, default=None, required=False,
                         help="full path to the executable binary ccs(PBCCS) file. "
                              "If not specified, it is assumed that ccs is "
                              "in the PATH.")
    sch_ccs.add_argument("--threads", "-t", type=int, default=None, required=False,
                         help="number of threads to call hifi reads, "
                              "default None -> means using all available processors")
    sch_ccs.add_argument("--min-passes", type=int, default=None, required=False,
                         help="CCS: Minimum number of full-length subreads required to generate CCS for a ZMW. "
                              "default None -> means using a default value set by CCS")
    sch_ccs.add_argument("--by-strand", action="store_true", default=False, required=False,
                         help="CCS: Generate a consensus for each strand.")
    sch_ccs.add_argument("--hd-finder", action="store_true", default=False, required=False,
                         help="CCS: Enable heteroduplex finder and splitting.")
    sch_ccs.add_argument("--log-level", action="store", default="WARN", required=False,
                         help="CCS: Set log level. "
                              "Valid choices: (TRACE, DEBUG, INFO, WARN, FATAL). [WARN]")

    sub_call_hifi.add_argument("--path_to_samtools", type=str, default=None, required=False,
                               help="full path to the executable binary samtools file. "
                               "If not specified, it is assumed that samtools is in "
                               "the PATH.")

    sub_call_hifi.set_defaults(func=main_call_hifi)

    # sub_align_hifi ============================================================================
    sah_input = sub_align_hifi.add_argument_group("INPUT")
    sah_input.add_argument("--hifireads", "-i", type=str, required=True,
                           help="path to hifireads.bam/sam/fastq_with_pulseinfo file as input")
    sah_input.add_argument("--ref", type=str, required=True,
                           help="path to genome reference to be aligned, in fasta/fa format. "
                                "If using bwa, the reference must have already been indexed.")

    sah_output = sub_align_hifi.add_argument_group("OUTPUT")
    sah_output.add_argument("--output", "-o", type=str, required=False,
                            help="output file path for alignment results, bam/sam supported. "
                                 "If not specified, the results will be saved in "
                                 "input_file_prefix.bam by default.")
    sah_output.add_argument("--header", action="store_true", default=False, required=False,
                            help="save header annotations from bam/sam. DEPRECATED")

    sah_align = sub_align_hifi.add_argument_group("ALIGN")
    sah_align.add_argument("--path_to_pbmm2", type=str, default=None, required=False,
                           help="full path to the executable binary pbmm2 file. "
                                "If not specified, it is assumed that pbmm2 is "
                                "in the PATH.")
    sah_align.add_argument("--minimap2", action="store_true", default=False, required=False,
                           help="use minimap2 instead of pbmm2 for alignment")
    sah_align.add_argument("--path_to_minimap2", type=str, default=None, required=False,
                           help="full path to the executable binary minimap2 file. "
                                "If not specified, it is assumed that minimap2 is "
                                "in the PATH.")
    sah_align.add_argument("--bestn", "-n", type=int, default=3, required=False,
                           help="retain at most n alignments in minimap2. "
                                "default 3, which means 2 secondary alignments are retained. "
                                "Do not use 2, cause -N1 is not suggested for high accuracy of alignment. "
                                "[This arg is for further extension.]")
    sah_align.add_argument("--bwa", action="store_true", default=False, required=False,
                           help="use bwa instead of pbmm2 for alignment")
    sah_align.add_argument("--path_to_bwa", type=str, default=None, required=False,
                           help="full path to the executable binary bwa file. If not "
                                "specified, it is assumed that bwa is in the PATH.")
    sah_align.add_argument("--path_to_samtools", type=str, default=None, required=False,
                           help="full path to the executable binary samtools file. "
                                "If not specified, it is assumed that samtools is in "
                                "the PATH.")
    sah_align.add_argument("--threads", "-t", type=int, default=5, required=False,
                           help="number of threads, default 5")

    sub_align_hifi.set_defaults(func=main_align_hifi)

    # sub_call_mods =============================================================================================
    scm_input = sub_call_mods.add_argument_group("INPUT")
    scm_input.add_argument("--input", "-i", action="store", type=str,
                           required=True,
                           help="input file, can be bam/sam, or features.tsv generated by "
                                "extract_features.py.")
    scm_input.add_argument("--holes_batch", type=int, default=50, required=False,
                           help="number of holes/hifi-reads in an batch to get/put in queues, default 50. "
                                "only used when --input is bam/sam")
    
    scm_output = sub_call_mods.add_argument_group("OUTPUT")
    scm_output.add_argument("--output", "-o", action="store", type=str, required=True,
                            help="the prefix of output files to save the predicted results. "
                                 "output files will be [--output].per_readsite.tsv/.modbam.bam")
    scm_output.add_argument("--gzip", action="store_true", default=False, required=False,
                            help="if compressing .per_readsite.tsv when --input is not in bam/sam format.")
    scm_output.add_argument("--keep_pulse", action="store_true", default=False, required=False,
                            help="if keeping ipd/pw tags in .modbam.bam when --input is in bam/sam format.")
    scm_output.add_argument("--no_sort", action="store_true", default=False, required=False,
                            help="don't sort .modbam.bam when --input is in bam/sam format.")

    scm_call = sub_call_mods.add_argument_group("CALL")
    scm_call.add_argument("--model_file", "-m", action="store", type=str, required=True,
                          help="file path of the trained model (.ckpt)")
    # model param
    scm_call.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s", "transencoder2s", 
                                   "attbilstm2s2", "attbigru2s2",],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "'transencoder2s', 'attbilstm2s2', 'attbigru2s2', "
                               "default: attbigru2s")
    scm_call.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    scm_call.add_argument('--is_npass', type=str, default="yes", required=False,
                          help="if using num_pass features, yes or no, default yes")
    scm_call.add_argument('--is_stds', type=str, default="no", required=False,
                          help="if using std features, yes or no, default no")
    
    scm_call.add_argument('--is_sn', type=str, default="no", required=False,
                          help="if using signal-to-noise-ratio features, yes or no, default no. "
                               "Effects both MODEL input and feature EXTRACTION")
    scm_call.add_argument('--is_map', type=str, default="no", required=False,
                          help="if using mapping features, yes or no, default no. "
                               "Effects both MODEL input and feature EXTRACTION, "
                               "only works in EXTRACTION-ALIGN-MODE")

    scm_call.add_argument('--class_num', type=int, default=2, required=False)
    scm_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    scm_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                          action="store", help="batch size, default 512")
    
    scm_callb = sub_call_mods.add_argument_group("CALL MODEL_HYPER RNN")
    # BiRNN model param
    scm_callb.add_argument('--layer_rnn', type=int, default=3,
                           required=False, help="BiRNN layer num, default 3")
    scm_callb.add_argument('--hid_rnn', type=int, default=256, required=False,
                           help="BiRNN hidden_size, default 256")
    
    scm_callt = sub_call_mods.add_argument_group("CALL MODEL_HYPER TRANSFORMER")
    # Transformer model param
    scm_callt.add_argument('--layer_trans', type=int, default=6, required=False,
                           help="TransformerEncoder nlayers, default 6")
    scm_callt.add_argument('--nhead', type=int, default=4, required=False,
                           help="TransformerEncoder nhead, default 4")
    scm_callt.add_argument('--d_model', type=int, default=256, required=False, 
                           help="TransformerEncoder input feature numbers, default 256")
    scm_callt.add_argument('--dim_ff', type=int, default=512, required=False,
                           help="TransformerEncoder dim_feedforward, default 512")

    scm_extract = sub_call_mods.add_argument_group("EXTRACTION")
    scm_extract.add_argument("--mode", type=str, default="denovo", required=False,
                             choices=["denovo", "align"],
                             help="denovo mode: extract features from unaligned/aligned hifi.bam without "
                                  "reference position info;\n"
                                  "align mode: extract features from aligned hifi.bam with "
                                  "reference position info. default: denovo")
    scm_extract.add_argument("--holeids_e", type=str, default=None, required=False,
                             help="file contains holeids to be extracted, default None")
    scm_extract.add_argument("--holeids_ne", type=str, default=None, required=False,
                             help="file contains holeids not to be extracted, default None")
    scm_extract.add_argument("--motifs", action="store", type=str,
                             required=False, default='CG',
                             help='motif seq to be extracted, default: CG. '
                                  'can be multi motifs splited by comma '
                                  '(no space allowed in the input str), '
                                  'or use IUPAC alphabet, '
                                  'the mod_loc of all motifs must be '
                                  'the same')
    scm_extract.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                             help='0-based location of the targeted base in the motif, default 0')
    scm_extract.add_argument("--methy_label", action="store", type=int,
                             choices=[1, 0], required=False, default=1,
                             help="the label of the interested modified bases, this is for training."
                                  " 0 or 1, default 1")
    scm_extract.add_argument("--norm", action="store", type=str, 
                             choices=["zscore", "min-mean", "min-max", "mad", "none"],
                             default="zscore", required=False,
                             help="method for normalizing ipd/pw in subread level. "
                                  "zscore, min-mean, min-max, mad, or none. default zscore")
    scm_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                             help="not use CodecV1 to decode ipd/pw")
    # scm_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
    #                          help="full path to the executable binary samtools file. "
    #                               "If not specified, it is assumed that samtools is in "
    #                               "the PATH.")

    scm_extract_ref = sub_call_mods.add_argument_group("EXTRACTION ALIGN_MODE")
    scm_extract_ref.add_argument("--ref", type=str, required=False,
                                 help="path to genome reference to be aligned, in fasta/fa format.")
    scm_extract_ref.add_argument("--mapq", type=int, default=1, required=False,
                                 help="MAPping Quality cutoff for selecting alignment items, default 1")
    scm_extract_ref.add_argument("--identity", type=float, default=0.0, required=False,
                                 help="identity cutoff for selecting alignment items, [0.0, 1.0], default 0.0")
    scm_extract_ref.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                                 help="not use supplementary alignment")
    scm_extract_ref.add_argument("--skip_unmapped", type=str, default="yes", required=False,
                                 help="if skipping unmapped sites in reads, yes or no, default yes")

    sub_call_mods.add_argument("--threads", "-p", action="store", type=int, default=10,
                               required=False, help="number of threads to be used, default 10.")
    sub_call_mods.add_argument("--threads_call", action="store", type=int, default=3,
                               required=False, help="number of threads used to call modifications with trained "
                                                    "models, no more than threads/3 is suggested. default 3.")
    sub_call_mods.add_argument('--tseed', type=int, default=1234,
                               help='random seed for torch')
    sub_call_mods.add_argument('--use_compile', type=str, default="no", required=False,
                               help="if using torch.compile, yes or no, default no ('yes' only works in pytorch>=2.0)")

    sub_call_mods.set_defaults(func=main_call_mods)

    # sub_extract ============================================================================
    se_input = sub_extract.add_argument_group("INPUT")
    se_input.add_argument("--input", "-i", type=str, required=True,
                          help="input file in bam/sam format, "
                               "can be unaligned hifi.bam/sam and aligned sorted hifi.bam/sam.")
    se_input.add_argument("--holeids_e", type=str, default=None, required=False,
                          help="file contains holeids/hifiids to be extracted, default None")
    se_input.add_argument("--holeids_ne", type=str, default=None, required=False,
                          help="file contains holeids/hifiids not to be extracted, default None")

    se_output = sub_extract.add_argument_group("OUTPUT")
    se_output.add_argument("--output", "-o", type=str, required=False,
                           help="output file path to save the extracted features. "
                                "If not specified, use input_prefix.tsv as default.")
    se_output.add_argument("--gzip", action="store_true", default=False, required=False,
                           help="if compressing the output using gzip")

    se_extract = sub_extract.add_argument_group("EXTRACTION")
    se_extract.add_argument("--mode", type=str, default="denovo", required=False,
                            choices=["denovo", "align"],
                            help="denovo mode: extract features from unaligned/aligned hifi.bam without "
                                 "reference position info;\n"
                                 "align mode: extract features from aligned hifi.bam with "
                                 "reference position info. default: denovo")
    se_extract.add_argument("--seq_len", type=int, default=21, required=False,
                            help="len of kmer. default 21")
    se_extract.add_argument("--motifs", action="store", type=str,
                            required=False, default='CG',
                            help='motif seq to be extracted, default: CG. '
                                 'can be multi motifs splited by comma '
                                 '(no space allowed in the input str), '
                                 'or use IUPAC alphabet, '
                                 'the mod_loc of all motifs must be '
                                 'the same')
    se_extract.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                            help='0-based location of the targeted base in the motif, default 0')
    se_extract.add_argument("--methy_label", action="store", type=int,
                            choices=[1, 0], required=False, default=1,
                            help="the label of the interested modified bases, this is for training."
                                 " 0 or 1, default 1")
    se_extract.add_argument("--norm", action="store", type=str, 
                            choices=["zscore", "min-mean", "min-max", "mad", "none"],
                            default="zscore", required=False,
                            help="method for normalizing ipd/pw in subread level. "
                                 "zscore, min-mean, min-max, mad, or none. default zscore")
    se_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                            help="not use CodecV1 to decode ipd/pw")
    # se_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
    #                         help="full path to the executable binary samtools file. "
    #                              "If not specified, it is assumed that samtools is in "
    #                              "the PATH.")
    se_extract.add_argument("--holes_batch", type=int, default=50, required=False,
                            help="number of holes/hifi-reads in an batch to get/put in queues, default 50")
    se_extract.add_argument("--is_sn", type=str, default="no", required=False,
                            help="if extracting signal-to-noise features, yes or no, default no")
    se_extract.add_argument("--is_map", type=str, default="no", required=False,
                            help="if extracting mapping features, yes or no, default no. only works in ALIGN-MODE")

    se_extract_ref = sub_extract.add_argument_group("EXTRACTION ALIGN_MODE")
    se_extract_ref.add_argument("--ref", type=str, required=False,
                                help="path to genome reference to be aligned, in fasta/fa format.")
    se_extract_ref.add_argument("--mapq", type=int, default=1, required=False,
                                help="MAPping Quality cutoff for selecting alignment items, default 1")
    se_extract_ref.add_argument("--identity", type=float, default=0.0, required=False,
                                help="identity cutoff for selecting alignment items, [0.0, 1.0], default 0.0")
    se_extract_ref.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                                help="not use supplementary alignment")
    se_extract_ref.add_argument("--skip_unmapped", type=str, default="yes", required=False,
                                help="if skipping unmapped sites in reads, yes or no, default yes")

    sub_extract.add_argument("--threads", type=int, default=5, required=False,
                             help="number of threads, default 5")

    sub_extract.set_defaults(func=main_extract)

    # sub_call_freq_txt =====================================================================================
    scf_input = sub_call_freqt.add_argument_group("INPUT")
    scf_input.add_argument('--input_path', '-i', action="append", type=str, required=True,
                           help='an output file from call_mods/call_modifications.py, or a directory contains '
                                'a bunch of output files. this arg is in "append" mode, can be used multiple times')
    scf_input.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                           help='a unique str which all input files has, this is for finding all input files '
                                'and ignoring the not-input-files in a input directory. if input_path is a file, '
                                'ignore this arg.')

    scf_output = sub_call_freqt.add_argument_group("OUTPUT")
    scf_output.add_argument('--result_file', '-o', action="store", type=str, required=True,
                            help='the file path to save the result')
    scf_output.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    scf_output.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    scf_output.add_argument("--gzip", action="store_true", default=False, required=False,
                            help="if compressing the output using gzip")

    scf_cal = sub_call_freqt.add_argument_group("CALL_FREQ")
    scf_cal.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                         help='this is to remove ambiguous calls. '
                              'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                              'means use all calls. range [0, 1], default 0.0.')
    scf_cal.add_argument('--rm_1strand', action='store_true', default=False,
                         help="abandon ccs reads with only 1 strand subreads [DEPRECATED]")
    scf_cal.add_argument('--refsites_only', action='store_true', default=False,
                         help="only keep sites which are target motifs in both reference and reads")
    scf_cal.add_argument("--motifs", action="store", type=str,
                         required=False, default='CG',
                         help='motif seq to be extracted, default: CG. '
                              'can be multi motifs splited by comma '
                              '(no space allowed in the input str), '
                              'or use IUPAC alphabet, '
                              'the mod_loc of all motifs must be '
                              'the same. [Only useful when --refsites_only is True]')
    scf_cal.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                         help='0-based location of the targeted base in the motif, default 0. '
                              '[Only useful when --refsites_only is True]')
    scf_cal.add_argument("--ref", type=str, required=False,
                         help="path to genome reference, in fasta/fa format. "
                              "[Only useful when --refsites_only is True]")

    scf_para = sub_call_freqt.add_argument_group("PARALLEL")
    scf_para.add_argument('--contigs', action="store", type=str, required=False, default=None,
                          help="a reference genome file (.fa/.fasta/.fna), used for extracting all "
                               "contig names for parallel; "
                               "or path of a file containing chromosome/contig names, one name each line; "
                               "or a string contains multiple chromosome names splited by comma."
                               "default None, which means all chromosomes will be processed at one time. "
                               "If not None, one chromosome will be processed by one subprocess.")
    scf_para.add_argument('--threads', action="store", type=int, required=False, default=1,
                          help="number of subprocesses used when --contigs is set. i.e., number of contigs processed "
                               "in parallel. default 1")

    sub_call_freqt.set_defaults(func=main_call_freqt)

    # sub_call_freq_bam =====================================================================================
    sub_call_freqb.add_argument('--threads', action="store", type=int, required=False, default=5,
                                help="number of subprocesses used. default 5")

    scfb_input = sub_call_freqb.add_argument_group("INPUT")
    scfb_input.add_argument('--input_bam', action="store", type=str, required=True,
                            help='input bam, should be aligned and sorted')
    scfb_input.add_argument("--ref", type=str, required=True,
                            help="path to genome reference, in fasta/fa format.")
    scfb_input.add_argument('--contigs', action="store", type=str, required=False, default=None,
                            help="path of a file containing chromosome/contig names, one name each line; "
                                 "or a string contains multiple chromosome names splited by comma."
                                 "default None, which means all chromosomes will be processed.")
    scfb_input.add_argument('--chunk_len', type=int, required=False, default=500000,
                            help="chunk length, default 500000")

    scfb_output = sub_call_freqb.add_argument_group("OUTPUT")
    scfb_output.add_argument('--output', '-o', action="store", type=str, required=True,
                             help='prefix of output file to save the results')
    scfb_output.add_argument('--bed', action='store_true', default=False,
                             help="save the result in bedMethyl format")
    scfb_output.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    scfb_output.add_argument("--gzip", action="store_true", default=False, required=False,
                             help="if compressing the output using gzip")

    scfb_callfreq = sub_call_freqb.add_argument_group("CALL_FREQ")
    scfb_callfreq.add_argument('--modtype', type=str, action="store", required=False, default="5mC",
                               choices=["5mC", ],
                               help='modification type, default 5mC.')
    scfb_callfreq.add_argument('--call_mode', type=str, action="store", required=False, default="count",
                               choices=["count", "aggregate"],
                               help='call mode: count, aggregate. default count.')
    scfb_callfreq.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                               help='this is to remove ambiguous calls (only for count-mode now). '
                                    'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                                    'means use all calls. range [0, 1], default 0.0.')
    scfb_callfreq.add_argument('--no_amb_cov', action="store_true", required=False, default=False,
                               help='when using prob_cf>0, DO NOT count ambiguous calls '
                                    'for calculating reads coverage')
    scfb_callfreq.add_argument("--hap_tag", type=str, action="store", required=False, default="HP",
                               help="haplotype tag, default HP")
    scfb_callfreq.add_argument("--mapq", type=int, default=1, required=False,
                               help="MAPping Quality cutoff for selecting alignment items, default 1")
    scfb_callfreq.add_argument("--identity", type=float, default=0.0, required=False,
                               help="identity cutoff for selecting alignment items, [0.0, 1.0], default 0.0")
    scfb_callfreq.add_argument("--no_supplementary", action="store_true", default=False, required=False,
                               help="not use supplementary alignment")
    scfb_callfreq.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    scfb_callfreq.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    scfb_callfreq.add_argument("--no_comb", action="store_true", default=False, required=False,
                               help="don't combine fwd/rev reads of one CG. [Only works when motifs is CG]")
    scfb_callfreq.add_argument('--refsites_only', action='store_true', default=False,
                               help="only keep sites which are target motifs in both reference and reads")
    scfb_callfreq.add_argument('--refsites_all', action='store_true', default=False,
                               help="output all covered sites which are target motifs in reference. "
                                    "--refsites_all is True, also means we do not output sites which "
                                    "are target motifs only in reads.")
    scfb_callfreq.add_argument("--no_hap", action="store_true", default=False, required=False,
                               help="don't call_freq on haplotypes ")
    scfb_callfreq.add_argument("--base_clip", action="store", type=int, required=False, default=0,
                               help='number of base clipped in each read, default 0')

    scfb_aggre = sub_call_freqb.add_argument_group("AGGREGATE_MODE")
    scfb_aggre.add_argument("--aggre_model", "-m", action="store", type=str, required=False,
                            help="file path of the aggregate model (.ckpt)")
    scfb_aggre.add_argument('--model_type', type=str, default="attbigru",
                            choices=["attbilstm", "attbigru"],
                            required=False,
                            help="type of model to use, 'attbigru', 'attbilstm', "
                                 "default: attbigru")
    scfb_aggre.add_argument('--seq_len', type=int, default=11, required=False,
                            help="len of sites used. default 11")
    scfb_aggre.add_argument('--class_num', type=int, default=1, required=False)
    scfb_aggre.add_argument('--layer_rnn', type=int, default=1,
                            required=False, help="BiRNN layer num, default 1")
    scfb_aggre.add_argument('--hid_rnn', type=int, default=32, required=False,
                            help="BiRNN hidden_size, default 32")
    scfb_aggre.add_argument('--bin_size', type=int, action="store", required=False, default=20,
                            help="histogram bin size, default 20")
    scfb_aggre.add_argument('--cov_cf', action="store", type=int, required=False,
                            default=4, help="coverage cutoff, to consider if use aggregate model to "
                                            "re-predict the modstate of the site")
    scfb_aggre.add_argument('--only_close', action="store_true", default=False, required=False,
                            help="[EXPERIMENTAL]")
    scfb_aggre.add_argument('--discrete', action="store_true", default=False, required=False,
                            help="[EXPERIMENTAL]")
    scfb_aggre.add_argument('--tseed', type=int, default=1234,
                            help='random seed for torch')

    sub_call_freqb.set_defaults(func=main_call_freqb)

    # sub_train =====================================================================================
    st_input = sub_train.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = sub_train.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=True)

    st_train = sub_train.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    st_train.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s", "transencoder2s", 
                                   "attbilstm2s2", "attbigru2s2",],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "'transencoder2s', 'attbilstm2s2', 'attbigru2s2', "
                               "default: attbigru2s")
    st_train.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    st_train.add_argument('--is_npass', type=str, default="yes", required=False,
                          help="if using num_pass features, yes or no, default yes")
    st_train.add_argument('--is_sn', type=str, default="no", required=False,
                          help="if using signal-to-noise-ratio features, yes or no, default no")
    st_train.add_argument('--is_map', type=str, default="no", required=False,
                          help="if using mapping features, yes or no, default no")
    st_train.add_argument('--is_stds', type=str, default="no", required=False,
                          help="if using std features, yes or no, default no")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    
    st_trainb = sub_train.add_argument_group("TRAIN MODEL_HYPER RNN")
    # BiRNN model param
    st_trainb.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    st_trainb.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size, default 256")
    
    st_traint = sub_train.add_argument_group("TRAIN MODEL_HYPER TRANSFORMER")
    # Transformer model param
    st_traint.add_argument('--layer_trans', type=int, default=6, required=False,
                          help="TransformerEncoder nlayers, default 6")
    st_traint.add_argument('--nhead', type=int, default=4, required=False,
                          help="TransformerEncoder nhead, default 4")
    st_traint.add_argument('--d_model', type=int, default=256, required=False, 
                          help="TransformerEncoder input feature numbers, default 256")
    st_traint.add_argument('--dim_ff', type=int, default=512, required=False,
                          help="TransformerEncoder dim_feedforward, default 512")

    st_training = sub_train.add_argument_group("TRAINING")
    # model training
    st_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                  "'Ranger' or 'LookaheadAdam', default Adam")
    st_training.add_argument('--batch_size', type=int, default=512, required=False)
    st_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                             choices=["StepLR", "ReduceLROnPlateau"],
                             help="StepLR or ReduceLROnPlateau, default StepLR")
    st_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001")
    st_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    st_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    st_training.add_argument('--lr_patience', type=int, default=0, required=False,
                             help="effective in ReduceLROnPlateau. default 0")
    st_training.add_argument('--lr_mode_strategy', type=str, default="last", required=False,
                             choices=["last", "mean", "max"],
                             help="effective in ReduceLROnPlateau. last, mean, or max, default last")
    st_training.add_argument("--max_epoch_num", action="store", default=50, type=int,
                             required=False, help="max epoch num, default 50")
    st_training.add_argument("--min_epoch_num", action="store", default=10, type=int,
                             required=False, help="min epoch num, default 10")
    st_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_training.add_argument('--step_interval', type=int, default=500, required=False)
    st_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")
    st_training.add_argument('--dl_offsets', action="store_true", default=False, required=False,
                             help="use file offsets loader")

    st_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    st_training.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')
    st_training.add_argument('--use_compile', type=str, default="no", required=False,
                             help="if using torch.compile, yes or no, default no ('yes' only works in pytorch>=2.0)")

    sub_train.set_defaults(func=main_train)

    # sub_train_multigpu =====================================================================================
    stm_input = sub_trainm.add_argument_group("INPUT")
    stm_input.add_argument('--train_file', type=str, required=True)
    stm_input.add_argument('--valid_file', type=str, required=True)

    stm_input = sub_trainm.add_argument_group("OUTPUT")
    stm_input.add_argument('--model_dir', type=str, required=True)

    stm_train = sub_trainm.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    stm_train.add_argument('--model_type', type=str, default="attbigru2s",
                           choices=["attbilstm2s", "attbigru2s", "transencoder2s", 
                                    "attbilstm2s2", "attbigru2s2",],
                           required=False,
                           help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                                "'transencoder2s', 'attbilstm2s2', 'attbigru2s2', "
                                "default: attbigru2s")
    stm_train.add_argument('--seq_len', type=int, default=21, required=False,
                           help="len of kmer. default 21")
    stm_train.add_argument('--is_npass', type=str, default="yes", required=False,
                           help="if using num_pass features, yes or no, default yes")
    stm_train.add_argument('--is_sn', type=str, default="no", required=False,
                           help="if using signal-to-noise-ratio features, yes or no, default no")
    stm_train.add_argument('--is_map', type=str, default="no", required=False,
                           help="if using mapping features, yes or no, default no")
    stm_train.add_argument('--is_stds', type=str, default="no", required=False,
                           help="if using std features, yes or no, default no")
    stm_train.add_argument('--class_num', type=int, default=2, required=False)
    stm_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    
    stm_trainb = sub_trainm.add_argument_group("TRAIN MODEL_HYPER RNN")
    # BiRNN model param
    stm_trainb.add_argument('--layer_rnn', type=int, default=3,
                           required=False, help="BiRNN layer num, default 3")
    stm_trainb.add_argument('--hid_rnn', type=int, default=256, required=False,
                           help="BiRNN hidden_size, default 256")
    
    stm_traint = sub_trainm.add_argument_group("TRAIN MODEL_HYPER TRANSFORMER")
    # Transformer model param
    stm_traint.add_argument('--layer_trans', type=int, default=6, required=False,
                           help="TransformerEncoder nlayers, default 6")
    stm_traint.add_argument('--nhead', type=int, default=4, required=False,
                           help="TransformerEncoder nhead, default 4")
    stm_traint.add_argument('--d_model', type=int, default=256, required=False, 
                           help="TransformerEncoder input feature numbers, default 256")
    stm_traint.add_argument('--dim_ff', type=int, default=512, required=False,
                           help="TransformerEncoder dim_feedforward, default 512")

    # model training
    stm_training = sub_trainm.add_argument_group("TRAINING")
    stm_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                 "Ranger", "LookaheadAdam"],
                              required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                   "'Ranger' or 'LookaheadAdam', default Adam")
    stm_training.add_argument('--batch_size', type=int, default=512, required=False)
    stm_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                              choices=["StepLR", "ReduceLROnPlateau"],
                              help="StepLR or ReduceLROnPlateau, default StepLR")
    stm_training.add_argument('--lr', type=float, default=0.001, required=False,
                              help="default 0.001. [lr should be lr*world_size when using multi gpus? "
                                   "or lower batch_size?]")
    stm_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                              help="default 0.1")
    stm_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                              help="effective in StepLR. default 1")
    stm_training.add_argument('--lr_patience', type=int, default=0, required=False,
                              help="effective in ReduceLROnPlateau. default 0")
    # stm_training.add_argument('--lr_mode_strategy', type=str, default="last", required=False,
    #                           choices=["last", "mean", "max"],
    #                           help="effective in ReduceLROnPlateau. last, mean, or max, default last")
    stm_training.add_argument("--max_epoch_num", action="store", default=50, type=int,
                              required=False, help="max epoch num, default 50")
    stm_training.add_argument("--min_epoch_num", action="store", default=10, type=int,
                              required=False, help="min epoch num, default 10")
    stm_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    stm_training.add_argument('--step_interval', type=int, default=500, required=False)
    stm_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                              help="default 0")

    stm_training.add_argument('--init_model', type=str, default=None, required=False,
                              help="file path of pre-trained model parameters to load before training")
    stm_training.add_argument('--tseed', type=int, default=1234,
                              help='random seed for pytorch')
    stm_training.add_argument('--use_compile', type=str, default="no", required=False,
                              help="if using torch.compile, yes or no, default no ('yes' only works in pytorch>=2.0)")

    stm_trainingp = sub_trainm.add_argument_group("TRAINING PARALLEL")
    stm_trainingp.add_argument("--nodes", default=1, type=int,
                               help="number of nodes for distributed training, default 1")
    stm_trainingp.add_argument("--ngpus_per_node", default=2, type=int,
                               help="number of GPUs per node for distributed training, default 2")
    stm_trainingp.add_argument("--dist-url", default="tcp://127.0.0.1:12315", type=str,
                               help="url used to set up distributed training")
    stm_trainingp.add_argument("--node_rank", default=0, type=int,
                               help="node rank for distributed training, default 0")
    stm_trainingp.add_argument("--epoch_sync", action="store_true", default=False,
                               help="if sync model params of gpu0 to other local gpus after per epoch")

    sub_trainm.set_defaults(func=main_trainm)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    sys.exit(main())
