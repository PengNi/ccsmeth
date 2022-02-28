#!/usr/bin/python
from __future__ import absolute_import

import sys
import argparse

from .utils.process_utils import str2bool
from .utils.process_utils import display_args

from ._version import VERSION


def main_align(args):
    from .align_subreads import align_subreads_to_genome

    display_args(args, True)
    align_subreads_to_genome(args)


def main_call_mods(args):
    from .call_modifications import call_mods

    display_args(args)
    call_mods(args)


def main_extract(args):
    from .extract_features import extract_subreads_features

    display_args(args, True)
    extract_subreads_features(args)


def main_train(args):
    from .train import train
    import time

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser(prog='ccsmeth',
                                     description="detecting methylation from PacBio CCS reads, "
                                                 "ccsmeth contains four modules:\n"
                                                 "\t%(prog)s align: align subreads to reference\n"
                                                 "\t%(prog)s call_mods: call modifications\n"
                                                 "\t%(prog)s extract: extract features from corrected (tombo) "
                                                 "fast5s for training or testing\n"
                                                 "\t%(prog)s train: train a model, need two independent "
                                                 "datasets for training and validating",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-v', '--version', action='version',
        version='ccsmeth version: {}'.format(VERSION),
        help='show ccsmeth version and exit.')

    subparsers = parser.add_subparsers(title="modules", help='ccsmeth modules, use -h/--help for help')
    sub_align = subparsers.add_parser("align", description="align using bwa/minimap2")
    sub_call_mods = subparsers.add_parser("call_mods", description="call modifications")
    sub_extract = subparsers.add_parser("extract", description="extract features from aligned subreads.")
    sub_train = subparsers.add_parser("train", description="train a model, need two independent datasets for training "
                                                           "and validating")

    # sub_align ============================================================================
    sa_input = sub_align.add_argument_group("INPUT")
    sa_input.add_argument("--subreads", "-i", type=str, required=True,
                          help="path to subreads.bam/sam/fastq_with_pulseinfo file as input")
    sa_input.add_argument("--ref", type=str, required=True,
                          help="path to genome reference to be aligned, in fasta/fa format. "
                               "If using bwa, the reference must have already been indexed.")

    sa_output = sub_align.add_argument_group("OUTPUT")
    sa_output.add_argument("--output", "-o", type=str, required=False,
                           help="output file path for alignment results, bam/sam supported. "
                                "If not specified, the results will be saved in "
                                "input_file_prefix.bam by default.")
    sa_output.add_argument("--header", action="store_true", default=False, required=False,
                           help="save header annotations from bam/sam. DEPRECATED")

    sa_align = sub_align.add_argument_group("ALIGN")
    sa_align.add_argument("--bestn", "-n", type=int, default=3, required=False,
                          help="retain at most n alignments in minimap2. "
                               "default 3, which means 2 secondary alignments are retained. "
                               "Do not use 2, cause -N1 is not suggested for high accuracy of alignment. "
                               "[This arg is for further extension, for now it is no use cause "
                               "we use only primary alignment.]")
    sa_align.add_argument("--bwa", action="store_true", default=False, required=False,
                          help="use bwa instead of minimap2 for alignment")
    sa_align.add_argument("--path_to_minimap2", type=str, default=None, required=False,
                          help="full path to the executable binary minimap2 file. "
                               "If not specified, it is assumed that minimap2 is "
                               "in the PATH.")
    sa_align.add_argument("--path_to_bwa", type=str, default=None, required=False,
                          help="full path to the executable binary bwa file. If not "
                               "specified, it is assumed that bwa is in the PATH.")
    sa_align.add_argument("--path_to_samtools", type=str, default=None, required=False,
                          help="full path to the executable binary samtools file. "
                               "If not specified, it is assumed that samtools is in "
                               "the PATH.")
    sa_align.add_argument("--threads", "-t", type=int, default=5, required=False,
                          help="number of threads, default 5")

    sub_align.set_defaults(func=main_align)

    # sub_call_mods =============================================================================================
    sc_input = sub_call_mods.add_argument_group("INPUT")
    sc_input.add_argument("--input", "-i", action="store", type=str,
                          required=True,
                          help="input file, can be aligned.bam/sam, or features.tsv generated by "
                               "extract_features.py. If aligned.bam/sam is provided, args in EXTRACTION "
                               "should (reference_path must) be provided.")
    sc_input.add_argument("--holes_batch", type=int, default=50, required=False,
                          help="number of holes in an batch to get/put in queues")

    sc_call = sub_call_mods.add_argument_group("CALL")
    sc_call.add_argument("--model_file", "-m", action="store", type=str, required=True,
                         help="file path of the trained model (.ckpt)")

    # model param
    sc_call.add_argument('--model_type', type=str, default="attbigru2s",
                         choices=["attbilstm", "attbigru", "bilstm", "bigru",
                                  "transencoder",
                                  "resnet18",
                                  "attbigru2s"],
                         required=False,
                         help="type of model to use, 'attbilstm', 'attbigru', "
                              "'bilstm', 'bigru', 'transencoder', 'resnet18', "
                              "'attbigru2s', "
                              "default: attbigru2s")
    sc_call.add_argument('--seq_len', type=int, default=21, required=False,
                         help="len of kmer. default 21")
    sc_call.add_argument('--is_stds', type=str, default="yes", required=False,
                         help="if using std features at ccs level, yes or no. default yes.")
    sc_call.add_argument('--class_num', type=int, default=2, required=False)
    sc_call.add_argument('--dropout_rate', type=float, default=0, required=False)

    sc_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                         action="store", help="batch size, default 512")

    # BiRNN/transformerencoder model param
    sc_call.add_argument('--n_vocab', type=int, default=16, required=False,
                         help="base_seq vocab_size (15 base kinds from iupac)")
    sc_call.add_argument('--n_embed', type=int, default=4, required=False,
                         help="base_seq embedding_size")

    # BiRNN model param
    sc_call.add_argument('--layer_rnn', type=int, default=3,
                         required=False, help="BiRNN layer num, default 3")
    sc_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                         help="BiRNN hidden_size for combined feature")

    # transformerencoder model param
    sc_call.add_argument('--layer_tfe', type=int, default=6,
                         required=False, help="transformer encoder layer num, default 6")
    sc_call.add_argument('--d_model_tfe', type=int, default=256,
                         required=False, help="the number of expected features in the "
                                              "transformer encoder/decoder inputs")
    sc_call.add_argument('--nhead_tfe', type=int, default=4,
                         required=False, help="the number of heads in the multiheadattention models")
    sc_call.add_argument('--nhid_tfe', type=int, default=512,
                         required=False, help="the dimension of the feedforward network model")

    sc_output = sub_call_mods.add_argument_group("OUTPUT")
    sc_output.add_argument("--output", "-o", action="store", type=str, required=True,
                           help="the file path to save the predicted result")

    sc_extract = sub_call_mods.add_argument_group("EXTRACTION")
    sc_extract.add_argument("--ref", type=str, required=False,
                            help="path to genome reference to be aligned, in fasta/fa format.")
    sc_extract.add_argument("--holeids_e", type=str, default=None, required=False,
                            help="file contains holeids to be extracted, default None")
    sc_extract.add_argument("--holeids_ne", type=str, default=None, required=False,
                            help="file contains holeids not to be extracted, default None")
    sc_extract.add_argument("--motifs", action="store", type=str,
                            required=False, default='CG',
                            help='motif seq to be extracted, default: CG. '
                                 'can be multi motifs splited by comma '
                                 '(no space allowed in the input str), '
                                 'or use IUPAC alphabet, '
                                 'the mod_loc of all motifs must be '
                                 'the same')
    sc_extract.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                            help='0-based location of the targeted base in the motif, default 0')
    sc_extract.add_argument("--methy_label", action="store", type=int,
                            choices=[1, 0], required=False, default=1,
                            help="the label of the interested modified bases, this is for training."
                                 " 0 or 1, default 1")
    sc_extract.add_argument("--mapq", type=int, default=20, required=False,
                            help="MAPping Quality cutoff for selecting alignment items, default 20")
    sc_extract.add_argument("--identity", type=float, default=0.8, required=False,
                            help="identity cutoff for selecting alignment items, default 0.8")
    sc_extract.add_argument("--two_strands", action="store_true", default=False, required=False,
                            help="after quality (mapq, identity) control, if then only using CCS reads "
                                 "which have subreads in two strands")
    sc_extract.add_argument("--comb_strands", action="store_true", default=False, required=False,
                            help="if combining features in two(+/-) strands of one site")
    sc_extract.add_argument("--depth", type=int, default=1, required=False,
                            help="(mean) depth (number of subreads) cutoff for "
                                 "selecting high-quality aligned reads/kmers "
                                 "per strand of a CCS, default 1.")
    sc_extract.add_argument("--norm", action="store", type=str, choices=["zscore", "min-mean", "min-max", "mad"],
                            default="zscore", required=False,
                            help="method for normalizing ipd/pw in subread level. "
                                 "zscore, min-mean, min-max or mad, default zscore")
    sc_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                            help="not use CodecV1 to decode ipd/pw")
    sc_extract.add_argument("--num_subreads", type=int, default=0, required=False,
                            help="info of max num of subreads to be extracted to output, default 0")
    sc_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
                            help="full path to the executable binary samtools file. "
                                 "If not specified, it is assumed that samtools is in "
                                 "the PATH.")
    sc_extract.add_argument("--seed", type=int, default=1234, required=False,
                            help="seed for randomly selecting subreads, default 1234")

    sub_call_mods.add_argument("--threads", "-p", action="store", type=int, default=10,
                               required=False, help="number of threads to be used, default 10.")
    sub_call_mods.add_argument("--threads_call", action="store", type=int, default=2,
                               required=False, help="number of threads used to call with trained models, "
                                                    "no more than threads/4 is suggested. default 2.")
    sub_call_mods.add_argument('--tseed', type=int, default=1234,
                               help='random seed for torch')

    sub_call_mods.set_defaults(func=main_call_mods)

    # sub_extract ============================================================================
    se_input = sub_extract.add_argument_group("INPUT")
    se_input.add_argument("--input", "-i", type=str, required=True,
                          help="alignment results in bam/sam format. "
                               "We assume that all items/reads are sorted by hole_ids "
                               "in aligned.bam, which generated by align_subreads.py from subreads.bam.")
    se_input.add_argument("--ref", type=str, required=True,
                          help="path to genome reference to be aligned, in fasta/fa format.")
    se_input.add_argument("--holeids_e", type=str, default=None, required=False,
                          help="file contains holeids to be extracted, default None")
    se_input.add_argument("--holeids_ne", type=str, default=None, required=False,
                          help="file contains holeids not to be extracted, default None")

    se_output = sub_extract.add_argument_group("OUTPUT")
    se_output.add_argument("--output", "-o", type=str, required=False,
                           help="output file path to save the extracted features. "
                                "If not specified, use input_prefix.tsv as default.")

    se_extract = sub_extract.add_argument_group("EXTRACT")
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
    se_extract.add_argument("--mapq", type=int, default=20, required=False,
                            help="MAPping Quality cutoff for selecting alignment items, default 20")
    se_extract.add_argument("--identity", type=float, default=0.8, required=False,
                            help="identity cutoff for selecting alignment items, default 0.8")
    se_extract.add_argument("--two_strands", action="store_true", default=False, required=False,
                            help="after quality (mapq, identity) control, if then only using CCS reads "
                                 "which have subreads in two strands")
    se_extract.add_argument("--comb_strands", action="store_true", default=False, required=False,
                            help="if combining features in two(+/-) strands of one site")
    se_extract.add_argument("--depth", type=int, default=1, required=False,
                            help="(mean) depth (number of subreads) cutoff for "
                                 "selecting high-quality aligned reads/kmers "
                                 "per strand of a CCS, default 1.")
    se_extract.add_argument("--norm", action="store", type=str, choices=["zscore", "min-mean", "min-max", "mad"],
                            default="zscore", required=False,
                            help="method for normalizing ipd/pw in subread level. "
                                 "zscore, min-mean, min-max or mad, default zscore")
    se_extract.add_argument("--no_decode", action="store_true", default=False, required=False,
                            help="not use CodecV1 to decode ipd/pw")
    se_extract.add_argument("--num_subreads", type=int, default=0, required=False,
                            help="info of max num of subreads to be extracted to output, default 0")
    se_extract.add_argument("--path_to_samtools", type=str, default=None, required=False,
                            help="full path to the executable binary samtools file. "
                                 "If not specified, it is assumed that samtools is in "
                                 "the PATH.")
    se_extract.add_argument("--holes_batch", type=int, default=50, required=False,
                            help="number of holes in an batch to get/put in queues")
    se_extract.add_argument("--seed", type=int, default=1234, required=False,
                            help="seed for randomly selecting subreads, default 1234")

    sub_extract.add_argument("--threads", type=int, default=5, required=False,
                             help="number of threads, default 5")

    sub_extract.set_defaults(func=main_extract)

    # sub_train =====================================================================================
    st_input = sub_train.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = sub_train.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=True)

    st_train = sub_train.add_argument_group("TRAIN")
    # model param
    st_train.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm", "attbigru", "bilstm", "bigru",
                                   "transencoder",
                                   "resnet18",
                                   "attbigru2s"],
                          required=False,
                          help="type of model to use, 'attbilstm', 'attbigru', "
                               "'bilstm', 'bigru', 'transencoder', 'resnet18', "
                               "'attbigru2s', "
                               "default: attbigru2s")
    st_train.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    st_train.add_argument('--is_stds', type=str, default="yes", required=False,
                          help="if using std features at ccs level, yes or no. default yes.")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiRNN/transformerencoder model param
    st_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    st_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")

    # BiRNN model param
    st_train.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    st_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size for combined feature")

    # transformerencoder model param
    st_train.add_argument('--layer_tfe', type=int, default=6,
                          required=False, help="transformer encoder layer num, default 6")
    st_train.add_argument('--d_model_tfe', type=int, default=256,
                          required=False, help="the number of expected features in the "
                                               "transformer encoder/decoder inputs")
    st_train.add_argument('--nhead_tfe', type=int, default=4,
                          required=False, help="the number of heads in the multiheadattention models")
    st_train.add_argument('--nhid_tfe', type=int, default=512,
                          required=False, help="the dimension of the feedforward network model")

    # model training
    st_train.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                             "Ranger"],
                          required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop' or 'Ranger', "
                                               "default Adam")
    st_train.add_argument('--batch_size', type=int, default=512, required=False)
    st_train.add_argument('--lr', type=float, default=0.001, required=False)
    st_train.add_argument('--lr_decay', type=float, default=0.1, required=False)
    st_train.add_argument('--lr_decay_step', type=int, default=1, required=False)
    st_train.add_argument("--max_epoch_num", action="store", default=50, type=int,
                          required=False, help="max epoch num, default 50")
    st_train.add_argument("--min_epoch_num", action="store", default=10, type=int,
                          required=False, help="min epoch num, default 10")
    st_train.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_train.add_argument('--tseed', type=int, default=1234,
                          help='random seed for pytorch')
    st_train.add_argument('--step_interval', type=int, default=500, required=False)

    st_train.add_argument('--init_model', type=str, default=None, required=False,
                          help="file path of pre-trained model parameters to load before training")

    sub_train.set_defaults(func=main_train)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    sys.exit(main())
