# ccsmeth
### Detecting DNA methylation from PacBio CCS reads

## Contents
- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation
ccsmeth is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/).
   - Prerequisites: \
       [Python3.*](https://www.python.org/) (version>=3.6)\
       [minimap2](https://github.com/lh3/minimap2) \
       [samtools](https://github.com/samtools/samtools)
   - Dependencies: \
       [numpy](http://www.numpy.org/) \
       [statsmodels](https://github.com/statsmodels/statsmodels/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.7.0?)

#### install ccsmeth from github (latest version):
```bash
# it is highly recommended to install ccsmeth in an virtual environment
conda create -n ccsmethenv python=3.7
# activate
conda activate ccsmethenv
# download and install ccsmethy from github
git clone https://github.com/PengNi/ccsmeth.git
cd ccsmeth
python setup.py install

# deactivate this environment
conda deactivate
```

## Trained models
See [models](https://github.com/PengNi/ccsmeth/tree/master/models):
   * _model_cpg_attbigru2s_hg002_15kb_s2.b21_epoch7.ckpt_: a CpG model trained using HG002 PacBio Sequel II (kit 2.0) CCS subreads.


## Quick start

```shell
# 1. align subreads (should have added minimap2 to $PATH)
ccsmeth align --subreads /path/to/subreads.bam \
  --ref /path/to/genome.fa \
  --threads 10 \
  --output /path/to/output.subreads.minimap2.bam

# 2. extract features
ccsmeth extract --input /path/to/output.subreads.minimap2.bam \
  --ref /path/to/genome.fa \
  --threads 10 --norm zscore --comb_strands --depth 1 \
  --output /path/to/output.subreads.minimap2.features.zscore.fb.depth1.tsv

# 3. call modifications
CUDA_VISIBLE_DEVICES=0 csmeth call_mods \
  --input /path/to/output.subreads.minimap2.features.zscore.fb.depth1.tsv \
  --model_file /path/to/ccsmeth/models/model_cpg_attbigru2s_hg002_15kb_s2.b21_epoch7.ckpt \
  --output /path/to/output.subreads.minimap2.features.zscore.fb.depth1.call_mods.tsv \
  --threads 10 --threads_call 2 --model_type attbigru2s
```


## Usage

Users can use `ccsmeth subcommands --help/-h` for help.

<!-- TODO: output file format explanation -->

#### 1. align subreads

```shell
ccsmeth align -h
usage: ccsmeth align [-h] --subreads SUBREADS --ref REF [--output OUTPUT]
                     [--header] [--bestn BESTN] [--bwa]
                     [--path_to_minimap2 PATH_TO_MINIMAP2]
                     [--path_to_bwa PATH_TO_BWA]
                     [--path_to_samtools PATH_TO_SAMTOOLS] [--threads THREADS]

align subreads using bwa/minimap2

optional arguments:
  -h, --help            show this help message and exit

INPUT:
  --subreads SUBREADS, -i SUBREADS
                        path to subreads.bam/sam/fastq_with_pulseinfo file as
                        input
  --ref REF             path to genome reference to be aligned, in fasta/fa
                        format. If using bwa, the reference must have already
                        been indexed.

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        output file path for alignment results, bam/sam
                        supported. If not specified, the results will be saved
                        in input_file_prefix.bam by default.
  --header              save header annotations from bam/sam. DEPRECATED

ALIGN:
  --bestn BESTN, -n BESTN
                        retain at most n alignments in minimap2. default 3,
                        which means 2 secondary alignments are retained. Do
                        not use 2, cause -N1 is not suggested for high
                        accuracy of alignment. [This arg is for further
                        extension, for now it is no use cause we use only
                        primary alignment.]
  --bwa                 use bwa instead of minimap2 for alignment
  --path_to_minimap2 PATH_TO_MINIMAP2
                        full path to the executable binary minimap2 file. If
                        not specified, it is assumed that minimap2 is in the
                        PATH.
  --path_to_bwa PATH_TO_BWA
                        full path to the executable binary bwa file. If not
                        specified, it is assumed that bwa is in the PATH.
  --path_to_samtools PATH_TO_SAMTOOLS
                        full path to the executable binary samtools file. If
                        not specified, it is assumed that samtools is in the
                        PATH.
  --threads THREADS, -t THREADS
                        number of threads, default 5
```

#### 2. extract features

```shell
ccsmeth extract -h
usage: ccsmeth extract [-h] --input INPUT --ref REF [--holeids_e HOLEIDS_E]
                       [--holeids_ne HOLEIDS_NE] [--output OUTPUT]
                       [--seq_len SEQ_LEN] [--motifs MOTIFS]
                       [--mod_loc MOD_LOC] [--methy_label {1,0}] [--mapq MAPQ]
                       [--identity IDENTITY] [--two_strands] [--comb_strands]
                       [--depth DEPTH] [--norm {zscore,min-mean,min-max,mad}]
                       [--no_decode] [--num_subreads NUM_SUBREADS]
                       [--path_to_samtools PATH_TO_SAMTOOLS]
                       [--holes_batch HOLES_BATCH] [--seed SEED]
                       [--threads THREADS]

extract features from aligned subreads.

optional arguments:
  -h, --help            show this help message and exit
  --threads THREADS     number of threads, default 5

INPUT:
  --input INPUT, -i INPUT
                        alignment results in bam/sam format. We assume that
                        all items/reads are sorted by hole_ids in aligned.bam,
                        which generated by align_subreads.py from
                        subreads.bam.
  --ref REF             path to genome reference to be aligned, in fasta/fa
                        format.
  --holeids_e HOLEIDS_E
                        file contains holeids to be extracted, default None
  --holeids_ne HOLEIDS_NE
                        file contains holeids not to be extracted, default
                        None

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        output file path to save the extracted features. If
                        not specified, use input_prefix.tsv as default.

EXTRACT:
  --seq_len SEQ_LEN     len of kmer. default 21
  --motifs MOTIFS       motif seq to be extracted, default: CG. can be multi
                        motifs splited by comma (no space allowed in the input
                        str), or use IUPAC alphabet, the mod_loc of all motifs
                        must be the same
  --mod_loc MOD_LOC     0-based location of the targeted base in the motif,
                        default 0
  --methy_label {1,0}   the label of the interested modified bases, this is
                        for training. 0 or 1, default 1
  --mapq MAPQ           MAPping Quality cutoff for selecting alignment items,
                        default 20
  --identity IDENTITY   identity cutoff for selecting alignment items, default
                        0.8
  --two_strands         after quality (mapq, identity) control, if then only
                        using CCS reads which have subreads in two strands
  --comb_strands        if combining features in two(+/-) strands of one site
  --depth DEPTH         (mean) depth (number of subreads) cutoff for selecting
                        high-quality aligned reads/kmers per strand of a CCS,
                        default 1.
  --norm {zscore,min-mean,min-max,mad}
                        method for normalizing ipd/pw in subread level.
                        zscore, min-mean, min-max or mad, default zscore
  --no_decode           not use CodecV1 to decode ipd/pw
  --num_subreads NUM_SUBREADS
                        info of max num of subreads to be extracted to output,
                        default 0
  --path_to_samtools PATH_TO_SAMTOOLS
                        full path to the executable binary samtools file. If
                        not specified, it is assumed that samtools is in the
                        PATH.
  --holes_batch HOLES_BATCH
                        number of holes in an batch to get/put in queues
  --seed SEED           seed for randomly selecting subreads, default 1234
```

#### 3. call modifications

```shell
ccsmeth call_mods -h
usage: ccsmeth call_mods [-h] --input INPUT [--holes_batch HOLES_BATCH]
                         --model_file MODEL_FILE
                         [--model_type {attbilstm,attbigru,bilstm,bigru,transencoder,resnet18,attbigru2s}]
                         [--seq_len SEQ_LEN] [--is_stds IS_STDS]
                         [--class_num CLASS_NUM] [--dropout_rate DROPOUT_RATE]
                         [--batch_size BATCH_SIZE] [--n_vocab N_VOCAB]
                         [--n_embed N_EMBED] [--layer_rnn LAYER_RNN]
                         [--hid_rnn HID_RNN] [--layer_tfe LAYER_TFE]
                         [--d_model_tfe D_MODEL_TFE] [--nhead_tfe NHEAD_TFE]
                         [--nhid_tfe NHID_TFE] --output OUTPUT [--ref REF]
                         [--holeids_e HOLEIDS_E] [--holeids_ne HOLEIDS_NE]
                         [--motifs MOTIFS] [--mod_loc MOD_LOC]
                         [--methy_label {1,0}] [--mapq MAPQ]
                         [--identity IDENTITY] [--two_strands]
                         [--comb_strands] [--depth DEPTH]
                         [--norm {zscore,min-mean,min-max,mad}] [--no_decode]
                         [--num_subreads NUM_SUBREADS]
                         [--path_to_samtools PATH_TO_SAMTOOLS] [--seed SEED]
                         [--threads THREADS] [--threads_call THREADS_CALL]
                         [--tseed TSEED]

call modifications

optional arguments:
  -h, --help            show this help message and exit
  --threads THREADS, -p THREADS
                        number of threads to be used, default 10.
  --threads_call THREADS_CALL
                        number of threads used to call with trained models, no
                        more than threads/4 is suggested. default 2.
  --tseed TSEED         random seed for torch

INPUT:
  --input INPUT, -i INPUT
                        input file, can be aligned.bam/sam, or features.tsv
                        generated by extract_features.py. If aligned.bam/sam
                        is provided, args in EXTRACTION should (reference_path
                        must) be provided.
  --holes_batch HOLES_BATCH
                        number of holes in an batch to get/put in queues

CALL:
  --model_file MODEL_FILE, -m MODEL_FILE
                        file path of the trained model (.ckpt)
  --model_type {attbilstm,attbigru,bilstm,bigru,transencoder,resnet18,attbigru2s}
                        type of model to use, 'attbilstm', 'attbigru',
                        'bilstm', 'bigru', 'transencoder', 'resnet18',
                        'attbigru2s', default: attbigru2s
  --seq_len SEQ_LEN     len of kmer. default 21
  --is_stds IS_STDS     if using std features at ccs level, yes or no. default
                        yes.
  --class_num CLASS_NUM
  --dropout_rate DROPOUT_RATE
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size, default 512
  --n_vocab N_VOCAB     base_seq vocab_size (15 base kinds from iupac)
  --n_embed N_EMBED     base_seq embedding_size
  --layer_rnn LAYER_RNN
                        BiRNN layer num, default 3
  --hid_rnn HID_RNN     BiRNN hidden_size for combined feature
  --layer_tfe LAYER_TFE
                        transformer encoder layer num, default 6
  --d_model_tfe D_MODEL_TFE
                        the number of expected features in the transformer
                        encoder/decoder inputs
  --nhead_tfe NHEAD_TFE
                        the number of heads in the multiheadattention models
  --nhid_tfe NHID_TFE   the dimension of the feedforward network model

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        the file path to save the predicted result

EXTRACTION:
  --ref REF             path to genome reference to be aligned, in fasta/fa
                        format.
  --holeids_e HOLEIDS_E
                        file contains holeids to be extracted, default None
  --holeids_ne HOLEIDS_NE
                        file contains holeids not to be extracted, default
                        None
  --motifs MOTIFS       motif seq to be extracted, default: CG. can be multi
                        motifs splited by comma (no space allowed in the input
                        str), or use IUPAC alphabet, the mod_loc of all motifs
                        must be the same
  --mod_loc MOD_LOC     0-based location of the targeted base in the motif,
                        default 0
  --methy_label {1,0}   the label of the interested modified bases, this is
                        for training. 0 or 1, default 1
  --mapq MAPQ           MAPping Quality cutoff for selecting alignment items,
                        default 20
  --identity IDENTITY   identity cutoff for selecting alignment items, default
                        0.8
  --two_strands         after quality (mapq, identity) control, if then only
                        using CCS reads which have subreads in two strands
  --comb_strands        if combining features in two(+/-) strands of one site
  --depth DEPTH         (mean) depth (number of subreads) cutoff for selecting
                        high-quality aligned reads/kmers per strand of a CCS,
                        default 1.
  --norm {zscore,min-mean,min-max,mad}
                        method for normalizing ipd/pw in subread level.
                        zscore, min-mean, min-max or mad, default zscore
  --no_decode           not use CodecV1 to decode ipd/pw
  --num_subreads NUM_SUBREADS
                        info of max num of subreads to be extracted to output,
                        default 0
  --path_to_samtools PATH_TO_SAMTOOLS
                        full path to the executable binary samtools file. If
                        not specified, it is assumed that samtools is in the
                        PATH.
  --seed SEED           seed for randomly selecting subreads, default 1234
```

The call_mods file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **readname**:  read name of the ccs read
   - **read_depth**:   subreads depth of the ccs read
   - **prob_0**:    [0, 1], the probability of the targeted base predicted as 0 (unmethylated)
   - **prob_1**:    [0, 1], the probability of the targeted base predicted as 1 (methylated)
   - **called_label**:  0/1, unmethylated/methylated
   - **k_mer**:   the kmer around the targeted base


#### 4. call modification frequency

```shell
python /path/to/ccsmeth/scripts/call_modification_frequency.py -h
usage: call_modification_frequency.py [-h] --input_path INPUT_PATH
                                      --result_file RESULT_FILE [--bed]
                                      [--sort] [--prob_cf PROB_CF]
                                      [--rm_1strand] [--file_uid FILE_UID]

calculate frequency of interested sites at genome level

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH, -i INPUT_PATH
                        a result file from call_modifications.py, or a
                        directory contains a bunch of result files.
  --result_file RESULT_FILE, -o RESULT_FILE
                        the file path to save the result
  --bed                 save the result in bedMethyl format
  --sort                sort items in the result
  --prob_cf PROB_CF     this is to remove ambiguous calls. if
                        abs(prob1-prob0)>=prob_cf, then we use the call. e.g.,
                        proc_cf=0 means use all calls. range [0, 1], default
                        0.0.
  --rm_1strand          abandon ccs reads with only 1 strand subreads
  --file_uid FILE_UID   a unique str which all input files has, this is for
                        finding all input files and ignoring the un-input-
                        files in a input directory. if input_path is a file,
                        ignore this arg.
```

The modification_frequency file can be either saved in [bedMethyl](https://www.encodeproject.org/data-standards/wgbs/) format (by setting `--bed`), or saved as a tab-delimited text file in the following format by default:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **prob_0_sum**:    sum of the probabilities of the targeted base predicted as 0 (unmethylated)
   - **prob_1_sum**:    sum of the probabilities of the targeted base predicted as 1 (methylated)
   - **count_modified**:    number of reads in which the targeted base counted as modified
   - **count_unmodified**:  number of reads in which the targeted base counted as unmodified
   - **coverage**:  number of reads aligned to the targeted base
   - **modification_frequency**:    modification frequency
   - **k_mer**:   the kmer around the targeted base

#### 5. train models

```shell
ccsmeth train -h
usage: ccsmeth train [-h] --train_file TRAIN_FILE --valid_file VALID_FILE
                     --model_dir MODEL_DIR
                     [--model_type {attbilstm,attbigru,bilstm,bigru,transencoder,resnet18,attbigru2s}]
                     [--seq_len SEQ_LEN] [--is_stds IS_STDS]
                     [--class_num CLASS_NUM] [--dropout_rate DROPOUT_RATE]
                     [--n_vocab N_VOCAB] [--n_embed N_EMBED]
                     [--layer_rnn LAYER_RNN] [--hid_rnn HID_RNN]
                     [--layer_tfe LAYER_TFE] [--d_model_tfe D_MODEL_TFE]
                     [--nhead_tfe NHEAD_TFE] [--nhid_tfe NHID_TFE]
                     [--optim_type {Adam,RMSprop,SGD,Ranger}]
                     [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY]
                     [--lr_decay_step LR_DECAY_STEP]
                     [--max_epoch_num MAX_EPOCH_NUM]
                     [--min_epoch_num MIN_EPOCH_NUM] [--pos_weight POS_WEIGHT]
                     [--tseed TSEED] [--step_interval STEP_INTERVAL]
                     [--init_model INIT_MODEL]

train a model, need two independent datasets for training and validating

optional arguments:
  -h, --help            show this help message and exit

INPUT:
  --train_file TRAIN_FILE
  --valid_file VALID_FILE

OUTPUT:
  --model_dir MODEL_DIR

TRAIN:
  --model_type {attbilstm,attbigru,bilstm,bigru,transencoder,resnet18,attbigru2s}
                        type of model to use, 'attbilstm', 'attbigru',
                        'bilstm', 'bigru', 'transencoder', 'resnet18',
                        'attbigru2s', default: attbigru2s
  --seq_len SEQ_LEN     len of kmer. default 21
  --is_stds IS_STDS     if using std features at ccs level, yes or no. default
                        yes.
  --class_num CLASS_NUM
  --dropout_rate DROPOUT_RATE
  --n_vocab N_VOCAB     base_seq vocab_size (15 base kinds from iupac)
  --n_embed N_EMBED     base_seq embedding_size
  --layer_rnn LAYER_RNN
                        BiRNN layer num, default 3
  --hid_rnn HID_RNN     BiRNN hidden_size for combined feature
  --layer_tfe LAYER_TFE
                        transformer encoder layer num, default 6
  --d_model_tfe D_MODEL_TFE
                        the number of expected features in the transformer
                        encoder/decoder inputs
  --nhead_tfe NHEAD_TFE
                        the number of heads in the multiheadattention models
  --nhid_tfe NHID_TFE   the dimension of the feedforward network model
  --optim_type {Adam,RMSprop,SGD,Ranger}
                        type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop'
                        or 'Ranger', default Adam
  --batch_size BATCH_SIZE
  --lr LR
  --lr_decay LR_DECAY
  --lr_decay_step LR_DECAY_STEP
  --max_epoch_num MAX_EPOCH_NUM
                        max epoch num, default 50
  --min_epoch_num MIN_EPOCH_NUM
                        min epoch num, default 10
  --pos_weight POS_WEIGHT
  --tseed TSEED         random seed for pytorch
  --step_interval STEP_INTERVAL
  --init_model INIT_MODEL
                        file path of pre-trained model parameters to load
                        before training
```


## Acknowledgements
- We thank Tse *et al.*, The Chinese University of Hong Kong (CUHK) Department of Chemical Pathology, for sharing their code and data, as reported in [Proc Natl Acad Sci USA 2021; 118(5): e2019768118](https://doi.org/10.1073/pnas.2019768118). We made use of their data and code for evaluation and comparison.
- We thank Akbari _et al._, as part of the code for haplotyping were taken from [NanoMethPhase](https://github.com/vahidAK/NanoMethPhase) of Akbari _et al._


License
=======
Copyright (C) 2021 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn), 
School of Computer Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
