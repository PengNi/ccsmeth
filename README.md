# ccsmeth

[![Python](https://img.shields.io/pypi/pyversions/ccsmeth)](https://www.python.org/)
[![GitHub-License](https://img.shields.io/github/license/PengNi/ccsmeth)](https://github.com/PengNi/ccsmeth/blob/master/LICENSE)

[![PyPI-version](https://img.shields.io/pypi/v/ccsmeth)](https://pypi.org/project/ccsmeth/)
[![PyPI-Downloads](https://pepy.tech/badge/ccsmeth)](https://pepy.tech/project/ccsmeth/)

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
       [pbccs](https://ccs.how/) (version>=6.3.0) \
       [pbmm2](https://github.com/PacificBiosciences/pbmm2) (version>=1.9.0) or [minimap2](https://github.com/lh3/minimap2) (version>=2.22-r1101) \
       [samtools](https://github.com/samtools/samtools) (version>=1.12)
   - Dependencies: \
       [numpy](http://www.numpy.org/) \
       [statsmodels](https://github.com/statsmodels/statsmodels/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.7.0?) \
       [tqdm](https://github.com/tqdm/tqdm) \
       [pysam](https://pysam.readthedocs.io/en/latest/installation.html) \
       [pybedtools](https://daler.github.io/pybedtools/) \
       [pytabix](https://github.com/slowkow/pytabix)

#### 1. install ccsmeth
```bash
# 1. it is highly recommended to install ccsmeth in an virtual environment
conda create -n ccsmethenv python=3.7
# activate
conda activate ccsmethenv
# deactivate this environment
conda deactivate

# 2. install ccsmeth
# install ccsmeth from github (latest version)
git clone https://github.com/PengNi/ccsmeth.git
cd ccsmeth
python setup.py install
# OR install ccsmeth using pip
pip install ccsmeth
```

#### 2. install necessary packages
Install necessary packages ([pbccs](https://ccs.how/), [pbmm2](https://github.com/PacificBiosciences/pbmm2) or [minimap2](https://github.com/lh3/minimap2), [samtools](https://github.com/samtools/samtools) in the same environment. Installing of those packages using [Bioconda](https://bioconda.github.io/) is recommended:
```shell
conda install pbccs pbmm2 samtools -c bioconda
```


## Trained models
See [models](https://github.com/PengNi/ccsmeth/tree/master/models):


## Quick start

```shell
# 1. call hifi reads if needed
# should have added pbccs to $PATH or the used environment
ccsmeth call_hifi --subreads /path/to/subreads.bam \
  --threads 10 \
  --output /path/to/output.hifi.bam

# 2. call modifications
# outputs: [--output].per_readsite.tsv, 
#          [--output].modbam.bam
CUDA_VISIBLE_DEVICES=0 ccsmeth call_mods \
  --input /path/to/output.hifi.bam \
  --model_file /path/to/ccsmeth/models/model.ckpt \
  --output /path/to/output.hifi.call_mods \
  --threads 10 --threads_call 2 --model_type attbigru2s

# 3. align hifi reads
# should have added pbmm2 to $PATH or the used environment
ccsmeth align_hifi \
  --hifireads /path/to/output.hifi.call_mods.modbam.bam \
  --ref /path/to/genome.fa \
  --output /path/to/output.hifi.call_mods.modbam.pbmm2.bam \
  --threads 10

# 4. call modification frequency
# outputs: [--output].count.all.bed
# if the input bam file contains haplotags, 
# there will be [--output].count.hp1/hp2.bed in outputs
ccsmeth call_freqb \
  --input_bam /path/to/output.hifi.call_mods.modbam.pbmm2.bam \
  --ref /path/to/genome.fa \
  --output /path/to/output.hifi.call_mods.modbam.pbmm2.freq \
  --threads 10 --sort --bed
```


## Usage

Users can use `ccsmeth subcommands --help/-h` for help.

#### 1. call hifi reads

```shell
ccsmeth call_hifi -h
usage: ccsmeth call_hifi [-h] --subreads SUBREADS [--output OUTPUT]
                         [--path_to_ccs PATH_TO_CCS] [--threads THREADS]
                         [--min-passes MIN_PASSES] [--by-strand] [--hd-finder]
                         [--path_to_samtools PATH_TO_SAMTOOLS]

call hifi reads from subreads.bam using CCS, save in bam/sam format cmd:
ccsmeth call_hifi -i input.subreads.bam

optional arguments:
  -h, --help            show this help message and exit
  --path_to_samtools PATH_TO_SAMTOOLS
                        full path to the executable binary samtools file. If
                        not specified, it is assumed that samtools is in the
                        PATH.

INPUT:
  --subreads SUBREADS, -i SUBREADS
                        path to subreads.bam file as input

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        output file path for alignment results, bam/sam
                        supported. If not specified, the results will be saved
                        in input_file_prefix.hifi.bam by default.

CCS ARG:
  --path_to_ccs PATH_TO_CCS
                        full path to the executable binary ccs(PBCCS) file. If
                        not specified, it is assumed that ccs is in the PATH.
  --threads THREADS, -t THREADS
                        number of threads to call hifi reads, default None ->
                        means using all available processors
  --min-passes MIN_PASSES
                        CCS: Minimum number of full-length subreads required
                        to generate CCS for a ZMW. default None -> means using
                        a default value set by CCS
  --by-strand           CCS: Generate a consensus for each strand.
  --hd-finder           CCS: Enable heteroduplex finder and splitting.
```

#### 2. call modifications

```shell
ccsmeth call_mods -h
usage: ccsmeth call_mods [-h] --input INPUT [--holes_batch HOLES_BATCH]
                         --model_file MODEL_FILE
                         [--model_type {attbilstm2s,attbigru2s}]
                         [--seq_len SEQ_LEN] [--is_npass IS_NPASS]
                         [--is_qual IS_QUAL] [--is_map IS_MAP]
                         [--is_stds IS_STDS] [--class_num CLASS_NUM]
                         [--dropout_rate DROPOUT_RATE]
                         [--batch_size BATCH_SIZE] [--n_vocab N_VOCAB]
                         [--n_embed N_EMBED] [--layer_rnn LAYER_RNN]
                         [--hid_rnn HID_RNN] --output OUTPUT [--gzip]
                         [--modbam MODBAM] [--mode {denovo,reference}]
                         [--holeids_e HOLEIDS_E] [--holeids_ne HOLEIDS_NE]
                         [--motifs MOTIFS] [--mod_loc MOD_LOC]
                         [--methy_label {1,0}]
                         [--norm {zscore,min-mean,min-max,mad}] [--no_decode]
                         [--loginfo LOGINFO] [--ref REF] [--mapq MAPQ]
                         [--identity IDENTITY] [--no_supplementary]
                         [--is_mapfea IS_MAPFEA]
                         [--skip_unmapped SKIP_UNMAPPED] [--threads THREADS]
                         [--threads_call THREADS_CALL] [--tseed TSEED]

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
                        input file, can be bam/sam, or features.tsv generated
                        by extract_features.py.
  --holes_batch HOLES_BATCH
                        number of holes/hifi-reads in an batch to get/put in
                        queues, default 50. only used when --input is bam/sam

CALL:
  --model_file MODEL_FILE, -m MODEL_FILE
                        file path of the trained model (.ckpt)
  --model_type {attbilstm2s,attbigru2s}
                        type of model to use, 'attbilstm2s', 'attbigru2s',
                        default: attbigru2s
  --seq_len SEQ_LEN     len of kmer. default 21
  --is_npass IS_NPASS   if using num_pass features, yes or no, default yes
  --is_qual IS_QUAL     if using base_quality features, yes or no, default no
  --is_map IS_MAP       if using mapping features, yes or no, default no
  --is_stds IS_STDS     if using std features, yes or no, default no
  --class_num CLASS_NUM
  --dropout_rate DROPOUT_RATE
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size, default 512
  --n_vocab N_VOCAB     base_seq vocab_size (15 base kinds from iupac)
  --n_embed N_EMBED     base_seq embedding_size
  --layer_rnn LAYER_RNN
                        BiRNN layer num, default 3
  --hid_rnn HID_RNN     BiRNN hidden_size for combined feature

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        the prefix of output files to save the predicted
                        results. output files will be
                        [--output].per_readsite.tsv/.modbam.bam
  --gzip                if compressing .per_readsite.tsv using gzip
  --modbam MODBAM       if generating modbam file when --input is in bam/sam
                        format. yes or no, default yes

EXTRACTION:
  --mode {denovo,reference}
                        denovo mode: extract features from unaligned hifi.bam;
                        reference mode: extract features from aligned
                        hifi.bam. default: denovo
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
  --norm {zscore,min-mean,min-max,mad}
                        method for normalizing ipd/pw in subread level.
                        zscore, min-mean, min-max or mad, default zscore
  --no_decode           not use CodecV1 to decode ipd/pw
  --loginfo LOGINFO     if printing more info of feature extraction on reads.
                        yes or no, default no

EXTRACTION REFERENCE_MODE:
  --ref REF             path to genome reference to be aligned, in fasta/fa
                        format.
  --mapq MAPQ           MAPping Quality cutoff for selecting alignment items,
                        default 10
  --identity IDENTITY   identity cutoff for selecting alignment items, default
                        0.8
  --no_supplementary    not use supplementary alignment
  --is_mapfea IS_MAPFEA
                        if extract mapping features, yes or no, default no
  --skip_unmapped SKIP_UNMAPPED
                        if skipping unmapped sites in reads, yes or no,
                        default yes
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

#### 3. align hifi reads

```shell
ccsmeth align_hifi -h
usage: ccsmeth align_hifi [-h] --hifireads HIFIREADS --ref REF
                          [--output OUTPUT] [--header]
                          [--path_to_pbmm2 PATH_TO_PBMM2] [--minimap2]
                          [--path_to_minimap2 PATH_TO_MINIMAP2]
                          [--bestn BESTN] [--bwa] [--path_to_bwa PATH_TO_BWA]
                          [--path_to_samtools PATH_TO_SAMTOOLS]
                          [--threads THREADS]

align hifi reads using pbmm2/minimap2/bwa, default pbmm2

optional arguments:
  -h, --help            show this help message and exit

INPUT:
  --hifireads HIFIREADS, -i HIFIREADS
                        path to hifireads.bam/sam/fastq_with_pulseinfo file as
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
  --path_to_pbmm2 PATH_TO_PBMM2
                        full path to the executable binary pbmm2 file. If not
                        specified, it is assumed that pbmm2 is in the PATH.
  --minimap2            use minimap2 instead of pbmm2 for alignment
  --path_to_minimap2 PATH_TO_MINIMAP2
                        full path to the executable binary minimap2 file. If
                        not specified, it is assumed that minimap2 is in the
                        PATH.
  --bestn BESTN, -n BESTN
                        retain at most n alignments in minimap2. default 3,
                        which means 2 secondary alignments are retained. Do
                        not use 2, cause -N1 is not suggested for high
                        accuracy of alignment. [This arg is for further
                        extension, for now it is no use cause we use only
                        primary alignment.]
  --bwa                 use bwa instead of pbmm2 for alignment
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

#### 4. call modification frequency from modbam file

```shell
ccsmeth call_freqb -h
usage: ccsmeth call_freqb [-h] [--threads THREADS] --input_bam INPUT_BAM --ref
                          REF [--contigs CONTIGS] [--chunk_len CHUNK_LEN]
                          --output OUTPUT [--bed] [--sort] [--gzip]
                          [--modtype {5mC}] [--call_mode {count,aggregate}]
                          [--prob_cf PROB_CF] [--hap_tag HAP_TAG]
                          [--mapq MAPQ] [--identity IDENTITY]
                          [--no_supplementary] [--motifs MOTIFS]
                          [--mod_loc MOD_LOC] [--no_comb] [--refsites_only]

call frequency of modifications at genome level from modbam.bam file

optional arguments:
  -h, --help            show this help message and exit
  --threads THREADS     number of subprocesses used. default 5

INPUT:
  --input_bam INPUT_BAM
                        input bam, should be aligned and sorted
  --ref REF             path to genome reference, in fasta/fa format.
  --contigs CONTIGS     path of a file containing chromosome/contig names, one
                        name each line; or a string contains multiple
                        chromosome names splited by comma.default None, which
                        means all chromosomes will be processed.
  --chunk_len CHUNK_LEN
                        chunk length, default 500000

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        prefix of output file to save the results
  --bed                 save the result in bedMethyl format
  --sort                sort items in the result
  --gzip                if compressing the output using gzip

CALL_FREQ:
  --modtype {5mC}       modification type, default 5mC.
  --call_mode {count,aggregate}
                        call mode: count, aggregate. default count.
  --prob_cf PROB_CF     this is to remove ambiguous calls. if
                        abs(prob1-prob0)>=prob_cf, then we use the call. e.g.,
                        proc_cf=0 means use all calls. range [0, 1], default
                        0.0.
  --hap_tag HAP_TAG     haplotype tag, default HP
  --mapq MAPQ           MAPping Quality cutoff for selecting alignment items,
                        default 10
  --identity IDENTITY   identity cutoff for selecting alignment items, default
                        0.8
  --no_supplementary    not use supplementary alignment
  --motifs MOTIFS       motif seq to be extracted, default: CG. can be multi
                        motifs splited by comma (no space allowed in the input
                        str), or use IUPAC alphabet, the mod_loc of all motifs
                        must be the same
  --mod_loc MOD_LOC     0-based location of the targeted base in the motif,
                        default 0
  --no_comb             dont combine fwd/rev reads of one CG. [Only works when
                        motifs is CG]
  --refsites_only       only keep sites which is a target motif in reference
```

The modification_frequency file can be either saved in [bedMethyl](https://www.encodeproject.org/data-standards/wgbs/) format (by setting `--bed`), or saved as a tab-delimited text file in the following format by default:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **pos_end**:   pos + 1
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **prob_0_sum**:    sum of the probabilities of the targeted base predicted as 0 (unmethylated)
   - **prob_1_sum**:    sum of the probabilities of the targeted base predicted as 1 (methylated)
   - **count_modified**:    number of reads in which the targeted base counted as modified
   - **count_unmodified**:  number of reads in which the targeted base counted as unmodified
   - **coverage**:  number of reads aligned to the targeted base
   - **modification_frequency**:    modification frequency
   - **k_mer**:   the kmer around the targeted base

#### 5. call modification frequency from per_readsite file

```shell
ccsmeth call_freqt -h
usage: ccsmeth call_freqt [-h] --input_path INPUT_PATH [--file_uid FILE_UID]
                          --result_file RESULT_FILE [--bed] [--sort] [--gzip]
                          [--prob_cf PROB_CF] [--rm_1strand] [--refsites_only]
                          [--motifs MOTIFS] [--mod_loc MOD_LOC] [--ref REF]
                          [--contigs CONTIGS] [--threads THREADS]

call frequency of modifications at genome level from per_readsite text files

optional arguments:
  -h, --help            show this help message and exit

INPUT:
  --input_path INPUT_PATH, -i INPUT_PATH
                        an output file from call_mods/call_modifications.py,
                        or a directory contains a bunch of output files. this
                        arg is in "append" mode, can be used multiple times
  --file_uid FILE_UID   a unique str which all input files has, this is for
                        finding all input files and ignoring the not-input-
                        files in a input directory. if input_path is a file,
                        ignore this arg.

OUTPUT:
  --result_file RESULT_FILE, -o RESULT_FILE
                        the file path to save the result
  --bed                 save the result in bedMethyl format
  --sort                sort items in the result
  --gzip                if compressing the output using gzip

CALL_FREQ:
  --prob_cf PROB_CF     this is to remove ambiguous calls. if
                        abs(prob1-prob0)>=prob_cf, then we use the call. e.g.,
                        proc_cf=0 means use all calls. range [0, 1], default
                        0.0.
  --rm_1strand          abandon ccs reads with only 1 strand subreads
                        [DEPRECATED]
  --refsites_only       only keep sites which is a target motif in reference
  --motifs MOTIFS       motif seq to be extracted, default: CG. can be multi
                        motifs splited by comma (no space allowed in the input
                        str), or use IUPAC alphabet, the mod_loc of all motifs
                        must be the same. [Only useful when --refsites_only is
                        True]
  --mod_loc MOD_LOC     0-based location of the targeted base in the motif,
                        default 0. [Only useful when --refsites_only is True]
  --ref REF             path to genome reference, in fasta/fa format. [Only
                        useful when --refsites_only is True]

PARALLEL:
  --contigs CONTIGS     a reference genome file (.fa/.fasta/.fna), used for
                        extracting all contig names for parallel; or path of a
                        file containing chromosome/contig names, one name each
                        line; or a string contains multiple chromosome names
                        splited by comma.default None, which means all
                        chromosomes will be processed at one time. If not
                        None, one chromosome will be processed by one
                        subprocess.
  --threads THREADS     number of subprocesses used when --contigs is set.
                        i.e., number of contigs processed in parallel. default
                        1
```
The format of the output file is the same as of `ccsmeth call_freqb`.

#### 6. extract features

```shell
ccsmeth extract -h
usage: ccsmeth extract [-h] [--threads THREADS] [--loginfo LOGINFO] --input
                       INPUT [--holeids_e HOLEIDS_E] [--holeids_ne HOLEIDS_NE]
                       [--output OUTPUT] [--gzip] [--mode {denovo,reference}]
                       [--seq_len SEQ_LEN] [--motifs MOTIFS]
                       [--mod_loc MOD_LOC] [--methy_label {1,0}]
                       [--norm {zscore,min-mean,min-max,mad}] [--no_decode]
                       [--holes_batch HOLES_BATCH] [--ref REF] [--mapq MAPQ]
                       [--identity IDENTITY] [--no_supplementary]
                       [--is_mapfea IS_MAPFEA] [--skip_unmapped SKIP_UNMAPPED]

extract features from hifi reads.

optional arguments:
  -h, --help            show this help message and exit
  --threads THREADS     number of threads, default 5
  --loginfo LOGINFO     if printing more info of feature extraction on reads.
                        yes or no, default no

INPUT:
  --input INPUT, -i INPUT
                        input file in bam/sam format, can be unaligned
                        hifi.bam/sam and aligned sorted hifi.bam/sam.
  --holeids_e HOLEIDS_E
                        file contains holeids/hifiids to be extracted, default
                        None
  --holeids_ne HOLEIDS_NE
                        file contains holeids/hifiids not to be extracted,
                        default None

OUTPUT:
  --output OUTPUT, -o OUTPUT
                        output file path to save the extracted features. If
                        not specified, use input_prefix.tsv as default.
  --gzip                if compressing the output using gzip

EXTRACTION:
  --mode {denovo,reference}
                        denovo mode: extract features from unaligned hifi.bam;
                        reference mode: extract features from aligned
                        hifi.bam. default: denovo
  --seq_len SEQ_LEN     len of kmer. default 21
  --motifs MOTIFS       motif seq to be extracted, default: CG. can be multi
                        motifs splited by comma (no space allowed in the input
                        str), or use IUPAC alphabet, the mod_loc of all motifs
                        must be the same
  --mod_loc MOD_LOC     0-based location of the targeted base in the motif,
                        default 0
  --methy_label {1,0}   the label of the interested modified bases, this is
                        for training. 0 or 1, default 1
  --norm {zscore,min-mean,min-max,mad}
                        method for normalizing ipd/pw in subread level.
                        zscore, min-mean, min-max or mad, default zscore
  --no_decode           not use CodecV1 to decode ipd/pw
  --holes_batch HOLES_BATCH
                        number of holes/hifi-reads in an batch to get/put in
                        queues, default 50

EXTRACTION REFERENCE_MODE:
  --ref REF             path to genome reference to be aligned, in fasta/fa
                        format.
  --mapq MAPQ           MAPping Quality cutoff for selecting alignment items,
                        default 10
  --identity IDENTITY   identity cutoff for selecting alignment items, default
                        0.8
  --no_supplementary    not use supplementary alignment
  --is_mapfea IS_MAPFEA
                        if extract mapping features, yes or no, default no
  --skip_unmapped SKIP_UNMAPPED
                        if skipping unmapped sites in reads, yes or no,
                        default yes
```

#### 7. train a new model

```shell
ccsmeth train -h
usage: ccsmeth train [-h] --train_file TRAIN_FILE --valid_file VALID_FILE
                     --model_dir MODEL_DIR
                     [--model_type {attbilstm2s,attbigru2s}]
                     [--seq_len SEQ_LEN] [--is_npass IS_NPASS]
                     [--is_qual IS_QUAL] [--is_map IS_MAP] [--is_stds IS_STDS]
                     [--class_num CLASS_NUM] [--dropout_rate DROPOUT_RATE]
                     [--n_vocab N_VOCAB] [--n_embed N_EMBED]
                     [--layer_rnn LAYER_RNN] [--hid_rnn HID_RNN]
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
  --model_type {attbilstm2s,attbigru2s}
                        type of model to use, 'attbilstm2s', 'attbigru2s',
                        default: attbigru2s
  --seq_len SEQ_LEN     len of kmer. default 21
  --is_npass IS_NPASS   if using num_pass features, yes or no, default yes
  --is_qual IS_QUAL     if using base_quality features, yes or no, default no
  --is_map IS_MAP       if using mapping features, yes or no, default no
  --is_stds IS_STDS     if using std features, yes or no, default no
  --class_num CLASS_NUM
  --dropout_rate DROPOUT_RATE
  --n_vocab N_VOCAB     base_seq vocab_size (15 base kinds from iupac)
  --n_embed N_EMBED     base_seq embedding_size
  --layer_rnn LAYER_RNN
                        BiRNN layer num, default 3
  --hid_rnn HID_RNN     BiRNN hidden_size for combined feature
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
