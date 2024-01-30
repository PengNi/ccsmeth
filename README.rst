ccsmeth
========


Changelog
-------------
v0.5.0
----------
bug fixes

change feature QUAL to SNratio, didn't work though?

hid '--n_vocab' and '--n_embed' in model params

add attbigru2s2/attbilstm2s2 models

update new version of pretrained model params


v0.4.1
----------
handle ValueError when fetching bam items in call_freqb module

change pytorch max version to 1.12.1

using logging instead of print/sys.stderr.write


v0.4.0
----------
optimieze call_mods module, faster generating of modbam file


v0.3.4
----------
replace 'numpy.float' by 'float" in extract_features and call_mods modules


v0.3.3
----------
more robust operation for MM/ML tags

update the call_mods model (v1 to v2), use shared params for both rnn and attention

update aggregate model (v2 to v2p)


v0.3.2
----------
fix bug (0-pos CG in reverse strand) of call_freqb module

add output check at the start of call_mods/call_freqb

update aggregate model


v0.3.1
----------
optimize call_mods module

add aggregate mode in call_freqb module

release a stable call_mods model and a stable call_freqb aggregate model


v0.3.0
----------
add multi-threads support for reading .bam by pysam

force mod prob discrete integer in [0, 255], which means mod prob in [0, 1), in ML tag

fix align_hifi module, enable minimap2 and bwa

update requirements


v0.2.3
----------
more options in train module

multi-gpu support in call_mods and trainm modules

add denoise module

Note: skip 0.2.2, cause 0.2.2 has been deleted in pypi (https://pypi.org/manage/project/ccsmeth/history/)


v0.2.1
----------
minor fixes

change default mode in extract module

improvements in call_freqb module


v0.2.0
----------
use hifi reads instead of subreads as input

modbam support

use (chrom, pos, strand) as key instead of (chrom, pos) to handle CG mismatch in CCS read when call_freq


v0.1.2
----------
add pbmm2 as default aligner

enable gzip output


v0.1.1
----------
bug fixes


v0.1.0
----------
initialize