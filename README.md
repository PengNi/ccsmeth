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
       [minimap2](https://github.com/lh3/minimap2)
   - Dependencies: \
       [numpy](http://www.numpy.org/) \
       [statsmodels](https://github.com/statsmodels/statsmodels/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.7.0?)

#### install from github (latest version)
```bash
git clone https://github.com/PengNi/ccsmeth.git
cd ccsmeth
python setup.py install
```

## Trained models
See [models](https://github.com/PengNi/ccsmeth/tree/master/models):
   * _model_cpg_attbigru2s_hg002_15kb_s2.b21_epoch7.ckpt_: a CpG model trained using HG002 PacBio Sequel II (kit 2.0) CCS subreads.


## Quick start


## Usage


## Acknowledgements
- We thank Tse *et al.*, The Chinese University of Hong Kong (CUHK) Department of Chemical Pathology, for sharing their code and data, as reported in [Proc Natl Acad Sci USA 2021; 118(5): e2019768118](https://doi.org/10.1073/pnas.2019768118). We made use of their data and code for evaluation and comparison.
- We thank Akbari _et al._, as part of the code for haplotyping were taken from [NanoMethPhase](https://github.com/vahidAK/NanoMethPhase) of Akbari _et al._


License
=========
Copyright (C) 2021 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn), 
School of Computer Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
