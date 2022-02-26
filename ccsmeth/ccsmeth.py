#!/usr/bin/python
from __future__ import absolute_import

import sys
import argparse

from .utils.process_utils import str2bool
from .utils.process_utils import display_args

from ._version import VERSION


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


if __name__ == '__main__':
    sys.exit(main())
