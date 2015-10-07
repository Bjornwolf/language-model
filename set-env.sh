#!/bin/bash

export PATH=/pio/os/anaconda/bin:$PATH
export OPENSSL_CONF=/home/bjornwolf/anaconda/ssl/openssl.cnf

export RNN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

export PYTHONPATH=$RNN/libs/Theano:$RNN/libs/blocks:$RNN/libs/blocks-extras:$RNN/libs/picklable-itertools:$RNN/libs/xmldict:$RNN/libs/fuel:$RNN/libs/dill/dill:/pio/lscratch/1/os/anaconda/lib/python2.7/site-packages:$RNN/libs/progressbar/progressbar:$PYTHONPATH
export PATH=$RNN/bin:$RNN/libs/blocks/bin:$RNN/libs/blocks-extras/bin:$RNN/libs/fuel/bin:$PATH
