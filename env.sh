#!/usr/bin/env bash

export RNN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"

export PYTHONPATH=$RNN/libs/blocks:$RNN/libs/blocks-extras:$RNN/libs/Theano:$RNN/libs/picklable-itertools:$PYTHONPATH
export PATH=$RNN/bin:$RNN/libs/blocks/bin:$RNN/libs/blocks-extras/bin:$RNN/libs/fuel/bin:$PATH
