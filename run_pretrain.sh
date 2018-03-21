#!/bin/sh
python3 -u pretrain.py -df /home/thistlethwaiten/data/hotknifedata.hdf5 -ne 100 -glr 1e-4 -o pretrain-1
python3 -u pretrain.py -df /home/thistlethwaiten/data/hotknifedata.hdf5 -ne 100 -glr 3e-5 -o pretrain-2
python3 -u pretrain.py -df /home/thistlethwaiten/data/hotknifedata.hdf5 -ne 100 -glr 1e-5 -o pretrain-3
