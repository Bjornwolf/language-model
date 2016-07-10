#!/bin/bash
touch comparer
echo "2x128\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/2x128.yaml 2x128 >> results_comparer
echo "2x256\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/2x256.yaml 2x256 >> results_comparer
echo "2x512\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/2x512.yaml 2x512 >> results_comparer
echo "3x128\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/3x128.yaml 3x128 >> results_comparer
echo "3x256\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/3x256.yaml 3x256 >> results_comparer
echo "3x512\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/3x512.yaml 3x512 >> results_comparer
echo "4x128\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/4x128.yaml 4x128 >> results_comparer
echo "4x256\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/4x256.yaml 4x256 >> results_comparer
echo "4x512\n" >> results_comparer
python comparer.py ../dev_phones/n100/true ~/lattices_for_fch/complete_test configs/mgr/4x512.yaml 4x512 >> results_comparer
