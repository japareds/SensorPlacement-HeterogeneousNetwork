#!/bin/bash
n_refst_range=(0 1 2 3 4)
n_empty_range=(7 8 9 10 11 12 13)

for n_empty in ${n_empty_range[@]}
do
	for n_refst in ${n_refst_range[@]}
	do
		python global_minimum_search.py --estimate -nr ${n_refst} -ne ${n_empty}
	done
done
