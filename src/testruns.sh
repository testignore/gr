#!/bin/bash

make clean
for i in `seq 1 10`; do
	echo "###"
	echo "### TEST RUN ${i} of 10"
	echo "###"
	touch testrun_${i}.started
	(make -j 10 check 2>&1 | tee -a testrun_${i}.log) && touch testrun_${i}.ok
done

