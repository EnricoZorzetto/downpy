#!/bin/bash
echo 'Running conus analysis'
FIRST=$(sbatch -o con1.out -e con1.err --wait --parsable conus1.q)
sleep 3s
echo $FIRST
n=$(cat "numberofjobs.txt")
echo "Number of jobs = $n"
nm=$((n-1))
sleep 3s
SECOND=$(sbatch -o con2.out -e con2.err --array 0-$nm --dependency=afterany:$FIRST --parsable conus2.q)
echo $SECOND
THIRD=$(sbatch  -o con3.out -e con3.err --dependency=afterany:$SECOND --parsable conus3.q)
echo $THIRD
exit 0

