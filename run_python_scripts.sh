#!/usr/bin/bash

# this script runs all the python scripts 
# in the current working directory

echo
echo running shell script : "${0}"

# put the path to the python interpreter 
path_to_python="/home/bawasthi/anaconda3/envs/ortools/bin/python"

python_scripts=$(ls *.py)
echo
count=1
for fls in $python_scripts
do
    echo "running script $count: $fls"
    "$path_to_python" "$fls"
    if [ "$?" -ne 0 ]
    then
        echo "script $fls didn't execute successfully"
        exit 1
    fi
    echo
    (( count++ ))
done
echo Done!
exit 0
