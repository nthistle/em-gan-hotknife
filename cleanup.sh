#!/bin/bash
echo WARNING! This script will delete all run results and logs in the current directory.
read -p "Are you sure you want to continue? [y/N] " -n 1 -r confirm
echo
if [[ $confirm =~ ^[Yy]$ ]]
then
    echo Deleting training results folders...
    rm -rf train-*
    echo Deleting logs...
    rm -rf *.log
    echo Done!
fi
