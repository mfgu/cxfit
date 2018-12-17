#!/bin/bash

dd0=$PWD
dd=`printf 'data/%s%02dLSJ' $1 $2`
if [ "x$3" == "x" ]; then
    fb=basis_000
    fn=`printf 'data/%s%02db.LS' $1 $2`
else
    fb=$3
    fn=`printf 'data/%s%02di.LS' $1 $2`
fi
jj2lsj=${HOME}/bin/jj2lsj
echo "$dd $fb $fn"
cd $dd
echo $fb > jj2lsj.in
echo 'Y' >> jj2lsj.in
echo 'Y' >> jj2lsj.in
$jj2lsj < jj2lsj.in > jj2lsj.out 2>&1
cp ${fb}.lsj.lbl $dd0/$fn

