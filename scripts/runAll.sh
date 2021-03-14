#!/bin/bash

mkdir hotRes
bash ./changingHotPercent.sh
mv *.tsv hotRes
tar czvf hotRes.tgz hotRes

mkdir thetaRes
bash ./changingTheta.sh
mv *.tsv thetaRes
tar czvf thetaRes.tgz thetaRes

mkdir valueRes
bash ./changingValueSize.sh
mv *.tsv valueRes
tar czvf valueRes.tgz valueRes

mkdir valueRes99
bash ./changingValueSize99.sh
mv *.tsv valueRes99
tar czvf valueRes99.tgz valueRes99

mkdir modelRes
bash ./changingWorkload.sh
mv *.tsv modelRes
tar czvf modelRes.tgz modelRes

mkdir results
mv *.tgz results
zip -r results.zip results