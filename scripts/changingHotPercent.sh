#!/bin/bash

kvcgExe=../cmake-build-relwithdebinfo/kvcg_simpl
soFile=../cmake-build-relwithdebinfo/libzipfianWorkload.so

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true,
  \"modelFile\" : \"./model.json\"
}"

echo $serverJson > server.json

theta="0.5"

workloadJson="{
  \"theta\" : $theta,
  \"range\" : 1000000000,
  \"n\" : 10000000,
  \"ops\" : 10000,
  \"keysize\" : 8,
  \"ratio\" : 95
}"

echo $workloadJson > workload.json

values=(0 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000)

for value in "${values[@]}"; do

  modelFileJson="{
    \"value\" : $value
  }"

  echo $modelFileJson > model.json

  $kvcgExe -l $soFile -f ./server.json -w ./workload.json > results_${value}.tsv

  sleep 5

done

rm server.json
rm workload.json
rm model.json
