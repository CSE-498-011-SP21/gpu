#!/bin/bash

kvcgExe=../cmake-build-relwithdebinfo/kvcg
soFile=../cmake-build-relwithdebinfo/libzipfianWorkload.so

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"


echo $serverJson > server.json

thetas=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

for theta in "${thetas[@]}"; do

  workloadJson="{
    \"theta\" : $theta,
    \"range\" : 1000000000,
    \"n\" : 10000000,
    \"ops\" : 10000,
    \"keysize\" : 8,
    \"ratio\" : 95
  }"

  echo $workloadJson > workload.json

  $kvcgExe -l $soFile -f ./server.json -w ./workload.json > results_$(echo $theta | tr "." "_").tsv

  sleep 5

done

rm server.json
rm workload.json
