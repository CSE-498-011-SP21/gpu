#!/bin/bash

kvcgExe=../cmake-build-relwithdebinfo/kvcg
soFile=../cmake-build-relwithdebinfo/libzipfianWorkload.so
mkvExe=../cmake-build-relwithdebinfo/megakv
mkvsoFile=../cmake-build-relwithdebinfo/libmkvzipfianWorkload.so

theta=0.5
valuesizes=(8 16 32 64 128 256 512)
ratios=(80 95 100)

function run_test {

  for ratio in "${ratios[@]}"; do
    for valuesize in "${valuesizes[@]}"; do

      workloadJson="{
        \"theta\" : $theta,
        \"range\" : 1000000000,
        \"n\" : 10000000,
        \"ops\" : 10000,
        \"keysize\" : $valuesize,
        \"ratio\" : $ratio
      }"

      echo $workloadJson > workload.json

      $kvcgExe -l $soFile -f ./server.json -w ./workload.json > ${1}_${valuesize}_${ratio}.tsv

      sleep 5

    done
  done

}

function run_mkv_test {

  for ratio in "${ratios[@]}"; do
    for valuesize in "${valuesizes[@]}"; do

      workloadJson="{
        \"theta\" : $theta,
        \"range\" : 1000000000,
        \"n\" : 10000000,
        \"ops\" : 10000,
        \"keysize\" : $valuesize,
        \"ratio\" : $ratio
      }"

      echo $workloadJson > workload.json

      $mkvExe -l $mkvsoFile -w ./workload.json > megakv_${valuesize}_${ratio}.tsv

      sleep 5

    done
  done

}


serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"


echo $serverJson > server.json

run_test "kvcg"

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : false
}"


echo $serverJson > server.json


run_test "gpu_only"

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 0,
  \"gpus\" : 0,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"

echo $serverJson > server.json

run_test "cpu_only"

rm server.json

run_mkv_test

rm workload.json
