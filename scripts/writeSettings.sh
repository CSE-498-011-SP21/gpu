kvcgExe=../cmake-build-relwithdebinfo/benchmark/kvcg
soFile=../cmake-build-relwithdebinfo/benchmark/libzipfianWorkload.so

# Number of times to run the benchmark
trials=5

keysize=8
ratio=95
theta=0.99

# The BTree can only function with one GPU, one CUDA stream, and <4B keys and values.
# It's easy to make keysize less than 4 bytes, but it's harder with values because those are actual pointers to data.
#
streams=1


workloadJson="{
        \"theta\" : $theta,
        \"range\" : 1000000000,
        \"n\" : 10000000,
        \"ops\" : 10000,
        \"keysize\" : $keysize,
        \"ratio\" : $ratio
      }"

echo $workloadJson > workload.json

serverJson="{\"cpu_threads\":  12,
  \"threads\" : 2,
  \"streams\" : $streams,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"
echo $serverJson > server.json

#for ((i=0; i<trials; i++))
#do
#  echo "TRIAL: $i"
#  $kvcgExe -l $soFile -f ./server.json -w ./workload.json > btree.tsv
#  echo "Sleeping for 1m"
#  sleep 1m
#done
#
#rm server.json
#rm workload.json
