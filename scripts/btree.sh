kvcgExe=../cmake-build-relwithdebinfo/benchmark/kvcg
soFile=../cmake-build-relwithdebinfo/benchmark/libzipfianWorkload.so

# Number of times to run the benchmark
trials=5

keysize=8
ratio=95
theta=0.5
# number of batches to run
batches=100
# operations per batch
n=10000
# range=maximum value for key (1Billion default)

# The BTree can only function with one GPU, one CUDA stream, and <4B keys.
streams=1


workloadJson="{
        \"theta\" : $theta,
        \"range\" : 1000000000,
        \"n\" : $n,
        \"ops\" : $batches,
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
  \"cache\" : false
}"
echo $serverJson > server.json

for ((i=0; i<trials; i++))
do
  echo "TRIAL: $i"
  $kvcgExe -l $soFile -f ./server.json -w ./workload.json > btree.tsv
  echo "Sleeping for 10s"
  sleep 10s
done

rm server.json
rm workload.json
