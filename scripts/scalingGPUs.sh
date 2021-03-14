kvcgExe=../cmake-build-relwithdebinfo/kvcg
soFile=../cmake-build-relwithdebinfo/libzipfianWorkload.so

keysize=8
ratio=95
theta=0.5

workloadJson="{
        \"theta\" : $theta,
        \"range\" : 1000000000,
        \"n\" : 10000000,
        \"ops\" : 10000,
        \"keysize\" : $valuesize,
        \"ratio\" : $ratio
      }"

echo $workloadJson > workload.json

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"
echo $serverJson > server.json

$kvcgExe -l $soFile -f ./server.json -w ./workload.json > singleGPU.tsv

serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 5,
  \"gpus\" : 2,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"
echo $serverJson > server.json


$kvcgExe -l $soFile -f ./server.json -w ./workload.json > dualGPU.tsv

rm server.json
rm workload.json
