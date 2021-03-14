serverJson="{\"cpu_threads\":  24,
  \"threads\" : 4,
  \"streams\" : 10,
  \"gpus\" : 1,
  \"train\" : false,
  \"size\" : 1000000,
  \"cache\" : true
}"

kvcgExe=../cmake-build-relwithdebinfo/kvcg_forever
soFile=../cmake-build-relwithdebinfo/libzipfianWorkloadSwitch.so

echo $serverJson > server.json

valuesize=8
ratio=95
theta=0.5

workloadJson="{
        \"theta\" : $theta,
        \"range\" : 20000000,
        \"n\" : 10000000,
        \"ops\" : 10000,
        \"keysize\" : $valuesize,
        \"ratio\" : $ratio
      }"

echo $workloadJson > workload.json

$kvcgExe -l $soFile -f ./server.json -w ./workload.json > modelchange.tsv

rm server.json
rm workload.json
