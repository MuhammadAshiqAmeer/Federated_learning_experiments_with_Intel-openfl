#! /bin/bash
# Read in the number of devices
#echo "Enter the number of devices to copy the file to:"
#read num_devices

# Read in the addresses of each device

#devices=()
#for ((i=1; i<=$num_devices; i++)); do
#  echo "Enter the address of device $i:"
#  read address
#  devices+=($address)
#done

devices=("tomsy@172.16.89.5,train_1.xlsx,test_1.xlsx" "tomsy@172.16.64.54,train_2.xlsx,test_2.xlsx" "cirmlab@172.16.88.248,train_3.xlsx,test_3.xlsx")

# Read in the location to save the file
#echo "Enter the location to save the file in collabs:"
save_location=$1

# Read in the location of the original file
#echo "Enter the location of the original files to send:"
#file_location=$1

# Loop through devices and copy the file to each one
for device in "${devices[@]}"; do
  scp envoy/`cut -d"," -f2 <<<$device` `cut -d"," -f1 <<<$device`:$1
  scp envoy/`cut -d"," -f3 <<<$device` `cut -d"," -f1 <<<$device`:$1
#  echo `cut -d"," -f2 $device` $device:$save_location
done
