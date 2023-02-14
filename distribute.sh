# Read in the number of devices
echo "Enter the number of devices to copy the file to:"
read num_devices

# Read in the addresses of each device
devices=()
for ((i=1; i<=$num_devices; i++)); do
  echo "Enter the address of device $i:"
  read address
  devices+=($address)
done

# Read in the location to save the file
echo "Enter the location to save the file in collabs:"
read save_location

# Read in the location of the original file
echo "Enter the location of the original files to send:"
read file_location

# Loop through devices and copy the file to each one
for device in "${devices[@]}"; do
  scp $file_location $device:$save_location
done
