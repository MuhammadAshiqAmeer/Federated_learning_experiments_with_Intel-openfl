#!/bin/bash

# Run the Python code
python mandatory_check.py

# Check if the Python code raised an error
if [ $? -eq 0 ]; then
  # Start envoy if all mandatory fields are present
  echo "All mandatory fields are present !"
  echo 'Name of envoy : '
  read envoy_name
  echo 'Name of configuration file :'
  read envoy_config
  echo 'Director FQDN/IP :'
  read director_fqdn
  echo 'Director listening port No :'
  read port
  fx envoy start -n $envoy_name --disable-tls --envoy-config-path $envoy_config -dh $director_fqdn -dp $port
else
  # Do something else if mandatory fields are missing
  echo "Some mandatory fields are missing !"
fi

