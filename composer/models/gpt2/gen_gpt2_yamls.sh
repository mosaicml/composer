#!/bin/bash

ssrs=(1.0 0.8 0.67 0.5 0.34 0.25) 

for i in "${ssrs[@]}"
do
	python scaling_laws_generator.py --hours 2  --per_device_batch_size 8 --num_devices 8 --ssr $i --output_file gpt2_38m_$i.yaml
  echo
done

for i in "${ssrs[@]}"
do
	python scaling_laws_generator.py --hours 24 --per_device_batch_size 3 --num_devices 8 --ssr $i --output_file gpt2_114m_$i.yaml
  echo
done

for i in "${ssrs[@]}"
do
	python scaling_laws_generator.py --hours 12 --per_device_batch_size 4 --num_devices 8 --ssr $i --output_file gpt2_85m_$i.yaml
  echo
done
