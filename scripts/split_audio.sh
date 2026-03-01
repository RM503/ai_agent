#!/bin/bash

if [ ! -f "$1" ]; then
  echo "Error: File '$1' not found."
  exit 1
fi

input_file_path=$1
basename="${input_file_path%.*}"
ext="${input_file_path##*.}"
ffmpeg -i "$input_file_path" -f segment -segment_time 600 -c copy "${basename}_%03d.${ext}"