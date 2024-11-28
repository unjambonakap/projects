#/bin/bash
set -euv

input=$1; shift

tmp_vid=/tmp/t1.mp4
moov_table_file=res.json
prog_dir=/home/benoit/programmation
samples_file=/tmp/t1_samples.json
probe_file=/tmp/t1_probe.json
data_file=data.json

#ffmpeg -i $input -s 1x1 -map_metadata 0 -c copy -copy_unknown -map 0:3 $tmp_vid
#node node/test.js $tmp_vid $moov_table_file
ffprobe -v quiet -print_format json -show_format -show_streams $input > $probe_file
$prog_dir/build/projects/uav/projects_uav_sample_tools.cpp --action=convert_gpmf --infile $moov_table_file --outfile=$samples_file

jq -s '{samples: .[0], metadata: .[1]}' $samples_file $probe_file > $data_file
#
#
