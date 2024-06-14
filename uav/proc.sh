#/bin/bash
set -euv

input=$1; shift

tmp_vid=/tmp/t1.mp4
moov_table_file=res.json
prog_dir=/home/benoit/programmation
samples_file=samples.json

ffmpeg -i $input -s 1x1 -map_metadata 0 -c copy -copy_unknown -map 0:3 $tmp_vid
node node/test.js $tmp_vid $moov_table_file

$prog_dir/build_debug/projects/uav/projects_uav_sample_tools.cpp --action=convert_gpmf --infile $moov_table_file --outfile=$samples_file

