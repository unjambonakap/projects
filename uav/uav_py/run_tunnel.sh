#!/bin/bash
set -ux

trap 'pkill -9 -P $$' SIGINT SIGTERM

mav_endpoint=$1; shift
plain_endpoint=$1; shift
conf_file=$1; shift
tool=./uav_py/tools.py

echo "{src: {sys: 10, comp: 11}, dst: {sys: 20, comp: 21}}" > $conf_file
cmd="$tool --config-file $conf_file --verbosity=DEBUG --actions=test $@"

socka_mav=$(mktemp  sock_mav.XXX)
socka_plain=$(mktemp sock_plain.XXX)
rm $socka_plain $socka_mav

socat UNIX-LISTEN:$socka_plain $plain_endpoint &
socat UNIX-LISTEN:$socka_mav $mav_endpoint &
sleep 0.2
$cmd --plain-endpoint $socka_plain --encoded-endpoint $socka_mav &



wait
