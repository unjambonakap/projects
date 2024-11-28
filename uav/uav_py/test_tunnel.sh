#!/bin/bash
set -ux

trap 'pkill -9 -P $$' SIGINT SIGTERM

conf_file=/tmp/conf_test_tunnel.yaml
tool=./uav_py/tools.py

echo "{src: {sys: 10, comp: 11}, dst: {sys: 20, comp: 21}}" > $conf_file
cmd="$tool --config-file $conf_file --verbosity=DEBUG --actions=test"

socka_toconn=/tmp/socka_toconn.sock
socka_res=/tmp/socka_res.sock
sockb_toconn=/tmp/sockb_toconn.sock
sockb_res=/tmp/sockb_res.sock

socat UNIX-LISTEN:$socka_toconn TCP-LISTEN:10000 &
socat UNIX-LISTEN:$socka_res TCP-LISTEN:11111 &
socat UNIX-LISTEN:$sockb_res TCP-LISTEN:11112 &
sleep 0.1
socat UNIX-LISTEN:$sockb_toconn TCP:localhost:10000 &
sleep 0.1
$cmd --plain-endpoint $socka_res --encoded-endpoint $socka_toconn &
$cmd --plain-endpoint $sockb_res --encoded-endpoint $sockb_toconn --toggle-dir &



wait
