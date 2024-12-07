#!/bin/bash


NGINX_RUNDIR=/tmp/nginx
mkdir -p NGINX_RUNDIR/logs
/home/benoit/repos/nginx/objs/nginx  -p $NGINX_RUNDIR -c $PWD/server/nginx.conf

