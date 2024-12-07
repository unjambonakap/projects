#!/bin/bash

mkdir out
gopy pkg --output out/ --name chdrft_cel ./app ./app/proto
#needs to be ran twice oO
gopy pkg --output out/ --name chdrft_cel ./app ./app/proto

