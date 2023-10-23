#!/usr/bin/env bash
if [[ $(docker image ls | grep locobot) ]]
then
	xhost + && docker run --rm --net host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${DISPLAY} -it locobot:latest
else
	docker build -t locobot .
fi
