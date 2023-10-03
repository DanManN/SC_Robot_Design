#!/usr/bin/env bash
if [ -f ros.sif ]
then
	apptainer run --bind $(pwd):/ros_ws ros.sif
else
	apptainer build ros.sif apptainer.def
fi
