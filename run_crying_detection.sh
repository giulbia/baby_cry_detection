#!/usr/bin/env bash

PREDICTION=1
PLAYING=0
CPT=0

function clean_up {

	# Perform program exit housekeeping
	echo ""
	echo "Thank you for using parenting 2.0"
	echo "Good Bye."
	stop_playing
	exit
}

trap clean_up SIGHUP SIGINT SIGTERM

function recording(){
	echo "Start Recording..."
	arecord -D plughw:1,0 -d 9 -f S16_LE -c1 -r44100 -t wav /opt/baby_cry_rpi/recording/signal_9s.wav
}


function predict() {
	echo "Predicting..."
	echo -n "What is the prediction ?"
	python /opt/baby_cry_rpi/script/make_prediction.py
	PREDICTION=$(cat /opt/baby_cry_rpi/prediction/prediction.txt)
	echo "Prediction is $PREDICTION"
}

function start_playing() {
	if [[ $PLAYING == 0 ]]; then
		echo "start playing"
                aplay -D plughw:0,0 /opt/baby_cry_rpi/lullaby/lullaby_classic.wav
		PLAYING=1
	fi
}

function stop_playing(){
	if [[ $PLAYING == 1 ]]; then
		echo "stop playing"
		PLAYING=0
	fi
}

echo "Welcome to Parenting 2.0"
echo ""
while true; do
	recording
	predict
	if [[ $PREDICTION == 0 ]]; then
		stop_playing
	else
		CPT=$(expr $CPT + 1)
		start_playing
	fi
echo "State of the Process PREDICTION = $PREDICTION, PLAYING=$PLAYING, COMPTEUR=$CPT"
done
clean_up
