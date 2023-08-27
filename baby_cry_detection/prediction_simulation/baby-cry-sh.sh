#!/usr/bin/env bash

function recording() {
    echo "Start Recording..."
sox -t coreaudio default "$filename" trim 0 9 rate 44100 channels 1 
}

filename="/Users/MT/Desktop/stg.wav"
PREDICTION_SCRIPT="/Users/MT/Desktop/baby_cry_detection/baby_cry_detection/prediction_simulation/prediction_simulation.py"
lullaby_file="/Users/MT/Desktop/baby_cry_detection/baby_cry_detection/prediction_simulation/strange-lullaby-28691.mp3"
log_file="/Users/MT/Desktop/output/prediction/prediction.txt"
PREDICTION=1
PLAYING=0
CPT=0
result=0

function predict() {
    echo "Predicting..."
    PREDICTION=$(python3 "$PREDICTION_SCRIPT" --file_name "$filename")
    if grep -q "1" "$log_file"; then
    result=1
    elif grep -q "0" "$log_file"; then
    result=0
    fi
}

function start_playing() {
    if [[ $PLAYING == 0 ]]; then
        echo "Start playing"
        play -q "$lullaby_file"
        PLAYING=1
    fi
}

function stop_playing(){
    if [[ $PLAYING == 1 ]]; then
        echo "Stop playing"
        PLAYING=0
    fi
}

function clean_up {
    # Perform program exit housekeeping
    echo ""
    echo "Thank you for using parenting 2.0"
    echo "Good Bye."
    stop_playing
    exit
}

trap clean_up SIGHUP SIGINT SIGTERM

echo "Welcome to Parenting 2.0"
echo ""

recording
predict

if [[ $result == 0 ]]; then
    stop_playing
    echo "Your Baby is relaxing"
else
    CPT=$(expr $CPT + 1)
    start_playing
fi

echo "Majority Vote Prediction: $PREDICTION"
echo "State of the Process PREDICTION=$PREDICTION, PLAYING=$PLAYING, COMPTEUR=$CPT"

clean_up
