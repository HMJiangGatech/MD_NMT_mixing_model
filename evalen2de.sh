#!/bin/bash

if [[ $# -ne 2  ]]; then
      echo "Usage:   \"bash evalen2de.sh SAVEDIR GPUID\""
      exit 0
else
    SAVEDIR=$1
    GPUID=$2
fi

echo "Model path" $SAVEDIR
echo "GPUID" $GPUID

MODELDIR=$SAVEDIR/model.pt
if [ -f $MODELDIR ]; then
  echo $MODELDIR "already exists"
else
  echo "Start averaging model"
  python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR \
     --num-epoch-checkpoints 10 --output $MODELDIR | grep 'Finish'
  echo "End averaging model"
fi

ARRAY=( "NEWS:valid"
        "TED:valid1" )

for data in "${ARRAY[@]}" ; do
    KEY="${data%%:*}"
    VALUE="${data##*:}"
    echo "Evaluate over" $KEY "Dataset"
    CUDA_VISIBLE_DEVICES=$GPUID fairseq-generate data-bin/mixed \
      --gen-subset $VALUE \
      --path $MODELDIR \
      --batch-size 128 --beam 5 --remove-bpe \
      --user-dir ./mymodule --quiet
done
