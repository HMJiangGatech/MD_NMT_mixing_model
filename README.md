# MTDA
Domain Adaptation of Machine Translation

#### Test Environment

- GPU: 1080Ti
- System: RedHat
- fairseq: see submodule

#### Data Preprocessing

```bash
cd dataset
bash prepare-wmt14en2de.sh
bash prepare-iwslt14en2de.sh
cd ..
WMTTEXT=dataset/wmt14_en_de
IWSLTTEXT=dataset/iwslt14.tokenized.en-de

python multidata_preprocess.py --source-lang en --target-lang de \
  --trainpref $WMTTEXT/train,$IWSLTTEXT/train \
  --validpref $WMTTEXT/valid,$IWSLTTEXT/valid \
  --testpref $WMTTEXT/test,$IWSLTTEXT/test \
  --destdir data-bin/mixed --thresholdtgt 0 --thresholdsrc 0
```

#### Training

```bash
SAVEDIR=checkpoints/NEWS
mkdir -p $SAVEDIR
CUDA_VISIBLE_DEVICES=0 python main.py data-bin/mixed \
  --multidatasource 0 -a transformer_iwslt_de_en\
  --optimizer adam --lr 0.0005 -s en -t de \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-epoch 50 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVEDIR \
  --task translation_da

SAVEDIR=checkpoints/TED
mkdir -p $SAVEDIR
CUDA_VISIBLE_DEVICES=0 python main.py data-bin/mixed \
  --multidatasource 1 -a transformer_iwslt_de_en\
  --optimizer adam --lr 0.0005 -s en -t de \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-epoch 50 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVEDIR \
  --task translation_da

SAVEDIR=checkpoints/Mixed
mkdir -p $SAVEDIR
CUDA_VISIBLE_DEVICES=0 python main.py data-bin/mixed \
  --multidatasource mixed --damethod naive -a transformer_iwslt_de_en\
  --optimizer adam --lr 0.0005 -s en -t de \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-epoch 50 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVEDIR \
  --task translation_da

SAVEDIR=checkpoints/Mixed_Ours
mkdir -p $SAVEDIR
CUDA_VISIBLE_DEVICES=0 python main.py data-bin/mixed \
  --multidatasource mixed --damethod bayesian -a transformer_da_bayes_iwslt_de_en\
  --optimizer adam --lr 0.0005 -s en -t de \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion cross_entropy_da --max-epoch 50 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir $SAVEDIR \
  --task translation_da
```

## Average 10 latest checkpoints:

```
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR \
   --num-epoch-checkpoints 10 --output $SAVEDIR/model.pt
```

##

## Evaluation

|           | NEWS      |  TED      |
|-----------|-----------|-----------|
|NEWS naive | | |
|TED naive  | 4.90      | 29.09     |
|Mix        | | |
|Ours       | | | 

