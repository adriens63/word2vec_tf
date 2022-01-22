# word2vec_tf   

Word2vec implementation from scratch


## W2V trained on Wiki_text_2 :  

To see the embeddings in the Tensorboard 3D projector, type :

```
tensorboard --logdir=./word2vec/weights/w2v_skip_gram_1/log_dir/ --port=8012
```

## Train a model :  

Install the requirements :   
```
pip install -r requirements.txt 
```

Launch the training :   
```
python -m word2vec.train --config ./word2vec/config/config.yml
```
