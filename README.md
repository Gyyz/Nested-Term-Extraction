# Term Extraction

## 1. File Discribtion:
`./data`: the dir contains the corpus

`./models`: the dir contains model python files, in it:

```bash
|--> `./charfeat`: the model to build char level feature. It is copied from [Jie's code](https://github.com/jiesutd/NCRFpp/tree/master/model), it contains:

	    |--> `charbigru.py`: the bi-directional gru model for char feature.
	    
	    |--> `charbilstm.py`: the bi-directional lstm model for char feature.
	    
	    |--> `charcnn.py`: the cnn pooling model for char feature
	    
|--> `./wordfeat`: the model to build word level and sequence level feature. It is copied from [Jie's code](https://github.com/jiesutd/NCRFpp/tree/master/model) and modified, it contains:

	    |--> `WordRep.py`: the model class to build word level features
	    
	    |--> `WordSeq.py`: the model class to build sequential features from word level features
```


​	|-->`FCRanking.py`: the model file for span classification based ranking model

`./saves`: the dir to save models & data & test output.

`./utils`: the dir contains some utils that load data, build vocab, attention functions and ect:

```bash
|--> `alphabet.py`: the tools to build vocab, It is copied from [Jie's code](https://github.com/jiesutd/NCRFpp/tree/master/model)

|--> `data.py`: the tools to load data and build vocab, It is copied from [Jie's code](https://github.com/jiesutd/NCRFpp/tree/master/model) and modified.

|--> `functions.py`: the python file that includes attention, softmax, masked softmax and ect. tools.
```

`main.py`: the python file to train and test model.

**Key files**: _FCRanking.py   main.py_


## 2. How to run:

### 2.1 Before Run, Data Pre-processing:

Please process the data into the jsonline format below:

`{"words": ["IL-2", "gene", "expression", "and", "NF-kappa", "B", "activation", "through", "CD28", "requires", "reactive", "oxygen", "production", "by", "5-lipoxygenase", "."], "tags": ["NN", "NN", "NN", "CC", "NN", "NN", "NN", "IN", "NN", "VBZ", "JJ", "NN", "NN", "IN", "NN", "."], "terms": [[0, 1, "G#DNA_domain_or_region"], [0, 2, "G#other_name"], [4, 5, "G#protein_molecule"], [4, 6, "G#other_name"], [8, 8, "G#protein_molecule"], [14, 14, "G#protein_molecule"]]}`

There are three keys:

<span style="color:red">"words"</span>: the tokenized sentence.

<span style="color:red">"tags"</span>: the POS-tag, not a must, you can modified in the load file `data.py`

<span style="color:red">"terms"</span>: the golden term spans. In it, [0, 1, "G#DNA_domain_or_region"] for example, the first two int number is a must. the third string can use a placeholder like '@' instead if you don't want to do detailed labelling.

Here we use the [GENIA corpus](http://www.geniaproject.org/genia-corpus)  and shared it in the `./data` dir in jsonlines format.

### 2.2 How to run and parameters:
1. train (you can change the parameters below, the parameter in <a style="color:red">()</a> is not a must):
  
   ```bash
   python main.py --status train (--early_stop 26 --dropout 0.5 --use_gpu False --gpuid 3 --max_lengths 5 --word_emb [YOUR WORD EMBEDDINGS DIR])
   ```
   
   
   
2.  test (Be noted that the parameters should be <span style="color:red">***strictly***</span> the same with those you used to train the model except the `--status`)

   ```bash
   python main.py --status test (--early_stop 26 --dropout 0.5 --use_gpu False --gpuid 3 --max_lengths 5 --word_emb [YOUR WORD EMBEDDINGS DIR])
   ```

   Please be noted that the path of data is written in `data.py` in `./data` dir.

3. 