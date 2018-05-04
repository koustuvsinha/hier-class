# Structured-Self-Attentive-Sentence-Embedding
An modified version of open-source implementation of the paper ``A Structured Self-Attentive Sentence Embedding'' published by IBM and MILA. 
https://arxiv.org/abs/1703.03130

source: https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding
We use it for classification on DBpedia dataset: 
* attention with 1-hop

| epoch  19 |  9600/ 9668 batches | ms/batch 301.02 | loss 0.2641 | pure loss 0.2606
-----------------------------------------------------------------------------------------
| evaluation | time: 77.50s | valid loss (pure) 0.3089 | Acc   0.9350
-----------------------------------------------------------------------------------------
finish loading test model
-----------------------------------------------------------------------------------------
| testing | valid loss (pure) 0.2828 | Acc   0.9383

* attention with 30 hops

| epoch  19 |  9600/ 9668 batches | ms/batch 303.60 | loss 1.7985 | pure loss 0.7682
-----------------------------------------------------------------------------------------
| evaluation | time: 76.02s | valid loss (pure) 0.4976 | Acc   0.8658
-----------------------------------------------------------------------------------------
finish loading test model
-----------------------------------------------------------------------------------------
| testing | valid loss (pure) 0.4717 | Acc   0.8695
-----------------------------------------------------------------------------------------

* BiLSTM with max pooling

| epoch   7 |  9600/ 9668 batches | ms/batch 529.58 | loss 0.3582 | pure loss 0.3582
-----------------------------------------------------------------------------------------
| evaluation | time: 167.54s | valid loss (pure) 0.3340 | Acc   0.9280
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
| test set result | valid loss (pure) 0.3178 | Acc   0.9289
-----------------------------------------------------------------------------------------
