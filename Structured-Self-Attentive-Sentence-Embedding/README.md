# Structured-Self-Attentive-Sentence-Embedding
An modified version of open-source implementation of the paper ``A Structured Self-Attentive Sentence Embedding'' published by IBM and MILA. 
https://arxiv.org/abs/1703.03130

source: https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding

We use it for classification on DBpedia dataset: 
* attention with 1-hop

| epoch  21 |  9600/ 9668 batches | ms/batch 307.81 | loss 0.2565 | pure loss 0.2534
-----------------------------------------------------------------------------------------
| evaluation | time: 74.88s | valid loss (pure) 0.3055 | Acc   0.9350
-----------------------------------------------------------------------------------------
finish loading test model
-----------------------------------------------------------------------------------------
| testing | valid loss (pure) 0.2803 | Acc   0.9388

* attention with 30 hops

| epoch  21 |  9600/ 9668 batches | ms/batch 304.29 | loss 1.7869 | pure loss 0.7578
-----------------------------------------------------------------------------------------
| evaluation | time: 75.66s | valid loss (pure) 0.4910 | Acc   0.8670
-----------------------------------------------------------------------------------------
finish loading test model
-----------------------------------------------------------------------------------------
| testing | valid loss (pure) 0.4669 | Acc   0.8713
-----------------------------------------------------------------------------------------

* BiLSTM with max pooling

| epoch   8 |  9600/ 9668 batches | ms/batch 539.25 | loss 0.3396 | pure loss 0.3396
-----------------------------------------------------------------------------------------
| evaluation | time: 169.82s | valid loss (pure) 0.3204 | Acc   0.9309
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
| test set result | valid loss (pure) 0.3042 | Acc   0.9312
-----------------------------------------------------------------------------------------
