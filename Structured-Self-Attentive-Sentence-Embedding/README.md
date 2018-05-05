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

| epoch  39 |  9600/ 9668 batches | ms/batch 161.31 | loss 1.7065 | pure loss 0.7000
-----------------------------------------------------------------------------------------
| evaluation | time: 45.76s | valid loss (pure) 0.4722 | Acc   0.8695
-----------------------------------------------------------------------------------------
finish loading test model
-----------------------------------------------------------------------------------------
| testing | valid loss (pure) 0.4489 | Acc   0.8755
-----------------------------------------------------------------------------------------

* BiLSTM with mean pooling

| epoch  15 |  9600/ 9668 batches | ms/batch 537.28 | loss 0.1909 | pure loss 0.1909
-----------------------------------------------------------------------------------------
| evaluation | time: 168.25s | valid loss (pure) 0.3319 | Acc   0.9459
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
| test set result | valid loss (pure) 0.3088 | Acc   0.9468
-----------------------------------------------------------------------------------------

* BiLSTM with max pooling
| epoch  23 |  9600/ 9668 batches | ms/batch 536.91 | loss 0.2345 | pure loss 0.2345
-----------------------------------------------------------------------------------------
| evaluation | time: 169.24s | valid loss (pure) 0.2680 | Acc   0.9416
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
| test set result | valid loss (pure) 0.2516 | Acc   0.9420
-----------------------------------------------------------------------------------------
