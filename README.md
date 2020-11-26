## Hierarchical Neural Attention-based Classification

Code and Data for the paper [A Hierarchical Neural Attention-based Text Classifier](http://www.aclweb.org/anthology/D18-1094) published as a short paper in EMNLP, 2018.

**Note: This repository is no longer actively maintained.**

### Dependencies

- Pytorch
- Pandas
- tqdm


### Dataset

- Web of Science dataset - [Drive link](https://drive.google.com/file/d/19vJLoo8g1E18pq2prmrFpFlXvsSoP1YE/view?usp=sharing)
- DBPedia dataset (curated using [get_taxonomy.py](https://github.com/koustuvsinha/hier-class/blob/master/data/get_taxonomy.py))
    - [Drive link](https://drive.google.com/file/d/1AloTMDSlujNu086UBeSGTeR1DyYwqPEW/view?usp=sharing)
    - Train test split - [Drive link](https://drive.google.com/open?id=1Yi3GnYe-I2F_jghK3Y3yHJpphHKDZhUE)

### Running the code

- Navigate into the directory `codes/app/`
- Run `python main.py --config_id <config name>`

### Experiment configs

To run, create an experiment config from the sample configs in `config/` folder.

### Questions

Please open an issue!

### License

MIT

