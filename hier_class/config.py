from sacred import Experiment

### All experiment configurations and values here
### Note the variants too

ex = Experiment()

@ex.config
def exp_config():
    gpu = 0
    use_gpu = True
    exp_name = 'wos_norm_level_self'
    embedding_dim = 300
    mlp_hidden_dim = 300
    use_embedding = False
    fix_embeddings = False
    embedding_file = '/home/ml/ksinha4/mlp/hier-class/data/glove.6B.100d.txt'
    embedding_saved = 'glove_embeddings.mod'
    load_model = False
    load_model_path = ''
    save_name = 'model_epoch_{}_step_{}.mod'
    optimizer = 'adam'
    lr = 1e-3
    lr_factor = 0.1
    lr_threshold = 1e-4
    lr_patience = 3
    clip_grad = 0.5
    momentum = 0.9
    dropout = 0.2
    log_interval = 200
    save_interval = 1000
    train_test_split = 0.9
    data_type = 'WIKI'
    data_loc = '/home/ml/ksinha4/mlp/hier-class/data/'
    data_path = 'wos_train_clean'
    file_name = 'wos_data_train.csv'
    test_file_name = 'wos_data_test.csv'
    test_output_name = 'wos_data_output.csv'
    #data_loc = '/home/ml/ksinha4/datasets/data_WOS/WebOfScience/WOS46985'
    tokenization = 'word'
    clean = True
    batch_size = 64
    epochs = 20
    level = -1
    levels = 2
    cat_emb_dim = 600
    tf_ratio=1
    tf_anneal=1
    validation_tf = 0
    weight_decay=1e-4
    temperature = 1
    loss_focus = [1,1,1]
    label_weights = [1,1,1]
    dynamic_dictionary = True
    max_vocab = 100000
    max_word_doc = -1
    decoder_ready = True
    prev_emb = False
    n_heads = [10,10,10]
    baseline = False # either False, or fast / bilstm
    debug = False
    attn_penalty_coeff = 0
    d_k = 64
    d_v = 64
    da = 350
    seed = 1111
    attention_type = 'scaled'
    use_attn_mask = False # use attention mask for scaled if required
    renormalize = 'level' # level -> for level masking, category -> for tree masking
    single_attention = True # for scaled attention use only one attention layer for all

@ex.named_config
def wos_variation_1():
    exp_name = 'small_heads'
    n_heads = [1,1,1]
    prev_emb = True

@ex.named_config
def wos_variation_2():
    exp_name = 'medium_heads'
    n_heads = [2,2,2]
    prev_emb = False

@ex.named_config
def wos_variation_3():
    exp_name = 'eight_headed'
    n_heads = [8,8,8]
    prev_emb = True

@ex.named_config
def wos_variation_4():
    exp_name = 'tree_masking_prev_true'
    renormalize = 'category'
    n_heads = [8,8,8]
    prev_emb = True

@ex.named_config
def wos_variation_5():
    exp_name = 'tree_masking_prev_false'
    renormalize = 'category'
    n_heads = [8,8,8]
    prev_emb = False

@ex.named_config
def wos_variation_6():
    exp_name = 'clean_tokenization'
    n_heads = [2,2,8]
    prev_emb = False






