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
    attn_penalty = False

### Web of Science experiments

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

@ex.named_config
def wos_variation_7():
    exp_name = 'self_attention_new'
    n_heads = [4,4,4]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    single_attention = False

@ex.named_config
def wos_variation_8():
    exp_name = 'self_attention_new_single'
    n_heads = [8,8,8]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 500
    single_attention = True
    attn_penalty = True

@ex.named_config
def wos_variation_9():
    exp_name = 'self_attention_new_single_low'
    n_heads = [8,8,8]
    prev_emb = False
    attention_type = 'self'
    embedding_dim = 100
    mlp_hidden_dim = 100
    cat_emb_dim = 100
    da = 200
    single_attention = True
    attn_penalty = True
    lr_patience=10

@ex.named_config
def wos_variation_10():
    exp_name = 'self_attention_high'
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10

@ex.named_config
def wos_variation_11():
    exp_name = 'wos_maxpool'
    prev_emb = True
    attention_type = 'no_attention'

@ex.named_config
def wos_variation_12():
    exp_name = 'wos_maxpool_sgd'
    prev_emb = True
    attention_type = 'no_attention'
    optimizer='sgd'
    lr=0.06
    lr_patience=30
    epochs=30

@ex.named_config
def wos_variation_13():
    exp_name = 'wos_self_sgd'
    prev_emb = False
    attention_type = 'self'
    optimizer='sgd'
    lr=0.1
    cat_emb_dim = 300
    single_attention = True
    attn_penalty = True
    lr_patience=30
    epochs=30

### DBpedia experiments

@ex.named_config
def dbp_base():
    data_loc = '/home/ml/ksinha4/mlp/hier-class/data/'
    data_path = 'db_sm_train_clean'
    file_name = 'df_small_train.csv'
    test_file_name = 'df_small_test.csv'
    test_output_name = 'df_small_outp.csv'
    levels = 3
    weight_decay = 1e-6

@ex.named_config
def dbp_variation_1():
    exp_name = 'db_self_attention_new_single'
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 150
    single_attention = True
    attn_penalty = True
    clean = True
    batch_size = 16


@ex.named_config
def dbp_variation_2():
    exp_name = 'db_no_attention'
    prev_emb = True
    attention_type = 'no_attention'
    clean = True
    n_heads = [2,2,4]

@ex.named_config
def dbp_variation_3():
    exp_name = 'db_scaled_attention'
    prev_emb = False
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [8, 8, 8]
    lr_patience = 30
    batch_size = 16

@ex.named_config
def dbp_variation_4():
    exp_name = 'db_scaled_attention_2'
    prev_emb = False
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [4, 4, 4]
    lr_patience = 30

@ex.named_config
def dbp_variation_5():
    exp_name = 'db_scaled_attention_2'
    prev_emb = True
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [4, 4, 4]
    lr_patience = 30

@ex.named_config
def dbp_variation_6():
    exp_name = 'db_scaled_attention_ugly'
    prev_emb = False
    attention_type = 'scaled'
    clean = False
    data_path = 'db_sm_train_ugly'
    single_attention = True
    n_heads = [2, 2, 2]
    lr_patience = 30