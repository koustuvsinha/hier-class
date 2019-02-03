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
    embedding_file = '/home/ml/ksinha4/mlp/hier-class/data/glove.6B.300d.txt'
    embedding_saved = 'glove_embeddings.mod'
    load_model = False
    load_model_path = ''
    save_name = 'model_epoch_{}_step_{}.mod'
    optimizer = 'adam'
    lr = 1e-3
    lr_factor = 0.1
    lr_threshold = 1e-4
    lr_patience = 1
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
    attn_penalty_coeff = 1
    d_k = 64
    d_v = 64
    da = 350
    seed = 1111
    attention_type = 'scaled'
    model_type = 'attentive' # attentive / pooled
    use_attn_mask = False # use attention mask for scaled if required
    renormalize = 'level' # level -> for level masking, category -> for tree masking
    single_attention = True # for scaled attention use only one attention layer for all
    attn_penalty = False
    use_rnn = True
    n_layers = 1
    multi_class = True
    detach_encoder = False
    teacher_forcing = True # teacher forcing by default should be true
    use_cat_emb = False
    overall = True
    pretrained_lm = False # use pretrained and tuned language model
    use_parent_emb = False

### Web of Science experiments
@ex.named_config
def wos_new_data():
    data_path = 'wos_train_n_clean_4'
    file_name = 'wos_data_n_train.csv'
    test_file_name = 'wos_data_n_test.csv'
    test_output_name = 'wos_data_n_output.csv'


@ex.named_config
def wos_variation_1():
    #score: l1:0.74, l2:0.32
    exp_name = 'small_heads'
    n_heads = [1,1,1]
    prev_emb = True
    n_layers = 2

@ex.named_config
def wos_variation_2():
    exp_name = 'medium_heads'
    n_heads = [2,2,2]
    prev_emb = False
    n_layers = 2

@ex.named_config
def wos_variation_3():
    exp_name = 'eight_headed'
    n_heads = [8,8,8]
    prev_emb = True
    n_layers = 2

@ex.named_config
def wos_variation_3_1():
    exp_name = 'eight_headed_penalty'
    n_heads = [8,8,8]
    prev_emb = True
    attn_penalty = True
    n_layers = 2

@ex.named_config
def wos_variation_4():
    exp_name = 'tree_masking_prev_true'
    renormalize = 'category'
    n_heads = [8,8,8]
    prev_emb = True
    n_layers = 2

@ex.named_config
def wos_variation_5():
    exp_name = 'tree_masking_prev_false'
    renormalize = 'category'
    n_heads = [8,8,8]
    prev_emb = False
    n_layers = 2

@ex.named_config
def wos_variation_6():
    exp_name = 'clean_tokenization'
    n_heads = [2,2,8]
    prev_emb = False
    n_layers = 2

@ex.named_config
def wos_variation_7():
    exp_name = 'self_attention_new'
    n_heads = [4,4,4]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    single_attention = False
    n_layers = 2

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
    n_layers = 2

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
    n_layers = 2

@ex.named_config
def wos_variation_10():
    # score: l1: 0.95, l2: 0.89
    exp_name = 'self_attention_high'
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10
    n_layers = 2

@ex.named_config
def wos_variation_11():
    # score: l1 : 0.93, l2: 0.75
    exp_name = 'wos_maxpool'
    prev_emb = True
    attention_type = 'no_attention'
    n_layers = 2

@ex.named_config
def wos_variation_12():
    exp_name = 'wos_maxpool_sgd'
    prev_emb = True
    attention_type = 'no_attention'
    optimizer='sgd'
    lr=0.06
    lr_patience=30
    epochs=30
    n_layers = 2

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
    n_layers = 2

@ex.named_config
def wos_variation_14():
    exp_name = 'wos_maxpool_first'
    prev_emb = False
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    lr_patience=30
    epochs=30
    use_embedding = True
    level = 1
    n_layers = 2

@ex.named_config
def wos_variation_15():
    exp_name = 'wos_pretrained'
    prev_emb = False
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    lr_patience=30
    epochs=30
    clean=True
    data_path = 'wos_data_new_clean'
    use_embedding = True
    n_layers = 2

@ex.named_config
def wos_variation_16():
    exp_name = 'wos_pretrained_2'
    prev_emb = True
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    optimizer = 'rmsprop'
    data_path = 'wos_data_new_clean'
    weight_decay = 0
    lr_patience=30
    epochs=30
    use_embedding = True
    n_layers = 2

@ex.named_config
def wos_variation_17():
    exp_name = 'wos_pretrained_2'
    prev_emb = True
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    optimizer = 'rmsprop'
    data_path = 'wos_data_new_clean'
    weight_decay = 0
    lr_patience=30
    epochs=30
    n_layers=2
    use_embedding = True
    n_layers = 2

@ex.named_config
def wos_variation_18():
    exp_name = 'wos_pretrained_single'
    prev_emb = True
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    optimizer = 'rmsprop'
    data_path = 'wos_data_new_clean'
    lr_factor=0.5
    epochs=30
    n_layers=2
    use_embedding = True
    level = 0
    renormalize = False

@ex.named_config
def wos_variation_19():
    exp_name = 'wos_pretrained_self'
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 100
    embedding_dim = 100
    mlp_hidden_dim = 100
    optimizer = 'rmsprop'
    data_path = 'wos_data_new_clean'
    lr_factor=0.5
    epochs=30
    n_layers=2
    use_embedding = True
    single_attention = True
    level = 0
    renormalize = False

@ex.named_config
def wos_variation_20():
    exp_name = 'wos_pretrained_full_vocab'
    prev_emb = True
    attention_type = 'no_attention'
    cat_emb_dim = 200
    embedding_dim = 100
    mlp_hidden_dim = 100
    optimizer = 'rmsprop'
    data_path = 'wos_data_new_clean'
    lr_factor=0.5
    epochs=30
    n_layers=2
    use_embedding = True
    level = 0
    renormalize = False
    data_path = 'wos_full_vocab'
    max_vocab = -1

@ex.named_config
def wos_variation_21():
    # score: l1=0.95, l2=0.858
    exp_name = 'wos_maxpool_classifier'
    prev_emb = True
    attention_type = 'no_attention'
    n_layers = 2

@ex.named_config
def wos_variation_22():
    # score:
    exp_name = 'self_attention_high_with_prev'
    n_heads = [15,15,15]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10
    n_layers = 2

@ex.named_config
def wos_variation_23():
    # score: l1: 0.891651, l2: 0.76478
    # score with pretrained LM: l1: 0.865305, l2: 0.72161
    # sc : l1:
    exp_name = 'self_attention_high_new'
    #n_heads = [15,15,15]
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01

@ex.named_config
def wos_variation_23_1():
    # sc : l1:
    exp_name = 'self_attention_high_new'
    #n_heads = [15,15,15]
    n_heads = [2,2,2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    optimizer = 'sgd'

@ex.named_config
def wos_variation_23_2():
    # sc : l1:
    exp_name = 'self_attention_high_new'
    #n_heads = [15,15,15]
    n_heads = [2,2,2]
    prev_emb = True
    use_parent_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01

@ex.named_config
def wos_variation_23_3():
    # sc : l1: 0.88815, l2: exact: 0.75875, overall: 0.75875
    exp_name = 'self_attention_high_p_false'
    #n_heads = [15,15,15]
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01

@ex.named_config
def wos_variation_23_4():
    # sc : l1: 0.890911, l2: exact: 0.78203, overall: 0.76446
    exp_name = 'self_attention_high_p_false_true'
    #n_heads = [15,15,15]
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    use_parent_emb = True

@ex.named_config
def wos_variation_23_5():
    # sc :
    exp_name = 'self_attention_vlow_p_false_true'
    #n_heads = [15,15,15]
    n_heads = [1,1,1]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.001
    use_parent_emb = True





@ex.named_config
def wos_variation_24():
    #
    exp_name = 'self_attention_high_treemask'
    # n_heads = [15,15,15]
    n_heads = [4, 4, 4]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    renormalize = 'category'

### Ablation Studies
@ex.named_config
def wos_variation_25():
    # score: l1: 0.88826, l2: 0.75970
    exp_name = 'self_attention_high_single'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    multi_class = False

@ex.named_config
def wos_variation_26():
    # score:
    exp_name = 'self_attention_high_single3'
    # n_heads = [15,15,15]
    n_heads = [3, 3, 3]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    multi_class = False
    dropout = 0.4

@ex.named_config
def wos_variation_27():
    # score:
    exp_name = 'self_attention_high_multi_10'
    # n_heads = [15,15,15]
    n_heads = [10, 10, 10]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    dropout = 0.4

@ex.named_config
def wos_variation_28():
    # score: l1: 0.87715, l2: 0.7727
    exp_name = 'wos_concatpool_nornn'
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    use_rnn = False
    use_embedding = True

@ex.named_config
def wos_variation_28_1():
    # score:
    exp_name = 'wos_maxpool'
    model_type = 'pooling'
    prev_emb = True
    attention_type = 'maxpool'
    cat_emb_dim = 300
    da = 400
    n_layers = 2
    use_rnn = True
    use_embedding = True

@ex.named_config
def wos_variation_28_2():
    # score:
    exp_name = 'wos_meanpool'
    model_type = 'pooling'
    prev_emb = True
    attention_type = 'meanpool'
    cat_emb_dim = 300
    da = 400
    n_layers = 2
    use_rnn = True
    use_embedding = True


@ex.named_config
def wos_variation_29():
    # score: l1: 87.36, l2: 75.98
    # with cat_emb prev: l1: 0.8739, l2: 0.7616
    exp_name = 'wos_concatpool_nornn'
    n_heads = [15,15,15]
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 5
    n_layers = 2
    use_rnn = False
    use_embedding = True

@ex.named_config
def wos_variation_29_1():
    # score: l1: 87.36, l2: 75.98
    exp_name = 'wos_concatpool_rnn'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 5
    n_layers = 2
    use_embedding = True

@ex.named_config
def wos_variation_30():
    # score:
    exp_name = 'wos_concatpool_nornn_fixed'
    n_heads = [15, 15, 15]
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 5
    n_layers = 2
    use_rnn = False
    use_embedding = True
    fix_embeddings = True

@ex.named_config
def wos_variation_31():
    # score: l1: 81.2, l3: 41.8
    exp_name = 'wos_meanpool_nornn'
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'meanpool'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    use_rnn = False
    use_embedding = True

@ex.named_config
def wos_variation_32():
    # score: l1: 0.88805, l2: 0.75917
    exp_name = 'self_attention_high_prev_emb'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.001

@ex.named_config
def wos_variation_32_1():
    # score: l1: 0.88572, l2: 0.75335
    exp_name = 'self_attention_high_prev_emb_single'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    multi_class = False
    lr = 0.001

@ex.named_config
def wos_variation_32_2():
    # score: l1: 0.89323, l2: 0.77462
    exp_name = 'self_attention_high_prev_emb_detach'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    attn_penalty_coeff = 0
    lr = 0.001
    detach_encoder = True


@ex.named_config
def wos_variation_33():
    # score: l1: 0.889535, l2: 0.750820 (e & o)
    exp_name = 'self_attention_prev_false'
    #n_heads = [15,15,15]
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.001

@ex.named_config
def wos_variation_34():
    #score: l1: 0.88710, l2: 0.541318
    exp_name = 'self_attention_temp_0.5'
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    temperature = 0.5

@ex.named_config
def wos_variation_34_1():
    exp_name = 'self_attention_temp_0.5_cat'
    n_heads = [2,2,2]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    temperature = 0.5
    renormalize = 'category'

@ex.named_config
def wos_variation_34_2():
    exp_name = 'self_attention_temp_only_0.5_cat'
    n_heads = [2,2,2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    temperature = 0.5
    renormalize = 'category'

@ex.named_config
def wos_variation_35():
    #score: l1: 0.8885, l2: 0.75949
    exp_name = 'self_attention_self_mistake'
    n_heads = [2,2,2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    lr = 0.01
    teacher_forcing = False


@ex.named_config
def wos_variation_36():
    #score: l1: 0.877684, l2: 0.770923
    exp_name = 'wos_concatpool_prev_true'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'

@ex.named_config
def wos_variation_36_1():
    #score: l1:
    exp_name = 'wos_concatpool_prev_true_nornn'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    use_rnn = False
    teacher_forcing = True

@ex.named_config
def wos_variation_37():
    #score: l1: 0.87969, l2: 0.75156
    exp_name = 'wos_concatpool_prev_true_cat'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    renormalize = 'category'

@ex.named_config
def wos_variation_38():
    #score: l1:  0.88329, l2: 0.818643
    exp_name = 'wos_concatpool_prev_true_always_correct'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    overall = False
    use_cat_emb = True

@ex.named_config
def wos_variation_39():
    #score: l1: 0.88879, l2: 0.82425
    exp_name = 'wos_concatpool_prev_true_always_correct_small_cat'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 100
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    use_cat_emb = True

@ex.named_config
def wos_variation_40():
    #score: l1: 0.88868, l2: 0.0.817162
    exp_name = 'wos_concatpool_prev_true_always_correct_small_cat_less_param'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 100
    embedding_dim = 100
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    overall = False
    use_cat_emb = True

@ex.named_config
def wos_variation_41():
    #score: l1: 0.88868, l2: 0.817162
    exp_name = 'wos_concatpool_prev_true_always_correct_small_cat_even_less_param'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 100
    embedding_dim = 100
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    overall = False
    use_cat_emb = True
    mlp_hidden_dim = 100

@ex.named_config
def wos_variation_42():
    #score: l1: 0.8744, l2: overall:  0.76700, exact: 0.77240
    exp_name = 'wos_concatpool_with_cat_new_emb'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 100
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    use_cat_emb = True
    use_parent_emb = True

@ex.named_config
def wos_variation_42_1():
    #score: l1: 0.886361, l2: overall: 0.7728, exact: 0.809226
    exp_name = 'wos_concatpool_with_only_parent'
    prev_emb = True
    attention_type = 'concat'
    cat_emb_dim = 100
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    model_type = 'pooling'
    use_cat_emb = False
    use_parent_emb = True

@ex.named_config
def wos_variation_42_2():
    #score: l1:
    exp_name = 'self_attention_with_only_parent'
    prev_emb = True
    n_heads = [2, 2, 2]
    attention_type = 'concat'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = False
    lr_patience = 10
    n_layers = 2
    use_cat_emb = False
    use_parent_emb = True
    attn_penalty_coeff = 0
    mlp_hidden_dim = 2000

@ex.named_config
def wos_variation_43():
    # score: l1: 0.8656, l2: overall: 0.72098, exact: 0.79600
    exp_name = 'self_attention_nornn_2'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    use_rnn = False

@ex.named_config
def wos_variation_44():
    # score: l1: 0.31213, l2 : 0.07385
    exp_name = 'self_attention_final_single'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0
    multi_class = True

@ex.named_config
def wos_variation_45():
    # score: l1: 0.89154, overall: 0.7499, exact: 0.78806
    exp_name = 'self_attention_small_hops'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0

@ex.named_config
def wos_variation_46():
    # score: l1:
    exp_name = 'self_attention_large_hops'
    # n_heads = [15,15,15]
    n_heads = [15, 15, 15]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 0

@ex.named_config
def wos_variation_47():
    # score: l1:
    exp_name = 'self_attention_with_penalty'
    # n_heads = [15,15,15]
    n_heads = [2, 2, 2]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 5
    n_layers = 2
    mlp_hidden_dim = 2000
    attn_penalty_coeff = 1
    use_parent_emb = True
    lr = 0.001

### Thesis ablation study
# Attention
#   Without previous layer encoding                 wos_variation_33
#	Without BiLSTM encoder - pure attention         wos_variation_43
#	With single final classifier                    wos_variation_44
#	With special parent encoding                    wos_variation_42_1
#	With detached BiLSTM encoder                    wos_variation_32_2
#	With low attention hops - 2                     wos_variation_45
#	With high attention hops - 15                   wos_variation_46
#   With attention penalty                          wos_variation_47
# Pooling
#   Without attention - max pooling                 wos_variation_28_1
#	Without attention - mean pooling                wos_variation_28_2
#	Without attention - concat pooling              wos_variation_36
#	Without BiLSTM encoder - pure concat pooling    wos_variation_36_1


### DBpedia experiments

@ex.named_config
def dbp_base():
    data_loc = '/home/ml/ksinha4/mlp/hier-class/data/'
    data_path = 'db_sm_train_pruned'
    file_name = 'df_small_train.csv'
    test_file_name = 'df_small_test.csv'
    test_output_name = 'df_small_outp.csv'
    levels = 3
    weight_decay = 1e-6

@ex.named_config
def dbp_variation_1():
    # score: l1-0.99129, l2-0.96161, l3-0.93606
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
    # score:
    exp_name = 'db_scaled_attention'
    prev_emb = False
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [8, 8, 8]
    batch_size=16

@ex.named_config
def dbp_variation_4():
    # slurm: 4759360
    exp_name = 'db_scaled_attention_2'
    prev_emb = False
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [4, 4, 4]
    lr_patience = 30

@ex.named_config
def dbp_variation_5():
    # slurm: 4759364
    exp_name = 'db_scaled_attention_2'
    prev_emb = True
    attention_type = 'scaled'
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

@ex.named_config
def dbp_variation_7():
    # score: l1-0.9933, l2-0.9681, l3-0.9458
    # slurm: 4759325
    exp_name = 'db_maxpool_classifier'
    prev_emb = True
    attention_type = 'no_attention'
    n_layers = 2

@ex.named_config
def dbp_variation_8():
    # score:
    # slurm: 4759348
    exp_name = 'db_self_attention_new_single'
    n_heads = [2,5,8]
    prev_emb = True
    attention_type = 'self'
    cat_emb_dim = 300
    da = 350
    single_attention = True
    attn_penalty = True
    clean = True

@ex.named_config
def dbp_variation_9():
    # score:
    exp_name = 'db_scaled_attention_small'
    prev_emb = False
    attention_type = 'scaled'
    clean = True
    single_attention = True
    n_heads = [1, 1, 1]

@ex.named_config
def dbp_variation_10():
    # score: l1: 0.9918, l2: 0.9588, l3: 0.9200
    exp_name = 'db_scaled_attention_original'
    prev_emb = True
    attention_type = 'scaled'
    clean = True
    single_attention = False
    attn_penalty=True
    n_heads = [2, 2, 8]

@ex.named_config
def dbp_variation_11():
    # slurm: 4763088,
    exp_name = 'db_self_layer_2'
    prev_emb = True
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 4]
    n_layers = 2

@ex.named_config
def dbp_variation_12():
    # slurm: 4763291
    exp_name = 'db_self_layer_2_pf'
    prev_emb = False
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 4]
    n_layers = 2

@ex.named_config
def dbp_variation_13():
    # slurm: 4773153
    exp_name = 'db_self_layer_2_pf_penal'
    prev_emb = False
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 4]
    attn_penalty=True
    n_layers = 2

@ex.named_config
def dbp_variation_14():
    # slurm: 4763320, datamachine
    exp_name = 'db_self_layer_2_penal'
    prev_emb = True
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 4]
    n_layers = 2
    attn_penalty=True

@ex.named_config
def dbp_variation_15():
    # slurm: 4773065
    exp_name = 'db_self_layer_2_high'
    prev_emb = True
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 8]
    n_layers = 2

@ex.named_config
def dbp_variation_16():
    # slurm: 4773072
    exp_name = 'db_self_layer_2_high_penal'
    prev_emb = True
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 8]
    n_layers = 2
    attn_penalty=True

@ex.named_config
def dbp_variation_17():
    # slurm: 4773080
    exp_name = 'db_self_layer_2_high_pf'
    prev_emb = False
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 8]
    n_layers = 2

@ex.named_config
def dbp_variation_18():
    # slurm: 4773102
    exp_name = 'db_self_layer_2_high_pf_penal'
    prev_emb = False
    attention_type = 'self'
    single_attention = True
    n_heads = [4, 4, 8]
    n_layers = 2
    attn_penalty = True

@ex.named_config
def dbp_variation_19():
    # score: l1:0.9916, l2: 0.9667, l3: 0.9479
    exp_name = 'db_scaled_masked'
    prev_emb = True
    attention_type = 'scaled'
    single_attention = False
    n_heads = [2, 2, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_20():
    # score: l1: 0.992, l2: 0.9665, l3: 0.9438
    exp_name = 'dbp_self_attention_high'
    n_heads = [15,15,15]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10
    n_layers = 2

@ex.named_config
def dbp_variation_21():
    # score:
    exp_name = 'db_scaled_masked_single'
    prev_emb = True
    attention_type = 'scaled'
    single_attention = True
    n_heads = [2, 2, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_22():
    # score:
    exp_name = 'db_scaled_masked_single_high'
    prev_emb = True
    attention_type = 'scaled'
    single_attention = True
    n_heads = [2, 2, 10]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True
    mlp_hidden_dim = 50

@ex.named_config
def dbp_variation_23():
    # score: l1: 0.9920, l2: 0.96641, l3: 0.94568
    exp_name = 'db_scaled_masked_single_noprev'
    prev_emb = False
    attention_type = 'scaled'
    single_attention = True
    n_heads = [2, 2, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_24():
    # score:
    exp_name = 'db_scaled_masked_multi_noprev'
    prev_emb = False
    attention_type = 'scaled'
    single_attention = False
    n_heads = [2, 4, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_25():
    # score:
    exp_name = 'db_scaled_masked_multi'
    prev_emb = True
    attention_type = 'scaled'
    single_attention = False
    n_heads = [2, 4, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_26():
    # score:
    exp_name = 'db_scaled_masked_single_prev'
    prev_emb = True
    attention_type = 'scaled'
    single_attention = True
    n_heads = [2, 2, 8]
    n_layers = 2
    attn_penalty = True
    lr_patience = 5
    use_attn_mask = True

@ex.named_config
def dbp_variation_27():
    #
    exp_name = 'dbp_self_attention_medium'
    n_heads = [8,8,8]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10
    n_layers = 2

@ex.named_config
def dbp_variation_28():
    #
    exp_name = 'dbp_self_attention_low'
    n_heads = [5,5,5]
    prev_emb = False
    attention_type = 'self'
    cat_emb_dim = 300
    da = 400
    single_attention = True
    attn_penalty = True
    lr_patience = 10
    n_layers = 2
