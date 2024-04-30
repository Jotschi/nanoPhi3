# train a miniature kleiner_astronaut model

out_dir = 'out-kleiner_astronaut'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 500
log_interval = 20 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'kleiner_astronaut'
wandb_run_name = 'mini-gpt2'

dataset = 'kleiner_astronaut'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 780
dropout = 0.2

learning_rate = 1e-5 # with baby networks can afford to go a bit higher
max_iters = 500000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
