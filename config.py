import argparse

parser = argparse.ArgumentParser()

# path argument
parser.add_argument("--data_folder",default=None,type=str,required=True,help="CoQA data folder")
parser.add_argument("--bert_dir",default=None,type=str,required=True,help="BERT pretrained model directory")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--train_filename",default="coqa-train-v1.0.json",type=str,help="CoQA trianset")
parser.add_argument("--dev_filename",default="coqa-dev-v1.0.json",type=str,help="CoQA devset")

# model argument
parser.add_argument("--max_seq_length",default=512,type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will ")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--do_restore", action='store_true', help="Whether to restore model for resuming training.")
parser.add_argument("--train_batch_size", default=20, type=int, help="Total batch size for training.")
parser.add_argument("--predict_batch_size", default=180, type=int, help="Total batch size for predictions.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=2, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                         "of training.")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--max_answer_length", default=30, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
# new add
parser.add_argument('--ema',
                    action='store_true',
                    help="Whether to use EMA")
parser.add_argument('--lr_layer_decay_rate',
                    type=float,
                    default=0.9,
                    help="learning rate layer-wise decay rate")
parser.add_argument('--em_loss',
                    action='store_true',
                    help="Whether to add EM loss for AT")
parser.add_argument('--vat_loss',
                    action='store_true',
                    help="Whether to add VAT loss for AT")
parser.add_argument('--store_step',
                    action='store_true',
                    help='store model param during training')
parser.add_argument('--l_at',
                    type=float,
                    default=1.0,
                    help='lambda_AT')
parser.add_argument('--l_em',
                    type=float,
                    default=1.0,
                    help='lambda_EM')
parser.add_argument('--l_vat',
                    type=float,
                    default=1.0,
                    help='lambda_VAT')
parser.add_argument('--epsilon_at',
                    type=float,
                    default=1.0,
                    help='epsilon_AT')
parser.add_argument('--epsilon_vat',
                    type=float,
                    default=1.0,
                    help='epsilon_VAT')
parser.add_argument('--xi',
                    type=float,
                    default=1.0,
                    help='xi used in VAT')

args= parser.parse_args()

