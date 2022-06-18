import os
import pickle
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from fairseq.models.roberta import RobertaModel
import json
from coqa_utils import CoQAReader, CoQAEvaluator, get_best_answer, RunningAverage, EMA
from BertWrapper import BertDataHelper
from model import RoBERTaCoQA
from batch_generator import BatchGenerator
from config import args
from logger import config_logger
from optimization import BertAdam,WarmupLinearSchedule

from transformers import RobertaTokenizer, RobertaTokenizerFast

MODEL_NAME = 'model.bin'
logger = config_logger("MAIN", "INFO")

def read_data(roberta):
    train_data,dev_data = [],[]
    coqa_reader = CoQAReader(-1)
    bert_data_helper = BertDataHelper(args.bert_dir, roberta=roberta, do_lowercase=False)

    if args.do_train:
        cached_train_feature_file = args.train_filename + '_{0}_{1}_{2}'.format(
            str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        if os.path.isfile(cached_train_feature_file):
            with open(cached_train_feature_file, "rb") as fh:
                logger.info("Open cached_train_feature_file,{}".format(cached_train_feature_file))
                train_data = pickle.load(fh)
        else:
            train_data = coqa_reader.read(args.data_folder + args.train_filename, 'train')
            train_data = bert_data_helper.convert(train_data)
            # logger.info("Saving cached_train_feature_file,{}".format(cached_train_feature_file))
            # with open(cached_train_feature_file, "wb") as fh:
                # pickle.dump(train_data, fh)
    if args.do_train or args.do_predict:
        cached_dev_feature_file = args.dev_filename + '_{0}_{1}_{2}'.format(
            str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        if os.path.isfile(cached_dev_feature_file):
            with open(cached_dev_feature_file, "rb") as fh:
                logger.info("Open cached_dev_feature_file,{}".format(cached_dev_feature_file))
                dev_data = pickle.load(fh)
        else:
            dev_data = coqa_reader.read(args.data_folder + args.dev_filename, 'dev')
            dev_data = bert_data_helper.convert(dev_data)
            # logger.info("Saving cached_dev_feature_file,{}".format(cached_dev_feature_file))
            # with open(cached_dev_feature_file, "wb") as fh:
                # pickle.dump(dev_data, fh)
        # dev_data = coqa_reader.read(args.data_folder + args.dev_filename, 'dev')
        # dev_data = bert_data_helper.convert(dev_data)
    return train_data,dev_data


def create_loader(train_data, dev_data):
    train_dataloader,dev_dataloader = None,None
    if args.do_train:
        train_batch_generator = BatchGenerator(train_data)
        train_dataloader = train_batch_generator(batch_size=args.train_batch_size, is_train=True, drop_last=True)

    dev_batch_generator = BatchGenerator(dev_data)
    dev_dataloader = dev_batch_generator(batch_size=args.predict_batch_size, is_train=False)
    return train_dataloader,dev_dataloader


def create_optimizer(optimizer_grouped_parameters, num_train_steps):
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps,
                         lr_layer_decay_rate=args.lr_layer_decay_rate)
    return optimizer


def main():
    logger.info("args:{}".format(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    roberta = RobertaModel.from_pretrained(args.bert_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    train_data, dev_data = read_data(roberta_tokenizer)
    # train_data = train_data[:300]
    # dev_data = dev_data[:300]

    train_dataloader, dev_dataloader = create_loader(train_data, dev_data)
    evaluator = CoQAEvaluator(os.path.join(args.data_folder, args.dev_filename))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    model = RoBERTaCoQA(roberta=roberta)

    if args.do_train:
        model = model.to(device)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_optimizer_file = os.path.join(args.output_dir, "pytorch_optimizer.bin")
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs +1
        logger.info("total train number:{}, total dev number:{}".format(len(train_data), len(dev_data)))
        del(train_data)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'names': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'names': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = create_optimizer(optimizer_grouped_parameters,num_train_steps)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            amp.register_float_function(torch, 'sigmoid')
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if args.ema:
            ema = EMA(model, 0.99)
            ema.register()
        start_epoch = 0
        best_eval_score = 0
        global_step = 0
        for epoch in range(start_epoch, args.num_train_epochs):
            logger.info("********* Running Training *********")
            model.train()
            step_loss = 0
            batches = []
            loss_avg = RunningAverage()
            for step, batch in enumerate(train_dataloader):
                
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                # ML
                # print("train inference 1")
                loss, _ = model(batch, answer_verification=True, optim_at=False)
                if n_gpu > 1:
                    loss = loss.mean()
                loss_avg.update(loss.item())
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # accumulate ML gradient and (if necessary) EM gradient
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                step_loss += loss.item()
                batches.append(batch)
                
                # At the end of each batch, do AT, VAT
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # assert ((optimizer.at_flag == False) and  (optimizer.vat_flag == False))
                    optimizer.at_flag = True
                    step_loss_advt = 0
                    step_loss_advvat = 0
                    at_done = False
                    vat_done = False
                    # logger.info('AT step')
                    optimizer.step() # backup and apply AT to embedding layer, at_flag = False
                    if not optimizer.at_flag:
                        at_done = True
                        for batch in batches:
                            # AT
                            at_loss, _ = model(batch, answer_verification=True, optim_at=True)
                            at_loss = args.l_at * at_loss
                            if n_gpu > 1:
                                at_loss = at_loss.mean()
                            loss_avg.update(at_loss.item())
                            if args.gradient_accumulation_steps > 1:
                                at_loss = at_loss / args.gradient_accumulation_steps
                            # accumulate AT gradient
                            if args.fp16:
                                with amp.scale_loss(at_loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                at_loss.backward()
                            step_loss_advt += at_loss.item()
                        # print('restore')
                        optimizer.restore_embd() # restore embedding layer
                        # print('step')
                    # else:
                    #     optimizer.at_flag = False
                        
                        assert ((optimizer.at_flag == False) and (optimizer.vat_flag == False))
                        
                        if args.vat_loss:
                            assert(optimizer.at_flag == False and optimizer.vat_flag == False)
                            optimizer.backup_grad()
                            
                            optimizer.zero_grad()
                            
                            optimizer.vat_flag = True
                            # logger.info('VAT step')
                            optimizer.step() # backup and apply VAT to embedding layer, vat_flag = False
                            # if  not optimizer.vat_flag:
                            if not optimizer.vat_flag:
                                for batch in batches:
                                    #VAT normal noise
                                    vat_loss, _ = model(batch, answer_verification=True, loss_func = 'kl', optim_at=True)
                                    if n_gpu > 1:
                                        vat_loss = vat_loss.mean()
                                    if args.gradient_accumulation_steps > 1:
                                        vat_loss = vat_loss / args.gradient_accumulation_steps
                                    # BP turbulance gradient
                                    # '''
                                    # print("vat loss:", vat_loss)
                                    if args.fp16:
                                        with amp.scale_loss(vat_loss, optimizer) as scaled_loss:
                                            # print("vat loss:", vat_loss)
                                            scaled_loss.backward()
                                    else:
                                        vat_loss.backward()
                                    # '''
                                
                                optimizer.restore_embd()
                                
                                
                                optimizer.at_flag = True
                                # logger.info('VAT g_KL step')
                                optimizer.step() # backup and apply g_KL to embedding layer
                                
                                optimizer.restore_grad() # restore grad to that before VAT
                                if not optimizer.at_flag: # check if fp16 fails
                                    vat_done = True
                                    for batch in batches:
                                        assert(optimizer.vat_flag == False)
                                        assert(optimizer.at_flag == False)
                                        vat_loss, _ = model(batch, answer_verification=True, loss_func='kl', optim_at=True)
                                        vat_loss = args.l_vat * vat_loss
                                        if n_gpu > 1:
                                            vat_loss = vat_loss.mean()
                                        loss_avg.update(vat_loss.item())
                                        if args.gradient_accumulation_steps > 1:
                                            vat_loss = vat_loss / args.gradient_accumulation_steps
                                        # accumulate VAT gradient
                                        if args.fp16:
                                            with amp.scale_loss(vat_loss, optimizer) as scaled_loss:
                                                scaled_loss.backward()
                                        else:
                                            vat_loss.backward()
                                        step_loss_advvat += vat_loss.item()
                                    optimizer.restore_embd()
                                else:
                                    optimizer.at_flag=False
                            else:
                                optimizer.vat_flag = False

                    # assert ((optimizer.at_flag == False) and (optimizer.vat_flag == False))
                    optimizer.at_flag = False
                    optimizer.vat_flag = False
                    # logger.info('loss step')
                    # optimizer.backup_grad()
                    # optimizer.zero_grad()
                    # optimizer.restore_grad()
                    # update parameters
                    if not args.vat_loss:
                        vat_done = True
                    if at_done and vat_done:
                        optimizer.step()
                    optimizer.zero_grad()
                    batches = []

                    if args.vat_loss:
                        model.clear_backup()
                    
                    # print()
                    if args.ema:
                        ema.update()
                    global_step +=1
                    if(step + 1) %(100 * args.gradient_accumulation_steps) == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        # If we save using the predefined names, we can load using `from_pretrained`
                        output_model_file = os.path.join(args.output_dir, MODEL_NAME+'_{}'.format(int((step+1)/args.gradient_accumulation_steps)))
                        torch.save(model_to_save.state_dict(), output_model_file)
                    if (step+1) % (10*args.gradient_accumulation_steps) == 0:
                        
                        logger.info("step: {}/{}/{}, loss: {:.4f}, at_loss: {:.4f}, vat_loss: {:.4f}, loss_avg: {:.4f}".format(int((step+1)/args.gradient_accumulation_steps),
                                                 int(num_train_steps/args.num_train_epochs),
                                                 epoch,
                                                 step_loss,
                                                 step_loss_advt,
                                                 step_loss_advvat,
                                                 loss_avg()))
                    step_loss = 0
                    step_loss_advt = 0
                    step_loss_advvat = 0

            model.eval()
            if args.ema:
                ema.apply_shadow()
            final_output = defaultdict(list)
            logger.info("********* Running Prediction *********")
            for batch in tqdm(dev_dataloader, desc='Prediction...'):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                with torch.no_grad():
                    loss, output = model(batch, answer_verification=True)
                for key in output.keys():
                    final_output[key] += [i for i in output[key]]
            if args.ema:
                ema.restore()
            result = get_best_answer(final_output, dev_data)
            scores = evaluator.get_score(result)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in scores.items())
            logger.info("- Eval metrics: " + metrics_string)
            # save best model
            eval_score = scores[evaluator.get_monitor()]
            if eval_score > best_eval_score:
                # Save a trained model, configuration, vocab
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(args.output_dir, MODEL_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                best_eval_score = eval_score
            torch.cuda.empty_cache()

    if args.do_predict:
        # When only predict: do not modify bert_dir, modify the output_dir to the model you wanna use instead
        if not torch.cuda.is_available():
            model_state_dict = torch.load(os.path.join(args.output_dir, MODEL_NAME), map_location=torch.device('cpu'))
        else:
            model_state_dict = torch.load(os.path.join(args.output_dir, MODEL_NAME))
        model.load_state_dict(model_state_dict, strict=False)
        model = model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.eval()
        final_output = defaultdict(list)
        logger.info("********* Running Prediction *********")
        for batch in tqdm(dev_dataloader, desc='Prediction...'):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            with torch.no_grad():
                loss, output = model(batch, answer_verification=True)
            for key in output.keys():
                final_output[key] += [i for i in output[key]]
        result = get_best_answer(final_output, dev_data)
        try:
            with open(args.output_dir+'prediction.json', 'w') as fh:
                fh.write(json.dumps(_to_CoQA_Pred(result), indent=2))
        except:
            print("export predictions fail!")
        scores = evaluator.get_score(result)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in scores.items())
        logger.info("- Eval metrics: " + metrics_string)
def _to_CoQA_Pred(pred_dict : dict):
    CoQA_pred = []
    for key in pred_dict:
        t = dict()
        t['id'] = key[0]
        t['turn_id'] = int(key[1])
        t['answer'] = pred_dict[key]
        CoQA_pred.append(t)
    return CoQA_pred

if __name__ == "__main__":
    main()












