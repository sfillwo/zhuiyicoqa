import re
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
from config import args
from logger import config_logger
from batch_generator import BatchGenerator

from transformers import RobertaTokenizer, RobertaTokenizerFast

from main_advtrain import _to_CoQA_Pred

MODEL_NAME = 'model.bin'
logger = config_logger("MAIN", "INFO")

article_text = """rapid repair hair mask
## what is it? We developed this product with JLo herself, so that you can experience how she repairs and hydrates her strands in your very own shower. This **[luxurious hair mask](https://www.forhers.com/hair-care/hair-mask)** is designed to not only be hydrating, but a luxurious scent experience as well. ## how do i apply it? Massage into your hair from root to tip and leave on for 2 to 5 minutes, twice a week. ## what’s in it? Key ingredients: * Coconut oil, shea butter, jojoba seed oil, murumuru seed butter, and cupuacu butter ## common ingredient questions: * No parabens * No phthalates * No sulfates * No mineral oil * Cruelty-free
"""

# todo - auto read and generate texts from file of tests
# maintain spacing, but reduces newlines
# remove extra formatting?

def _prep(content):
    shrunk_content = re.sub("[ \t]+", " ", content)
    shrunk_content = re.sub("\n *\n *\n+[\n ]*", "\n\n", shrunk_content)
    if " " in shrunk_content:
        shrunk_content = shrunk_content.replace(" ", " ")
    return shrunk_content

INTERACTIVE = False

if __name__ == '__main__':
    logger.info("args:{}".format(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    roberta = RobertaModel.from_pretrained(args.bert_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    coqa_reader = CoQAReader(-1)
    bert_data_helper = BertDataHelper(args.bert_dir, roberta=roberta_tokenizer, do_lowercase=False)

    def run_model(data):
        data = bert_data_helper.convert(data)

        batch_generator = BatchGenerator(data)
        dataloader = batch_generator(batch_size=args.predict_batch_size, is_train=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        model = RoBERTaCoQA(roberta=roberta)

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
        for batch in tqdm(dataloader, desc='Prediction...'):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            with torch.no_grad():
                loss, output = model(batch, answer_verification=True)
            for key in output.keys():
                final_output[key] += [i for i in output[key]]
        result = get_best_answer(final_output, data)
        final_result = _to_CoQA_Pred(result)
        context["data"][0]["answers"][-1]["input_text"] = final_result[0]["answer"]
        print(json.dumps(final_result, indent=2))
        print()

    if INTERACTIVE:
        context = {"data": [{"id": "0", "story": article_text, "questions": [], "answers": []}]}
        while True:
            data = coqa_reader.get_input(context)
            run_model(data)
    else:
        with open("data/eval.json") as f:
            eval = json.load(f)
        for num_docs, d in eval.items():
            context = {"data": []}
            docs = d["docs"]
            utters = d["utterances"]
            story = "\n".join([f"{doc['article']}\n{_prep(doc['text'])}" for doc in docs])
            context["data"].append({"id": num_docs, "story": story, "questions": [], "answers": []})
            for turn_id, utter in enumerate(utters):
                context["data"][-1]["questions"].append({"input_text": utter, "turn_id": turn_id+1})
                context["data"][-1]["answers"].append({'span_start': 0, 'span_end': 1, 'span_text': 'xxx', 'input_text': 'yyy', 'turn_id': turn_id+1})
                data = coqa_reader._read(context, "test")
                run_model(data)

