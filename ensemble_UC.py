import torch.nn as nn
import sys
import logging
import argparse
from prettytable import PrettyTable
import torch
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main_eval_uc(token, token_UC, model_teacher, args):
    # Load transformers' model checkpoint
    model = model_teacher
    tokenizer = token
    tokenizer_UC = token_UC
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            batch_UC = tokenizer_UC.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            batch_UC = tokenizer_UC.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
            batch_UC[k] = batch_UC[k].to(device)
        # Get raw embeddings
        with torch.no_grad():
            if model_teacher.model_type == 'BERT':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, input_ids_UC=input_ids,
                                attention_mask_UC=attention_mask, token_type_ids_UC=token_type_ids)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                input_ids_UC = batch_UC["input_ids"].to(device)
                attention_mask_UC = batch_UC["attention_mask"].to(device)
                token_type_ids_UC = batch_UC["token_type_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, input_ids_UC=input_ids_UC,
                                attention_mask_UC=attention_mask_UC, token_type_ids_UC=token_type_ids_UC)
        return outputs.cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


class Ensemble_UC(nn.Module):
    def __init__(self, dual1_path, dual2_path, UC_path, weight, model_type):
        super(Ensemble_UC, self).__init__()
        self.dual1_path = dual1_path
        self.dual2_path = dual2_path
        self.UC_path = UC_path
        self.weight = weight
        self.EncoderI = AutoModel.from_pretrained(dual1_path)
        self.EncoderII = AutoModel.from_pretrained(dual2_path)
        self.UC = AutoModel.from_pretrained(UC_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dual1_path)
        self.tokenizer_UC = AutoTokenizer.from_pretrained(UC_path)
        self.model_type = model_type

    def forward(self, input_ids=None, attention_mask=None, input_ids_UC=None, attention_mask_UC=None,
                token_type_ids_UC=None):
        if self.model_type == 'BERT':
            outputs_1 = self.EncoderI(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids_UC,
                                      output_hidden_states=True,
                                      return_dict=True).last_hidden_state[:, 0]
            outputs_2 = self.EncoderII(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids_UC,
                                       output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
            outputs_UC = self.UC(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids_UC,
                                 output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
            return (1.0 - self.weight) * (outputs_1 + outputs_2) + self.weight * outputs_UC
        else:
            outputs_1 = self.EncoderI(input_ids=input_ids, attention_mask=attention_mask,
                                      output_hidden_states=True,
                                      return_dict=True).last_hidden_state[:, 0]
            outputs_2 = self.EncoderII(input_ids=input_ids, attention_mask=attention_mask,
                                       output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
            outputs_UC = self.UC(input_ids=input_ids_UC, attention_mask=attention_mask_UC,
                                 token_type_ids=token_type_ids_UC,
                                 output_hidden_states=True, return_dict=True).last_hidden_state[:, 0]
            return (1.0 - self.weight) * (outputs_1 + outputs_2) + self.weight * outputs_UC


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='BERT', choices=['BERT', 'RoBERTa'])
    parser.add_argument("--model_path1", type=str)
    parser.add_argument("--model_path2", type=str)
    parser.add_argument("--ensemble_UC", type=str)
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")

    args = parser.parse_args()
    if args.model_type == 'BERT':
        JTCSE_UC = Ensemble_UC(dual1_path=args.model_path1, dual2_path=args.model_path2,
                               UC_path=args.ensemble_UC,
                               weight=0.7, model_type=args.model_type)

    else:
        JTCSE_UC = Ensemble_UC(dual1_path=args.model_path1, dual2_path=args.model_path2,
                               UC_path=args.ensemble_UC,
                               weight=0.7, model_type=args.model_type)

    main_eval_uc(token=JTCSE_UC.tokenizer, token_UC=JTCSE_UC.tokenizer_UC, model_teacher=JTCSE_UC, args=args)
