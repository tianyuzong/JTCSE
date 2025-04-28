import argparse
from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import TrainDataset, TestDataset
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import pandas as pd
import time
from model_dual import unsup_infonce_loss, icnce, LTN_loss
from eval_jtcse import eval_main
from transformers import AutoTokenizer, AutoModel
from transformers_cp.models.bert.modeling_bert import My_BertModel, BertModel


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, optimizer, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]

            # if "bert" in args.pretrain_tokenizer:
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)
            jtcse_out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              hidden_states_add_tag=True)

            last_hidden_state = jtcse_out.last_hidden_state[:, 0, :]
            pooler_output = jtcse_out.pooler_output

            last_hidden_state_add = jtcse_out.last_hidden_state_add[:, 0, :]
            pooler_output_add = jtcse_out.pooler_output_add

            temp_1 = jtcse_out.temp_1
            temp_2 = jtcse_out.temp_2
            L_nce = unsup_infonce_loss(last_hidden_state, device) + unsup_infonce_loss(last_hidden_state_add, device)

            random_data = random.randint(0, 1)
            # Randomly provide positive samples to each other
            if random_data == 0:
                L_icnce = icnce(temp_1[:, 0, :], temp_2[:, 0, :], device) + icnce(last_hidden_state,
                                                                                  last_hidden_state_add, device)
            else:
                L_icnce = icnce(temp_2[:, 0, :], temp_1[:, 0, :], device) + icnce(last_hidden_state_add,
                                                                                  last_hidden_state, device)

            L_ictm = LTN_loss(last_hidden_state, last_hidden_state_add, pooler_output, pooler_output_add,
                              device) + LTN_loss(last_hidden_state_add, last_hidden_state, pooler_output_add,
                                                 pooler_output, device)

            loss = L_nce + L_icnce + L_ictm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                corrcoef = evaluate(model, dev_loader, device, args.pretrain_tokenizer)
                logger.info('loss:{}, corrcoef: {} in step {} epoch {}'.format(loss, corrcoef, step, epoch))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'simcse.pt'))
                    logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch))


def evaluate(model, dataloader, device, type):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            # if "bert" in type:
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(input_ids=source_input_ids, attention_mask=source_attention_mask,
                                token_type_ids=source_token_type_ids, hidden_states_add_tag=True)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(input_ids=target_input_ids, attention_mask=target_attention_mask,
                                token_type_ids=target_token_type_ids, hidden_states_add_tag=True)

            sim = F.cosine_similarity(
                source_pred.last_hidden_state[:, 0, :] + source_pred.last_hidden_state_add[:, 0, :],
                target_pred.last_hidden_state[:, 0, :] + target_pred.last_hidden_state_add[:, 0, :],
                dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def load_train_data_unsupervised(tokenizer, args):
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            feature = tokenizer([line, line], max_length=args.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
            feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_train_data_supervised(tokenizer, args):
    logger.info('loading supervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-supervised.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    df = pd.read_csv(args.train_file, sep=',')
    logger.info("len of train data:{}".format(len(df)))
    rows = df.to_dict('reocrds')
    for row in tqdm(rows):
        sent0 = row['sent0']
        sent1 = row['sent1']
        hard_neg = row['hard_neg']
        feature = tokenizer([sent0, sent1, hard_neg], max_length=args.max_len, truncation=True, padding='max_length',
                            return_tensors='pt')
        feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_eval_data(tokenizer, args, mode):
    assert mode in ['dev', 'test'], 'mode should in ["dev", "test"]'
    logger.info('loading {} data'.format(mode))
    output_path = os.path.dirname(args.output_path)
    eval_file_cache = join(output_path, '{}.pkl'.format(mode))
    if os.path.exists(eval_file_cache) and not args.overwrite_cache:
        with open(eval_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of {} data:{}".format(mode, len(feature_list)))
            return feature_list

    if mode == 'dev':
        eval_file = args.dev_file
    else:
        eval_file = args.test_file
    feature_list = []
    with open(eval_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of {} data:{}".format(mode, len(lines)))
        for line in tqdm(lines):
            line = line.strip().split("\t")
            assert len(line) == 7 or len(line) == 9
            score = float(line[4])
            data1 = tokenizer(line[5].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
            data2 = tokenizer(line[6].strip(), max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')

            feature_list.append((data1, data2, score))
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = My_BertModel.from_pretrained(args.pretrain_tokenizer, calayers_num=args.num_ca_layers).to(device)
    logger.info(get_parameter_number(model))
    sub_bert1 = AutoModel.from_pretrained(args.pretrain_subencoder_1).to(device)
    sub_bert2 = AutoModel.from_pretrained(args.pretrain_subencoder_2).to(device)

    model.embeddings.load_state_dict(sub_bert1.embeddings.state_dict())
    model.encoder.layer.load_state_dict(sub_bert1.encoder.layer.state_dict())
    model.pooler.load_state_dict(sub_bert1.pooler.state_dict())

    model.embeddings_add.load_state_dict(sub_bert2.embeddings.state_dict())
    model.pooler_add.load_state_dict(sub_bert2.pooler.state_dict())
    model.encoder.layer_add.load_state_dict(sub_bert2.encoder.layer.state_dict())
    logger.info('Loading Successful!')
    logger.info(args.output_path)

    if args.do_train:
        assert args.train_mode in ['supervise', 'unsupervise'], \
            "train_mode should in ['supervise', 'unsupervise']"
        if args.train_mode == 'supervise':
            train_data = load_train_data_supervised(tokenizer, args)
        elif args.train_mode == 'unsupervise':
            train_data = load_train_data_unsupervised(tokenizer, args)
        train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = load_eval_data(tokenizer, args, 'dev')
        dev_dataset = TestDataset(dev_data, tokenizer, max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                    num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, train_dataloader, dev_dataloader, optimizer, args)
    if args.do_predict:
        test_data = load_eval_data(tokenizer, args, 'test')
        test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                     num_workers=args.num_workers)
        model.load_state_dict(torch.load(join(args.output_path, 'simcse.pt')))
        model.eval()
        corrcoef = evaluate(model, test_dataloader, args.device, args.pretrain_tokenizer)
        logger.info('testset corrcoef:{}'.format(corrcoef))
        eval_main(tokenizer, model, args.device)
    logger.info("begin saving")
    sub_bert1.embeddings.load_state_dict(model.embeddings.state_dict())
    sub_bert1.encoder.layer.load_state_dict(model.encoder.layer.state_dict())
    sub_bert1.pooler.load_state_dict(model.pooler.state_dict())

    sub_bert2.embeddings.load_state_dict(model.embeddings_add.state_dict())
    sub_bert2.encoder.layer.load_state_dict(model.encoder.layer_add.state_dict())
    sub_bert2.pooler.load_state_dict(model.pooler_add.state_dict())
    logger.info("Saved done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='JTCSE_BERT_OUTPUT')
    parser.add_argument("--lr", type=float, default=1.3e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_ca_layers", type=int, default=6, choices=[1, 2, 3, 4, 6, 12],
                        help="Equally spaced cross-attention layers, need to be divisible by 12 EncoderLayers")
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--train_file", type=str, default="data/Wiki_for_JTCSE.txt")
    parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
    parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")
    parser.add_argument("--pretrain_subencoder_1", type=str,
                        default="Pretrained_encoder/BERT/JTCSE_bert_pretrained_encoder_1")
    parser.add_argument("--pretrain_subencoder_2", type=str,
                        default="Pretrained_encoder/BERT/JTCSE_bert_pretrained_encoder_2")
    parser.add_argument("--pretrain_tokenizer", type=str,
                        default="bert-base-uncased/")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='pooler to use')
    parser.add_argument("--train_mode", type=str, default='unsupervise', choices=['unsupervise', 'supervise'],
                        help="unsupervise or supervise")
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=True)

    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, args.train_mode, args.pretrain_tokenizer,
                            'bsz-{}-lr-{}'.format(args.batch_size_train, args.lr))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    main(args)
