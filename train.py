import json
import logging
import os
import numpy as np
import data_process
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import PCL
from argparse import ArgumentParser


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    logger = logging.getLogger('Training FewRel')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.DEBUG)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHeader)
    return logger


def set_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, help="temperature of softmax")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int, help='training epochs')
    parser.add_argument("--lr", type=float, help='learning rate')
    parser.add_argument("--unseen", type=int, help='Number of unseen class')
    # file_path
    parser.add_argument("--dataset_path", type=str, help='where data stored')
    parser.add_argument("--dataset", type=str, default='fewrel', choices=['fewrel', 'wikizsl'], help='original dataset')
    parser.add_argument("--rel2id_path", type=str)
    parser.add_argument("--rel_split_seed", type=str)
    parser.add_argument("--relation_description", type=str, help='relation descriptions of manual design')
    # model and cuda config
    parser.add_argument("--visible_device", type=str, default='0', help='the device on which this model will run')
    parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased',
                        help='huggingface pretrained model')
    parser.add_argument("--sentence_model", type=str, default='stsb-bert-base', help='huggingface pretrained model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    # dataset, relation_description, rel2id path
    args.log_path = f'./log/split_{args.rel_split_seed}_ratio_{str(args.neg_ratio)}.log'
    args.dataset_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
    args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                                  args.relation_description)
    args.rel2id_file = os.path.join(args.rel2id_path, f'{args.dataset}_rel2id',
                                    f'{args.dataset}_rel2id_{str(args.unseen)}_{args.rel_split_seed}.json')

    # set seed
    set_seed(args.seed)
    logger = set_logger(args.log_path)
    logger.info(f'log_file_path: {args.log_path}')

    # cuda set
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device

    with open(args.rel2id_file, 'r', encoding='utf-8') as r2id:
        relation2idx = json.load(r2id)
        train_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['test']
        train_idx2relation, test_idx2relation, = dict((v, k) for k, v in train_relation2idx.items()), \
                                                 dict((v, k) for k, v in test_relation2idx.items())

    train_label, test_label = list(train_relation2idx.keys()), list(test_relation2idx.keys())

    # load rel_description
    with open(args.relation_description_file, 'r', encoding='utf-8') as rd:
        relation_desc = json.load(rd)
        train_desc = [i for i in relation_desc if i['relation'] in train_label]
        test_desc = [i for i in relation_desc if i['relation'] in test_label]
    # load data
    with open(args.dataset_file, 'r', encoding='utf-8') as d:
        raw_data = json.load(d)
        training_data = [i for i in raw_data if i['relation'] in train_label]
        test_data = [i for i in raw_data if i['relation'] in test_label]

    # print info
    logger.info(f'args_config: lr:{args.lr},batch_size:{args.batch_size},epochs:{args.epochs},'
                f'neg_ratio:{args.neg_ratio},temperature:{args.temperature}')
    logger.info('there are {} kinds of relation in train.'.format(len(set(train_label))))
    logger.info('there are {} kinds of relation in test.'.format(len(set(test_label))))
    logger.info('the length of train data is {} '.format(len(training_data)))
    logger.info('the length of test data is {} '.format(len(test_data)))

    # load description
    train_rel2vec, test_rel2vec = data_process.generate_attribute(args, train_desc, test_desc)

    # load model
    config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
    config.pretrained_model = args.pretrained_model
    config.temperature = args.temperature
    model = PCL.from_pretrained(args.pretrained_model, config=config)
    # model = model.to(torch.float64)
    model = model.cuda()

    trainset = data_process.FewRelDataset(args, 'train', training_data, train_rel2vec, train_relation2idx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=data_process.create_mini_batch, shuffle=True)

    # To evaluate the inference time
    test_batchsize = 10 * args.unseen

    testset = data_process.FewRelDataset(args, 'test', test_data, test_rel2vec, test_relation2idx)
    testloader = DataLoader(testset, batch_size=test_batchsize, collate_fn=data_process.create_mini_batch, shuffle=False)

    train_y_attr, train_y_e1, train_y_e2 = [], [], []
    for i in train_label:
        train_y_attr.append(train_rel2vec[i][0])
        train_y_e1.append(train_rel2vec[i][1])
        train_y_e2.append(train_rel2vec[i][2])
    train_y_attr = torch.tensor(np.array(train_y_attr)).cuda()
    train_y_e1 = torch.tensor(np.array(train_y_e1)).cuda()
    train_y_e2 = torch.tensor(np.array(train_y_e2)).cuda()

    test_y_attr, test_y, test_y_e1, test_y_e2 = [], [], [], []
    for i, test in enumerate(test_data):
        label = int(test_relation2idx[test['relation']])
        test_y.append(label)
    for i in test_label:
        test_y_attr.append(test_rel2vec[i][0])
        test_y_e1.append(test_rel2vec[i][1])
        test_y_e2.append(test_rel2vec[i][2])
    test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
        test_y_e2)

    # optimizer and scheduler
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    test_pt, test_rt, test_f1t = 0.0, 0.0, 0.0
    best_test_pt, best_test_rt, best_test_f1t = 0.0, 0.0, 0.0
    for epoch in range(args.epochs):
        logger.info(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
        running_loss = 0.0
        for step, data in enumerate(trainloader):
            tokens_tensors, attention_mask, marked_head, marked_tail, marked_mask, \
            relation_emb, labels_ids = [t.cuda() for t in data]
            optimizer.zero_grad()
            outputs, out_sentence_emb, e1_h, e2_h = model(input_ids=tokens_tensors,
                                                          attention_mask=attention_mask,
                                                          marked_head=marked_head,
                                                          marked_tail=marked_tail,
                                                          mask_mask=marked_mask,
                                                          input_relation_emb=relation_emb,
                                                          train_y_attr=train_y_attr,
                                                          train_y_e1=train_y_e1,
                                                          train_y_e2=train_y_e2,
                                                          labels=labels_ids
                                                          )

            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 500 == 0:
                logger.info(f'[step {step}] ' + f'running_loss: {running_loss / (step + 1)}')

        logger.info('============== EVALUATION ON Test DATA ==============')
        preds, e1_hs, e2_hs = extract_relation_emb(model, testloader)
        test_pt, test_rt, test_f1t = evaluate(preds.cpu(), e1_hs.cpu(), e2_hs.cpu(),
                                              test_y_attr, test_y_e1, test_y_e2, test_y)
        if best_test_f1t < test_f1t:
            best_test_pt = test_pt
            best_test_rt = test_rt
            best_test_f1t = test_f1t
        logger.info(f'[test] precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')
        logger.info(
            f'[test] best precision: {best_test_pt:.4f}, recall: {best_test_rt:.4f}, f1 score: {best_test_f1t:.4f}\n')
