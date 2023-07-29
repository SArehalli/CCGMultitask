import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import numpy as np
import tqdm
import pickle

from model import MultiTaskModel

from train_augment import train_augment, evaluate_lm, evaluate_ccg, train_lm
from data import tag_dataset, tag_lm, augment_tag_lm, AugmentDataset, BatchSampler

parser = argparse.ArgumentParser()

parser.add_argument("--train", type=str)
parser.add_argument("--data_lm", type=str)
parser.add_argument("--data_ccg", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--save", type=str)
parser.add_argument("--load", type=str)
parser.add_argument("--opt", type=str)
parser.add_argument("--w2idx", type=str)

parser.add_argument("--lm_weight", type=float)
parser.add_argument("--hid_dim", type=int)
parser.add_argument("--emb_dim", type=int)
parser.add_argument("--seq_len", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--batch_size_aug", type=int)
parser.add_argument("--earlystop", type=int)
parser.add_argument("--patience", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--log_interval", type=int)

parser.add_argument("--ccg_pred_offset", type=int, default=0)

parser.add_argument("--dropout", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--clip", type=float)

parser.add_argument("--cuda", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--prog", action="store_true")
parser.add_argument("--strip_feat", action="store_true")
parser.add_argument("--save_all", action="store_true")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.eval:
    with open(args.load + ".w2idx", "rb") as w2idx_f:
        w2idx = pickle.load(w2idx_f)
    with open(args.load + ".c2idx", "rb") as c2idx_f:
        c2idx = pickle.load(c2idx_f)

    test_data_lm = augment_tag_lm(args.data_lm + "test.txt", None,
                             args.seq_len, w2idx=w2idx)
    test_sampler_lm = BatchSampler(test_data_lm, args.batch_size)
    test_loader_lm = DataLoader(test_data_lm, batch_sampler=test_sampler_lm)

    test_data_ccg = tag_dataset(args.data_ccg + "ccg.24.common", args.data_ccg + "categories", 
                             args.seq_len, pred_offset=args.ccg_pred_offset, w2idx=w2idx, c2idx=c2idx)
    test_sampler_ccg = BatchSampler(test_data_ccg, args.batch_size)
    test_loader_ccg = DataLoader(test_data_ccg, batch_sampler=test_sampler_ccg)

    mt_model = MultiTaskModel(len(w2idx), args.emb_dim, args.hid_dim, 
                             [len(w2idx), len(c2idx)],
                             args.n_layers, dropout = args.dropout)

    if args.load is not None:
        device = torch.device("cuda" if args.cuda else "cpu")
        mt_model.load_state_dict(torch.load(args.load + ".pt", map_location=device))
    if args.cuda:
        mt_model.cuda()
    
    lm_loss = evaluate_lm(mt_model, test_loader_lm, args.batch_size, nn.NLLLoss(), cuda=args.cuda, wrap=tqdm.tqdm)
    ccg_loss, one_best_ccg = evaluate_ccg(mt_model, test_loader_ccg, args.batch_size,
                                          nn.NLLLoss(), cuda=args.cuda, wrap=tqdm.tqdm)
    _, five_best_ccg = evaluate_ccg(mt_model, test_loader_ccg, args.batch_size,
                                          nn.NLLLoss(), cuda=args.cuda, nbest=5, wrap=tqdm.tqdm)
    print("TEST: lm ppl {:6.2f} \t| ccg 1-best {:.5f} \t| ccg 5-best {:.5f}".format(np.exp(lm_loss), one_best_ccg, five_best_ccg))

if args.train == "ccglm-base":
    print(args)
    w2idx = None
    c2idx = None
    if args.load is not None:
        with open(args.load + ".w2idx", "rb") as w2idx_f:
            w2idx = pickle.load(w2idx_f)
        with open(args.load + ".c2idx", "rb") as c2idx_f:
            c2idx = pickle.load(c2idx_f)

    train_data_ccglm = tag_lm(args.data_ccg + "ccg.02-21.common", args.data_ccg + "categories",  
                             args.seq_len, w2idx=w2idx, c2idx=c2idx, strip_feat=args.strip_feat)
    train_sampler_ccglm = BatchSampler(train_data_ccglm, args.batch_size)
    train_loader_ccglm = DataLoader(train_data_ccglm, batch_sampler=train_sampler_ccglm)

    valid_data_ccglm = tag_lm(args.data_ccg + "ccg.24.common", args.data_ccg + "categories",  
                             args.seq_len, w2idx=train_data_ccglm.w2idx, c2idx=train_data_ccglm.c2idx, strip_feat=args.strip_feat)
    valid_sampler_ccglm = BatchSampler(valid_data_ccglm, args.batch_size)
    valid_loader_ccglm = DataLoader(valid_data_ccglm, batch_sampler=valid_sampler_ccglm)

    test_data_ccglm = tag_lm(args.data_ccg + "ccg.23.common", args.data_ccg + "categories",  
                             args.seq_len, w2idx=train_data_ccglm.w2idx, c2idx=train_data_ccglm.c2idx, strip_feat=args.strip_feat)
    test_sampler_ccglm = BatchSampler(test_data_ccglm, args.batch_size)
    test_loader_ccglm = DataLoader(test_data_ccglm, batch_sampler=test_sampler_ccglm)

    mt_model = MultiTaskModel(len(train_data_ccglm.categories), args.emb_dim, args.hid_dim, 
                             [len(train_data_ccglm.categories)],
                             args.n_layers, dropout = args.dropout)

    if args.load is not None:
        device = torch.device("cuda" if args.cuda else "cpu")
        mt_model.load_state_dict(torch.load(args.load + ".pt", map_location=device))
        
    with open(args.save + ".w2idx", "wb") as w2idx_f:
        pickle.dump(train_data_ccglm.w2idx, w2idx_f)
    with open(args.save + ".c2idx", "wb") as w2idx_f:
        pickle.dump(train_data_ccglm.c2idx, w2idx_f)

    if args.cuda:
        mt_model.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(mt_model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(mt_model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    epoch = 0
    if args.load is not None:
        opt_dict = torch.load(args.load + ".opt")
        optimizer.load_state_dict(opt_dict["optimizer"])
        epoch = opt_dict["epoch"]
        
    
    for name, param in mt_model.named_parameters():
        if param.requires_grad:
                print(name, param.size())

    losses = train_lm(mt_model, optimizer, 
                           train_loader_ccglm, valid_loader_ccglm, 
                           args.batch_size,
                           nn.NLLLoss(), args.clip, 
                           args.log_interval, args.epochs, args.save, 
                           patience=args.patience, cuda=args.cuda, init_epoch=epoch, save_all=args.save_all)


if args.train == "augment":
    print(args)

    w2idx = None
    if args.load is not None:
        with open(args.load + ".w2idx", "rb") as w2idx_f:
            w2idx = pickle.load(w2idx_f)
        with open(args.load + ".c2idx", "rb") as c2idx_f:
            c2idx = pickle.load(c2idx_f)
        
    # Train data is handled differently, so that we can loop over one iterator

    train_data_lm = augment_tag_lm(args.data_lm + "train.txt", args.data_ccg + "ccg.02-21.common",  
                             args.seq_len, w2idx=w2idx)
    train_data_ccg = tag_dataset(args.data_ccg + "ccg.02-21.common", args.data_ccg + "categories", 
                             args.seq_len, pred_offset=args.ccg_pred_offset, w2idx=train_data_lm.w2idx, strip_feat=args.strip_feat)
    train_data = AugmentDataset(train_data_lm, train_data_ccg)
    train_sampler = BatchSampler(train_data, args.batch_size)

    train_loader = DataLoader(train_data, batch_sampler=train_sampler)


    valid_data_lm = augment_tag_lm(args.data_lm + "valid.txt", None, 
                             args.seq_len, w2idx=train_data_lm.w2idx)
    valid_sampler_lm = BatchSampler(valid_data_lm, args.batch_size)
    valid_loader_lm = DataLoader(valid_data_lm, batch_sampler=valid_sampler_lm)

    test_data_lm = augment_tag_lm(args.data_lm + "test.txt", None,
                             args.seq_len, w2idx=train_data_lm.w2idx)
    test_sampler_lm = BatchSampler(test_data_lm, args.batch_size)
    test_loader_lm = DataLoader(test_data_lm, batch_sampler=test_sampler_lm)

    valid_data_ccg = tag_dataset(args.data_ccg + "ccg.24.common", args.data_ccg + "categories", 
                             args.seq_len, pred_offset=args.ccg_pred_offset, w2idx=train_data_lm.w2idx, strip_feat=args.strip_feat)
    valid_sampler_ccg = BatchSampler(valid_data_ccg, args.batch_size)
    valid_loader_ccg = DataLoader(valid_data_ccg, batch_sampler=valid_sampler_ccg)

    test_data_ccg = tag_dataset(args.data_ccg + "ccg.23.common", args.data_ccg + "categories", 
                             args.seq_len, pred_offset=args.ccg_pred_offset, w2idx=train_data_lm.w2idx, strip_feat=args.strip_feat)
    test_sampler_ccg = BatchSampler(test_data_ccg, args.batch_size)
    test_loader_ccg = DataLoader(test_data_ccg, batch_sampler=test_sampler_ccg)

    mt_model = MultiTaskModel(len(train_data_lm.vocab), args.emb_dim, args.hid_dim, 
                             [len(train_data_lm.vocab), len(train_data_ccg.categories)],
                             args.n_layers, dropout = args.dropout)

    if args.load is not None:
        device = torch.device("cuda" if args.cuda else "cpu")
        mt_model.load_state_dict(torch.load(args.load + ".pt", map_location=device))
        
    with open(args.save + ".w2idx", "wb") as w2idx_f:
        pickle.dump(train_data_lm.w2idx, w2idx_f)
    with open(args.save + ".c2idx", "wb") as w2idx_f:
        pickle.dump(train_data_ccg.c2idx, w2idx_f)

    if args.cuda:
        mt_model.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(mt_model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(mt_model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    epoch = 0
    if args.load is not None:
        opt_dict = torch.load(args.load + ".opt")
        optimizer.load_state_dict(opt_dict["optimizer"])
        epoch = opt_dict["epoch"]
        
    
    for name, param in mt_model.named_parameters():
        if param.requires_grad:
                print(name, param.size())

    losses = train_augment(mt_model, optimizer, args.lm_weight, 
                           train_loader, valid_loader_lm, 
                           valid_loader_ccg, args.batch_size,
                           nn.NLLLoss(), args.clip, 
                           args.log_interval, args.epochs, args.save, 
                           patience=args.patience, cuda=args.cuda, init_epoch=epoch, save_all=args.save_all)

    lm_loss = evaluate_lm(mt_model, test_loader_lm, args.batch_size, nn.NLLLoss(), cuda=args.cuda)
    ccg_loss, one_best_ccg = evaluate_ccg(mt_model, test_loader_ccg, args.batch_size,
                                          nn.NLLLoss(), cuda=args.cuda)
    print("TEST: lm ppl {:6.2f} \t| ccg 1-best {:.5f}".format(np.exp(lm_loss), one_best_ccg))

