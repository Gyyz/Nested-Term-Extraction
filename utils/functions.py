# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06

from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
seed_num = 626
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def _start_shell(local_ns=None):
    ''''''
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

def cint(ins):
    if isinstance(ins, (float, str)):
        return int(ins)
    if isinstance(ins, (list, tuple)):
        return [int(itm) for itm in ins]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%.4f"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def reformat_input_data(batch_data, use_gpu=True, if_train=True):
    ''''''
    batch_size = len(batch_data)
    words = [inst[0] for inst in batch_data]
    chars = [inst[1] for inst in batch_data]
    pos_tags = [inst[2] for inst in batch_data]
    labels = [inst[3] for inst in batch_data]
    sentTexts = [inst[4] for inst in batch_data]
    # span_label = [label[0] for label in labels]
    # sequ_label = [label[1] for label in labels]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    pos_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()

    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)

    for idx, (seq, tag, seqlen) in enumerate(zip(words, pos_tags, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        pos_seq_tensor[idx, :seqlen] = torch.LongTensor(tag)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)

    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    pos_seq_tensor = pos_seq_tensor[word_perm_idx]
    labels = [labels[idx] for idx in word_perm_idx]
    sentTexts = [sentTexts[idx] for idx in word_perm_idx]
    mask = mask[word_perm_idx]

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)

    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]

    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if use_gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        pos_seq_tensor = pos_seq_tensor.cuda()
        mask = mask.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()

    return word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sentTexts


def random_embedding(vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, epsilon=1e-45) -> torch.Tensor:

    vector = vector + (mask + epsilon).log()
    return F.softmax(vector, dim=dim)

def domasking(lengths, maxlen=None, dtype=None) -> torch.Tensor:
    ''''''
    maxlen = maxlen or lengths.max().item()
    batch_size = len(lengths)
    mask = np.zeros((batch_size, maxlen), dtype=dtype)
    for idx, lent in enumerate(lengths):
        mask[idx, :lent] = 1
    return torch.Tensor(mask)

def checkratio(total_words, reindex, goldenIDs, totalgoldens):
    ratios = [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
    Ks = [(rat*total_words).floor().int() for rat in ratios]
    reindex.cpu()
    filter_indexes = [reindex[:K].tolist() for K in Ks]
    def single_check(rein):
        shared = 0
        for itm in rein:
            if itm in goldenIDs:
                shared += 1
        return shared
    TPs = [single_check(rei) for rei in filter_indexes]
    precisions = [float(A)/float(B) for A, B in zip(TPs, Ks)]
    recalls = [float(A)/totalgoldens for A in TPs]
    f1s = [2*A*B/(A+B) for A, B in zip(precisions, recalls)]

    for ratio, K, TP, prec, reca, f1 in zip(ratios, Ks, TPs, precisions, recalls,f1s):
        print('{}\t{}\t{}\t{}\t{:0.4f}\t{:0.4f}\t{:0.4f}'.format(ratio, K, TP, totalgoldens, prec, reca, f1))

class TermAttention(nn.Module):
    def __init__(self, hidden_dim, gpu=True):
        super(TermAttention, self).__init__()
        self.gpu = gpu
        self.hiddenDim = hidden_dim
        self.termWeight = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)).cuda()) if gpu else nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))
        if gpu:
            self.attSeq = self.attSeq.cuda()

    def forward(self, hidden_states, mask):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_hidden = hidden_states * self.termWeight
        attention_score = self.attSeq(attention_hidden)
        mask = mask.unsqueeze(-1)
        attention_score = masked_log_softmax(attention_score, mask=mask, dim=1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x #, attention_score

class QueryAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(QueryAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, query, hidden_states):

        if hidden_states.size(0) == 0:
            return torch.zeros(self.hiddenDim)
        query = self.linear_in(query)
        attention_score = torch.matmul(hidden_states, query)
        attention_score = F.softmax(attention_score, dim=0).view(hidden_states.size(0), 1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.sum(scored_x, dim=0)
        # condensed_x = self.tanh(condensed_x)
        return condensed_x


class MaskedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, gpu=True):
        super(MaskedQueryAttention, self).__init__()
        self.gpu = gpu
        self.hiddenDim = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear_out = nn.Linear(2*hidden_dim, hidden_dim)
        if gpu:
            self.linear_in = self.linear_in.cuda()
            self.linear_out = self.linear_out.cuda()

    def forward(self, query, hidden_states, mask):
        if self.gpu:
            query = query.cuda()
            hidden_states = hidden_states.cuda()
            mask = mask.cuda()
        query = self.linear_in(query)
        attention_score = torch.matmul(hidden_states, query)
        mask = mask.float()
        attention_score = masked_log_softmax(attention_score, mask=mask, dim=1).view((hidden_states.size(0), hidden_states.size(1), 1))
        scored_x = hidden_states * attention_score
        combined_x = torch.cat((hidden_states, scored_x), dim=-1)
        condensed_x = self.linear_out(combined_x)
        return condensed_x #, attention_score


class SeqAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SeqAttention, self).__init__()
        self.hiddenDim = hidden_dim
        self.SeqWeight = nn.Parameter(torch.Tensor(np.random.randn(self.hiddenDim)))
        self.attSeq = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, hidden_states, mask):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_hidden = hidden_states * self.SeqWeight
        attention_score = self.attSeq(attention_hidden)
        mask = mask.unsqueeze(-1).float()
        attention_score = attention_score * mask
        attention_score = masked_log_softmax(attention_score, mask, dim=1).view(hidden_states.size(0), hidden_states.size(1), 1)
        scored_x = hidden_states * attention_score
        condensed_x = torch.tanh(scored_x)
        return condensed_x #, attention_score


class getElmo(nn.Module):
    def __init__(self, layer=2, dropout=0, out_dim=100, gpu=True):
        super(getElmo, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.dropout = dropout
        self.gpu = gpu
        self.Elmo = Elmo(options_file, weight_file, layer, dropout=dropout)
        self.Elmo.eval()
        self.layers2one = nn.Linear(layer, 1).cuda() if self.gpu else nn.Linear(layer, 1)
        self.optLinear = nn.Linear(1024, out_dim).cuda() if self.gpu else nn.Linear(1024, out_dim)

    def forward(self, texts):
        word_idxs = batch_to_ids(texts).cuda() if self.gpu else batch_to_ids(texts)
        elmo_embs = self.Elmo.forward(word_idxs)
        elmo_reps = torch.stack(elmo_embs['elmo_representations'], dim=-1).cuda() if self.gpu else torch.stack(elmo_embs['elmo_representations'], dim=-1)
        elmo_decrease_layer = self.layers2one(elmo_reps).squeeze()
        elmo_fit_hidden = self.optLinear(elmo_decrease_layer)
        mask = elmo_embs['mask']

        return elmo_fit_hidden, mask
