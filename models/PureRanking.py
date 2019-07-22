# -*- coding: utf-8 -*-

from models.wordfeat.WordSeq import WordSequence
from utils.data import Data
from utils.functions import reformat_input_data, masked_log_softmax, TermAttention, MaskedQueryAttention, random_embedding, domasking, getElmo
import torch
import torch.optim as optim
from copy import copy
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


class FeatureModel(nn.Module):
    def __init__(self, data):
        super(FeatureModel, self).__init__()
        print('Building the Feature Model ...')
        if data.use_char:
            print('char feature extractor:', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)
        self.gpu = data.HP_gpu
        self.hidden_dim = data.HP_hidden_dim
        self.average_batch = data.average_batch_loss
        self.max_span = data.term_span
        self.pos_embedding_dim = data.pos_emb_dim
        self.useSpanLen = data.useSpanLen
        self.useElmo = data.use_elmo
        self.pos_as_feature = data.pos_as_feature

        self.WordHiddenFeatures = WordSequence(data)

        self.TermQuery = nn.Parameter(torch.Tensor(np.random.randn(self.hidden_dim)))
        self.QueryAttention = MaskedQueryAttention(data.HP_hidden_dim, gpu=self.gpu)

        self.TermAttention = TermAttention(self.hidden_dim, gpu=self.gpu)

        self.hidden2Vec = nn.Linear(self.hidden_dim * self.max_span, data.HP_hidden_dim)

        self.feature_dim = self.hidden_dim * 5

        if self.pos_as_feature:
            self.feature_dim += self.pos_embedding_dim * 5
            self.posLSTM = nn.LSTM(self.pos_embedding_dim, self.pos_embedding_dim,
                                   num_layers=1, batch_first=True, bidirectional=True)
            self.posSeq2Vec = nn.Linear(self.pos_embedding_dim * self.max_span, self.pos_embedding_dim)
            self.posEmbedding = nn.Embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)
            self.posEmbedding.weight.data.copy_(torch.from_numpy(random_embedding(data.ptag_alphabet.size(),
                                                                                  self.pos_embedding_dim)))

        if self.useElmo:
            self.feature_dim += self.hidden_dim
            self.elmoAttention = TermAttention(self.hidden_dim, gpu=self.gpu)
            self.elmo2Vec = nn.Linear(self.hidden_dim * self.max_span, self.hidden_dim)
            self.elmoEmb = getElmo(layer=2, dropout=data.HP_dropout, out_dim=self.hidden_dim, gpu=self.gpu)

        if self.useSpanLen:
            self.feature_dim += data.spamEm_dim
            self.spanLenEmb = nn.Embedding(self.max_span+1, data.spamEm_dim)
            self.spanLenEmb.weight.data.copy_(torch.from_numpy(random_embedding(self.max_span+1, data.spamEm_dim)))

        if self.gpu:
            self.posSeq2Vec = self.posSeq2Vec.cuda()
            self.elmo2Vec = self.elmo2Vec.cuda()
            self.hidden2Vec = self.hidden2Vec.cuda()

    def reformat_labels(self, golden_labels):
        ''''''
        golden_spans = []
        golden_class = []
        golden_term_num = []
        for sentSpan in golden_labels:
            golden_spans.append([(itm[0], itm[1]+1) for itm in sentSpan])
            golden_class.append([itm[2] for itm in sentSpan])
            sent_term_num = len(sentSpan) if sentSpan[0][1] != -1 else 0
            golden_term_num.append(sent_term_num)
        return golden_spans, golden_class, golden_term_num

    def get_candidate_span_pairs(self, seq_lengths):
        ''''''
        if self.gpu:
            seq_lengths = seq_lengths.cpu()
        sents_lengths = [np.arange(seq_len) for seq_len in seq_lengths]
        candidate_starts = [sent.reshape(-1, 1).repeat(self.max_span, 1) for sent in sents_lengths]
        span_lengths = np.arange(self.max_span)
        candidate_ends = [copy(sent_itm + span_lengths) for sent_itm in candidate_starts]
        candidate_ends = [np.array(np.minimum(canEnd, sentLen-1)) for sentLen, canEnd in zip(seq_lengths, candidate_ends)]
        spanPairs = []
        for canStarts, canEnds in zip(candidate_starts, candidate_ends):
            sentSpanPairs = []
            for wordStarts, wordEnds in zip(canStarts, canEnds):
                tmp_spans = [(start, end+1) for start, end in zip(wordStarts, wordEnds)]
                tmp_spans = list(set(tmp_spans))
                sentSpanPairs.extend(tmp_spans)
            spanPairs.append(sentSpanPairs)
        return spanPairs

    def get_sentSliceResult(self, sentence_slice, results):
        sentSliceRes = []
        for itm in sentence_slice:
            start = itm[0]
            end = itm[-1]
            sentSliceRes.append(results[start:end+1])
        return sentSliceRes

    def getGoldIndex(self, golden_spans, sentSpanCandi, sentSlice):
        ''''''
        golden_IDs = []
        goldenSentIDs = [[] for _ in range(len(sentSpanCandi))]
        OOVSpan = 0
        for sentID, (gold_, candi_, canIDs) in enumerate(zip(golden_spans, sentSpanCandi, sentSlice)):
            for gold in gold_:
                try:
                    tmp_ID = canIDs[candi_.index(gold)]
                except ValueError:
                    OOVSpan += 1
                    continue
                golden_IDs.append(tmp_ID)
                goldenSentIDs[sentID].append(tmp_ID)
        return golden_IDs, goldenSentIDs, OOVSpan

    def get_span_features(self, hidden_states, span_pairs, seq_lengths, pfeature='HIDDEN'):
        '''
        '''
        assert hidden_states.size(0) == len(span_pairs)
        if pfeature == 'HIDDEN':
            pad_hidden = torch.zeros(self.hidden_dim)
        elif pfeature == 'POS':
            pad_hidden = torch.zeros(self.pos_embedding_dim)
        elif pfeature == 'ELMO':
            pad_hidden = torch.zeros(self.hidden_dim)

        # from hidden
        flat_spanSErep = []
        flat_HiddenSeq = []
        flat_spanTarAtt = []

        flat_spanPairs = []
        flat_spanLens = [] # mask
        flat_sentIds = [] # the pair mapping to the sentence id

        sent_num = len(span_pairs)
        sentence_slice = [[] for _ in range(sent_num)]
        spanPairID = 0

        for sent_id, (seHiddens, sentPair, seqLen) in enumerate(zip(hidden_states, span_pairs, seq_lengths)):
            sentHid = seHiddens[:seqLen]
            for pairs in sentPair:
                pairVec = torch.mean(sentHid[pairs[0]: pairs[1]], dim=0).unsqueeze(-1) # [hidden_dim]
                pairTermScore = F.softmax(torch.matmul(sentHid, pairVec), dim=0)
                pairTvec = torch.sum(sentHid * pairTermScore, dim=0, keepdim=False).view(1, -1)
                flat_spanTarAtt.append(pairTvec)
                if pairs[1] - pairs[0] == self.max_span:
                    flat_HiddenSeq.append(sentHid[pairs[0]: pairs[1]].unsqueeze(0))
                else:
                    pad_for_seq = pad_hidden.unsqueeze(0).repeat(self.max_span+pairs[0]-pairs[1], 1)
                    flat_HiddenSeq.append(torch.cat([sentHid[pairs[0]: pairs[1]], pad_for_seq], dim=0).unsqueeze(0))
                flat_spanSErep.append(torch.cat((sentHid[pairs[0]], sentHid[pairs[1]-1]), dim=0).unsqueeze(0))
                flat_sentIds.append(sent_id)
                flat_spanPairs.append(pairs)
                flat_spanLens.append(pairs[1]-pairs[0])
                sentence_slice[sent_id].append(spanPairID)
                spanPairID += 1

        flat_mask = domasking(flat_spanLens, maxlen=self.max_span)

        flat_HiddenSeq = torch.cat(flat_HiddenSeq, dim=0)

        # self attention over span sequence --> span node
        if pfeature == 'HIDDEN':
            flat_spanNode = self.TermAttention(flat_HiddenSeq, flat_mask) # attention node [num_span, hidden]
        elif pfeature == 'POS':
            flat_spanNode = self.posAttention(flat_HiddenSeq, flat_mask)
        elif pfeature == "ELMO":
            flat_spanNode = self.elmoAttention(flat_HiddenSeq, flat_mask)

        # function condense MLP on span sequence --> span head
        if pfeature == 'HIDDEN':
            flat_spanHead = self.hidden2Vec(flat_HiddenSeq.view(flat_HiddenSeq.size(0), -1))
        elif pfeature == 'POS':
            flat_spanHead = self.posSeq2Vec(flat_HiddenSeq.view(flat_HiddenSeq.size(0), -1))
        elif pfeature == 'ELMO':
            flat_spanHead = self.elmo2Vec(flat_HiddenSeq.view(flat_HiddenSeq.size(0), -1))

        # start and end word representation
        flat_spanSErep = torch.cat(flat_spanSErep, dim=0)

        # term vector targeted attention over sentence sequence
        flat_spanTarAtt = torch.cat(flat_spanTarAtt, dim=0)

        if pfeature == 'HIDDEN':
            return flat_spanNode, flat_spanHead, flat_spanSErep, flat_spanTarAtt, flat_spanPairs, flat_mask, flat_spanLens, flat_sentIds, sentence_slice, sent_num
        else:
            return flat_spanNode, flat_spanHead, flat_spanSErep, flat_spanTarAtt

    def forward(self, word_inputs, pos_inputs, word_seq_lengths,
                char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, sent_texts):
        hidden_features, word_rep = self.WordHiddenFeatures(word_inputs, pos_inputs, word_seq_lengths, char_inputs,
                                                              char_seq_lengths, char_seq_recover)
        hidden_features = self.QueryAttention(self.TermQuery, hidden_features, mask)

        golden_labels = [bat_[0] for bat_ in batch_label]
        golden_spans, golden_class, golden_term_num = self.reformat_labels(golden_labels)

        spanPairs = self.get_candidate_span_pairs(seq_lengths=word_seq_lengths)

        flat_spanNode, flat_spanHead, flat_spanSErep, flat_spanTarAtt, \
        flat_spanPairs, flat_mask, flat_spanLens, flat_sentIds, sentence_slice, \
        sent_num = self.get_span_features(hidden_features, spanPairs, word_seq_lengths)

        sentence_span_candidate = self.get_sentSliceResult(sentence_slice, flat_spanPairs)
        flat_golden_indexes, sent_gold_indexes, oovNum = self.getGoldIndex(golden_spans, sentence_span_candidate,
                                                                           sentence_slice)

        flat_spanLens = torch.Tensor(flat_spanLens).long().cuda() if self.gpu else torch.Tensor(flat_spanLens).long()
        lenEmbeddings = self.spanLenEmb(flat_spanLens)

        spanEmbs = [flat_spanNode, flat_spanHead, flat_spanSErep, flat_spanTarAtt, lenEmbeddings]

        if self.pos_as_feature:
            pos_embs = self.pos_embedding(pos_inputs)
            pos_embs = self.POSLSTM(pos_embs)
            pflat_spanNode, pflat_spanHead, pflat_spanSErep, pflat_spanTarAtt = \
                self.get_span_features(pos_embs, spanPairs, word_seq_lengths, pfeature='POS')
            spanEmbs.append(pflat_spanNode)
            spanEmbs.append(pflat_spanHead)
            spanEmbs.append(pflat_spanSErep)
            spanEmbs.append(pflat_spanTarAtt)

        if self.useElmo:
            elmo_features, elmo_mask = self.elmoEmb(sent_texts)
            eflat_spanNode, eflat_spanHead, eflat_spanSErep, eflat_spanTarAtt = \
                self.get_span_features(elmo_features, spanPairs, word_seq_lengths, pfeature='ELMO')
            spanEmbs.append(eflat_spanNode)
            spanEmbs.append(eflat_spanHead)
            spanEmbs.append(eflat_spanSErep)
            spanEmbs.append(eflat_spanTarAtt)

        spanEmbs = torch.cat(spanEmbs, dim=-1)

        return spanEmbs, flat_sentIds, flat_spanPairs, flat_golden_indexes, sentence_slice

    def __str__(self):
        return 'feature'


class PureRanker(nn.Module):
    def __init__(self, data):
        super(PureRanker, self).__init__()
        self.SpanFeature = FeatureModel(data)
        self.hiddenDim = data.HP_hidden_dim
        self.feature_dim = self.SpanFeature.feature_dim
        self.FeatVec2Class = nn.Sequential(nn.Linear(self.feature_dim, self.hiddenDim),
                                           nn.ReLU(True),
                                           nn.Linear(self.hiddenDim, 1))
        self.drop = nn.Dropout(data.HP_dropout, inplace=True)
        self.gpu = data.HP_gpu

    def forward(self, word_inputs, pos_inputs, word_seq_lengths,
                char_inputs, char_seq_lengths, char_seq_recover,
                batch_label, mask, sent_texts, training=True):

        spanEmbs, flat_sentIds, flat_spanPairs, flat_golden_indexes, sentence_slice = \
            self.SpanFeature(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                             char_seq_recover, batch_label, mask, sent_texts)
        if training:
            spanEmbs = self.drop(spanEmbs)

        flat_neg_indexes = list(set(range(len(flat_spanPairs))) - set(flat_golden_indexes))

        spanClasses = self.FeatVec2Class(spanEmbs)

        sig = spanClasses.sigmoid()

        gold_sig = sig[flat_golden_indexes]
        nega_sig = sig[flat_neg_indexes]

        gold_mean_loss = gold_sig.mean()
        nega_mean_loss = nega_sig.mean()

        loss = 1 - gold_mean_loss + nega_mean_loss
        # the negative samples are too much times of the golden, so we'd better calculate them separately
        total_words = word_seq_lengths.sum().float()
        K = (total_words * 0.23).floor().int()

        sorted_score, reindex = sig.sort(0, descending=True)
        filter_index = reindex[:K]

        rTP = 0  # the true positive in ranking
        for itm in filter_index:
            if itm in flat_golden_indexes:
                rTP += 1
        print('loss {:.4f}'.format(loss.item()), 'result {}/{}={:.4f}'.format(rTP, K.item(), float(rTP) / K.item()))

        return loss

    def __str__(self):
        return 'pureranker'

if __name__ == '__main__':
    data = Data()
    model = PureRanker(data)
    instance = data.all_instances
    batch_size = data.HP_batch_size
    roptim = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    batches = len(instance) // batch_size + 1
    for e in range(20):
        for i in range(batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = end if end < len(instance) else len(instance)
            batch_instance = instance[start: end]
            if not batch_instance:
                continue
            batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
            batch_char_seq_lengths, batch_char_seq_recover, batch_labels, batch_mask, batch_texts \
                = reformat_input_data(batch_instance, data.HP_gpu, True)


            loss = model.forward(batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_char_seq, batch_char_seq_lengths,
                                 batch_char_seq_recover, batch_labels, batch_mask, batch_texts)
            loss.backward()
            roptim.step()
            model.zero_grad()