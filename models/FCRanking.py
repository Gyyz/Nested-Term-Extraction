# -*- coding: utf-8 -*-

from models.wordfeat.WordSeq import WordSequence
from utils.data import Data
from utils.functions import reformat_input_data, masked_log_softmax, TermAttention, MaskedQueryAttention, random_embedding, domasking, getElmo, checkratio
import torch
import torch.optim as optim
from copy import copy
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
seed_num = 626
torch.manual_seed(seed_num)
np.random.seed(seed_num)


class FeatureModel(nn.Module):
    def __init__(self, data):
        super(FeatureModel, self).__init__()
        print('Building the Feature Model ...')
        if data.use_char:
            print('char feature extractor:', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)
        self.data = data
        self.gpu = data.HP_gpu
        self.hidden_dim = data.HP_hidden_dim
        self.average_batch = data.average_batch_loss
        self.max_span = data.term_span
        self.pos_embedding_dim = data.pos_emb_dim
        self.useSpanLen = data.useSpanLen
        self.useElmo = data.use_elmo
        self.elmodim = data.elmodim
        self.pos_as_feature = data.pos_as_feature

        self.WordHiddenFeatures = WordSequence(data)

        self.TermQuery = nn.Parameter(torch.Tensor(np.random.randn(self.hidden_dim)))
        self.QueryAttention = MaskedQueryAttention(data.HP_hidden_dim, gpu=self.gpu)

        self.TermAttention = TermAttention(self.hidden_dim, gpu=self.gpu)

        self.hidden2Vec = nn.Linear(self.hidden_dim * self.max_span, data.HP_hidden_dim)

        self.feature_dim = self.hidden_dim * 2

        if self.data.use_sentence_att:
            self.feature_dim += self.hidden_dim
        if self.data.args.use_head:
            self.feature_dim += self.hidden_dim
        if self.data.args.use_span_node:
            self.feature_dim += self.hidden_dim

        if self.pos_as_feature:
            self.feature_dim += self.pos_embedding_dim * 5
            self.posLSTM = nn.LSTM(self.pos_embedding_dim, self.pos_embedding_dim // 2,
                                   num_layers=1, batch_first=True, bidirectional=True)
            self.posAttention = TermAttention(self.pos_embedding_dim, gpu=self.gpu)
            self.posSeq2Vec = nn.Linear(self.pos_embedding_dim * self.max_span, self.pos_embedding_dim)

            self.posEmbedding = nn.Embedding(data.ptag_alphabet.size(), self.pos_embedding_dim)
            self.posEmbedding.weight.data.copy_(torch.from_numpy(random_embedding(data.ptag_alphabet.size(),
                                                                                  self.pos_embedding_dim)))

        if self.useElmo:
            self.feature_dim += self.elmodim * 5
            self.elmoAttention = TermAttention(self.elmodim, gpu=self.gpu)
            self.elmo2Vec = nn.Linear(self.elmodim * self.max_span, self.elmodim)
            self.elmoEmb = getElmo(layer=2, dropout=data.HP_dropout, out_dim=self.elmodim, gpu=self.gpu)
            self.ElmoTermQuery = nn.Parameter(torch.Tensor(np.random.randn(self.elmodim)))
            self.ElmoQueryAttention = MaskedQueryAttention(self.elmodim, gpu=self.gpu)

        if self.useSpanLen:
            self.feature_dim += data.spamEm_dim
            self.spanLenEmb = nn.Embedding(self.max_span+1, data.spamEm_dim)
            self.spanLenEmb.weight.data.copy_(torch.from_numpy(random_embedding(self.max_span+1, data.spamEm_dim)))

        if self.gpu:
            if self.pos_as_feature:
                self.posSeq2Vec = self.posSeq2Vec.cuda()
            if self.useElmo:
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
            pad_hidden = torch.zeros(self.hidden_dim).cuda() if self.gpu else torch.zeros(self.hidden_dim)
        elif pfeature == 'POS':
            pad_hidden = torch.zeros(self.pos_embedding_dim).cuda() if self.gpu else torch.zeros(self.pos_embedding_dim)
        elif pfeature == 'ELMO':
            pad_hidden = torch.zeros(self.elmodim).cuda() if self.gpu else torch.zeros(self.elmodim)

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

        flat_mask = domasking(flat_spanLens, maxlen=self.max_span).cuda() if self.gpu else domasking(flat_spanLens, maxlen=self.max_span)

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
            pos_embs = self.posEmbedding(pos_inputs)
            pos_feas, *_ = self.posLSTM(pos_embs)
            pflat_spanNode, pflat_spanHead, pflat_spanSErep, pflat_spanTarAtt = \
                self.get_span_features(pos_feas, spanPairs, word_seq_lengths, pfeature='POS')
            spanEmbs.append(pflat_spanNode)
            spanEmbs.append(pflat_spanHead)
            spanEmbs.append(pflat_spanSErep)
            spanEmbs.append(pflat_spanTarAtt)

        if self.useElmo:
            elmo_features, elmo_mask = self.elmoEmb(sent_texts)
            elmo_features = self.ElmoQueryAttention(self.ElmoTermQuery, elmo_features, elmo_mask)
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


class Classifier(nn.Module):
    def __init__(self, data):
        super(Classifier, self).__init__()
        self.SpanFeature = FeatureModel(data)
        self.hiddenDim = data.HP_hidden_dim
        self.feature_dim = self.SpanFeature.feature_dim
        self.FeatVec2Class = nn.Sequential(nn.Linear(self.feature_dim, self.hiddenDim),
                                           nn.ReLU(True),
                                           nn.Linear(self.hiddenDim, 2))
        self.LossFunc = nn.CrossEntropyLoss()
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

        gold_span_classes = spanClasses[flat_golden_indexes]
        nega_span_classes = spanClasses[flat_neg_indexes]

        gold_targets = torch.ones((gold_span_classes.size(0)), dtype=torch.long).cuda() \
            if self.gpu else torch.ones((gold_span_classes.size(0)), dtype=torch.long)
        nega_targets = torch.zeros((nega_span_classes.size(0)), dtype=torch.long).cuda() \
            if self.gpu else torch.zeros((nega_span_classes.size(0)), dtype=torch.long)

        # the negative samples are too much times of the golden, so we'd better calculate them separately
        gold_loss = self.LossFunc(gold_span_classes, gold_targets)
        nega_loss = self.LossFunc(nega_span_classes, nega_targets)

        loss = gold_loss + nega_loss

        PredResults = torch.argmax(spanClasses, dim=-1) # all candidate predicts
        cTPFP = PredResults.sum().float().item() # tp + fp
        cTP = PredResults[flat_golden_indexes].sum().float().item() # tp
        cTPFN = float(len(flat_golden_indexes))

        # prepare dumping classification results and ranking
        initID = 0
        goldenIDs = []  # golden ID in predicted positive IDs
        globalIDs = [] # the span that is classified into true spans
        for pid, pix in enumerate(PredResults):
            if pix == 1:
                globalIDs.append(pid)
                if pid in flat_golden_indexes:
                    goldenIDs.append(initID)
                initID += 1 # the index in positive predicted
        negative_IDs = [_idx for _idx in range(len(globalIDs)) if _idx not in goldenIDs]

        classified_emb = spanEmbs[globalIDs]
        cpredict_spans = [[] for _ in range(len(sentence_slice))] # dump the spans for each sentence
        classified_spans = [flat_spanPairs[_idx] for _idx in globalIDs]
        classified_sent_ids = [flat_sentIds[_idx] for _idx in globalIDs]
        for sent_id, _span in zip(classified_sent_ids, classified_spans):
            cpredict_spans[sent_id].append([int(_span[0]), int(_span[1])-1])

        return loss, cTP, cTPFP, cTPFN, classified_emb, classified_spans, classified_sent_ids, goldenIDs, negative_IDs, cpredict_spans

    def __str__(self):
        return 'Classifier'


class niceRanking(nn.Module):
    def __init__(self, data):
        super(niceRanking, self).__init__()
        self.classifier = Classifier(data)
        self.feature_dim = self.classifier.feature_dim
        self.ranking = data.ranking
        self.spanRegression = nn.Sequential(nn.Linear(self.feature_dim, 200), nn.ReLU(), nn.Linear(200, 100), nn.Linear(100, 1))
        self.gpu = data.HP_gpu
        self.termratio = data.termratio
        self.drop = nn.Dropout(data.HP_dropout)
        self.silence = data.silence

    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                batch_label, mask, sent_texts, training=True, rankingflag=False, dump_result=False):

        loss, cTP, cTPFP, cTPFN, classified_emb, classified_spans, \
        classified_sent_ids, goldenIDs, negative_IDs, cpredict_spans = self.classifier(word_inputs,
                                                                                       pos_inputs,
                                                                                       word_seq_lengths,
                                                                                       char_inputs,
                                                                                       char_seq_lengths,
                                                                                       char_seq_recover,
                                                                                       batch_label,
                                                                                       mask,
                                                                                       sent_texts,
                                                                                       training)

        if rankingflag and classified_emb.size()[0] != 0:

            ranking_scores = self.spanRegression(classified_emb)
            ranking_scores = ranking_scores.sigmoid().view(-1) #=sigmoid()
            golden_ranking = 1-ranking_scores[goldenIDs].mean()
            negative_ranking = ranking_scores[negative_IDs].mean()

            loss = golden_ranking + negative_ranking

            total_words = word_seq_lengths.sum().float()
            K = (total_words * self.termratio).floor().int()

            sorted_score, reindex = ranking_scores.sort(0, descending=True)
            #if not training:
            #    checkratio(total_words, reindex, goldenIDs, cTPFN)
            # sorted_score, filter_index = torch.topk(ranking_scores, K, dim=0) #reindex[:K]
            filter_index = reindex[:K]

            rTP = 0 # the true positive in ranking
            for itm in filter_index:
                if itm in goldenIDs:
                    rTP += 1
            rTPFP = float(K)
            rTPFN = float(len(goldenIDs))

            if dump_result:
                rpredict_spans = [[] for _ in range(word_inputs.size()[0])]  # dump the spans for each sentence
                if len(classified_spans) > 0:
                    ranking_spans = [classified_spans[_idx] for _idx in filter_index]
                    ranking_sent_ids = [classified_sent_ids[_idx] for _idx in filter_index]
                    for _rspan, _sent_id in zip(ranking_spans, ranking_sent_ids):
                        rpredict_spans[_sent_id].append([int(_rspan[0]), int(_rspan[1])-1])

                return loss, rTP, rTPFP, rTPFN, cTP, cTPFP, cTPFN, cpredict_spans, rpredict_spans

            return loss, rTP, rTPFP, rTPFN, cTP, cTPFP, cTPFN

        return loss, cTP, cTPFP, cTPFN

    def __str__(self):
        return 'NiceRanking'

if __name__ == '__main__':

    data = Data()
    instances = data.all_instances
    batched = instances[:100]
    span_seq = niceRanking(data)
    optimizer = optim.Adam(span_seq.parameters(), lr=0.01, weight_decay=data.HP_l2)
    for epo in range(2):
        for idx in range(len(instances) // 100):
            batched = instances[idx * 100:(idx + 1) * 100]
            span_seq.train()
            word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sentTexts = reformat_input_data(
                batched, use_gpu=False)
            reps = span_seq(word_seq_tensor, pos_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths,
                            char_seq_recover, labels, mask, sent_texts=sentTexts)
            reps.backward()
            optimizer.step()
            span_seq.zero_grad()
        epo += 1

    span_seq.classifier.eval()
    # span_seq.classifier.weight.requires_grad = False
    for para in span_seq.classifier.parameters():
        para.requires_grad = False

    raning_para = span_seq.spanRegression.parameters()
    roptim = optim.Adam(raning_para, lr=0.01, weight_decay=data.HP_l2)

    for epo in range(6):
        for idx in range(len(instances) // 100):
            batched = instances[idx * 100:(idx + 1) * 100]
            word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sentTexts = reformat_input_data(
                batched, use_gpu=False)
            reps = span_seq(word_seq_tensor, pos_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths,
                                char_seq_recover, labels, mask, sent_texts=sentTexts, training=True, rankingflag=True)
            reps.backward()
            roptim.step()
            span_seq.zero_grad()
