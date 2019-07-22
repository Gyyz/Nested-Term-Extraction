# -*- coding: utf-8 -*-
# @Author: Yuze Gao
# @Date:   2019-04-10 18:17:45

from __future__ import print_function
from __future__ import absolute_import
import torch.nn as nn
from models.wordfeat.WordSeq import WordSequence
from utils.data import Data
from utils.functions import *
from copy import copy
import torch
import numpy as np
import torch.nn.functional as F


class MultiLabelSeq(nn.Module):
    def __init__(self, data):
        super(MultiLabelSeq, self).__init__()
        print('build span ranking model ...')
        print('use_char: ', data.use_char)
        if data.use_char:
            print('char feature extractor: ', data.char_feature_extractor)
        print('word feature extractor: ', data.word_feature_extractor)
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.word_hidden_features = WordSequence(data)
        self.seqLabel_num = data.seqlabel_alphabet.size()
        self.hidden2labels = nn.Linear(data.HP_hidden_dim, data.seqlabel_alphabet_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

    def format_labels(self, vectors, labels):
        ''''''
        assert vectors.size()[0] == len(labels)
        for idx, words in enumerate(labels):
            for idy, word in enumerate(words):
                for idz, tag in enumerate(word):
                    vectors[idx][idy][tag] = 1
        return vectors


    def forward(self, word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, sent_text, training=True):
        hidden_features, word_rep = self.word_hidden_features(word_inputs, pos_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size, max_seq_len = word_inputs.size()
        predicted_label_logits = self.hidden2labels(hidden_features)
        golden_labels = [bat_[1] for bat_ in batch_label]

        padding_zeros = torch.zeros(batch_size, max_seq_len, self.seqLabel_num)
        golden_labels_vector = self.format_labels(copy(padding_zeros), golden_labels)


        # get BCEloss
        loss = self.loss_fun(predicted_label_logits, golden_labels_vector)

        # get_predicted_labels
        re_mask = copy(mask).unsqueeze_(-1).cpu().data.numpy()
        padding_ones = torch.ones_like(golden_labels_vector).cpu().data.numpy()
        padding_zeros = padding_zeros.cpu().data.numpy()
        total_labels = golden_labels_vector.sum()

        # using sigmoid
        predicted_sigmoid = torch.sigmoid(predicted_label_logits)
        predicted_labels = predicted_sigmoid.ge(0.51)
        predicted_labels = predicted_labels.cpu().data.numpy()
        golden_labels_vec = copy(golden_labels_vector).cpu().data.numpy()
        predicted_right = np.greater((predicted_labels + golden_labels_vec + padding_ones), 2) * re_mask
        right_labels_num = np.sum(predicted_right) # true positive number 1 --> 1
        true_false = np.equal((predicted_labels + golden_labels_vec + padding_zeros), 0) * re_mask # 0 --> 0
        true_false_num = np.sum(true_false)
        false_positive = np.equal((predicted_labels - golden_labels_vec), 1) * re_mask # 0 --> 1
        false_pos_num = np.sum(false_positive)
        false_negative = np.equal((golden_labels_vec - predicted_labels), 1) * re_mask # 1 --> 0
        false_neg_num = np.sum(false_negative)

        assert right_labels_num+false_neg_num == total_labels
        recall = right_labels_num / (right_labels_num + false_neg_num) if (right_labels_num + false_neg_num) != 0 else 0
        precision = right_labels_num / (right_labels_num + false_pos_num) if (right_labels_num + false_pos_num) != 0 else 0
        F1 = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (right_labels_num + true_false_num) / (right_labels_num + true_false_num + false_neg_num + false_pos_num) if (right_labels_num + true_false_num + false_neg_num + false_pos_num) != 0 else 0

        print(precision, recall, F1, accuracy)
        # using softmax
        predicted_softmax = F.softmax(predicted_label_logits, dim=-1)
        soft_predicted = predicted_softmax.gt(0.36)
        soft_predicted = soft_predicted.cpu().data.numpy()
        soft_right = np.greater((soft_predicted + golden_labels_vec + padding_ones), 2) * re_mask
        soft_right_num = np.sum(soft_right)

        return loss, right_labels_num/total_labels

    def __str__(self):
        return 'SeqLabelling'

if __name__ == '__main__':
    data = Data()
    instances = data.all_instances
    batched = instances[:100]
    print(data.seqlabel_alphabet.instances)
    word_seq_tensor, pos_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sent_text = reformat_input_data(
       batched, False, True)
    span_seq = MultiLabelSeq(data)
    reps = span_seq(word_seq_tensor, pos_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths, char_seq_recover, labels, mask, sent_text)
    print(reps)