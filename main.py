# -*- coding: utf-8 -*-
# @Author: Yuze
# @Date:   2019-04-15 11:47:52
# @Last Modified by:   Yuze,     Contact: yuze.gao@outlook.com
# @Last Modified time: 2019-06-18 17:14:39
# ------------------------------------------------------------------------

from models.FCRanking import niceRanking
from utils.data import Data
from utils.functions import reformat_input_data, bcolors
from termcolor import colored
import torch
import torch.optim as optim
import random
import argparse
import sys, time, os, json
import numpy as np

seed_num = 626
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    # print(" Learning rate is set as:\r", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def model_save(model, data, stype='cls'): # cls ==> classification; rnk ==> ranking
    model_name = model.__str__()
    path = data.model_save_dir
    if not os.path.exists(path):
        os.mkdir(path)
    new_path = os.path.join(path, '_Drop_{}_MaxL_{}_HD_{}_GL_{}_POS_{}_ELMO_{}_{}'.format(data.HP_dropout,
                                                                                          data.term_span,
                                                                                          data.HP_hidden_dim,
                                                                                          True if data.word_emb_dir is not None else None,
                                                                                          data.pos_as_feature,
                                                                                          data.use_elmo,
                                                                                          data.termratio))
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    ck_filename = model_name + '_{}'.format(stype)
    filename = os.path.join(new_path, ck_filename)
    torch.save(model.state_dict(), filename)


def model_load(model, data, stype='cls'): # cls ==> classification; rnk ==> ranking
    path = data.model_save_dir
    model_name = model.__str__()
    new_path = os.path.join(path, '_Drop_{}_MaxL_{}_HD_{}_GL_{}_POS_{}_ELMO_{}_{}'.format(data.HP_dropout,
                                                                                          data.term_span,
                                                                                          data.HP_hidden_dim,
                                                                                          True if data.word_emb_dir is not None else None,
                                                                                          data.pos_as_feature,
                                                                                          data.use_elmo,
                                                                                          data.termratio))
    ck_filename = model_name + '_{}'.format(stype)
    filename = os.path.join(new_path, ck_filename)
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    return model


def evaluate(model, data, devData, rankingflag=False):
    # print('\n' + bcolors.WARNING + 'Evaluating' + bcolors.ENDC)
    random.shuffle(devData)
    batch_size = data.HP_batch_size
    dev_batchs = len(devData) // batch_size + 1
    total_loss = []
    total_right = []
    total_pretrue = []
    total_true = []

    total_golden = []

    cls_right = []
    cls_pretrue = []

    for batch_idx in range(dev_batchs):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        end = end if end < len(devData) else len(devData)
        batch_instance = devData[start: end]
        if not batch_instance:
            continue
        batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
        batch_char_seq_lengths, batch_char_seq_recover, batch_labels, batch_mask, batch_texts \
            = reformat_input_data(batch_instance, data.HP_gpu, True)

        if rankingflag:
            loss, TP, TPFP, TPFN, cTP, cTPFP, cTPFN = model.forward(batch_words_seq,
                                                                    batch_pos_seq,
                                                                    batch_seq_lengths,
                                                                    batch_char_seq,
                                                                    batch_char_seq_lengths,
                                                                    batch_char_seq_recover,
                                                                    batch_labels,
                                                                    batch_mask,
                                                                    batch_texts,
                                                                    training=False,
                                                                    rankingflag=True)
            total_golden.append(cTPFN)
            cls_right.append(cTP)
            cls_pretrue.append(cTPFP)

        else:
            loss, TP, TPFP, TPFN, *_ = model.forward(batch_words_seq,
                                                     batch_pos_seq,
                                                     batch_seq_lengths,
                                                     batch_char_seq,
                                                     batch_char_seq_lengths,
                                                     batch_char_seq_recover,
                                                     batch_labels,
                                                     batch_mask,
                                                     batch_texts, training=False)

        total_loss.append(loss.item())
        total_right.append(TP)
        total_pretrue.append(TPFP)
        total_true.append(TPFN)

    avg_loss = np.mean(total_loss)
    precision = np.mean(total_right) / np.mean(total_pretrue)
    recall = np.mean(total_right) / np.mean(total_true)
    f1 = 2 * precision * recall / (precision + recall)

    trecall = np.mean(total_right) / np.mean(total_golden) if rankingflag else 0
    tf1 = 2 * precision * trecall / (precision + trecall) if rankingflag else 0

    cls_precision = np.mean(cls_right) / np.mean(cls_pretrue) if rankingflag else precision
    cls_recall = np.mean(cls_right) / np.mean(total_golden) if rankingflag else recall
    cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if rankingflag else f1

    return avg_loss, precision, recall, f1, trecall, tf1, cls_precision, cls_recall, cls_f1


def training(model, data, trainData, devData):
    ''''''
    print(bcolors.OKBLUE+'Traing Model ... '+bcolors.ENDC)
    # choose an optimizer
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)

    best_dev_loss = 100
    early_stop_count = 0
    best_dev_loc = []
    for _idx in range(data.HP_epoch):
        print("\nStart training the Classifier at Epoch / Total <==> {} / {}\n".format(_idx+1, data.HP_epoch))
        if data.optimizer.lower() == 'sgd':
            optimizer = lr_decay(optimizer, _idx, data.HP_lr_decay, data.HP_lr)
        random.shuffle(trainData)
        model.zero_grad()
        batch_size = data.HP_batch_size
        train_batchs = len(trainData) // batch_size + 1
        total_loss = []
        total_right = []
        total_pretrue = []
        total_true = []
        for batch_idx in range(train_batchs):
            print('Classifier Model at ' +
                  colored('Epoch {} <==>'.format(_idx+1), 'red', 'on_grey') +
                  ' Batch / Total: ' +
                  bcolors.OKGREEN +
                  bcolors.UNDERLINE +
                  bcolors.BOLD +
                  '# {} / {} #'.format(batch_idx+1, train_batchs) +
                  bcolors.ENDC,
                  end='\r')

            model.train()
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            end = end if end < len(trainData) else len(trainData)
            batch_instance = trainData[start: end]
            if not batch_instance:
                continue
            batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
            batch_char_seq_lengths, batch_char_seq_recover, batch_labels,  batch_mask, batch_texts \
                = reformat_input_data(batch_instance, data.HP_gpu, True)

            loss, cTP, cTPFP, cTPFN = model.forward(batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_char_seq,
                                                    batch_char_seq_lengths, batch_char_seq_recover, batch_labels,
                                                    batch_mask, batch_texts)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total_loss.append(loss.item())
            total_right.append(cTP)
            total_pretrue.append(cTPFP)
            total_true.append(cTPFN)

            cavg_loss = np.mean(total_loss)
            cavg_precision = np.mean(total_right) / np.mean(total_pretrue)
            cavg_recall = np.mean(total_right) / np.mean(total_true)
            cavg_f1 = 2 * cavg_precision * cavg_recall / (cavg_precision + cavg_recall)

            if (batch_idx + 1) % data.print_every == 0 and batch_idx > 0:
                print(
                    colored("Classifier Model results at Epoch {} Batch {}: "
                           "#Average Loss: {:0.4f}  "
                           "#Average Prec: {:0.3f}  "
                           "#Average Recl: {:0.3f}  "
                           "#Average F1-V: {:0.3f}".format(_idx+1, batch_idx+1,
                                                           cavg_loss,
                                                           cavg_precision,
                                                           cavg_recall,
                                                           cavg_f1), 'green')
                      )

            if (batch_idx + 1) % data.evaluate_every == 0 and batch_idx > 0:
                davg_loss, dprecision, drecall, df1, *_ = evaluate(model, data, devData)
                # print('Evaluating ...', end='\r')
                print(
                    colored("DEV results at Epoch {} Batch {}:"
                            " #Average Loss: {:0.4f}  "
                            "#Average Prec: {:0.3f}  "
                            "#Average Recl: {:0.3f}  "
                            "#Average F1-V: {:0.3f}".format(_idx+1, batch_idx+1,
                                                           davg_loss,
                                                           dprecision,
                                                           drecall,
                                                           df1), 'cyan'))

                if davg_loss < best_dev_loss:
                    best_dev_loss = davg_loss
                    best_dev_loc = [_idx+1, batch_idx+1]
                    early_stop_count = 0
                    model_save(model.classifier, data)
                    print('Best Dev Result Occur at Epoch {} Batch {} and Saved'.format(best_dev_loc[0], best_dev_loc[1]))
                else:
                    early_stop_count += 1
                    if early_stop_count >= data.earlystop:
                        print('Classification Model early stop at Epoch {} Batch {}'.format(_idx+1, batch_idx))
                        print('Best Dev Result Occurred at Epoch {} Batch {}'.format(best_dev_loc[0], best_dev_loc[1]))
                        break
        if early_stop_count >= data.earlystop:
            break

    model.classifier = model_load(model.classifier, data)
    model.classifier.eval()
    for para in model.classifier.parameters():
        para.requires_grad = False
    paras = model.spanRegression.parameters()
    roptim = optim.Adam(paras, lr=data.HP_lr, weight_decay=data.HP_l2)

    rbest_dev_loc = []
    rbest_dev_loss = 100
    rearly_stop_count = 0
    for _ridx in range(data.HP_epoch):
        print("\nStart training the Ranker at Epoch / Total <==> {} / {} \n".format(_ridx+1, data.HP_epoch))
        random.shuffle(trainData)
        rtotal_loss = []
        rtotal_right = []
        rtotal_pretrue = []
        rtotal_true = []
        rtotal_golden = []

        cls_right = []
        cls_pretrue = []

        model.zero_grad()
        batch_size = data.HP_batch_size
        rtrain_batchs = len(trainData) // batch_size + 1

        for rbatch_idx in range(rtrain_batchs):
            print('Ranking Model at ' +
                  colored('Epoch {} <==>'.format(_ridx+1), 'red', 'on_grey') +
                  ' Batch / Total: ' +
                  bcolors.OKGREEN +
                  bcolors.UNDERLINE +
                  bcolors.BOLD +
                  '# {} / {} #'.format(rbatch_idx+1, rtrain_batchs) +
                  bcolors.ENDC,
                  end='\r')

            model.train()
            model.classifier.eval()
            rstart = rbatch_idx * batch_size
            rend = (rbatch_idx + 1) * batch_size
            rend = rend if rend < len(trainData) else len(trainData)
            rbatch_instance = trainData[rstart: rend]
            if not rbatch_instance:
                continue
            batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
            batch_char_seq_lengths, batch_char_seq_recover, batch_labels,  batch_mask, batch_texts \
                = reformat_input_data(rbatch_instance, data.HP_gpu, True)

            loss, rTP, rTPFP, rTPFN, cTP, cTPFP, cTPFN = model.forward(batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_char_seq,
                                                    batch_char_seq_lengths, batch_char_seq_recover, batch_labels,
                                                    batch_mask, batch_texts, training=True, rankingflag=True)
            loss.backward()
            roptim.step()
            model.zero_grad()

            rtotal_loss.append(loss.item())
            rtotal_right.append(rTP)
            rtotal_pretrue.append(rTPFP)
            rtotal_true.append(rTPFN)
            rtotal_golden.append(cTPFN)

            cls_right.append(cTP)
            cls_pretrue.append(cTPFP)

            ravg_loss = np.mean(rtotal_loss)
            ravg_precision = np.mean(rtotal_right) / np.mean(rtotal_pretrue)
            ravg_recall = np.mean(rtotal_right) / np.mean(rtotal_true)
            ravg_f1 = 2 * ravg_precision * ravg_recall / (ravg_precision + ravg_recall)
            travg_recall = np.mean(rtotal_right) / np.mean(rtotal_golden)
            travg_f1 = 2 * ravg_precision * travg_recall / (ravg_precision + travg_recall)

            cls_precision = np.mean(cls_right) / np.mean(cls_pretrue)
            cls_recall = np.mean(cls_right) / np.mean(rtotal_golden)
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall)

            if (rbatch_idx + 1) % data.print_every == 0 and rbatch_idx > 0:
                print(colored("Ranking model results at Epoch {} Batch {}: "
                              "#Average Loss: {:0.4f}  "
                              "#Average Prec: {:0.3f}  "
                              "#Average Recl: {:0.3f}  "
                              "#Average F1-V: {:0.3f}  "
                              "#Actual Recall: {:0.3f}  "
                              "#Actual F1: {:0.3f}"
                              "\n#Its CLF #Prec: {:0.3f}  "
                              "#Recl: {:0.3f}  "
                              "#F1-V: {:0.3f}".format(_ridx+1,
                                                      rbatch_idx+1,
                                                      ravg_loss,
                                                      ravg_precision,
                                                      ravg_recall,
                                                      ravg_f1,
                                                      travg_recall,
                                                      travg_f1,
                                                      cls_precision,
                                                      cls_recall,
                                                      cls_f1), 'green')
                      )

            if (rbatch_idx + 1) % data.evaluate_every == 0 and rbatch_idx > 0:
                davg_loss, dprecision, drecall, df1, dtrecall, dtf1, dcls_precision, dcls_recall, dcls_f1 = evaluate(model, data, devData, rankingflag=True)
                # print('Evaluating ...', end='\r')
                print(colored("DEV results: #Average Loss: {:0.4f}  "
                              "#Average Prec: {:0.3f}  "
                              "#Average Recl: {:0.3f}  "
                              "#Average F1-V: {:0.3f}  "
                              "#Actual AVG Recl: {:0.3f}  "
                              "#Actual AVG F1: {:0.3f}  "
                              "\n#Its CLF #Prec: {:0.3f}  "
                              "#Recl: {:0.3f}  "
                              "#F1-V: {:0.3f}".format(davg_loss,
                                                      dprecision,
                                                      drecall,
                                                      df1,
                                                      dtrecall,
                                                      dtf1,
                                                      dcls_precision,
                                                      dcls_recall,
                                                      dcls_f1), 'cyan'))

                if davg_loss < rbest_dev_loss:
                    rbest_dev_loss = davg_loss
                    rbest_dev_loc = [_ridx+1, rbatch_idx+1, davg_loss, dprecision, drecall, df1, dtrecall, dtf1, dcls_precision, dcls_recall, dcls_f1]
                    rearly_stop_count = 0
                    model_save(model, data, 'rnk')
                    print('Best Dev Result Occur at Epoch {} Batch {} and Saved'.format(rbest_dev_loc[0],
                                                                                        rbest_dev_loc[1]))
                else:
                    rearly_stop_count += 1
                    if rearly_stop_count >= data.earlystop:
                        print('Ranking Model early stop at Epoch {} Batch {}'.format(_ridx+1, rbatch_idx))
                        print('Best Dev Result Occurred at Epoch {} Batch {}'.format(rbest_dev_loc[0], rbest_dev_loc[1]))
                        break
        if rearly_stop_count >= data.earlystop:
            break
    return rbest_dev_loc

def test(data, model, testData):
    print('Testing the data')
    model = model_load(model, data, stype='rnk')
    model.eval()
    batch_size = data.HP_batch_size
    test_batchs = len(testData) // batch_size + 1
    total_loss = []
    total_right = []
    total_pretrue = []
    total_true = []
    total_golden = []
    cls_right = []
    cls_pretrue = []

    batch_results = []

    for batch_idx in range(test_batchs):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        end = end if end < len(testData) else len(testData)
        batch_instance = testData[start: end]
        if not batch_instance:
            continue
        batch_words_seq, batch_pos_seq, batch_seq_lengths, batch_seq_recover, batch_char_seq, \
        batch_char_seq_lengths, batch_char_seq_recover, batch_labels, batch_mask, batch_texts \
            = reformat_input_data(batch_instance, data.HP_gpu, True)

        loss, TP, TPFP, TPFN, cTP, cTPFP, cTPFN, cpredict, rpredict = model.forward(batch_words_seq,
                                                                                    batch_pos_seq,
                                                                                    batch_seq_lengths,
                                                                                    batch_char_seq,
                                                                                    batch_char_seq_lengths,
                                                                                    batch_char_seq_recover,
                                                                                    batch_labels,
                                                                                    batch_mask,
                                                                                    batch_texts,
                                                                                    training=False,
                                                                                    rankingflag=True,
                                                                                    dump_result=True)

        total_golden.append(cTPFN)
        total_loss.append(loss.item())
        total_right.append(TP)
        total_pretrue.append(TPFP)
        total_true.append(TPFN)
        cls_right.append(cTP)
        cls_pretrue.append(cTPFP)

        batch_results.append([batch_texts, batch_labels, cpredict, rpredict])

    # &&&&&&&&&&&&&&& Dump Test Result &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    test_jsonlines = []
    for batch_test in batch_results: # batch result
        sents, goldens, classificatons, rankings = batch_test
        for sent, gold, clas, rank in zip(sents, goldens, classificatons, rankings):
            test_jsonlines.append({
                'sentence': ' '.join(sent),
                'golden': gold[0],
                'classifier': clas,
                'ranker': rank
            })
    test_filename = 'Test_dropout_{}_lr_{}_MaxLen_{}_HD_{}_POS_{}_ELMO_{}.txt'.format(data.HP_dropout,
                                                                                      data.HP_lr,
                                                                                      data.term_span,
                                                                                      data.HP_hidden_dim,
                                                                                      data.pos_as_feature,
                                                                                      data.use_elmo)
    out_filedir = os.path.join(data.output_dir, test_filename)
    with open(out_filedir, 'w') as tfilout:
        for lin in test_jsonlines:
            json.dump(lin, tfilout)
            tfilout.write('\n')

    avg_loss = np.mean(total_loss)
    precision = np.mean(total_right) / np.mean(total_pretrue)
    recall = np.mean(total_right) / np.mean(total_true)
    f1 = 2 * precision * recall / (precision + recall)
    trecall = np.mean(total_right) / np.mean(total_golden)
    tf1 = 2 * precision * trecall / (precision + trecall)

    cls_precision = np.mean(cls_right) / np.mean(cls_pretrue)
    cls_recall = np.mean(cls_right) / np.mean(total_golden)
    cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall)

    return [avg_loss, precision, recall, f1, trecall, tf1, cls_precision, cls_recall, cls_f1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Term Extraction')
    # data and saves
    parser.add_argument('--data_dir', type=str, default='./data/gene_term_format_by_sentence.json', help='Path to data')
    parser.add_argument('--savemodel', default="saves/model/")
    parser.add_argument('--savedset', default=None, help='Dir of saved data setting')
    parser.add_argument('--output', help='The test result output dir', default='saves/output/')

    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='test')
    parser.add_argument('--config', help='Configuration File', default='None')
    parser.add_argument('--wordemb', help='Embedding for words', default=None)
    parser.add_argument('--charemb', help='Embedding for chars', default=None)
    parser.add_argument('--hwdim', type=int, default=100, help='The hidden dim of word embedding and LSTM')
    parser.add_argument('--elmodim', type=int, default=100, help='The hidden dim of Elmo Feature')
    parser.add_argument('--posdim', type=int, default=100, help='The hidden dim of POS Feature')

    parser.add_argument('--use_gpu', type=bool, default=False, help='whether use gpu')
    parser.add_argument('--gpuid', type=int, default=3)
    # parser.add_argument('--model', help='Choose a model', choices=['spanc', 'multi', 'spancr'], default='spanc')

    parser.add_argument('--dropout', type=float, default=0.5, help='The probability of a number replaced by zero')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--term_ratio', type=float, default=0.23, help='The ratio of total words to be terms')
    parser.add_argument('--early_stop', type=int, default=36, help='The patience of bad dev result')
    parser.add_argument('--max_length', type=int, default=5, help='The maximum length of the term span')
    parser.add_argument('--evaluate_every', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--use_pos', type=bool, default=False, help='Use POS-tagging feature')
    parser.add_argument('--use_elmo', type=bool, default=False, help='Use Elmo feature embeddings')

    args = parser.parse_args()

    data = Data(args)
    #torch.cuda.is_available()
    if data.HP_gpu:
        torch.cuda.set_device(args.gpuid)

    instances = data.all_instances
    total_num = len(instances)
    ratio = data.data_ratio
    train_ratio = round(ratio[0] * total_num)
    dev_ratio = round((ratio[0] + ratio[1]) * total_num)
    train_instances = instances[:train_ratio+1]
    dev_instances = instances[train_ratio+1:dev_ratio+1]
    test_instances = instances[dev_ratio+1:]
    if args.savedset is not None:
        data.data_save_file = args.savedset
        data.save(args.savedset)

    model = niceRanking(data)
    if data.HP_gpu:
        model.cuda()
    if args.status == 'train':
        dev_best_result = training(model, data, train_instances, dev_instances)
        test_result = test(data, model, test_instances)
    elif args.status == 'test':
        test_result = test(data, model, test_instances)

    print(bcolors.BOLD+colored('\n\nSummary of this training/testing:', 'red')+bcolors.ENDC)
    print(colored('Settings:\n', 'blue'))
    print('\tDropout:\t{}'.format(data.HP_dropout))
    print('\tHidden_Dim:\t{}'.format(data.HP_hidden_dim))
    print('\tLearning Rate:\t{}'.format(data.HP_lr))
    print('\tLR Decay:\t{}'.format(data.HP_lr_decay))
    print('\tMax_Span_Len:\t{}'.format(data.term_span))
    print('\tSeed:\t{}'.format(seed_num))
    print(colored('\nDev & Test Results', 'yellow'))
    print('\tLoss\tPrec\tRec\tF1\tTRec\tTF1\tCLS_Prec\tCLS_Rec\tCLS_F1')
    if args.status == 'train':
        print('Dev:\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}'.format(*dev_best_result[2:]))

    print('Test:\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}'.format(*test_result))

    summary_filename = './logs/'+'Dropout_{}_lr_{}_MaxLen_{}_HD_{}_POS_{}_ELMO_{}_{}.txt'.format(data.HP_dropout,
                                                                                                 data.HP_lr,
                                                                                                 data.term_span,
                                                                                                 data.HP_hidden_dim,
                                                                                                 data.pos_as_feature,
                                                                                                 data.use_elmo,
                                                                                                 args.status)
    with open(summary_filename, 'w') as filout:
        filout.write('Summary of this training/testing:\n')
        filout.write('Settings:\n')
        filout.write('\tDropout:\t{}\n'.format(data.HP_dropout))
        filout.write('\tHidden_Dim:\t{}\n'.format(data.HP_hidden_dim))
        filout.write('\tLearning Rate:\t{}\n'.format(data.HP_lr))
        filout.write('\tLR Decay:\t{}\n'.format(data.HP_lr_decay))
        filout.write('\tMax_Span_Len:\t{}\n'.format(data.term_span))
        filout.write('\tSeed:\t{}\n'.format(seed_num))
        filout.write('\nDev & Test Results\n')
        filout.write('\tLoss\tPrec\tRec\tF1\tTRec\tTF1\tCLS_Prec\tCLS_Rec\tCLS_F1\n')
        if args.status == 'train':
            filout.write('Dev:\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\n'.format(*dev_best_result[2:]))
        filout.write('Test:\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\t{:0.4f}\n'.format(*test_result))
    print('Summary is written to {}'.format(summary_filename))
