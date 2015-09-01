# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:54:54 2015

@author: wangwenya
"""
import numpy as np
from rnn.propagation import *
import cPickle
from sklearn import metrics

def evaluate(data_split, model_file, d, c, mixed = False):
    
    dic_file = open("w2v_output_comma.txt", "r")
    dic = dic_file.readlines()

    dictionary = {}

    for line in dic:
        word_vector = line.split(",")
        word = word_vector[0]
    
        vector_list = []
        for element in word_vector[1:len(word_vector)-1]:
            vector_list.append(element)
        
        vector = np.asarray(vector_list)
        dictionary[word] = vector
    
    labelCorrect = 0
    labelIncorrect = 0
    num_nodes = 0
    labelConfusion = np.zeros((c, c))
    
#    vocab, rel_list, tree_dict = \
#        cPickle.load(open(data_split, 'rb'))
    
    rel_list, tree_dict = \
        cPickle.load(open(data_split, 'rb'))
        
    #train_trees = tree_dict['train']
    #test_trees = tree_dict['test']
    test_trees = tree_dict
    
    params, vocab, rel_list1, rel_list2 = cPickle.load(open(model_file, 'rb'))
    (rel_dict1, rel_dict2, Wv_1, Wv_2, Wc, b_1, b_2, b_c, We) = params
    
    bad_trees = []
    for ind, tree in enumerate(test_trees):
        if tree.get(0).is_word == 0:
            # print tree.get_words()
            bad_trees.append(ind)
            continue

    # print 'removed', len(bad_trees)
    for ind in bad_trees[::-1]:
        #test_trees.pop(ind)
        test_trees = np.delete(test_trees, ind)
        
    true = []
    predict = []    
    
    for tree in test_trees:
        for node in tree.get_nodes():
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape( (d, 1) )
            elif node.word.lower in dictionary.keys():
                if mixed:
                    node.vec = (dictionary[node.word.lower()].append(2 * np.random.rand(50) - 1)).reshape( (d, 1) )
                else:
                    node.vec = dictionary[node.word.lower()]
            else:
                node.vec = np.random.rand(d,1)
            
        forward_prop(params, tree, d, c, labels=False)
        for node in tree.get_nodes():
            num_nodes += 1
            max = 0
            predict_label = node.predict_label
            for entry in predict_label:
                if entry > max:
                    max = entry
            node.prediction = np.nonzero(predict_label==(max))[0][0]
            #add prediction to predict list
            predict.append(node.prediction)
            true.append(node.trueLabel)
            """
            if node.trueLabel == node.prediction:
                labelCorrect += 1
            else:
                labelIncorrect += 1
                
            labelConfusion[node.trueLabel][node.prediction] += 1
            """
    """            
    accuracy = float(labelCorrect) / float(num_nodes)
    recall = float(labelConfusion[1][1]) / (labelConfusion[1][0] + labelConfusion[1][1] + labelConfusion[1][2] + labelConfusion[1][3])
    precision = float(labelConfusion[1][1]) / (labelConfusion[0][1] + labelConfusion[1][1] + labelConfusion[2][1] + labelConfusion[3][1])
    F1 = 2 * recall * precision / (recall + precision)
    """
    #use sklearn performance metrics
    #target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print (metrics.classification_report(true, predict))
    print "Confusion matrix from sklearn: \n", metrics.confusion_matrix(true,predict)
    
    print metrics.precision_recall_fscore_support(true, predict, average = 'macro')
    """
    print "Tested number of labels: ", num_nodes
    print "Labels correct: ", labelCorrect
    print "Labels incorrect:", labelIncorrect

    print "Accuracy: ", accuracy
    print "Confusion matrix: ", labelConfusion
    print "Precision: ", precision
    print "recall: ", recall
    print "F1 score: ", F1
    """
            
    #print predict