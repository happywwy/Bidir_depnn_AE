# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:28:35 2015

@author: wangwenya
"""

from dtree_util import *
import gen_util as gen
import sys, cPickle, random, os
from numpy import *

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None
    
wordnet_lemmatizer = WordNetLemmatizer()


f = open('data_semEval/raw_parses_res', 'r')
sentence_file = open('data_semEval/sentence_res', 'r')
data = f.readlines()
plist = []
tree_dict = []
vocab = []
rel_list = []

label_file = open('data_semEval/aspectTerm_res', 'r')
#opinion_positive_file = open('opinion-lexicon/positive-words.txt', 'r')
#opinion_negative_file = open('opinion-lexicon/negative-words.txt', 'r')
#
#opinion_positive = opinion_positive_file.read().splitlines()
#opinion_negative = opinion_negative_file.read().splitlines()
label_sentence = open('data_semEval/addsenti_res.txt', 'r')

for line in data:
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

                    #print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
            
        plist.append((rel,final_deps))

    else:
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind

        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word#.lower()

        tree = dtree(nodes)
        
 
        #add code corresponding to customer review data       
        #labels = label_file.readline().strip()
        
        opinion_words = []
        sentence = sentence_file.readline().strip()

        terms = word_tokenize(sentence)
        pos_review = nltk.pos_tag(terms)
        
        labeled_sent = label_sentence.readline().strip()
        if '##' in labeled_sent:
            opinion_line = labeled_sent.split('##')[1].strip()
            opinions = opinion_line.split(',')
            for opinion in opinions:
                opinion_word = opinion.strip().split(' ')[:-1]
                for word in opinion_word:
                    opinion_words.append(word)
        
        word_pos = {}
        for item in pos_review:
            word_pos[item[0]] = penn_to_wn(item[1])
        '''    
        #check if the sentence has aspects
        if label[0] != '':
        
            aspects = label[0].split(',')
            for aspect in aspects:
                aspect = aspect.split('[')[0].strip()
                for word in aspect.split(' '):
                    if word != '':
                        aspect_words.append(word.strip())
            
        if len(label) > 3:
            sentiments = label[2:-1]
            for sentiment in sentiments:
                sentiment = sentiment.strip()
                sentiment = sentiment.split(' ')
                
                for word in sentiment[:-1]:
                    opinion_words.append(word)
                
                """
                if sentiment[len(sentiment) - 1] == '-1':
                    for word in sentiment[:-1]:
                        negative_words.append(word)
                else:
                    for word in sentiment[:-1]:
                        positive_words.append(word)
                """
        '''
            
        
        aspect_term = label_file.readline().rstrip()
        
        if aspect_term != 'NIL':
            #aspect_term += " "
            aspects = aspect_term.split(' ')
        
            '''
            for term in nodes:
                ind = nodes.index(term)
                
                if term in word_pos.keys():
                
                    if term in aspects:
                        tree.get(ind).trueLabel = 1
                    elif term in opinion_positive:
                        tree.get(ind).trueLabel = 2
                    elif term in opinion_negative:
                        tree.get(ind).trueLabel = 2
                        
                    tree.get(ind).word = term.lower()
                    if word_pos[term] != None:
                        tree.get(ind).word = wordnet_lemmatizer.lemmatize(term.lower(), word_pos[term])
                    
                    
            '''
        
            for term in nodes:
                ind = nodes.index(term)
                
              
                if term in word_pos.keys():
                    
                    if term in aspects:
                        tree.get(ind).trueLabel = 1
                
                    elif term.lower() in opinion_words:
                        tree.get(ind).trueLabel = 2
                    
                    tree.get(ind).word = term.lower()
                    
                    if word_pos[term] != None:
                        tree.get(ind).word = wordnet_lemmatizer.lemmatize(term.lower(), word_pos[term])                   
            

        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel)

        tree_dict.append(tree)  
        
        for node in tree.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())
                
            node.ind = vocab.index(node.word.lower())
            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []
        
        
#word_embedding = gen.gen_word_embeddings(25, len(vocab))


print 'rels: ', len(rel_list)
print 'vocab: ', len(vocab)

cPickle.dump((vocab, rel_list, tree_dict), open("data_semEval/final_input_res", "wb"))
#cPickle.dump(word_embedding, open("initial_We_res", "wb"))


