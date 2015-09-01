# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:36:32 2015

@author: wangwenya
"""

f = open("util/data_semEval/addsenti_res.txt", "r")

sentences = f.read().splitlines()
new_sentences = open("util/data_semEval/sentence_res", "w")

for sentence in sentences:
    split_sentence = sentence.split('##')
    if split_sentence[0] != '':
        new_sentences.write(split_sentence[0])
        new_sentences.write('\n')

new_sentences.close()        



