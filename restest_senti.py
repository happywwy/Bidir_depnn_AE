# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:57:51 2015

@author: happywwy1991
"""

new_file = open("restest_senti", "w")

f = open("parsedSentence_restest", "r")
sentences = f.read().splitlines()

positive_file = open("util/opinion-lexicon/positive-words.txt", "r")
negative_file = open("util/opinion-lexicon/negative-words.txt", "r")

positive = positive_file.read().splitlines()
negative = negative_file.read().splitlines()

new = []

for line in sentences:
    tokens = line.split()
    new_file.write(line)
    new_file.write(';')
    for token in tokens:
        token = token.lower()
        if token in positive:
            new_file.write(token)
            new_file.write('+' + ' ')
        elif token in negative:
            new_file.write(token)
            new_file.write('-' + ' ')
    new_file.write('\n')
    
    
new_file.close()