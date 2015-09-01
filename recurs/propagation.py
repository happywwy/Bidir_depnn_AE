import numpy as np
from util.math_util import *
import random

# - QANTA's forward propagation. the labels argument indicates whether
#   you want to compute errors and deltas at each node or not. for training,
#   you obviously want those computations to occur, but during testing they
#   unnecessarily slow down feature computation

#define softmax function
def softmax(v):
    v = np.array(v)
    max_v = np.amax(v)
    e = np.exp(v - max_v)
    dist = e / np.sum(e)

    return dist

    
def der_tanh(x):
    return 1-np.tanh(x)**2

#def forward_prop(params, tree, d, labels=True):
def forward_prop(params, tree, d, c, labels=True):

    # node.finished = 0
    tree.reset_finished()

    to_do = tree.get_nodes()

    #(rel_dict, Wv, b, We) = params
    (rel_dict1, rel_dict2, Wv_1, Wv_2, Wc, b_1, b_2, b_c, We) = params

    # - wrong_ans is 100 randomly sampled wrong answers for the objective function
    # - only need wrong answers when computing error

    """
    if labels:
        random.shuffle(tree.ans_list)
        wrong_ans = [We[:, ind] for ind in tree.ans_list[0:100]]
    """

    # forward prop
    while to_do:
        curr = to_do.pop(0)

        # node is leaf
        if len(curr.kids) == 0:

            # activation function is the normalized tanh
            # compute hidden state
            curr.p = tanh(Wv_1.dot(curr.vec) + b_1)
            #curr.p_norm = curr.p / linalg.norm(curr.p)
            #curr.ans_error = 0.0
            curr.label_error = 0.0
            curr.label_delta = 0.0
            # wwy add classification
            #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)

        else:

            # - root isn't a part of this! 
            # - more specifically, the stanford dep. parser creates a superficial ROOT node
            #   associated with the word "root" that we don't want to consider during training
            # 'root' is the last one to be popped
            if len(to_do) == 0:
                # 'root' only has one kid, which is the root word
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p
                #curr.p_norm = tree.get(ind).p_norm
                #curr.ans_error = 0.
                curr.label_error = 0.
                curr.label_delta = 0.
                #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)
                continue

            # check if all kids are finished
            all_done = True
            for ind, rel in curr.kids:
                if tree.get(ind).finished == 0:
                    all_done = False
                    break

            # if not, push the node back onto the queue
            if not all_done:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                kid_sum = zeros( (d, 1) )
                for ind, rel in curr.kids:
                    curr_kid = tree.get(ind)

                    try:
                        #kid_sum += rel_dict[rel].dot(curr_kid.p_norm)
                        #wwy only count direct influence
                        kid_sum += rel_dict1[rel].dot(curr_kid.p)
                        #kid_sum += rel_dict[rel].dot(curr_kid.vec)

                    # - this shouldn't happen unless the parser spit out a seriously 
                    #   malformed tree
                    except KeyError:
                        print 'forward propagation error'
                        print tree.get_words()
                        print curr.word, rel, tree.get(ind).word
                
                kid_sum += Wv_1.dot(curr.vec)
                curr.p = tanh(kid_sum + b_1)
                #curr.p_norm = curr.p / linalg.norm(curr.p)
                
                #add prediction
                #curr.predict_label = softmax(Wc.dot(curr.p) + b_c)

        curr.finished = 1
        
        
    #add another path from top to bottom
    tree.reset_finished()

    to_do = tree.get_nodes()
    
    while to_do:
        curr = to_do.pop(0)

        # node is root or top
        if len(curr.parent) == 0:
            #root
            if curr.word == 'ROOT':
            
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p
                #add final vector representation
                curr.fin = tanh(Wv_2.dot(curr.p) + b_2)
            #isolated
            else:
                curr.fin = tanh(Wv_2.dot(curr.p) + b_2)
            
            curr.predict_label = softmax(Wc.dot(curr.fin) + b_c)


        else:
            
            par_ind, par_rel = curr.parent[0]
            
            #parent is root
            if tree.get(par_ind).word == 'ROOT':
                curr.fin = tanh(Wv_2.dot(curr.p) + b_2)
                curr.predict_label = softmax(Wc.dot(curr.fin) + b_c)
                curr.finished = 1


            elif tree.get(par_ind).finished == 0:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                par_sum = zeros( (d, 1) )

                curr_par = tree.get(par_ind)
                try:
                    #kid_sum += rel_dict[rel].dot(curr_kid.p_norm)
                    par_sum += rel_dict2[par_rel].dot(curr_par.fin)
                    # - this shouldn't happen unless the parser spit out a seriously 
                    #   malformed tree
                except KeyError:
                    print 'forward propagation error'
                    print tree.get_words()
                    print curr.word, rel, tree.get(ind).word
                
                par_sum += Wv_2.dot(curr.p)
                curr.fin = tanh(par_sum + b_2)
                curr.predict_label = softmax(Wc.dot(curr.fin) + b_c)


        # error and delta
        if labels:

            curr.label_error = 0.0
            curr.label_delta = zeros( (c, 1) )
            true_label = zeros( (c, 1) )
            for i in range(c):
                if curr.trueLabel == i:
                    true_label[i] = 1
                    
            curr.true_class = true_label
                    
            curr.label_delta = curr.predict_label - curr.true_class
            curr.label_error = - (np.multiply(log(curr.predict_label), curr.true_class).sum())

        curr.finished = 1


# computes gradients for the given tree and increments existing gradients
#def backprop(params, tree, d, len_voc, grads):
def backprop(params, tree, d, c, len_voc, grads, mixed = False):
    import numpy as np

    #(rel_dict, Wv, b) = params
    (rel_dict1, rel_dict2, Wv_1, Wv_2, Wc, b_1, b_2, b_c) = params
    
    #add backpropagation from leaf node
    tree.reset_finished()
    to_do = []
    for node in tree.get_nodes():
        node.delta_prop = 0.0
        if len(node.kids) == 0:
            to_do.append((node, zeros((d, 1))))
            
    while to_do:
        curr = to_do.pop(0)
        node = curr[0]
        delta_up = curr[1]

        
        delta_Wc = node.label_delta.dot(node.fin.T) 
        delta_bc = node.label_delta
        
        #delta_node
        delta = Wc.T.dot(node.label_delta)
        curr_der = der_tanh(node.fin)
        #node.delta_node = np.multiply(delta, curr_der)
        node.delta_node = np.multiply(delta + delta_up, curr_der)
        
        #node.delta_full = delta_up + node.delta_node
        node.delta_full = node.delta_node
        
        #add delta from node.fin to node.p
        node.delta_fin_p = Wv_2.T.dot(node.delta_full)
        node.finished = 1 
        
        #if node has parent
        if len(node.parent) > 0:
            par_ind, par_rel = node.parent[0]
            #parent is not root
            if tree.get(par_ind).word != 'ROOT':
                curr_par = tree.get(par_ind)
            
                grads[1][par_rel] += node.delta_full.dot(curr_par.fin.T)
                #to_do.append((curr_par, rel_dict2[par_rel].T.dot(node.delta_full)))
                
                grads[3] += node.delta_full.dot(node.p.T)
                grads[4] += delta_Wc
                grads[6] += node.delta_full
                grads[7] += delta_bc
                
                curr_par.delta_prop += rel_dict2[par_rel].T.dot(node.delta_full)
                
                #check if parent has already be appended
                all_done = True
                for ind, rel in curr_par.kids:
                    if tree.get(ind).finished == 0:
                        all_done = False
                        break
                    
                if all_done:
                    to_do.append((curr_par, curr_par.delta_prop))
                    

                    
            #if parent is root
            else:
                grads[3] += node.delta_full.dot(node.p.T)
                grads[4] += delta_Wc
                grads[6] += node.delta_full
                grads[7] += delta_bc
                
        #top node but not root        
        elif len(node.parent) == 0 and node.word != 'ROOT':
            grads[3] += node.delta_full.dot(node.p.T)
            grads[4] += delta_Wc
            grads[6] += node.delta_full
            grads[7] += delta_bc
                
          

    #start the backpropagation from root
    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)

    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, zeros((d, 1)) ) ]

    while to_do:
        curr = to_do.pop()
        node = curr[0]
        #parent delta
        delta_down = curr[1]
        
        #delta_Wc
        #delta_Wc = node.label_delta.dot(node.p.T)    
        #delta_bc = node.label_delta
        
        #delta_node
        #delta = Wc.T.dot(node.label_delta)
        #curr_der = der_tanh(node.p)
        #node.delta_node = np.multiply(delta, curr_der)
        
        node.delta_end = delta_down + node.delta_fin_p

        # non-leaf node
        if len(node.kids) > 0:
            

            #act = pd + node.ans_delta
            #node.delta_i = df.dot(act)

            for ind, rel in node.kids:

                curr_kid = tree.get(ind)
                #W_rel
                #grads[0][rel] += node.delta_i.dot(curr_kid.p_norm.T)
                grads[0][rel] += node.delta_end.dot(curr_kid.p.T)
                #change to word vector
                #grads[0][rel] += node.delta_full.dot(curr_kid.vec.T)
                #to_do.append( (curr_kid, rel_dict[rel].T.dot(node.delta_i) ) )
                to_do.append( (curr_kid, rel_dict1[rel].T.dot(node.delta_end) ) )

            #grads[1] += node.delta_i.dot(node.vec.T)
            grads[2] += node.delta_end.dot(node.vec.T)
            #grads[2] += delta_Wc
            #grads[2] += node.delta_i
            grads[5] += node.delta_end
            #grads[4] += delta_bc
            #grads[3][:, node.ind] += Wv.T.dot(node.delta_i).ravel()
            if mixed:
                grads[8][50:, node.ind] += Wv_1.T.dot(node.delta_end).ravel()[50:]
            else:
                grads[8][:, node.ind] += Wv_1.T.dot(node.delta_end).ravel()

        # leaf
        else:
            #act = pd + node.ans_delta
            #df = dtanh(node.p)

            #node.delta_i = df.dot(act)
            #grads[1] += node.delta_i.dot(node.vec.T)
            grads[2] += node.delta_end.dot(node.vec.T)
            #grads[2] += delta_Wc
            #grads[2] += node.delta_i
            grads[5] += node.delta_end
            #grads[4] += delta_bc
            #grads[3][:, node.ind] += Wv.T.dot(node.delta_i).ravel()
            if mixed:
                grads[8][50:, node.ind] += Wv_1.T.dot(node.delta_end).ravel()[50:]
            else:
                grads[8][:, node.ind] += Wv_1.T.dot(node.delta_end).ravel()
