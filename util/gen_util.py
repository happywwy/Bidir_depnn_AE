
import numpy as np

# - given a vector containing all parameters, return a list of unrolled parameters
# - specifically, these parameters, as described in section 3 of the paper, are:
#   - rel_dict, dictionary of {dependency relation r: composition matrix W_r}
#   - Wv, the matrix for lifting a word embedding to the hidden space
#   - b, bias term
#   - We, the word embedding matrix
#def unroll_params(arr, d, len_voc, rel_list):
def unroll_params(arr, d, c, len_voc, rel_list1, rel_list2):

    mat_size = d * d
    #classification
    matClass_size = c * d
    rel_dict1 = {}
    rel_dict2 = {}
    ind = 0

    for r in rel_list1:
        rel_dict1[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size
        
    for r in rel_list2:
        rel_dict2[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv_1 = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    Wv_2 = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    #Wc
    Wc = arr[ind : ind + matClass_size].reshape( (c, d) )
    ind += matClass_size

    b_1 = arr[ind : ind + d].reshape( (d, 1) )
    ind += d
    
    b_2 = arr[ind : ind + d].reshape( (d, 1) )
    ind += d
    
    #b_c
    b_c = arr[ind : ind + c].reshape( (c, 1) )
    ind += c

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    #return [rel_dict, Wv, b, We]
    return [rel_dict1, rel_dict2, Wv_1, Wv_2, Wc, b_1, b_2, b_c, We]


# roll all parameters into a single vector
def roll_params(params, rel_list1, rel_list2):
    #(rel_dict, Wv, b, We) = params
    (rel_dict1, rel_dict2, Wv_1, Wv_2, Wc, b_1, b_2, b_c, We) = params

    rels_1 = np.concatenate( [rel_dict1[key].ravel() for key in rel_list1] )
    rels_2 = np.concatenate( [rel_dict2[key].ravel() for key in rel_list2] )
    #return concatenate( (rels, Wv.ravel(), b.ravel(), We.ravel() ) )
    return np.concatenate( (rels_1, rels_2, Wv_1.ravel(), Wv_2.ravel(), Wc.ravel(), b_1.ravel(), b_2.ravel(), b_c.ravel(), We.ravel() ) )


# randomly initialize all parameters
#def gen_dtrnn_params(d, rels):
def gen_dtrnn_params(d, c, rels):
    """
    Returns (dict_1{rels:[mat]}, dict_2{rels:[mat]}, Wv_1, Wv_2, Wc, b_1, b_2, b_c)
    """
    r = np.sqrt(6) / np.sqrt(2 * d + 1)
    r_Wc = 1.0 / np.sqrt(d)
    rel_dict1 = {}
    rel_dict2 = {}
    for rel in rels:
        rel_dict1[rel] = np.random.rand(d, d) * 2 * r - r
        rel_dict2[rel] = np.random.rand(d, d) * 2 * r - r

    return (
	    rel_dict1,
          rel_dict2,
	    np.random.rand(d, d) * 2 * r - r,
          np.random.rand(d, d) * 2 * r - r,
          np.random.rand(c, d) * 2 * r_Wc - r_Wc,
	    np.zeros((d, 1)),
          np.zeros((d, 1)),
          np.random.rand(c, 1)
          )
 
 
#generate word embedding matrix
def gen_word_embeddings(d, total_num):

    for ind in range(total_num):
        word_vec = np.random.rand(d, 1)
        if ind == 0:
            word_embedding = word_vec
        else:
            word_embedding = np.c_[word_embedding, word_vec]
     
    return word_embedding


# returns list of zero gradients which backprop modifies
#def init_dtrnn_grads(rel_list, d, len_voc):
def init_dtrnn_grads(rel_list1, rel_list2, d, c, len_voc):

    rel_grads1 = {}
    rel_grads2 = {}
    for rel in rel_list1:
	  rel_grads1[rel] = np.zeros( (d, d) )
   
    for rel in rel_list2:
	  rel_grads2[rel] = np.zeros( (d, d) )

    return [
	    rel_grads1,
          rel_grads2,
	    np.zeros((d, d)),
          np.zeros((d, d)),
          np.zeros((c, d)),
	    np.zeros((d, 1)),
          np.zeros((d, 1)),
          np.zeros((c, 1)),
	    np.zeros((d, len_voc))
	    ]


# random embedding matrix for gradient checks
def gen_rand_we(len_voc, d):
    r = np.sqrt(6) / np.sqrt(51)
    we = np.random.rand(d, len_voc) * 2 * r - r
    return we
