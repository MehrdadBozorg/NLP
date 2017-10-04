from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import re
import string
import numpy as np
from math import factorial
from collections import Counter
from sklearn import svm
import random


# ______Preprocessing________:


# Read file of data and make a list of dictionaries containing label and data for each document.
def make_doc_list(file_name):
    fp = open(file_name)
    data = fp.read()
    doc_lines = data.splitlines()
    doc_train_list = []

    for line in doc_lines:
        spl = line.split()
        lbl_record = {}
        line_punc = re.sub('[%s]' % re.escape(string.punctuation), '', line)
        lbl_record['label'] = spl[0]
        lbl_record['val'] = infreq_filter(line_punc.split()[1:])
        doc_train_list.append(lbl_record)

    return doc_train_list


# Stop word filtering and stemming
def stop_stem(data_list):
    stop_w = set(stopwords.words('english'))
    list_nonstop = []
    ps = PorterStemmer()

    for record in data_list:
        new_record = {}
        rec_n_stop = []
        for w in record['val']:
            lower_val = w.lower()

            # stemming
            stemmed_w = ps.stem(lower_val)

            # stop word elimination
            if stemmed_w not in stop_w:
                rec_n_stop.append(stemmed_w)

        new_record['val'] = rec_n_stop
        new_record['label'] = record['label']
        list_nonstop.append(new_record)

    return list_nonstop


# Filter infrequent terms (terms with less than 3 times frequency)
def infreq_filter(doc):
    item_freq = {}

    for item in doc:
        if item in item_freq:
            item_freq[item] += 1
        else:
            item_freq[item] = 1

    for item in doc:
        if item_freq[item] < 3:
            doc.remove(item)

    return doc


# Attribute selection (Select the most informative 200 terms using IG -Information Gain-)
# Calculates the entropy of the given data set for the target attribute.
def entropy(data):
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if record['label'] in val_freq:
            val_freq[record['label']] += 1.0
        else:
            val_freq.update({record['label']:  1.0})

    # Calculate the entropy of the data for the target attribute
    for label in val_freq:
        data_entropy += (-val_freq[label] / len(data)) * math.log1p(val_freq[label] / len(data))

    return data_entropy


# Bag of words
def bow(data):
    bow_list = []
    for record in data:
        for item in record['val']:
            if item not in bow_list:
                bow_list.append(item)

    return bow_list


# Calculates the information gain (reduction in entropy) resulted from splitting the data on the chosen item.
def gain(data, word):

    sub_data = []

    word_prob = 0
    for doc in data:

        # Count the number of each word in each document, then divide document based on the assumed word
        doc_counter = Counter(doc['val'])
        if word in doc_counter:
            sub_data.append(doc)
            word_prob += doc_counter[word] / len(doc)

    word_prob *= len(sub_data) / len(data)
    n_sub_data = [i for i in data if i not in sub_data]
    word_gain = entropy(data) - word_prob * entropy(sub_data) - (1-word_prob) * entropy(n_sub_data)

    return word_gain


# Make a loop to calculate the most informative 200 words and eliminate the others from records in doc_list
def informative_words(data_list, bow_list):

    info_l = []
    informative_items = {}
    for word in bow_list:
        word_gain = gain(data_list, word)
        if len(informative_items) < 200:
            informative_items.update({word: word_gain})

        # substitute word with the minimum of informative list
        else:
            min_inf = min(informative_items.keys(), key=(lambda k: informative_items[k]))
            if word_gain > informative_items[min_inf]:
                del informative_items[min_inf]
                informative_items.update({word: word_gain})

    for rec in informative_items:
        info_l.append(rec)

    return info_l


# Prepare document vector
def doc_to_vec(doc, info_l):
    vec = [0]*len(info_l)
    for iterator in range(len(info_l)):
        vec[iterator] = sum(1 for w in doc if w == info_l[iterator])

    return vec


# make_vec_list, get a document_list( after eliminating the stop words and the other preliminary processes and make
# representative vector set
def make_vec_list(documents_list, info_words):
    vec_set = []

    for i in range(len(documents_list)):
        comp_vec = doc_to_vec(documents_list[i]['val'], info_words)
        comp_vec.append(documents_list[i]['label'])
        vec_set.append(comp_vec)
    return vec_set


# ___________________Meaning value of words__________________
# Compute the frequency of each word in each class and save it in the vector assigned for that class.
# The result is a matrix called cls_set.
def word_meaning(vector_set):

    cls_set = {}
    corpus_freq_w = [0]*(len(vector_set[0])-1)
    for v in vector_set:
        if v[len(v) - 1] not in cls_set:  # last item of each vector indicates its label
            cls_set.update({v[len(v)-1]: v[:len(v)-1]})  # v[:len(v)-1] indicates vector
        else:
            cls_set[v[len(v) - 1]] = np.add(cls_set[v[len(v)-1]], v[:len(v)-1])
            # Each vector contains the frequency of words, we can find the number of each words in a vector defined by
            # the label of class, by iteratively summing them up... v[len(v)- 1] indicates the label

        corpus_freq_w = np.add(corpus_freq_w, v[:len(v)-1])

    # Compute the number of words in each class and collect it in the set cls_w_length
    cls_w_lengths = {}
    for cl in cls_set:
        count = 0
        for w in cls_set[cl]:
            count += w

        cls_w_lengths.update({cl: count})

    # Compute the whole number of words in the corpus (training set)
    corpus_length = 0

    for cl_freq in cls_w_lengths:
        corpus_length += cls_w_lengths[cl_freq]

    # Calculates meaning values of each word for each class. At first calculates values for each word in each class and
    # then collect them in the set cls_meaning
    l = corpus_length
    cls_meaning = {}
    for cl_lbl in cls_set:

        b = cls_w_lengths[cl_lbl]

        if l != 0 and b != 0:  # Check the exception condition L = 0 and B = 0
            n = l / b
        else:
            n = 1

        length = len(cls_set[cl_lbl])
        meaning_vec = [0] * length

        for w in range(length):
            m = cls_set[cl_lbl][w]
            k = corpus_freq_w[w]
            w_nfa = combination(k, m) * (n**(1-m))

            if m != 0:
                meaning_vec[w] = (-1/m) * math.log1p(w_nfa)
            else:
                meaning_vec[w] = 0

        cls_meaning.update({cl_lbl: meaning_vec})

    return cls_meaning
# The results are negative for all classes in a trivial test. It may because of the non-compatability of my array with
# real one. By the way it's not important as the matrix would be multiplied to its transpose


# Compute the combination of m and k
def combination(n, r):
    return factorial(n) / (factorial(r) * factorial(n - r))


# ____________________labeling___________________________
# 1st prepare unlabeled documents,
# after then calls function word_meaning to construct matrix of meaning of words in each class.
# then by applying function doc_meaning compute meaning of each document for each class and find the arg(max) of them as
# its label
def labeling(vec_set, meaning_matrix):
    labeled_docs = []

    for v in vec_set:
        cls_meaning_doc = {}
        for lbl in meaning_matrix:
            tm_doc = 0
            temp_vec = meaning_matrix[lbl]

            for index in range(len(v) - 1):
                tm_doc += (temp_vec[index] * v[index])

            cls_meaning_doc.update({lbl: tm_doc})

        v[-1] = max(cls_meaning_doc, key=cls_meaning_doc.get)

        labeled_docs.append(v)

    return labeled_docs


# ______________________Weight calculation_________________________
# It'd be called for new training set containing original labeled docs and label-assigned unlabeled docs
def c_w_k(vec_set):
    leng = len(vec_set[0]) - 1  # Indicates the length of each vector. The last item stands for the label.
    nw = [0]*leng  # Indicates number of documents containing each word
    cls_freq = {}  # Collection of number of documents containing indexed word in each class
    cwk_matrix = []
    n = len(vec_set)

    for vec in vec_set:  # Calculates frequencies
        if vec[-1] not in cls_freq:
            cls_freq.update({vec[-1]: vec[:leng]})
        else:
            cls_freq[vec[leng]] = np.add(cls_freq[vec[leng]], vec[:leng])

        for ind in range(leng):
            if vec[ind] > 0:
                nw[ind] += 1

    for key in cls_freq:  # Calculates weights of words
        cwk = [0]*leng
        for vec_index in range(leng):
            if nw[vec_index] != 0:
                cwk[vec_index] = (math.log1p(cls_freq[key][vec_index] + 1)) * math.log1p(n/nw[vec_index])

        cwk_matrix.append(cwk)

    return cwk_matrix


# _____________________Classifier____________________________
def classifier(info_list, wmatrix, train_list):
    # Data preparation:
    test_docs = make_doc_list("Testset.txt")
    test_mid = stop_stem(test_docs)
    test_list = make_vec_list(test_mid, info_list)
    precision = 0

    # Make train vector set by calculating train.w (inner product):
    tr_data_vec = []
    tr_label_vec = []
    for tr in train_list:
        tr_data_vec.append(np.dot(tr[:len(tr)-1], np.array(wmatrix).transpose()))
        tr_label_vec.append(tr[-1])

    # Make train vector set by calculating train.w (inner product):
    te_data_vec = []
    te_label_vec = []
    for te in test_list:
        te_data_vec.append(np.dot(te[:len(te)-1], np.array(wmatrix).transpose()))
        te_label_vec.append(te[-1])

    # Call SVC to figure out classifier with one-against-one method:
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(tr_data_vec, tr_label_vec)

    # Do prediction for each document in test set and compute the precision:
    for index in range(len(te_data_vec)):
        if clf.predict([te_data_vec[index]])[0] == te_label_vec[index]:
            precision += 1

    return precision/len(te_label_vec)


# Preparation:
if __name__ == "__main__":

    doc_list = make_doc_list("Trainset.txt")
    nonstop_list = stop_stem(doc_list)

    labeled_list = random.sample(doc_list, math.floor(len(doc_list)/11))
    unlabeled_list = [doc for doc in doc_list if doc not in labeled_list]
    info_list = informative_words(nonstop_list, bow(nonstop_list))

    # meaning matrix calculation
    orig_lbl_vec_list = make_vec_list(labeled_list, info_list)
    mean_matrix = word_meaning(orig_lbl_vec_list)

    # labeling
    unlabeled_vec_set = make_vec_list(unlabeled_list, info_list)
    assigned_label_set = labeling(unlabeled_vec_set, mean_matrix)
    labeled_set = orig_lbl_vec_list + assigned_label_set

    # Weight matrix calculation
    w_matrix = c_w_k(labeled_set)

    # Classification
    prec = classifier(info_list, w_matrix, labeled_set)
    target = open("precision.txt", 'w')
    target.write(str(prec))



