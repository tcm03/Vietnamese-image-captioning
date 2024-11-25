#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>
#
# Modified Date: Sun Nov 24, 2024
#
# Modifier: Minh Canh Tu <tcm03>

import copy
from collections import defaultdict
import numpy as np



class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''

        # self.ngrams = [set() for _ in range(n)] # set of ngrams of length n (0-indexed)

        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        # self.cook_append(test, refs)
        # self.ref_len = None

    # def precook(self, s, n=4, build_ngrams=False):
    #     """
    #     Takes a string as input and returns an object that can be given to
    #     either cook_refs or cook_test. This is optional: cook_refs and cook_test
    #     can take string arguments as well.
    #     :param s: string : sentence to be converted into ngrams
    #     :param n: int    : number of ngrams for which representation is calculated
    #     :return: term frequency vector for occuring ngrams
    #     """
    #     words = s.split()
    #     counts = [defaultdict(int) for _ in range(n)]
    #     for k in range(n):
    #         for i in range(len(words)-k+1):
    #             ngram = tuple(words[i:i+k])
    #             if build_ngrams:
    #                 self.ngrams[k-1].add(ngram)
    #             counts[k][ngram] += 1
    #     return counts

    # def cook_refs(self, refs, n=4): ## lhuang: oracle will call with "average"
    #     '''Takes a list of reference sentences for a single segment
    #     and returns an object that encapsulates everything that BLEU
    #     needs to know about them.
    #     :param refs: list of string : reference sentences for some image
    #     :param n: int : number of ngrams for which (ngram) representation is calculated
    #     :return: result (list of dict)
    #     '''
    #     return [self.precook(ref, n, True) for ref in refs]

    # def cook_test(self, test, n=4):
    #     '''Takes a test sentence and returns an object that
    #     encapsulates everything that BLEU needs to know about it.
    #     :param test: list of string : hypothesis sentence for some image
    #     :param n: int : number of ngrams for which (ngram) representation is calculated
    #     :return: result (dict)
    #     '''
    #     return self.precook(test, n)

    # def cook_append(self, test, refs):
    #     '''called by constructor and __iadd__ to avoid creating new instances.'''

    #     if refs is not None:
    #         self.crefs.append(self.cook_refs(refs))
    #         if test is not None:
    #             self.ctest.append(self.cook_test(test)) ## N.B.: -1
    #         else:
    #             self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            # self.cook_append(other[0], other[1])
            self.ctest.append(other[0])
            self.crefs.append(other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            ngrams = set()
            for ref in refs:
                ref_words = ref.split()
                for k in range(1, self.n+1):
                    for i in range(len(ref_words)-k+1):
                        ngram = tuple(ref_words[i:i+k])
                        ngrams.add(ngram)
            for ngram in ngrams:
                self.document_frequency[ngram] += 1

    def tfidf_score(self, ngram, sent):
            """
            Description: create vector of tfidf weights
            Args:
                ngram: n-gram as a tuple of n words from sent
                sent: candidate sentence
            """
            cand_words = sent.split()
            # count number of times the ngram occurs in candidate
            h = 0
            for i in range(len(cand_words)-len(ngram)+1):
                if tuple(cand_words[i:i+len(ngram)]) == ngram:
                    h += 1
            hs = len(cand_words) - len(ngram) + 1
            tf = float(h)/hs

            N = len(self.crefs) # number of reference sentences (or documents)
            df = self.document_frequency[ngram]
            idf = float(np.log(N / (df+1.0))) # smoothing factor = 1
            return tf * idf
    
    def cosine_sim(self, n, cand_sent, ref_sent):
        """
        Description: Compute cosine similarity
        Args:
            n: length of n-gram
            cand_sent: candidate sentence
            ref_sent: reference sentence
        """
        cand_ngrams = set()
        cand_words = cand_sent.split()
        for i in range(len(cand_words)-n+1):
            cand_ngrams.add(tuple(cand_words[i:i+n]))
        ref_ngrams = set()
        ref_words = ref_sent.split()
        for i in range(len(ref_words)-n+1):
            ref_ngrams.add(tuple(ref_words[i:i+n]))
        common_ngrams = cand_ngrams.intersection(ref_ngrams)

        cand_norm = 0.
        for ngram in cand_ngrams:
            tfidf = self.tfidf_score(ngram, cand_sent)
            cand_norm += tfidf**2
        cand_norm = float(np.sqrt(cand_norm))
        ref_norm = 0.
        for ngram in ref_ngrams:
            tfidf = self.tfidf_score(ngram, ref_sent)
            ref_norm += tfidf**2
        ref_norm = float(np.sqrt(ref_norm))
        dot_product = 0.
        for ngram in common_ngrams:
            tfidf_cand = self.tfidf_score(ngram, cand_sent)
            tfidf_ref = self.tfidf_score(ngram, ref_sent)
            dot_product += tfidf_ref * min(tfidf_cand, tfidf_ref)

        l_cand = len(cand_words)
        len_ref = len(ref_words)
        delta = float(l_cand - len_ref)
        exp = np.e**(-(delta**2)/(2*self.sigma**2))
        return dot_product / (cand_norm * ref_norm) * exp

    def compute_cider_n(self, n, cand_sent, ref_cents):
        m = len(ref_cents)
        cider = 0.
        for ref_cent in ref_cents:
            cider += self.cosine_sim(n, cand_sent, ref_cent)
        cider = 10./m * cider
        return cider
    
    def compute_score(self):
        """
        Compute CIDEr score
        """
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = 0.
        for i, (test, refs) in enumerate(zip(self.ctest, self.crefs)):
            # compute cider for each sentence
            cider = 0.
            for n in range(self.n):
                cider += 1./self.n * self.compute_cider_n(n+1, test[0], refs)
            score += cider
        score = score / len(self.ctest)
        return score

    # def compute_score(self, option=None, verbose=0):
    #     # compute idf
    #     self.compute_doc_freq()
    #     # assert to check document frequency
    #     assert(len(self.ctest) >= max(self.document_frequency.values()))
    #     # compute cider score
    #     score = self.compute_cider()
    #     # debug
    #     # print score
    #     return np.mean(np.array(score)), np.array(score)
