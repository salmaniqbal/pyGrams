from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

import os
import scripts.utils.utils as ut
import zipfile
import pandas as pd
from math import sqrt, log


class FilterTerms(object):
    def __init__(self, tfidf_obj, user_ngrams, file_name='models/glove/glove2vec.6B.50d.txt', threshold=None, use_autostop=False):
        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_obj.feature_names
        self.__file_name = file_name
        self.__ngram_weights_vec = list(np.ones(len(self.__tfidf_ngrams)))

        self.__tf_normalized = tfidf_obj.tf_matrix
        self.__tfidf_score = tfidf_obj.tfidf_matrix
        self.__ngram_count = tfidf_obj.ngram_counts
        self.__idf = tfidf_obj.idf


        self.__user_ngrams = user_ngrams
        self.__file_name = file_name
        self.__ngram_weights_vec = list(np.ones(len(tfidf_obj.feature_names)))
        if user_ngrams is not None and len(user_ngrams)>0:
            if not os.path.isfile(file_name):
                with zipfile.ZipFile(file_name+".zip","r") as zip_ref:
                    zip_ref.extractall("models/glove/")
            self.__model = KeyedVectors.load_word2vec_format(self.__file_name)
            self.__ngram_weights_vec = self.__get_embeddings_vec(threshold)
        if use_autostop:
            auto_stop_weights = self.__get_autostop_vec()

    @property
    def ngram_weights_vec(self):
        return self.__ngram_weights_vec

    def __get_embeddings_vec(self, threshold):
        embeddings_vect = []
        for term in tqdm(self.__tfidf_ngrams, desc='Evaluating terms distance with: ' + ' '.join(self.__user_ngrams), unit='term',
                         total=len(self.__tfidf_ngrams)):
            compare = []
            for ind_term in term.split():
                for user_term in self.__user_ngrams:
                    try:
                        similarity_score = self.__model.similarity(ind_term, user_term)
                        compare.append(similarity_score)
                    except:
                        compare.append(0.0)
                        continue

            max_similarity_score = max(similarity_score for similarity_score in compare)
            embeddings_vect.append(max_similarity_score)
        embeddings_vect_norm = ut.normalize_array(embeddings_vect, return_list=True)
        if threshold is not None:
            return [float(x>threshold) for x in embeddings_vect_norm]
        return embeddings_vect

    def collect_vector_for_feature(self, csc_mat):
        mp_vec = []
        for j in range(csc_mat.shape[1]):
            start_idx_ptr = csc_mat.indptr[j]
            end_idx_ptr = csc_mat.indptr[j + 1]
            mpj = 0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                pij = csc_mat.data[i]
                mpj += pij
            mp_vec.append(mpj)
        return np.array(mp_vec)

    def __get_autostop_vec(self):
        ngrams = self.__tfidf_ngrams
        tf_norm = self.__tf_normalized
        tfidf_score = self.__tfidf_score
        ngram_count = self.__ngram_count

        word_list_zipf = []
        word_list_aggregation = []
        word_list_tie = []
        word_list_sat = []

        tf_norm = tf_norm.tocsc()
        tfidf = tfidf_score.tocsc()
        count = ngram_count.tocsc()

        if not tf_norm.getformat() == 'csc':
            print('problem')

        N = tf_norm.shape[0]
        mp_vec = self.collect_vector_for_feature(tf_norm) / N
        tfidf_vec = self.collect_vector_for_feature(tfidf)
        ngram_vec = self.collect_vector_for_feature(count)
        probabilities_vec = mp_vec * N

        variance_vec = []
        for j in range(tf_norm.shape[1]):
            start_idx_ptr = tf_norm.indptr[j]
            end_idx_ptr = tf_norm.indptr[j + 1]
            vpj = 0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                pij = tf_norm.data[i]
                vpj += (pij - mp_vec[j]) ** 2
            variance = sqrt(vpj / N)
            variance_vec.append(variance)

        entropy_vec = []
        for j in range(tf_norm.shape[1]):
            start_idx_ptr = tf_norm.indptr[j]
            end_idx_ptr = tf_norm.indptr[j + 1]
            entropy_j = 0
            # iterate through rows with non-zero entries
            for i in range(start_idx_ptr, end_idx_ptr):
                # row_idx = csc_mat.indices[i]
                pij = tf_norm.data[i]
                entropy_j += pij * log(1 / pij)
            entropy_vec.append(entropy_j)

        sat_vec = mp_vec / np.array(variance_vec)

        word_list_general = zip(ngrams, ngram_vec, ut.normalize(probabilities_vec), self.__idf, ut.normalize(tfidf_vec), ut.normalize(probabilities_vec),
                                ut.normalize(variance_vec), ut.normalize(entropy_vec), ut.normalize(sat_vec))
        df = pd.DataFrame(word_list_general,
                          columns=['Word', 'Count', 'TF', 'IDF', 'TFIDF', 'Probability', 'Variance', 'Entropy', 'SAT'])
        # threshold_stopwords = 0.3 * df
        # print(df)

        filter_tf1 = df.loc[df['Count'] <= 1]
        filter_tf1.to_csv(os.path.join('outputs', 'stoplists', 'word_list_tf1_100.txt'), sep='\t', index=False, header=False,
                          columns=['Word'])
        # print(filter_tf1)

        filter_tf = df.sort_values('TF', ascending=False).head(1129)
        filter_tf.to_csv(os.path.join('outputs', 'stoplists', 'word_list_tf_100.txt'), sep='\t', index=False, header=False,
                         columns=['Word'])
        # print(filter_tf)

        filter_idf = df.sort_values('IDF', ascending=True).head(1129)
        filter_idf.to_csv(os.path.join('outputs', 'stoplists', 'word_list_idf_100.txt'), sep='\t', index=False, header=False,
                          columns=['Word'])
        # print(filter_idf)

        filter_tfidf = df.sort_values('Probability', ascending=False).head(1129)
        filter_tfidf.to_csv(os.path.join('outputs', 'stoplists', 'word_list_tfidf_100.txt'), sep='\t', index=False, header=False,
                            columns=['Word'])
        # print(filter_tfidf)

        filter_variance = df.sort_values('Variance', ascending=False).head(1129)
        filter_variance.to_csv(os.path.join('outputs', 'stoplists', 'word_list_variance_100.txt'), sep='\t', index=False,
                               header=False, columns=['Word'])
        # print(filter_variance)

        filter_entropy = df.sort_values('Entropy', ascending=False).head(1129)
        filter_entropy.to_csv(os.path.join('outputs', 'stoplists', 'word_list_entropy_100.txt'), sep='\t', index=False, header=False,
                              columns=['Word'])
        # print(filter_entropy)

        list_sat = df.sort_values('SAT', ascending=False).head(1129)
        list_sat.to_csv(os.path.join('outputs', 'stoplists', 'word_list_result_sat_100.txt'), sep='\t', index=False, header=False,
                        columns=['Word'])
        # print(list_sat)

        merge_tf_idf = pd.merge(filter_tf, filter_idf, on=['Word'], how='inner')
        merge_tf_entropy = pd.merge(filter_tf, filter_entropy, on=['Word'], how='inner')
        merge_tfidf_entropy = pd.merge(filter_tfidf, filter_entropy, on=['Word'], how='inner')

        list_tie = pd.merge(merge_tf_idf, merge_tfidf_entropy, on=['Word'], how='inner')
        stop_list_tie = list_tie.loc[:, ["Word"]]
        stop_list_tie.to_csv(os.path.join('outputs', 'stoplists', 'word_list_result_tie_100.txt'), sep='\t', index=False,
                             header=False, columns=['Word'])
        # print(stop_list_tie)

        list_agg = pd.merge(merge_tf_entropy, filter_variance, on=['Word'], how='inner')
        stop_list_agg = list_agg.loc[:, ["Word"]]
        stop_list_agg.to_csv(os.path.join('outputs', 'stoplists', 'word_list_result_agg_100.txt'), sep='\t', index=False,
                             header=False, columns=['Word'])
        # print(stop_list_agg)

        list_zipf = pd.concat([merge_tf_idf, filter_tf1], sort=False)
        list_zipf.to_csv(os.path.join('outputs', 'stoplists', 'word_list_result_zipf_100.txt'), sep='\t', index=False, header=False,
                         columns=['Word'])
        # print(list_zipf)

        list_zipf_modified = pd.merge(filter_tfidf, filter_tf1, on=['Word'], how='inner')
        stop_list_zm = list_zipf_modified.loc[:, ["Word"]]
        stop_list_zm.to_csv(os.path.join('outputs', 'stoplists', 'word_list_result_zipf_modified_100.txt'), sep='\t', index=False,
                            header=False, columns=['Word'])
        # print(list_zipf_modified)

        method_merge_tie_agg = pd.merge(list_tie, list_agg, on=['Word'], how='inner')
        stop_list_tie_agg = method_merge_tie_agg.loc[:, ["Word"]]
        stop_list_tie_agg.to_csv(os.path.join('outputs', 'stoplists', 'method_word_list_result_tie_agg_100.txt'), sep='\t',
                                 index=False, header=False, columns=['Word'])

        method_merge_sat_zipf = pd.merge(list_tie, list_agg, on=['Word'], how='inner')
        stop_list_sat_zipf = method_merge_sat_zipf.loc[:, ["Word"]]
        stop_list_sat_zipf.to_csv(os.path.join('outputs', 'stoplists', 'method_word_list_result_sat_zipf_100.txt'), sep='\t',
                                  index=False, header=False, columns=['Word'])

        final_stop_list = pd.merge(method_merge_tie_agg, method_merge_sat_zipf, on=['Word'], how='inner')
        stop_list = final_stop_list.loc[:, ["Word"]]
        stop_list.to_csv(os.path.join('outputs', 'stoplists', 'stop_list_100.txt'), sep='\t', index=False, header=False,
                         columns=['Word'])
        print(stop_list)

        # test_final_stop_list = pd.merge(method_merge_tie_agg, method_merge_sat_zipf, on=['Word'], how='inner')
        # test_stop_list = test_final_stop_list.loc[:, ["Word"]]
        # test_stop_list.to_csv('/Users/zara/results/test_stop_list.txt', sep='\t', index=False, header=False, columns=['Word'])
        # print(ngram_count)

        ###Empty List
        # final_stop_list_modified = pd.merge(final_stop_list, stop_list_zm, on=['Word'], how='inner')
        # stop_list_v2 = final_stop_list_modified.loc[:, ["Word"]]
        # stop_list_v2.to_csv('/Users/zara/results/stop_list_v2.txt', sep='\t', index=False, header=False, columns=['Word'])

        # final_stop_list_con = pd.concat(final_stop_list, list_zipf_modified, sort=False)
        #  stop_list_v3 = final_stop_list_con.loc[:, ["Word"]]
        #  stop_list_v3.to_csv('/Users/zara/results/stop_list_v3.txt', sep='\t', index=False, header=False, columns=['Word'])
        pd.set_option('display.max_row', len(stop_list))

