from math import log10, floor
from os import path

import scripts.data_factory as data_factory
import scripts.output_factory as output_factory
from scripts.documents_filter import DocumentsFilter
from scripts.filter_terms import FilterTerms
from scripts.text_processing import (
    LemmaTokenizer,
    WordAnalyzer,
)
from scripts.tfidf_mask import TfidfMask
from scripts.tfidf_reduce import TfidfReduce
from scripts.tfidf_wrapper import tfidf_subset_from_features, tfidf_from_text
from scripts.utils import utils


def proc(
    data_filename,
    docs_mask_dict,
    pick_method="sum",
    ngram_range=(1, 3),
    text_header="abstract",
    cached_folder_name=None,
    max_df=0.1,
    user_ngrams=None,
    prefilter_terms=0,
    terms_threshold=None,
    output_name=None,
    calculate_timeseries=None,
    m_steps_ahead=5,
    emergence_index="porter",
    exponential=False,
    nterms=50,
    patents_per_quarter_threshold=20,
    sma=None,
):

    # calculate or fetch tf-idf mat
    # NOTE: Will execute each time with default arguments
    dataframe = data_factory.get(data_filename)
    utils.checkdf(dataframe, calculate_timeseries, docs_mask_dict, text_header)
    utils.remove_empty_documents(dataframe, text_header)

    cached_folder_name = path.join("cached", output_name + f"-mdf-{max_df}")
    dates = None
    # seems like the tfidf_object TODO: ??
    tfidf_obj = tfidf_from_text(
        text_series=dataframe[text_header],
        ngram_range=ngram_range,
        max_document_frequency=max_df,
        tokenizer=LemmaTokenizer(),
        min_df=floor(log10(dataframe.shape[0])),
    )

    tfidf_mask_obj = TfidfMask(
        tfidf_obj, ngram_range=ngram_range, uni_factor=0.8, unbias=True
    )
    tfidf_obj.apply_weights(tfidf_mask_obj.tfidf_mask)

    tfidf_reduce_obj = TfidfReduce(tfidf_obj.tfidf_matrix, tfidf_obj.feature_names)
    term_score_mp = tfidf_reduce_obj.extract_ngrams_from_docset("mean_prob")
    num_tuples_to_retain = min(prefilter_terms, len(term_score_mp))

    term_score_entropy = tfidf_reduce_obj.extract_ngrams_from_docset("entropy")
    term_score_variance = tfidf_reduce_obj.extract_ngrams_from_docset("variance")

    feature_subset_mp = sorted([x[1] for x in term_score_mp[:num_tuples_to_retain]])
    feature_subset_variance = sorted(
        [x[1] for x in term_score_variance[:num_tuples_to_retain]]
    )
    feature_subset_entropy = sorted(
        [x[1] for x in term_score_entropy[:num_tuples_to_retain]]
    )

    feature_subset = (
        set(feature_subset_mp)
        .union(set(feature_subset_variance))
        .union(feature_subset_entropy)
    )

    number_of_ngrams_before = len(tfidf_obj.feature_names)
    tfidf_obj = tfidf_subset_from_features(tfidf_obj, sorted(list(feature_subset)))
    number_of_ngrams_after = len(tfidf_obj.feature_names)
    print(
        f"Reduced number of terms by pre-filtering from {number_of_ngrams_before:,} "
        f"to {number_of_ngrams_after:,}"
    )

    cpc_dict = utils.cpc_dict(dataframe)

    utils.pickle_object("tfidf", tfidf_obj, cached_folder_name)
    utils.pickle_object("dates", dates, cached_folder_name)
    utils.pickle_object("cpc_dict", cpc_dict, cached_folder_name)

    # NOTE: change from cached to outputs
    outputs_folder_name = cached_folder_name.replace("cached", "outputs")
    print(f"Applying documents filter...")

    # docs weights( column, dates subset + time, citations etc.)
    doc_filters = DocumentsFilter(
        dates, docs_mask_dict, cpc_dict, tfidf_obj.tfidf_matrix.shape[0],
    ).doc_filters

    # todo: build up list of weight functions (left with single remaining arg etc via partialfunc)
    #  combine(list, tfidf) => multiplies weights together, then multiplies across tfidf (if empty, no side effect)

    # todo: this is another weight function...

    # term weights - embeddings
    print(f"Applying terms filter...")
    filter_terms_obj = FilterTerms(
        tfidf_obj.feature_names, user_ngrams, threshold=terms_threshold
    )
    term_weights = filter_terms_obj.ngram_weights_vec

    # todo: replace tfidf_mask with isolated functions: clean_unigrams, unbias_ngrams;
    #  these operate directly on tfidf
    #  Hence return nothing - operate in place on tfidf.
    print(f"Creating a masked tfidf matrix from filters...")
    # tfidf mask ( doc_ids, doc_weights, embeddings_filter will all merge to a single mask in the future)
    tfidf_mask_obj = TfidfMask(tfidf_obj, ngram_range=ngram_range, uni_factor=0.8)
    tfidf_mask_obj.update_mask(doc_filters, term_weights)
    tfidf_mask = tfidf_mask_obj.tfidf_mask

    # todo: this mutiply and remove null will disappear - maybe put weight combiner last so it can remove 0 weights
    # mask the tfidf matrix

    tfidf_masked = tfidf_mask.multiply(tfidf_obj.tfidf_matrix)

    tfidf_masked, dates = utils.remove_all_null_rows_global(tfidf_masked, dates)
    print(
        f"Processing TFIDF matrix of {tfidf_masked.shape[0]:,}"
        f" / {tfidf_obj.tfidf_matrix.shape[0]:,} documents"
    )

    # todo: no advantage in classes - just create term_count and extract_ngrams as functions

    tfidf_reduce_obj = TfidfReduce(tfidf_masked, tfidf_obj.feature_names)
    timeseries_data = None

    # if other outputs
    term_score_tuples = tfidf_reduce_obj.extract_ngrams_from_docset(pick_method)
    term_score_tuples = utils.stop(
        term_score_tuples,
        WordAnalyzer.stemmed_stop_word_set_uni,
        WordAnalyzer.stemmed_stop_word_set_n,
        tuples=True,
    )
    return (
        term_score_tuples,
        outputs_folder_name,
        tfidf_reduce_obj,
        docs_mask_dict["date"],
        pick_method,
        data_filename,
    )


def output(
    # need to be returned by proc
    term_score_tuples,
    outputs_folder_name,
    tfidf_reduce_obj,
    date_dict,
    pick_method,
    data_filename,
    # end
    output_types,
    wordcloud_title=None,
    outname=None,
    nterms=50,
    n_nmf_topics=0,
):
    for output_type in output_types:
        output_factory.create(
            output_type,
            term_score_tuples,
            outputs_folder_name,
            emergence_list=[],
            wordcloud_title=wordcloud_title,
            tfidf_reduce_obj=tfidf_reduce_obj,
            name=outname,
            nterms=nterms,
            timeseries_data=None,
            date_dict=date_dict,
            pick=pick_method,
            doc_pickle_file_name=data_filename,
            nmf_topics=n_nmf_topics,
            timeseries_outputs=None,
            method="porter",
        )
