#!/usr/bin/env python
import argparse
import os
import time
import pickle

from scripts.pipeline import Pipeline
from scripts.utils.pygrams_exception import PygramsException

def rubbish_create_folds():
    paths = [
        os.path.join("outputs", "reports"),
        os.path.join("outputs", "wordclouds"),
        os.path.join("outputs", "table"),
        os.path.join("outputs", "emergence"),
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)

def main():
    rubbish_create_folds()

    # TODO: stop loading in the default arguments
    with open("args.pkl", "rb") as handle:
        args = pickle.load(handle)

    outputs = ['reports', 'json_config']

    docs_mask_dict = {'filter_by': 'intersection', 'cpc': None, 'cite': None, 'columns': None, 'date': None, 'timeseries_date': None, 'date_header': None}
    terms_mask_dict = None

    doc_source_file_name = os.path.join(args.path, args.doc_source)

    pipeline = Pipeline(
        'data/USPTO-random-1000.pkl.bz2',
        {'filter_by': 'intersection', 'cpc': None, 'cite': None, 'columns': None, 'date': None, 'timeseries_date': None, 'date_header': None},
        pick_method='sum',
        ngram_range=(1, 3),
        text_header='abstract',
        cached_folder_name=None,
        max_df=0.05,
        user_ngrams=[],
        prefilter_terms=100000,
        terms_threshold=0.75,
        output_name='out',
        calculate_timeseries=False,
        m_steps_ahead=5,
        emergence_index='porter',
        exponential=False,
        nterms=25,
        patents_per_quarter_threshold=15,
        sma='savgol'
    )

    pipeline.output(
        outputs,
        wordcloud_title="Popular Terms",
        outname="out",
        nterms=250,
        n_nmf_topics=0,
    )

    outputs_name = pipeline.outputs_folder_name

    print(outputs_name)

try:
    start = time.time()
    main()
    end = time.time()
    diff = int(end - start)
    hours = diff // 3600
    minutes = diff // 60
    seconds = diff % 60

    print("")
    print(f"pyGrams query took {hours}:{minutes:02d}:{seconds:02d} to complete")
except PygramsException as err:
    print(f"pyGrams error: {err.message}")
