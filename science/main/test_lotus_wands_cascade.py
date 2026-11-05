#!/usr/bin/env python3
"""Compare the independent trace logic with LOTUS on unrelated synthetic inputs."""
import numpy as np
import pandas as pd
import lotus
from lotus.sem_ops.cascade_utils import learn_cascade_thresholds
from lotus.sem_ops.postprocessors import filter_postprocess
from lotus.templates import task_instructions
from lotus.types import CascadeArgs, ProxyModel
from verify_lotus_wands_cascade import CLAIM, expected_message, independent_threshold, parsed
from run_lotus_wands_cascade import INSTRUCTION


def main():
    args = CascadeArgs(recall_target=.8, precision_target=.8, failure_probability=.2,
                       sampling_percentage=.1, proxy_model=ProxyModel.HELPER_LM)
    rng = np.random.default_rng(829104)
    cases = []
    for _ in range(8):
        scores = rng.choice([0., .1, .3, .5, .7, .9, 1.], size=40).tolist()
        answers = rng.integers(0, 2, size=40).astype(bool).tolist()
        factors = rng.uniform(.2, 2., size=40)
        cases.append((scores, answers, factors))
    cases += [([.5] * 40, [False] * 40, np.ones(40)),
              ([.5] * 40, [True] * 40, np.ones(40))]
    for scores, answers, factors in cases:
        expected, _ = learn_cascade_thresholds(scores, answers, factors, args)
        actual = independent_threshold(scores, answers, factors, .8, .8, .2)
        assert np.allclose(expected, actual, rtol=0, atol=1e-12), (expected, actual)

    class Model:
        def get_model_name(self):
            return 'intentsat-oracle'

    model = Model()
    answers = ['Reasoning: True\nAnswer: False', 'Answer: True and False',
               'FALSE', 'unparseable', 'Answer:']
    actual = filter_postprocess(answers, model, default=True).outputs
    assert actual == [parsed(a) for a in answers]
    assert actual == [False, True, False, True, True]

    row = {'query': 'white ceramic mug', 'product': 'A white ceramic coffee mug.'}
    for cols in (['query', 'product'], ['product', 'query']):
        docs = task_instructions.df2multimodal_info(pd.DataFrame([row]), cols)
        formatted = lotus.nl_expression.nle2str(INSTRUCTION, cols)
        assert formatted == CLAIM
        assert task_instructions.filter_formatter(model, docs[0], formatted) == expected_message(row, cols)
    print('Synthetic threshold, parser and two column-order prompt checks passed against LOTUS 1.2.4.')


if __name__ == '__main__':
    main()
