import pandas as pd
import numpy as np
import json
from sklearn.exceptions import NotFittedError
import pkg_resources

STANDARDS_FILE = pkg_resources.resource_filename(
    'stopsignalmetrics', 'data/standards.json')

JSON_DICT = {}
CSV_DICT = {}
for source in ['mturk', 'inlab']:
    CSV_DICT[source] = {}
    JSON_DICT[source] = pkg_resources.resource_filename(
                'stopsignalmetrics',
                'data/{}.json'.format(source))
    for level in ['group', 'individual']:
        CSV_DICT[source][level] = pkg_resources.resource_filename(
                'stopsignalmetrics',
                'data/{}_{}.csv'.format(source, level))


class Computer:
    """Parent class for computing metrics."""
    def __init__(self):
        self._raw_data = None
        self._transformed_data = None
        standards = self._load_json()
        self._cols = standards['columns']
        self._codes = standards['key_codes']

    def fit(self, data_df):
        return self._is_preprocessed(data_df)

    def transform(self):
        try:
            assert self._raw_data is not None
        except AssertionError:
            raise NotFittedError('Data must first be loaded using .fit()')
        return(self._transformed_data)

    def fit_transform(self, data_df):
        self.fit(data_df)
        return(self._transformed_data)

    def _is_preprocessed(self, data_df):
        """check that dataset matches standard."""
        assert isinstance(data_df, pd.core.frame.DataFrame),\
            'data must be in the form of a pandas dataframe.'
        missable_columns = [
            'ID', 'response', 'correct_response', 'choice_accuracy'
            ]
        for key in [key for key in self._cols.keys() if \
            key not in missable_columns]:
            assert self._cols[key] in data_df.columns, \
                'missing {} from data df columns'.format(self._cols[key])

        condition_codes = data_df[self._cols['condition']].unique()
        for cond in ['go', 'stop']:
            assert self._codes[cond] in condition_codes,\
                'missing {} from column: {}.'.format(
                self._cols[cond], self._cols["condition"])

        # check that all unique non-nan values in the accuracy column 
        # can be mapped onto the standard codes for correct or incorrect.
        if 'choice_accuracy' in data_df.columns:
            acc_codes = data_df[self._cols['choice_accuracy']].unique()
            acc_codes = [i for i in acc_codes if i==i]
            standard_acc_codes = [self._codes['correct'], self._codes['incorrect']]
            for acc_code in acc_codes:
                assert acc_code in standard_acc_codes,\
                    '{} present in {} column.'. format(
                        acc_code, self._cols["choice_accuracy"]
                    )

        return True

    def _load_json(self, filepath=STANDARDS_FILE):
        with open(filepath) as json_file:
            json_dict = json.load(json_file)
            self._replace_none(json_dict)
            return json_dict

    def _replace_none(self, any_dict):
        for k, v in any_dict.items():
            if v is None:
                any_dict[k] = np.nan
            elif type(v) == type(any_dict):
                self._replace_none(v)


class MultiLevelComputer(Computer):
    """Parent class for computing metrics at individual or group level."""
    def __init__(self):
        super().__init__()

    def fit(self, data_df, level='individual'):
        assert level in ['individual', 'group']
        self._level = level
        fit_dict = {
            'individual': self._fit_individual,
            'group': self._fit_group
        }
        fit_dict[self._level](data_df)
        return self

    def fit_transform(self, data_df, level='individual'):
        self.fit(data_df, level=level)
        return self._transformed_data

    def _fit_individual(self, data_df):
        return self._is_preproccessed(data_df)

    def _fit_group(self, data_df):
        return self._is_preproccessed(data_df)
