import pandas as pd
import numpy as np
import json
from sklearn.exceptions import NotFittedError
import pkg_resources

STANDARDS_FILE = pkg_resources.resource_filename(
    'stopsignalmetrics', 'data/standards.json')


class Computer:
    """Parent class for computing metrics."""
    def __init__(self):
        self._raw_data = None
        self._transformed_data = None
        standards = self._load_standards()
        self._cols = standards['columns']
        self._codes = standards['key_codes']

    def fit(self, data_df):
        return self._is_preprocessed(data_df)

    def transform(self):
        try:
            assert self.raw_data is not None
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

        for key in [key for key in self._cols.keys() if key != 'ID']:
            assert self._cols[key] in data_df.columns, \
                'missing {} from data df columns'.format(self._cols[key])

        condition_codes = data_df[self._cols['condition']].unique()
        for cond in ['go', 'stop']:
            assert self._codes[cond] in condition_codes,\
                'missing {} from column: {}.'.format(
                self._cols[cond], self._cols["condition"])

        acc_codes = data_df[self._cols['choice_accuracy']].unique()
        acc_codes = np.asarray(acc_codes)[~np.isnan(acc_codes)]
        standard_acc_codes = [self._codes['correct'], self._codes['incorrect']]
        for acc_code in acc_codes:
            assert acc_code in standard_acc_codes,\
                '{} present in {} column.'. format(
                    acc_code, self._cols["choice_accuracy"]
                )
        return True

    def _load_standards(self):
        with open(STANDARDS_FILE) as json_file:
            standards = json.load(json_file)
            self._replace_none(standards)
            return standards

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
