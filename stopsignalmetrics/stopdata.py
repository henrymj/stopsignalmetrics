import json
import numpy as np
import pandas as pd
from .base import Computer, STANDARDS_FILE


class StopData(Computer):
    """Class for converitng a dataset to a standard for computation."""

    def __init__(self, var_dict=None, compute_acc_col=True):
        super().__init__()
        self._add_var_dict(var_dict)
        self._compute_acc_col = compute_acc_col

    def fit(self, data_df):
        assert isinstance(data_df, pd.core.frame.DataFrame),\
            'data must be in the form of a pandas dataframe.'
        self._raw_data = data_df.copy()
        assert self._check_variables_in_raw_data()
        self._map_raw_data_to_standard()
        return self

    # private functions
    def _add_var_dict(self, var_dict=None):
        """Save passed in variables for mapping to standard."""
        # add variable dictionaries, supplementing anything missing
        # with the standards defined in the json
        with open(STANDARDS_FILE) as json_file:
            stndrds = json.load(json_file)
        if var_dict is None:
            var_dict = stndrds
        else:
            for level in stndrds.keys():
                if level not in var_dict.keys():
                    var_dict[level] = stndrds[level].copy()
                else:
                    for key in stndrds[level].keys():
                        if key not in var_dict[level].keys():
                            var_dict[level][key] = stndrds[level][key]
        self._map_cols = var_dict['columns']
        self._map_codes = var_dict['key_codes']
        self._variable_dict = var_dict
        self._standards = stndrds

    def _check_variables_in_raw_data(self):
        """Make sure a mapping is possible."""
        # make sure that all of the necessary variables are present
        # or mapped via the variable dict
        for key in [key for key in self._map_cols.keys()
                    if key not in ['block', 'choice_accuracy', 'ID']]:
            assert self._map_cols[key] in self._raw_data.columns,\
                 'missing {} from raw data df columns'.format(
                    self._map_cols[key])

        condition_codes = self._raw_data[self._map_cols['condition']].unique()
        for cond in ['go', 'stop']:
            assert self._map_codes[cond] in condition_codes,\
                ('missing {} from column: '.format(self._map_codes[cond]),
                 self._map_cols["condition"])

        if self._map_cols['choice_accuracy'] in self._raw_data.columns:
            acc_codes = self._raw_data[
                self._map_cols['choice_accuracy']].unique()
            for acc in ['correct', 'incorrect']:
                assert self._map_codes[acc] in acc_codes,\
                    ('missing {} from column: '.format(self._map_codes[acc]),
                     self._map_cols["choice_accuracy"])
        return True

    def _map_raw_data_to_standard(self):
        """Map data to standard."""
        data_df = self._raw_data.copy()

        # if only 1 RT col, split into 2
        if self._map_cols['goRT'] == self._map_cols['stopRT']:
            data_df[self._standards['columns']['goRT']] = np.where(
                data_df[self._map_cols['condition']] == self._map_codes['go'],
                data_df[self._map_cols['goRT']],
                np.nan)
            data_df[self._standards['columns']['stopRT']] = np.where(
                data_df[self._map_cols['condition']] ==
                self._map_codes['stop'],
                data_df[self._map_cols['stopRT']],
                np.nan)
            del data_df[self._map_cols['goRT']]
        else:
            data_df.loc[
                data_df[self._map_cols['condition']] !=
                self._map_codes['go'],
                self._map_cols['goRT']] = np.nan
            data_df.loc[
                data_df[self._map_cols['condition']] !=
                self._map_codes['stop'],
                self._map_cols['stopRT']] = np.nan

        # drop SSDs of non-stop Trials
        data_df.loc[
            data_df[self._map_cols['condition']] != self._map_codes['stop'],
            self._map_cols['SSD']] = np.nan

        # add block column if not present
        if self._map_cols['block'] not in data_df.columns:
            data_df[self._map_cols['block']] = 1

        # recompute choice accuracy
        if (self._map_cols['choice_accuracy'] not in self._raw_data.columns) |\
                self._compute_acc_col:
            corr_code = self._map_codes['correct']
            incorr_code = self._map_codes['incorrect']
            data_df[self._map_cols['choice_accuracy']] = np.where(
                data_df[self._map_cols['response']] == data_df[
                    self._map_cols['correct_response']],
                corr_code,
                incorr_code)

        # map columns, key codes to standard
        rename_column_dict = {self._map_cols[col]: self._standards['columns']
                              [col] for col in self._map_cols.keys()}
        data_df = data_df.rename(columns=rename_column_dict)
        standardize_codes_dict = {self._map_codes[code]: self._standards
                                  ['key_codes'][code] for code
                                  in self._map_codes.keys()}
        for col in [self._standards['columns']['condition'],
                    self._standards['columns']['choice_accuracy']]:
            data_df[col] = data_df[col].map(
                standardize_codes_dict
                ).fillna(data_df[col])

        assert self._is_preprocessed(data_df)
        self._transformed_data = data_df
