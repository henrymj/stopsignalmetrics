import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from .base import Computer, MultiLevelComputer


class Sequence(Computer):
    def __init__(self):
        super().__init__()
        self._acceptable_index_types = (
            str,
            list,
            np.ndarray,
            pd.core.indexes.numeric.Int64Index,
            pd.core.indexes.numeric.Float64Index,
            pd.core.indexes.numeric.IntegerIndex,
            pd.core.indexes.numeric.UInt64Index,
            pd.core.indexes.numeric.NumericIndex,
            pd.core.indexes.numeric.ABCFloat64Index,
            pd.core.indexes.numeric.ABCInt64Index,
            pd.core.indexes.numeric.ABCUInt64Index,
            )

    def fit(self, data_df, indices):
        """Get trial triplets centered on indices."""
        assert self._is_preprocessed(data_df)
        assert isinstance(indices, self._acceptable_index_types)
        self._raw_data = data_df.copy()

        if type(indices) == str:
            indices = self._raw_data.query(indices).index
        indices = indices[(indices > self._raw_data.index.min()) &
                          (indices < self._raw_data.index.max())]

        sequence_df = pd.DataFrame()
        sequence_df['trial_index'] = indices
        for shift, pfix in [(-1, 'pre'), (0, 'curr'), (1, 'post')]:
            for col in self._raw_data.columns:
                sequence_df['{}_{}'.format(pfix, col)] = self._raw_data.loc[
                    indices+shift, col].values

        # block match
        sequence_df = sequence_df[sequence_df['pre_block'] ==
                                  sequence_df['post_block']]

        # ID match
        if 'pre_ID' in sequence_df.columns:
            sequence_df = sequence_df[sequence_df['pre_ID'] ==
                                      sequence_df['post_ID']]

        self._transformed_data = sequence_df.reset_index(drop=True)
        return self

    def fit_transform(self, data_df, indices):
        self.fit(data_df, indices)
        return(self._transformed_data)


class PostStopSlow(Computer):
    def __init__(self, correct_go_only=True, filter_columns=True):
        super().__init__()
        self._correct_go_only = correct_go_only,
        self._filter_columns = filter_columns

    def fit(self, data_df, stop_type='all'):
        """Compare go RTs before and after stop trials."""
        assert self._is_preprocessed(data_df)
        assert stop_type in ['all', 'success', 'fail'], \
            "Can only exmine 3 types of stop trials: 'all', 'success', 'fail'."
        self._raw_data = data_df.copy()
        sequence_df = Sequence().fit_transform(
            self._raw_data,
            "{}=='{}'".format(self._cols['condition'],
                              self._codes['stop'])
            )

        if self._filter_columns:
            sequence_df = sequence_df.filter(
                regex='|'.join([self._cols[key] for key
                               in self._cols.keys()] +
                               ['trial_index']))

        keep_idx = ((sequence_df['pre_condition'] == "go") &
                    (sequence_df['post_condition'] == "go") &
                    (sequence_df['pre_goRT'] > 0) &
                    (sequence_df['post_goRT'] > 0))

        if self._correct_go_only:
            keep_idx = (keep_idx &
                        (sequence_df['pre_choice_accuracy'] == 1) &
                        (sequence_df['post_choice_accuracy'] == 1))

        stop_fail_idx = sequence_df['curr_stopRT'] > 0
        if stop_type == 'fail':
            keep_idx = (keep_idx &
                        stop_fail_idx)
        if stop_type == 'success':
            keep_idx = (keep_idx &
                        ~np.array(stop_fail_idx))

        sequence_df = sequence_df[keep_idx]
        self._diff_list = (sequence_df['post_goRT'] - sequence_df['pre_goRT'])
        self._mean_pss = self._diff_list.mean()
        self._transformed_data = sequence_df.reset_index(drop=True)
        return self

    def fit_transform(self, data_df, stop_type='all'):
        self.fit(data_df, stop_type=stop_type)
        return(self._transformed_data)

    def get_diff_list(self):
        """Get differences between goRTs before and after some stop trials."""
        try:
            assert self._differences is not None
        except AssertionError:
            raise NotFittedError('Data must first be loaded using .fit()')
        return(self._differences)

    def get_mean_pss(self):
        """Get mean post stop slowing."""
        try:
            assert self._mean_pss is not None
        except AssertionError:
            raise NotFittedError('Data must first be loaded using .fit()')
        return(self._mean_pss)


class Violations(MultiLevelComputer):
    def __init__(self, mean_thresh=200, n_pair_thresh=2,
                 ssd_quantity_thresh=5, verbose=False):
        super().__init__()
        self._mean_thresh = mean_thresh
        self._n_pair_thresh = n_pair_thresh
        self._ssd_quantity_thresh = ssd_quantity_thresh
        self._verbose = verbose

    def get_mean_below_thresh(self):
        """Get subject's mean violation at SSDs below a threshold."""
        try:
            assert self._transformed_data is not None
        except AssertionError:
            raise NotFittedError('Data must first be loaded using .fit()')
        return(self._transformed_data.loc[
            self._transformed_data[self._cols["SSD"]] <
            self._mean_thresh,
            'mean_violation']).mean()

    # private functions
    def _fit_individual(self, data_df):
        """Find the mean violation at each SSD for an individual."""
        assert self._is_preprocessed(data_df)
        self._raw_data = data_df.copy()
        seq_df = Sequence().fit_transform(self._raw_data,
                                          "condition=='stop' & stopRT>0")

        # filter to keep only previous Go trials, no omissions
        keep_idx = ((seq_df['pre_condition'] == 'go') &
                    (seq_df['pre_goRT'] > 0))
        seq_df = seq_df[keep_idx]

        # build up violation info per ssd
        info = []
        SSDs = seq_df['curr_SSD'].unique()
        for ssd in SSDs:
            ssd_df = seq_df[seq_df['curr_SSD'] == ssd].copy()
            info.append([ssd,
                         ssd_df.shape[0],
                         (ssd_df['curr_stopRT'] - ssd_df['pre_goRT']).mean(),
                         ssd_df['curr_stopRT'].mean(),
                         ssd_df['pre_goRT'].mean()])

        # convert to df and threshold
        info_df = pd.DataFrame(info,
                               columns=['SSD',
                                        'n_go_stopfail_pairs',
                                        'mean_violation',
                                        'mean_stopFailureRT',
                                        'mean_precedingGoRT'])
        va_df = info_df.query('n_go_stopfail_pairs >= {}'.format(
            self._n_pair_thresh))
        self._transformed_data = va_df.sort_values(
            by=self._cols["SSD"]
            ).set_index('SSD')

    def _fit_group(self, data_df):
        """Find the mean violation at each SSD for each individual."""
        assert self._is_preprocessed(data_df)
        self._raw_data = data_df.copy()
        violation = Violations()
        group_va_df = self._raw_data.groupby('ID').apply(
            violation.fit_transform)
        group_va_df = group_va_df.reset_index()
        all_ssdvals = group_va_df['SSD'].unique()
        all_ssdvals.sort()
        for ssd in all_ssdvals:
            group_va_df_ssd = group_va_df.query('SSD == %d' % ssd)
            if self._verbose:
                print(ssd, 'n subs:', group_va_df_ssd.shape[0])
            if group_va_df_ssd.shape[0] < self._ssd_quantity_thresh:
                group_va_df = group_va_df.query('SSD != %d' % ssd)
                if self._verbose:
                    print('\tdropping', ssd)
        group_va_df = group_va_df.sort_values(
            ['ID', 'SSD']).reset_index(drop=True)
        self._transformed_data = group_va_df
