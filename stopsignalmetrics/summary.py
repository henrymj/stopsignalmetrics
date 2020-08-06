import json
import pandas as pd
from base import MultiLevelComputer
from ssrtmodel import SSRTmodel
from sequence import PostStopSlow, Violations


class Summary(MultiLevelComputer):
    def __init__(self, ssrt_model='replacement',
                 pss_correct_go_only=True, pss_filter_columns=True,
                 violations_mean_thresh=200, violations_ssd_quantity_thresh=5,
                 violations_n_pair_thresh=2, violations_verbose=False):
        super().__init__()
        self._SSRTmodel = SSRTmodel(model=ssrt_model)
        self._PostStopSlow = PostStopSlow(
            correct_go_only=pss_correct_go_only,
            filter_columns=pss_filter_columns)
        self._Violations = Violations(
            mean_thresh=violations_mean_thresh,
            ssd_quantity_thresh=violations_ssd_quantity_thresh,
            n_pair_thresh=violations_n_pair_thresh,
            verbose=violations_verbose)
        self.args = {
            'ssrt_model': ssrt_model,
            'pss_correct_go_only': pss_correct_go_only,
            'pss_filter_columns': pss_filter_columns,
            'violations_mean_thresh': violations_mean_thresh,
            'violations_ssd_quantity_thresh': violations_ssd_quantity_thresh,
            'violations_n_pair_thresh': violations_n_pair_thresh,
            'violations_verbose': violations_verbose
        }

        with open('standards.json') as json_file:
            standards = json.load(json_file)
        self._cols = standards['columns']
        self._codes = standards['key_codes']

    def _fit_individual(self, data_df):
        """Calculate all available metrics for an individual."""
        self._raw_data = data_df.copy()
        metrics = self._SSRTmodel.fit_transform(self._raw_data).copy()
        metrics['post_stop_slow'] = self._get_mean_pss()
        metrics['post_stop_success_slow'] = self._get_mean_pss(
            stop_type='success')
        metrics['post_stop_fail_slow'] = self._get_mean_pss(
            stop_type='fail')
        metrics['mean_violation'] = self._Violations.fit(
            self._raw_data).get_mean_below_thresh()
        self._transformed_data = metrics.copy()

    def _fit_group(self, data_df):
        """Calculate all available metrics for a group."""
        self._raw_data = data_df.copy()
        summary_helper = Summary(**self.args)
        metrics = self._raw_data.groupby('ID').apply(
            summary_helper.fit_transform).apply(pd.Series)
        if self._SSRTmodel.model == 'all':
            metrics = pd.concat([metrics['SSRT'].apply(
                pd.Series).add_prefix('SSRT_'), metrics],
                1)
            del metrics['SSRT']
        self._transformed_data = metrics.copy()

    def _get_mean_pss(self, stop_type='all'):
        """Get a subject's mean PSS, after fitting to a stop type."""
        return self._PostStopSlow.fit(self._raw_data,
                                      stop_type=stop_type).get_mean_pss()
