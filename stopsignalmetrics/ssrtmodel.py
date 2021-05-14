import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from .base import MultiLevelComputer


class SSRTmodel(MultiLevelComputer):
    def __init__(self, model='replacement'):
        assert model in ['replacement', 'omission',
                         'integration', 'mean', 'all']
        super().__init__()
        self.model = model
        self._metrics = None
        self._qa = None

    # UNFINISHED
    def check_behavior(self):
        try:
            assert self.coefficients_ is not None
        except AssertionError:
            raise NotFittedError('Model must first be fitted using .fit()')
        # compute QA metrics
        self._qa = pd.DataFrame()

    def _fit_individual(self, data_df, max_RT=None):
        """Get SSRT and related metrics for an individual."""
        assert self._is_preprocessed(data_df)
        # fit the model for a single subject
        self._raw_data = data_df.copy()
        self._metrics = {
            'SSRT': None,
            'mean_SSD': None,
            'p_respond': None,
            'max_RT': max_RT,
            'mean_go_RT': None,
            'sd_go_RT': None,
            'mean_stopfail_RT': None,
            'sd_stopfail_RT': None,
            'omission_count': None,
            'omission_rate': None,
            'go_acc': None,
            'stopfail_acc': None,
        }
        self._calc_RTs()

        self._calc_mean_SSD()
        self._calc_p_respond()
        self._calc_omission_nums()
        if 'choice_accuracy' in self._raw_data.columns:
            self._calc_accs()
        if self._metrics['max_RT'] is None:
            _ = self._calc_max_RT()
        if (self._metrics['p_respond'] is not None) and (self._metrics['p_respond'] > 0 and self._metrics['p_respond'] < 1) :
            self._calc_SSRT()
        self._transformed_data = self._metrics.copy()
        return self

    def _fit_group(self, data_df):
        """Get SSRT and related metrics for group data."""
        assert self._is_preprocessed(data_df)
        self._raw_data = data_df.copy()

        self._metrics = {}
        groupmaxRT = self._calc_max_RT()

        group_metrics = data_df.groupby('ID').apply(
            lambda x: SSRTmodel(model=self.model)
            ._fit_individual(x, max_RT=groupmaxRT)
            .transform()).apply(pd.Series)
        if self.model == 'all':
            group_metrics = pd.concat([group_metrics['SSRT']
                                      .apply(pd.Series).add_prefix('SSRT_'),
                                      group_metrics], 1)
            del group_metrics['SSRT']
        self._transformed_data = group_metrics

    # private functions
    def _calc_SSRT(self):
        """ Calculate the SSRT via 4 supported methods."""
        goRTs = self._get_all_goRTs(sort=True)
        P_respond = self._metrics['p_respond']

        nrt_dict = {
            'mean': lambda : np.mean(goRTs),
            'integration': lambda : self._get_nth_RT(P_respond,
                                                     goRTs
                                                     ),
            'omission': lambda : self._get_nth_RT(P_respond/(1-self._metrics['omission_rate']), # corrected P(resp)
                                                  goRTs
                                                  ),
            'replacement': lambda : self._get_nth_RT(P_respond,     
                                                     np.concatenate((
            goRTs,
            [self._metrics['max_RT']] * self._metrics['omission_count'])) # appending max_RT to replace omissions
                                                     ),
        }
        # out_dict = {
        #     key: nrt - self._metrics['mean_SSD']
        #     for key, nrt in [
        #         ('mean', np.mean(goRTs)),
        #         ('integration', self._get_nth_RT(P_respond, goRTs)),
        #         ('omission', self._get_nth_RT(corrected_P_respond, goRTs)),
        #         ('replacement', self._get_nth_RT(P_respond, goRTs_w_replacements)),
        #     ]
        # }

        if self.model == 'all':
            self._metrics['SSRT'] = {k: func() - self._metrics['mean_SSD'] for k, func in nrt_dict.items()}
        else:
            self._metrics['SSRT'] = nrt_dict[self.model]() - self._metrics['mean_SSD']

    def _calc_p_respond(self):
        """Calculate the P(repsond|signal) of a dataset."""

        stop_idx = self._raw_data['condition'] == 'stop'
        num_stop_trials = self._raw_data.loc[stop_idx, 'stopRT'].shape[0]
        if num_stop_trials > 0:
            num_stop_failures = self._raw_data.loc[stop_idx &
                                                (self._raw_data['stopRT'].notnull()),
                                                'stopRT'].shape[0]
            p_respond = num_stop_failures / num_stop_trials
            self._metrics['p_respond'] = p_respond

    def _calc_mean_SSD(self):
        """Calculate the mean SSD of a dataset."""
        self._metrics['mean_SSD'] = self._raw_data['SSD'].mean()

    def _calc_RTs(self):
        """Find mean go and stop-fail RTs."""
        goRTs = self._get_all_goRTs()
        if len(goRTs) > 0:
            self._metrics['mean_go_RT'] = np.mean(goRTs)
            self._metrics['sd_go_RT'] = np.std(goRTs)
        stopfailRTs = self._raw_data.loc[
            (self._raw_data['condition'] == 'stop') &
            (self._raw_data['stopRT'].notnull()),
            'stopRT']
        if len(stopfailRTs) > 0:
            self._metrics['mean_stopfail_RT'] = stopfailRTs.mean()
            self._metrics['sd_stopfail_RT'] = stopfailRTs.std()

    def _calc_accs(self):
        """Calculate go and stop-failure Choice Accuracies."""
        self._metrics['go_acc'] = self._raw_data.loc[
            (self._raw_data['condition'] == 'go') &
            (self._raw_data['goRT'].notnull()),
            'choice_accuracy'].mean()

        self._metrics['stopfail_acc'] = self._raw_data.loc[
            (self._raw_data['condition'] == 'stop') &
            (self._raw_data['stopRT'].notnull()),
            'choice_accuracy'].mean()

    def _calc_omission_nums(self):
        """Get omission_count and omission_rate, respectively."""
        num_go_trials = self._raw_data.loc[
            self._raw_data['condition'] == 'go', 'condition'].shape[0]
        num_go_responses = self._raw_data.loc[
            (self._raw_data['condition'] == 'go') &
            (self._raw_data['goRT'].notnull()),
            'condition'].shape[0]

        omission_count = num_go_trials - num_go_responses
        omission_rate = omission_count/num_go_trials
        self._metrics['omission_count'] = omission_count
        self._metrics['omission_rate'] = omission_rate

    def _calc_max_RT(self):
        """Calculate participant's max RT."""
        self._metrics['max_RT'] = self._raw_data.loc[:, 'goRT'].max()
        return self._metrics['max_RT']

    def _get_all_goRTs(self, sort=False):
        """Get RTs, sorted ascendingly."""

        go_idx = ((self._raw_data['condition'] == 'go') &
                  (self._raw_data['goRT'].notnull()))
        goRTs = self._raw_data.loc[go_idx, 'goRT'].values
        if sort:
            goRTs.sort()
        return goRTs

    def _get_nth_RT(self, P_respond, goRTs):
        """Get nth RT based P(response|signal) and sorted go RTs."""
        nth_index = int(np.rint(P_respond*len(goRTs))) - 1
        if nth_index < 0:
            nth_RT = goRTs[0]
        elif nth_index >= len(goRTs):
            nth_RT = goRTs[-1]
        else:
            nth_RT = goRTs[nth_index]
        return nth_RT
