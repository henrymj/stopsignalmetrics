import numpy as np
import pandas as pd


class SimulateData():

    def __init__(self, model='independent_race',
                 variable_mu_stop=False,
                 trigger_failures=False,
                 guesses=False,
                 graded_mu_go=False):
        self.model = model
        self.variable_mu_stop = variable_mu_stop
        self.trigger_failures = trigger_failures
        self.guesses = guesses
        self.graded_mu_go = graded_mu_go
        trial_iterators = {
            'independent_race': self._independent_race_trial,
            'interactive_race': self._interactive_race_trial,
            'blocked_input': self._blocked_input_trial
        }
        self._trial_iter = trial_iterators[model]

    def simulate(self, params={}):
        params = self._add_param_defaults(params)
        data_dict = self._init_data_dict()
        self._set_n_guesses_per_type(params)
        for ssd_idx, SSD in enumerate(params['SSDs']):
            data_dict = self._simulate_guesses(data_dict, params, SSD, ssd_idx)
            data_dict = self._simulate_stop_trials(data_dict, params,
                                                   SSD, ssd_idx)
        data_dict = self._simulate_go_trials(data_dict, params)

        data_df = pd.DataFrame.from_dict(data_dict)
        data_df['block'] = 0
        data_df['goRT'] = np.where(data_df['condition'] == 'go',
                                   data_df['RT'],
                                   np.nan)
        data_df['stopRT'] = np.where(data_df['condition'] == 'stop',
                                     data_df['RT'],
                                     np.nan)
        del data_df['RT']
        return data_df

    def _simulate_guesses(self, data_dict, params, SSD, ssd_idx):
        guess_RTs = params['guess_function'](
            int(self._n_guesses[ssd_idx])
        )
        if SSD is None:
            for trial_idx, guess_RT in enumerate(guess_RTs):
                trial = self._init_trial_dict(params, trial_idx)
                trial['RT'] = guess_RT
                data_dict = self._update_data_dict(data_dict, trial)
        else:
            stop_init_time = SSD + params['nondecision_stop']
            for trial_idx, guess_RT in enumerate(guess_RTs):
                trial = self._init_trial_dict(params, trial_idx,
                                              SSD=SSD,
                                              stop_init_time=stop_init_time)
                stop_accum = 0
                for time in range(1, trial['max_time']+1):
                    if time >= trial['stop_init_time']:
                        stop_accum = self._at_least_0(
                            stop_accum + trial['mu_stop'] +
                            np.random.normal(loc=0, scale=trial['noise_stop'])
                        )
                        trial['process_stop'].append(stop_accum)
                    if stop_accum > trial['threshold']:
                        break

                if guess_RT <= time:
                    trial['RT'] = guess_RT

                trial['accum_stop'] = stop_accum
                data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _simulate_stop_trials(self, data_dict, params, SSD, ssd_idx):
        stop_init_time = SSD + params['nondecision_stop']
        for trial_idx in range(int(self._n_guesses[ssd_idx]),
                               params['n_trials']):
            trial = self._init_trial_dict(params, trial_idx,
                                          SSD=SSD,
                                          stop_init_time=stop_init_time)
            data_dict = self._trial_iter(data_dict, trial)
        return data_dict

    def _independent_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
            if stop_accum > trial['threshold']:
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _interactive_race_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    stop_accum + trial['mu_stop'] +
                    np.random.normal(loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break
        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _blocked_input_trial(self, data_dict, trial):
        go_accum = 0
        stop_accum = 0
        for time in range(1, trial['max_time']+1):
            if time >= trial['stop_init_time']:
                stop_accum = self._at_least_0(
                    trial['mu_stop'] + np.random.normal(
                        loc=0, scale=trial['noise_stop'])
                )
                trial['process_stop'].append(stop_accum)
            if time >= trial['nondecision_go']:
                go_accum = self._at_least_0(
                    go_accum + trial['mu_go'] -
                    trial['inhibition_interaction']*stop_accum +
                    np.random.normal(loc=0, scale=trial['noise_go'])
                )
                trial['process_go'].append(go_accum)
            if go_accum > trial['threshold']:
                trial['RT'] = time
                break

        trial['accum_go'] = go_accum
        trial['accum_stop'] = stop_accum
        return self._update_data_dict(data_dict, trial)

    def _simulate_go_trials(self, data_dict, params):
        data_dict = self._simulate_guesses(data_dict, params, None, -1)
        for trial_idx in range(int(self._n_guesses[-1]),
                               params['n_trials']):
            trial = self._init_trial_dict(params, trial_idx, condition='go')
            go_accum = 0
            stop_accum = 0
            for time in range(1, trial['max_time']+1):
                if time >= trial['nondecision_go']:
                    go_accum = self._at_least_0(
                        go_accum + trial['mu_go'] +
                        np.random.normal(loc=0, scale=trial['noise_go'])
                    )
                    trial['process_go'].append(go_accum)
                if go_accum > trial['threshold']:
                    trial['RT'] = time
                    break

            trial['accum_go'] = go_accum
            trial['accum_stop'] = stop_accum
            data_dict = self._update_data_dict(data_dict, trial)
        return data_dict

    def _set_n_guesses_per_type(self, params):
        # TODO: ADD ASSERTIONS TO CHECK FOR CORRECT USES, clean up!!!
        # TODO: allow for guessing on go trials
        num_types = len(params['SSDs']) + 1
        if self.guesses:
            if type(params['p_guess']) == float:
                p_guess_per_type = [params['p_guess']] * num_types
            elif type(params['p_guess']) in [list, np.ndarray]:
                if len(params['p_guess']) == 1:
                    p_guess_per_type = params['p_guess'] * num_types
                else:
                    p_guess_per_type = params['p_guess']
        else:
            p_guess_per_type = [0] * num_types

        if len(p_guess_per_type) == len(params['SSDs']):
            p_guess_per_type.append(0.0)
        assert(len(p_guess_per_type) == num_types)
        # TODO: clean up this line -
        # if 0 is returned, it's viewed as an int,
        # not a float, so it needs to be converted
        self._n_guesses = np.rint([float(p * params['n_trials'])
                                   for p in p_guess_per_type])

    def _get_mu_stop(self, params):
        mu_stop = params['mu_stop']
        if self.variable_mu_stop:
            mu_stop = mu_stop+np.random.normal(
                loc=0, scale=params['noise_stop']*.7)
        if self.trigger_failures and np.random.uniform(0, 1) <\
                params['p_trigger_fail']:
            mu_stop = 0
        return self._at_least_0(mu_stop)

    def _get_mu_go(self, params, SSD):
        # TODO: make more dynamic, pass max_SSD
        mu_go = params['mu_go']
        if self.graded_mu_go and SSD is not None:
            mu_go = self._log_mu_go(mu_go, SSD)
        return mu_go

    def _log_mu_go(self, mu_go, SSD, max_SSD=550):
        if SSD > max_SSD:
            SSD = max_SSD
        return self._at_least_0((np.log(SSD)/np.log(max_SSD)) * mu_go)

#     def _linear_mu_go(self, mu_go, SSD, max_SSD=550):
#         if SSD > max_SSD:
#             SSD = max_SSD
#         return self.at_least_0((SSD/max_SSD) * mu_go)

    def _add_param_defaults(self, params):
        # TODO: move default dict to json, read in
        default_dict = {'mu_go': .25,
                        'mu_stop': .6,
                        'noise_go': 1.15,
                        'noise_stop': 1.45,
                        'threshold': 95,
                        'nondecision_go': 50,
                        'nondecision_stop': 50,
                        'inhibition_interaction': .5,
                        'SSDs': np.arange(0, 600, 50),
                        'n_trials': 1000,
                        'max_time': 1000,
                        'p_trigger_fail': 0,
                        'p_guess': 0,
                        'guess_function': lambda x: np.random.uniform(
                            200, 400, x),
                        }

        for key in default_dict:
            if key not in params:
                params[key] = default_dict[key]
        return params

    def _init_data_dict(self):
        return {
            'condition': [],
            'SSD': [],
            'trial_idx': [],
            'RT': [],
            'mu_go': [],
            'mu_stop': [],
            'accum_go': [],
            'accum_stop': [],
            'process_go': [],
            'process_stop': [],
            }

    def _init_trial_dict(self, params, trial_idx,
                         SSD=None, stop_init_time=np.nan, condition='stop'):
        trial = {
                'condition': condition,
                'SSD': SSD,
                'trial_idx': trial_idx,
                'mu_go': self._get_mu_go(params, SSD),
                'mu_stop': self._get_mu_stop(params),
                'stop_init_time': stop_init_time,
                'noise_go': params['noise_go'],
                'noise_stop': params['noise_stop'],
                'nondecision_go': params['nondecision_go'],
                'inhibition_interaction': params['inhibition_interaction'],
                'threshold': params['threshold'],
                'max_time': params['max_time'],
                'accum_go': np.nan,
                'accum_stop': np.nan,
                'process_go': [],
                'process_stop': [],
                'RT': np.nan
            }
        return trial

    def _update_data_dict(self, data_dict, update_dict):
        for key in data_dict.keys():
            data_dict[key].append(update_dict[key])
        return data_dict

    def _at_least_0(self, num):
        if num < 0:
            num = 0
        return num