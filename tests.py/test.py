"""
tests for original code to perform in context of refactor

"""

from stopsignalmetrics import StopData, SSRTmodel, PostStopSlow, Violations, StopSummary
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def stopdata_init():
    variable_dict = {
       "columns": {
          "ID": "worker_id",
          "block": "current_block",
          "condition": "SS_trial_type",
          "SSD": "SS_delay",
          "goRT": "rt",
          "stopRT": "rt",
          "response": "key_press",
          "correct_response": "correct_response",
          "choice_accuracy": "choice_accuracy"
       },
       "key_codes": {
          "go": "go",
          "stop": "stop",
          "correct": 1,
          "incorrect": 0
       }
    }
    
    return(StopData(var_dict=variable_dict))


@pytest.fixture(scope="session")
def raw_example_data():
    eg_subj_file = 'examples/example_data/' +\
        'stop_signal_single_task_network_A3QAHF4UUBM7ZO.csv'
    return(pd.read_csv(eg_subj_file, index_col=0))


@pytest.fixture(scope="session")
def preprocessed_example_data(stopdata, raw_example_data):
    return(stopdata.fit_transform(raw_example_data))


@pytest.fixture(scope="session")
def raw_group_data():
    eg_group_file = 'examples/example_data/stop_signal_single_task_network.csv'
    eg_group_df = pd.read_csv(eg_group_file, index_col=0)
    return(eg_group_df.reset_index())


@pytest.fixture(scope="session")
def preprocessed_group_data(stopdata, raw_group_data):
    return(stopdata.fit_transform(raw_group_data))


@pytest.fixture(scope="session")
def stop_data_group2_init():
    # this was known as "group2" in the original notebook
    variable_dict = {
       "columns": {
          "ID": "Subject",
          "block": "Block",
          "condition": "TrialType",
          "SSD": "StopSignalDelay",
          "goRT": "GoRT",
          "stopRT": "StopFailureRT",
          "response": "response",
          "correct_response": "CorrectResponse",
          "choice_accuracy": "choice_accuracy"
       },
       "key_codes": {
          "go": "go",
          "stop": "stop",
          "correct": 1,
          "incorrect": 0
       }
    }

    StopData(var_dict=variable_dict)
    return(StopData(var_dict=variable_dict))


@pytest.fixture(scope="session")
def group2_raw_data(ssrtc_group2):
    # init ssrtc, load in data
    eg_group2_file = 'examples/example_data/DataFixedSSDs2.xlsx'
    group2_df = pd.read_excel(eg_group2_file)
    group2_df = group2_df.replace(
        r'^\s*$', np.nan, regex=True
        ).replace('?', np.nan)

    # build up "correct" responses for stop trials
    addon_file = 'examples/example_data/' +\
        'FixedSSD2StopTrialChoiceAccuracyInput.xlsx'
    addon_df = pd.read_excel(addon_file)

    group2_df['StopTrialCorrectResponse'] = np.nan

    # circle response
    for shape in ['Circle', 'Rhombus', 'Square', 'Triangle']:
        # this is a hairball...
        group2_df.loc[(group2_df['TrialType'] == 'stop') &
                      (addon_df['Unnamed: 5'] == f'{shape.lower()}.bmp'),
                      'StopTrialCorrectResponse'] = addon_df.loc[
                          (group2_df['TrialType'] == 'stop') &
                          (addon_df['Unnamed: 5'] == f'{shape.lower()}.bmp'),
                          f'{shape}Response']

    # combine go and stop into single columns
    group2_df['response'] = np.where(
        group2_df['GoTrialResponse'].isnull(),
        group2_df['StopTrialResponse'],
        group2_df['GoTrialResponse'])
    group2_df['CorrectResponse'] = np.where(
        group2_df['GoTrialCorrectResponse'].isnull(),
        group2_df['StopTrialCorrectResponse'],
        group2_df['GoTrialCorrectResponse'])
    group2_df['CorrectResponse'] = group2_df['CorrectResponse'].str.lower()
    return(group2_df)


@pytest.fixture(scope="session")
def group2_preprocessed_data(stop_data2, group2_raw_data):
    return(stop_data2.fit_transform(group2_raw_data))


def test_class(compy):
    assert compy is not None


def test_load_raw_data(raw_example_data):
    assert raw_example_data is not None


def test_preprocess_data(preprocessed_example_data):
    assert preprocessed_example_data is not None


def test_view_params(compy):
    compy.view_params()


def test_ssrt_omission(preprocessed_example_data):
    assert np.allclose(282.6296296296296,
                       SSRTmodel(model='omission').fit_transform(
                           preprocessed_example_data)['SSRT']
                      )


def test_ssrt_integration(preprocessed_example_data):
    assert np.allclose(279.6296296296296,
                       SSRTmodel(model='integration').fit_transform(
                           preprocessed_example_data)['SSRT']
                      )


def test_ssrt_mean(compy, preprocessed_example_data):
    assert np.allclose(301.53228449688623,
                       SSRTmodel(model='mean').fit_transform(
                           preprocessed_example_data)['SSRT']
                      )


def test_load_group_data(raw_group_data):
    assert raw_group_data is not None


def test_preprocess_group_data(preprocessed_group_data):
    assert preprocessed_group_data is not None


def test_group_max_RT(compy, preprocessed_group_data):
    assert np.allclose(1738.0, SSRTmodel().fit_transform(
                           preprocessed_example_data)['max_RT'])


def test_keyword_init(ssrtc_group2):
    assert ssrtc_group2 is not None


def test_group2_raw_data(group2_raw_data):
    assert group2_raw_data is not None


def test_group2_preprocessed_data(group2_preprocessed_data):
    assert group2_preprocessed_data is not None


def test_group2_summary(group2_preprocessed_data):
    summary_df2 = StopSummary().fit_transform(
        group2_preprocessed_data,
        level='group')
    # spot checks - use data for subject 10
    subject_index = 10
    reference_values = {
        'go_RT': 423.6401291016676,
        'stop_RT': 427.4141414141414,
        'post_stop_slow': -9.549295774647888,
        'P(respond|signal)': 0.75,
        'mean_SSD': 250.0,
        'go_corr_RT': 423.52793296089385,
        'go_ACC': 0.9628832705755783,
        'stop_ACC': 0.9696969696969697,
        'SSRT_replacement': 210.0,
        'SSRT_omission': 210.0,
        'SSRT_integration': 209.0,
        'SSRT_mean': 173.64012910166758}

    for variable in reference_values:
        assert np.allclose(summary_df2.loc[subject_index, variable],
                           reference_values[variable],
                           atol=1e-5)


def test_calc_group_violations(group2_preprocessed_data):
    va_df = Violations().fit_transform(
        group2_preprocessed_data,
        level='group')
    # spot check
    row_idx = 253
    reference_values = {
        'Subject': 24.0,
        'ssd': 300.0,
        'n_matched_go': 25.0,
        'mean_violation': -8.16,
        'mean_stopFailureRT': 454.68,
        'mean_precedingGoRT': 462.84}
    for variable in reference_values:
        assert np.allclose(va_df.loc[row_idx, variable],
                           reference_values[variable],
                           atol=1e-5)
