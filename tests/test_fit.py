# test fit and transform 
# These tests checkstop data's ability to load in and preprocess data
# check simulatedata's ability to simulate data
# check fit and transform on the other classes

 from stopsignalmetrics import StopData, SSRTmodel, PostStopSlow, Violations, StopSummary
 import numpy as np
 import pandas as pd
 import pytest

# Same beginning set up as test_smoke.py?

@pytest.fixture(scope="session")
def stopdata():
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

# taken from test.py lines 37 to 41
# unsure whether to include
@pytest.fixture(scope="session")
def raw_example_data():
    eg_subj_file = 'examples/example_data/' +\
        'stop_signal_single_task_network_A3QAHF4UUBM7ZO.csv'
    return(pd.read_csv(eg_subj_file, index_col=0))


@pytest.fixture(scope="session")
def preprocessed_example_data(stopdata, raw_example_data):
    return(stopdata.fit_transform(raw_example_data))

@pytest.fixture(scope="session")
def ssrtmodel():
    return(SSRTmodel())


@pytest.fixture(scope="session")
def sequence():
    return(Sequence())


@pytest.fixture(scope="session")
def pss():
    return(PostStopSlow())


@pytest.fixture(scope="session")
def violations():
    return(Violations())


@pytest.fixture(scope="session")
def stopsummary():
    return(StopSummary())



# does it get answer right using generate_test_df
# unsure about line 79 and 80
# unsure about how to test load in? see two functions after this
@pytest.fixture(scope="session")
def simulated_data(stopdata):
    return(generate_test_df(
        stopdata[raw_group_data], stopdata[preprocessed_group_data])


def test_load_group_data(raw_group_data)
		test_df= generate_test_df(
    		data, var_dict = StopData.load(source='inlab', level='group', return_clean=False) 


def test_preprocess_group_data(preprocessed_group_data):
	test_df= generate_test_df(
    		data, var_dict = StopData.load(source='inlab', level='group', return_clean=False) 


def test_ssrtmodel_fit(simulated_data, stopdata):
    ssrtmodel.fit(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], ssrtmodel.preprocessed_group_data)


def test_ssrtmodel_transform(simulated_data, stopdata):
    ssrtmodel.transform(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], ssrtmodel.preprocessed_group_data)

# does this data frame include rt?
def test_ssrtmodel_checkfail(simulated_data, stopdata):
    with pytest.raises(ValueError):
        raw_example_data.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:]) 

# def test_ssrtmodel_fit_transform(simulated_data, stopdata):


@pytest.fixture(scope="session")
	def test_stopdata(stopdata):
	preprocessed_example_data(stopdata, raw_example_data):
	   return(stopdata.fit_transform(raw_example_data))


def test_sequence_fit(simulated_data, stopdata):
    sequence.fit(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], sequence.preprocessed_group_data)


def test_sequence_transform(simulated_data, stopdata):
    sequence.transform(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], sequence.preprocessed_group_data)

def test_sequence_checkfail(simulated_data, stopdata):
    with pytest.raises(ValueError):
        raw_example_data.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:]) 



def test_pss_fit(simulated_data, stopdata):
    pss.fit(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], pss.preprocessed_group_data)


def test_pss_transform(simulated_data, stopdata):
    pss.transform(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], pss.preprocessed_group_data)

def test_pss_checkfail(simulated_data, stopdata):
    with pytest.raises(ValueError):
        raw_example_data.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:]) 


def test_violations_fit(simulated_data, stopdata):
    violations.fit(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], violations.preprocessed_group_data)

def test_violations_transform(simulated_data, stopdata):
    sequence.transform(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], sequence.preprocessed_group_data)

def test_violations_checkfail(simulated_data, stopdata):
    with pytest.raises(ValueError):
        raw_example_data.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:]) 


def test_stopsummary_fit(simulated_data, stopdata):
    stopsummary.fit(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], stopsummary.preprocessed_group_data)


def test_stopsummary_transform(simulated_data, stopdata):
    stopsummary.transform(simulated_data.rt, simulated_data.accuracy)
     assert np.allclose(stopdata[preprocessed_group_data], stopsummary.preprocessed_group_data)

def test_stopsummary_checkfail(simulated_data, stopdata):
    with pytest.raises(ValueError):
        raw_example_data.fit(simulated_data.rt,
                simulated_data.accuracy.loc[1:]) 

