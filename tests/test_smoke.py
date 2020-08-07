"""
smoke tests for testing setup
"""

from stopsignalmetrics import StopData, SSRTmodel, Sequence,\
   PostStopSlow, Violations, StopSummary
import pytest


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


def test_stopdata(stopdata):
    assert stopdata is not None


def test_ssrtmodel(ssrtmodel):
    assert ssrtmodel is not None


def test_sequence(sequence):
    assert sequence is not None


def test_pss(pss):
    assert pss is not None


def test_violations(violations):
    assert violations is not None


def test_stopsummary(stopsummary):
    assert stopsummary is not None
