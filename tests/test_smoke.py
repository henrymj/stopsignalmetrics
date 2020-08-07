"""
smoke tests for testing setup
"""

from stopsignalmetrics import StopData, SSRTmodel, PostStopSlow, Violations, StopSummary
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

def test_stopdata_init(stopdata_init):
    assert stopdata_init is not None