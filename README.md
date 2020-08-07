# stopsignalmetrics

![Python package](https://github.com/henrymj/stopsignalmetrics/workflows/Python%20package/badge.svg)

This is a package to streamline common computations on behavioral data from experiments using Stop Signal tasks. It is made up of multiple classes which focus on different types of metrics. All classes follow the scikit-learn `fit`, `transform` schema.

__0. `StopData` - Preprocessing and Standardization.__
This class will be initialized with a nested dictionary, mapping columns (e.g. the SSD and RT columns) and key_codes (e.g. labels for stop and go trials in the condition column) from the current data onto a standard. See stopsignalmetrics/standards.json or the examples to get a sense of this mapping. It will also compute choice accuracy if a choice accuracy column is not found, or `compute_acc_col=True` is passed in at intialization.

__1. `SSRTmodel` - Stop Signal Reaction Time (SSRT) Computation.__
The `SSRTmodel` class contains 4 methods of Stop Signal Reaction Time (SSRT) computation:

_1. Integration with Replacement ("replacement")_  
This method replaces go omissions with the max reaction time before getting the nth_RT. This is the recommended method from [Verbruggen et al. (2019)](10.7554/eLife.46323). The other 3 methods are included for completeness and comparison, but in general we agree with Verbruggen et al. 2019 that the Integration with Replacement method is preferred. 

_2. Integration with Omission Rate Adjustment ("omission")_  
This method uses the omission rate on go trials to adjust the P(respond | signal) before getting the nth_index.

_3. Integration ("integration")_  
A vanilla version of the above two methods which makes no adjustment based on omissions.

_4. Mean ("mean")_  
SSRT = mean_go_RT - mean_SSD. This method is based upon the assumption that the race between the go and stop process is tied, which should be the case when the common 1-up-1-down tracking method (Levitt, 1971) is used. 

Addionally, fitting 

__2. `Sequence` - Examining trial-by-trial Fluctuations.__
This module is designed to analyze the data in the format of triplets of trials, with the central trials being chosen based on a research-question-based criteria (e.g. stop-failures). There are currently 3 classes.

_1. `Sequence`_  
This class will produces dataframes with triples of trials centered on trials based on an array-like list of indices or a query string. It is the backbone of the following methods.

_2. `Post Stop Slowing`_  
This class examines the change in go reaction times after a stop trial (i.e., RT on the trial immediately preceding a stop trial and subtracting it from RT on the trial immediately following a stop trial) . By default it will use all stop trials, but users can specify focusing on stop-success or stop-failure trials.

_3. `Violations`_  
This class compares stop-failure RTs to the preceding trial's correct go RTs, calculating a "mean violation" per SSD, with some thresholding to reduce noise.

__3. `Summary` - Describing a stop dataset.__  
The `Summary` class computes every metric currently available, including a mean post-stop-slowing for each stop type ('all', 'success', 'fail'). It also attempts to compute a mean violation, thresholding at SSDs < 200ms.

__Notes__  

This package assumes that non-responses (omissions; correct stops) are coded as values <= 0 or NaNs.

Sequential functions assume that each trial is one row. This is not necessarily the case for other functions, which can handle irrelevant rows related to cue/instruction presentations or ITIs.
  
__Bibliography__  
Verbruggen, F., Aron, A. R., Band, G. P., Beste, C., Bissett, P. G., Brockett, A. T., ... & Colzato, L. S. (2019). A consensus guide to capturing the ability to inhibit actions and impulsive behaviors in the stop-signal task. Elife, 8, e46323.
