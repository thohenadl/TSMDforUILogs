# TSMDforUILogs
Time Series Motif Discovery for User Interaction Logs

Where is the routine? A question asked once a business is looking for automation possibilites.
With this approach the question is no longer a manual search mission.
With the TSMD approach large sets of user interaction data, i.e., merged logs, are scanned easy and fast.
This read-me file describes the three Jupyter notebooks that are relevant for the paper "Where is the Motif"?

## The approach

![Detailed Approach Visualisation](images/Approach.png)

## Single Log Discovery Notebook
The single log discovery notebook is setup to process a single User Interaction log that can contain a routine.
The **first section** imports all necessary functions, including the util.util and other native Python libraries.
The **second section** configured the parameters, which have to be set to make the approach work.
Please specifiy your file, the columns you aligned to the reference model, and the window size parameter particulary.
The **third section** does execute the encoding, discovery, and visualisation. Do not change anything in there to have the approach working.

## Validation Log Creation Notebook

## Experiment Notebook