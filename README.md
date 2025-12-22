# Time Series Routine Discovery
Time Series Routine Discovery (TSRD) for User Interaction Logs

Where is the routine? What routines do we actually do?
Question asked once a business is looking for automation possibilites.
With this approach the question is no longer a pure manual search mission.
With the TSMD approach large sets of user interaction data, i.e., merged logs, are scanned easy and fast.
This read-me file describes the three Jupyter notebooks that are relevant for the paper "Enabling Routine Discovery from High-Noise UI Logs: A Time Series Data Mining Approach"?

## The Approach

![Detailed Approach Visualisation](images/approach_png.png)

The approach has four sequential steps. You will first get to know them and in the section on the Jupyter Notebooks you will understand, how to use the approach and recreate our experiment:

1. The approach takes as an input a UI log. This log is tokenized by using Re-Pair Grammar Rules. The Grammar Rules are a tool to identify variable length motifs in time series data. Senin et al. have shown this in [GrammarViz 3.0](https://dl.acm.org/doi/abs/10.1145/3051126).
2. Based on the so-called Grammar Cores from 1., we can filter the time series to only contain routine candidate sections from the UI log. The approach identifies app switch and process switch patterns in the UI log and extends the Grammar Cores. Afterwards, the log is reduced to only contain these extended cores.
3. In step 3 the reduced log is encoded using Word2vec as described already by [Hohenadl 2025](https://link.springer.com/chapter/10.1007/978-3-032-02936-2_20). Afterwards, [LoCoMotif](https://github.com/ML-KULeuven/locomotif) is applied to identify variable length time series motifs in the reduced, embedded UI log.
4. The output from step 3 is mapped against the Grammar Cores of step 1 to filter all potential motifs for real automation worthy candidates.

The final result is a set of sets. Each set contains n candidates already clustered by the similarity.
More details on every step are available in the paper.

In the next section you will understand die Notebooks in the [Jupyter Notebooks folder](/JupyterNotebooks/).

# Jupyter Notebooks Prepared

There are 5 notebooks ready to be used.
Three notebooks are important for the reproduction of the main experiment: Testing the recall, precision and accuracy of the time series approach overall.
"01" Notebook: The main notebook relevant for the experiment in the paper
"02" Notebook: Test of different encoding approaches, some used for validation, some only for trial and error
"03" Notebook: Discovering routines in a single UI log
"04" Notebook: Creation of the validation dataset for the experiment in "01"
"05" Notebooks: Support code that helps transforming an smartRPA Log into an Action Logger Log

## "01" Experiment.ipynb Notebook
To execute the experiment and gather the insights as presented in the paper, just two steps are necessary:

1. Put the correct path in the parameters
2. Specify the list of window sizes that should be tested

Afterwards, the experiment can be executed and will create a .csv file containing the parameters as defined in the publication.
Once the file is converted into an XLSX file and the data is seperated into columns, the pivot function allows for creation of the graphs visualized in the paper.

## "03" SingleLogDiscovery.ipynb Notebook
The single log discovery notebook is setup to process a single User Interaction log that can contain a routine.
The **first section** imports all necessary functions, including the util.util and other native Python libraries.
The **second section** configured the parameters, which have to be set to make the approach work.
Please specifiy your file, the columns you aligned to the reference model, and the window size parameter particulary.
The **third section** does execute the encoding, discovery, and visualisation. Do not change anything in there to have the approach working.

## "04" ValidationLogCreation.ipynb Notebook
As described in the evaluation section of the paper, the experiment relies on a set of arteficially created user interaction logs.
You can create your own set and follow the process of the log creation by utilizing the "validationLogCreation" notebook.

<details>

<summary>Setting up the parameters as in the paper</summary>

To create validation data as in the publication use the following parameter setup

```randomness = [1] # Length of sampling sequence, when creating the baseline log (1=> only one event inserted, 2=> sequences of 2 from all possible events inserted ...)
motifs = [1] # how many different motifs should be inserted into the log
occurances = [10,15,20,30,60] # Number of motif appearances in the log
lengthMotifs = [5,10,15,20,25] # Length of the Motifs to be inserted
percentageMotifsOverLog = [10,5,2.5,1] # Percentage representation of the Motif in the log
shuffles = [0,10,20] # Percentage by which the inserted routine should be shuffled
```

</details>



## Experiment Results

The experiment results from the paper are available in the file **2025 Overall Experiment Results.xlsx**. 
In this file you will find two sheets. The first sheet contains the collected experiment results on which the evaluation section is based on. The second sheet contains the figures presented in the evaluation section of the paper. This file contains the all collected values from the experiment as outlined in the paper:

![Detailed Approach Visualisation](images/ResultTable.png)

You can reproduce the experiment by executin the **experiment** notebook after you have (a) selected the synthetic validation data or (b) run the **Validation Log Creation** notebook.
The validation data to compare the discovered result by the approach automatically is stored in the file **validationDataPercentage.csv**

## Real World Experiment

The real-world process was designed based on experience in small and medium size enterprises and reflects a common, yet simplified, version of an accounts payable process.
The instruction given to the auther is stored in the **logs/Banking/** folder: [Real World Accounts Payable Instruction](logs/Banking/RealWorldProcessInstruction.pdf)
The generated UI logs (SmartRPA/Tockler) are available in the same folder. 
The logs are anonymized to not reflect any author data.

## Additional notebooks: encodingAnalysis.ipynb & smartRPA-2-ActionLogger.ipynb

These repositories were used for generic research and ideation.

**encodingAnalysis.ipynb** was used to test different encoding methods and present the differences identified.
**smartRPA-2-ActionLogger.ipynb** was used to transform the baseline data into the formats required for comparing our approach with Agostinelli et als. and Leno et als. approaches.