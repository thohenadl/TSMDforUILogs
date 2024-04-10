# From Rebmann

from collections import Counter

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from util.const import COMPLETE_INDICATORS, OPERATIONS_ID, TERMS_FOR_MISSING, LABEL, INDEX, CASEID
from util.stringutil import preprocess_label
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

np.seterr(invalid='ignore')


# The maximum length of a vector
max_length = 1000

def tokenize(word):
    cleaned = preprocess_label(word).split(" ")
    return cleaned

# ----- Boundary Event Util ----



class UIPatternMiner:

    def __init__(self, events, case_ids, timeStamps, context_attributes, value_attributes, semantic_attributes):
        # GIVEN
        self.vec_len = 120
        self.frequent_item_sets = None
        self.binary_encoded_log = None
        self.case_id = None
        self.set_case_id(events.columns,case_ids.values())
        self.timeStamp = None
        self.set_timeStamp(events.columns,timeStamps.values())
        self.context_attributes = None
        self.set_context_attributes(events, context_attributes)
        self.value_attributes = value_attributes
        self.semantic_attributes = None
        self.set_semantic_attributes(events, semantic_attributes)
        # the input events
        self.events = events
        # CACHES
        # Buffers all events per index
        self.event_buffer = dict()
        # Buffers all event classes in order of appearance
        self.event_class_buffer = list()
        # Keeps track of unique event classes
        self.unique_event_classes = list()
        # The indices where the identified micro-tasks end
        self.boundaries = list()
        # This stores vectorized micro tasks per boundary index (boundary of the completed micro-task)
        self.buffered_vecs = dict()
        # This keeps track of how often a micro-task occurs globally
        self.vec_count = dict()
        # This stores vectorized micro-tasks per task instance
        self.vec_per_task = dict()

    def set_context_attributes(self,df: pd.DataFrame, context_parameters: list):
        """
        Sets the context parameters based on the dataframe and a list of possible context parameters.
        
        Arg(s):
        df (DataFrame): The event log
        context_parameters (list of Strings): Possible context parameter column names in df
        
        Sets:
        self.context_attributes
        
        Return(s):
        None"""
        actual_context_params = [col for col in context_parameters if col in df.columns]
        self.context_attributes = actual_context_params

    def set_semantic_attributes(self, df: pd.DataFrame, semantic_parameters: list):
        """
        Sets the semantic parameters based on the dataframe and a list of possible context parameters.
        
        Arg(s):
        df (DataFrame): The event log
        semantic_parameters (list of Strings): Possible context parameter column names in df
        
        Sets:
        self.semantic_attributes
        
        Return(s):
        None"""
        actual_semantic_params = [col for col in semantic_parameters if col in df.columns]
        self.semantic_attributes = actual_semantic_params

    def set_case_id(self,columns: list,case_ids: dict):
        """
        Sets the case ID column in the object based on a list of possible case ID column names and a dictionary
        mapping each case ID column name to a case identifier string.

        Parameters:
        columns (list): A list of column names in a Pandas DataFrame.
        case_ids (dict): A dictionary mapping each possible case ID column name to a case identifier string.

        Raises:
        ValueError: If no case ID column is found in the list of columns for the provided case identifiers.

        Returns:
        None.
        """
        for case_id in case_ids:
            if case_id in columns:
                self.case_id = case_id
                return
        # If no case id is identified raise an error
        raise ValueError("No case id column found for provided case identifiers in the provided dataframe.")
    
    def set_timeStamp(self,columns: list,timeStamps: dict):
        """
        Sets the time stamp column in the object based on a list of possible time stamp column names and a dictionary
        mapping each time stamp column name to a case identifier string.

        Parameters:
        columns (list): A list of column names in a Pandas DataFrame.
        case_ids (dict): A dictionary mapping each possible case ID column name to a case identifier string.

        Raises:
        ValueError: If no case ID column is found in the list of columns for the provided case identifiers.

        Returns:
        None.
        """
        for stamp in timeStamps:
            if stamp in columns:
                self.timeStamp = stamp
                return
        # If no case id is identified raise an error
        raise ValueError("No time stamp column found for provided case identifiers in the provided dataframe.")

    def prepare_feature_vec(self, uis_in_seg):
        vector = []
        counts = Counter(uis_in_seg)
        for unique_op in self.unique_event_classes:
            if unique_op in counts:
                vector.append(1 if counts[unique_op] > 0 else 0)# one-hot encoding
            else:
                vector.append(0)
        if len(vector) < max_length:
            vector = vector + [0 for _ in range(self.vec_len - len(self.unique_event_classes))]
        return vector

    def get_micro_tasks(self):
        if len(self.events) == 0:
            return self.boundaries
        for index, row in self.events.iterrows():
            task_inst_id = row[self.case_id] #str(row[LABEL]) + str(row[INDEX])
            # Derive the event class and add it to the buffers
            tup = self.create_tup(row)
            row[OPERATIONS_ID] = tup
            self.event_class_buffer.append(tup)
            self.event_buffer[index] = row
            if tup not in self.unique_event_classes:
                self.unique_event_classes.append(tup)
            if any(preprocess_label(str(row[sem_col])) in COMPLETE_INDICATORS or
                   (len(preprocess_label(str(row[sem_col])).split(" ")) == 2 and
                    preprocess_label(str(row[sem_col])).split(" ")[1] in COMPLETE_INDICATORS)
                   for sem_col in self.semantic_attributes):
                if len(self.boundaries) == 0:
                    curr = 0
                else:
                    curr = self.boundaries[-1]
                self.boundaries.append(index)
                vec = self.prepare_feature_vec([self.event_buffer[i][OPERATIONS_ID] for i in range(curr, index)])
                self.buffered_vecs[index] = vec
                if tuple(vec) not in self.vec_count:
                    self.vec_count[tuple(vec)] = 1
                else:
                    self.vec_count[tuple(vec)] += 1
                if task_inst_id not in self.vec_per_task:
                    self.vec_per_task[task_inst_id] = []
                self.vec_per_task[task_inst_id].append(tuple(vec))
        self.boundaries.append(len(self.events)-1)
        vec = self.prepare_feature_vec([self.event_buffer[i][OPERATIONS_ID] for i in range(self.boundaries[-1], len(self.events)-1)])
        self.buffered_vecs[len(self.events)-1] = vec
        if tuple(vec) not in self.vec_count:
            self.vec_count[tuple(vec)] = 1
        else:
            self.vec_count[tuple(vec)] += 1
        if task_inst_id not in self.vec_per_task:
            self.vec_per_task[task_inst_id] = []
        self.vec_per_task[task_inst_id].append(tuple(vec))
        return self.boundaries

    def create_tup(self, row, embedding=False):
        tup = []
        for att in self.context_attributes:
            if row[att] not in TERMS_FOR_MISSING:
                tup.append(str(row[att]))
        return tup if embedding else "*#*".join(tup)

    def create_value_tup(self, row):
        tup = []
        for att in self.value_attributes:
            if str(row[att]) not in TERMS_FOR_MISSING:
                tup.append(str(row[att]))
        return tup

    def log_encoding(self) -> pd.DataFrame:
        """
        Return a binary encoding for the ui log, i.e. the one-hot encoding stating whether a micro task is contained
        or not inside each trace.
        Returns
        -------
        binary_encoded_log
            the one-hot encoding of the input log, made over activity names or resources depending on 'dimension' value.
        """
        te = TransactionEncoder()
        dataset = self.get_micro_tasks_for_pattern_mining()
        te_array = te.fit(dataset).transform(dataset)
        self.binary_encoded_log = pd.DataFrame(te_array, columns=te.columns_)
        return self.binary_encoded_log

    def compute_frequent_itemsets(self, min_support: float,  algorithm: str = 'fpgrowth',
                                  len_itemset: int = None) -> None:
        self.log_encoding()
        if algorithm == 'fpgrowth':
            frequent_itemsets = fpgrowth(self.binary_encoded_log, min_support=min_support, use_colnames=True)
        elif algorithm == 'apriori':
            frequent_itemsets = apriori(self.binary_encoded_log, min_support=min_support, use_colnames=True)
        else:
            raise RuntimeError(f"{algorithm} algorithm not supported. Choose between fpgrowth and apriori")
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        if len_itemset is None:
            self.frequent_item_sets = frequent_itemsets
        else:
            self.frequent_item_sets = frequent_itemsets[(frequent_itemsets['length'] <= len_itemset)]

    def get_micro_tasks_for_pattern_mining(self):
        if len(self.vec_per_task) == 0:
            raise RuntimeError("No micro-tasks found! Cannot continue with data preparation.")
        dataset = []
        for task_id, vecs in self.vec_per_task.items():
            if len(vecs) < 2:
                # print("Skipping " + task_id + " because it does not even contain two microtasks.")
                continue
            print("adding " + str(len(vecs)) + " micro-tasks for " + task_id)
            dataset.append([",".join(str(v) for v in vec) for vec in vecs])
        print(dataset)
        return dataset

    def get_attribute(self, attr_name: str) -> object:
        """
        Gets the value of the specified attribute of this UserInteraction instance.

        Args:
            attr_name (str): The name of the attribute to get the value of.

        Returns:
            Any: The value of the specified attribute.
        """
        return getattr(self, attr_name)

    