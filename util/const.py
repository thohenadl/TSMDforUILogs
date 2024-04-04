# From Rebmann

import os

dirname = os.path.dirname(__file__)

path_to_files = os.path.join(dirname, 'logs')
log_dir = 'uilogs'
output_dir = 'output'

COMPLETE_INDICATORS_FULL = ["submit", "save", "ok", "confirm", "apply", "add", "cancel", "close", "delete", "done",
                            "download", "finish", "next", "ok", "post", "reject", "send", "update",
                            "upload", "fertig", "speichern", "anwenden", "bernehmen"]

COMPLETE_INDICATORS = ["submit", "save", "ok", "confirm", "apply", "bernehmen"]

OVERHEAD_INDICATORS = ["reload", "refresh", "open", "login", "log in", "username", "password", "signin", "sign in", "sign out", "log out", "sign up", "anmeldung"]

TERMS_FOR_MISSING = ['MISSING', 'UNDEFINED', 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown',
                     'other', 'others', 'na', 'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined',
                     'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
                     'nil', 'null', '', ' ']

NE_CATS = ["PERSON", "CARDINAL"]

# labels
LABEL = "Task"
# INDEX and CASEID refer to the case notion in the log
INDEX = "idx"
CASEID = "case:concept:name"
# A column name for a flag of a micro task limitation
MICROTASK = "micro_task"
# A column name for the action class id
USERACTIONID = 'actionID'
OPERATIONS_ID = "operations_id"
PRED_LABEL = "pred_label"
TIMESTAMP = "timeStamp"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

APPLICATION = "targetApp"

khost = 'localhost'
kport = ':9092'

test_log_name = "L1.csv"
cross_val_logs = ["L"]

# separators
RESULT_CSV_SEP = ";"

# Remove one set of context_attributes by using the comment function
context_attributes_ActionLogger = ["eventType", "target.name", "targetApp", "target.workbookName", "target.sheetName", "target.innerText", "target.tagName"]
context_attributes_smartRPA = ["concept:name","category","application","concept:name","tag_category","tag_type","tag_title"]
context_attributes = context_attributes_smartRPA + context_attributes_ActionLogger
# Attributes that classify the type of object interacted with, i.e. tag_type in smartRPA can be "Submit" or the field value
semantic_attributes_ActionLogger = ["target.innerText", "target.name"] # "tag_type"
semantic_attributes_smartRPA = ["tag_type", "tag_value", "tag_attributes"]
semantic_attributes = semantic_attributes_ActionLogger + semantic_attributes_smartRPA
value_attributes = ["target.innerText", "url", "target.value", "content"]

case_ids = {
        "L1.csv": "idx",
        "L2.csv": "idx",
        "L2_cleaned.csv": "idx",
        "Example Log für FP ALG.csv": "CaseID",
        "2023-01-17_16-06-31 - Banking - Sparkasse  - 1.csv": "CaseID",
        "concatenated_file.csv": "case:concept:name",
        "invoiceFast.csv":"caseID",
        "StudentRecord_segmented.csv": "timeStamp",
        "Reimbursement_segmented.csv": "timeStamp",
        "log1_segmented.csv":"caseID",
        "log2_segmented.csv":"caseID",
        "log3_segmented.csv":"caseID",
        "log4_segmented.csv":"caseID",
        "log5_segmented.csv":"caseID",
        "log6_segmented.csv":"caseID",
        "log7_segmented.csv":"caseID",
        "log8_segmented.csv":"caseID",
        "log9_segmented.csv":"caseID",
        "evaluation_log1_cases_2.csv":"caseID",
        "evaluation_log1_cases_5.csv":"caseID",
        "evaluation_log2_cases_5.csv":"caseID",
        "evaluation_log5_cases_3.csv":"caseID",
        "evaluation_log5_cases_5.csv":"caseID"
        }

timeStamps = {
    "L2.csv": "timeStamp",
    "L2_cleaned.csv": "timeStamp",
    "Example Log für FP ALG.csv": "timestamp",
    "concatenated_file.csv": "time:timestamp",
    "invoiceFast.csv": "timestamp",
    "StudentRecord_segmented.csv": "timeStamp",
    "Reimbursement_segmented.csv": "timeStamp",
    "log1_segmented.csv":"timestamp",
    "log2_segmented.csv":"timestamp",
    "log3_segmented.csv":"timestamp",
    "log4_segmented.csv":"timestamp",
    "log5_segmented.csv":"timestamp",
    "log6_segmented.csv":"timestamp",
    "log7_segmented.csv":"timestamp",
    "log8_segmented.csv":"timestamp",
    "log9_segmented.csv":"timestamp",
    "evaluation_log1_mt_2.csv":"timestamp"
    }