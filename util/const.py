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

conceptNames = {
    'beforeSaveWorkbook','urlHashChange','contextMenu','clickCheckboxButton','clickRadioButton','navigateTo','link','typed','form','reload','clickTextField',
    'clickButton','clickLink','selectOptions','selectText','submit','changeField','doubleClick','dragElement','cancelDialog','fullscreen','attachTab',
    'detachTab','newBookmark','removeBookmark','modifyBookmark','moveBookmark','startDownload','erasedDownload','installBrowserExtension','uninstallBrowserExtension',
    'enableBrowserExtension','disableBrowserExtension','closedNotification','clickedNotification','newWindow','closeWindow','newTab','closeTab','moveTab',
    'mutedTab','unmutedTab','pinnedTab','unpinnedTab','audibleTab','zoomTab','changeHistory','created','modified','deleted','Mount','Unmount','moved',
    'programOpen','programClose','selectFile','selectFolder','hotkey','insertUSB','printSubmitted','openFile','openFolder','copy','paste','cut','openWindow','closeWindow',
    'resizeWindow','newWorkbook','openWorkbook','addWorksheet','saveWorkbook','printWorkbook','closeWorkbook','activateWorkbook','deactivateWorkbook','modelChangeWorkbook',
    'newChartWorkbook','afterCalculate','selectWorksheet','deleteWorksheet','doubleClickCellWithValue','doubleClickEmptyCell','rightClickCellWithValue',
    'rightClickEmptyCell','sheetCalculate','editCellSheet','deselectWorksheet','followHiperlinkSheet','pivotTableValueChangeSheet','getRange',
    'getCell','worksheetTableUpdated','addinInstalledWorkbook','addinUninstalledWorkbook','XMLImportWorkbook','XMLExportWorkbook','activateWindow',
    'deactivateWindow','doubleClickWindow','rightClickWindow','newDocument','openDocument','changeDocument','saveDocument','printDocument','activateWindow',
    'deactivateWindow','rightClickPresentation','doubleClickPresentation','newPresentation','newPresentationSlide','closePresentation','savePresentation',
    'openPresentation','printPresentation','slideshowBegin','nextSlideshow','clickNextSlideshow','previousSlideshow','slideshowEnd','SlideSelectionChanged',
    'startupOutlook','quitOutlook','receiveMail','sendMail','logonComplete','newReminder'
    }

dtype = {
    'event_src_path': str,
    'event_dest_path': str,
    'clipboard_content': str,
    'mouse_coord': str,  # May benefit from a custom format string
    'workbook': str,
    'current_worksheet': str,
    'worksheets': str,
    'sheets': str,
    'cell_content': str,
    'cell_range': str,
    'cell_range_number': str,  # May benefit from a custom format string
    'window_size': str,  # May benefit from a custom format string
    'slides': str,
    'effect': str,
    'hotkey': str,
    'id': str,
    'title': str,
    'description': str,
    'browser_url': str,
    'eventQual': str,
    'tab_moved_from_index': str,  # May benefit from a custom format string (if numeric)
    'tab_moved_to_index': str,  # May benefit from a custom format string (if numeric)
    'newZoomFactor': str,  # May benefit from a custom format string (if numeric)
    'oldZoomFactor': str,  # May benefit from a custom format string (if numeric)
    'tab_pinned': str,  # May convert to boolean later if analysis requires
    'tab_audible': str,  # May convert to boolean later if analysis requires
    'tab_muted': str,  # May convert to boolean later if analysis requires
    'window_ingognito': str,  # May convert to boolean later if analysis requires
    'file_size': str,  # May benefit from a custom format string (if numeric)
    'tag_category': str,
    'tag_type': str,
    'tag_name': str,
    'tag_title': str,
    'tag_value': str,
    'tag_checked': str,  # May convert to boolean later if analysis requires
    'tag_html': str,
    'tag_href': str,
    'tag_innerText': str,
    'tag_option': str,
    'tag_attributes': str,
    'xpath': str,
    'xpath_full': str
}