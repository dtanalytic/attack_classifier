import os
import re
import sys
import json
import argparse
import traceback

from pandas import DataFrame
from Model import ReportModel

parser = argparse.ArgumentParser(description='ATTCK Analyser')
parser.add_argument('-b', '--base-path', action='store', dest='base_path', required=False)
parser.add_argument('-o', '--output-dir', action='store', dest='outdir_path', required=False)
result = parser.parse_args()

REPORTS_DIRS = list()
EXTRACTED_REPORTS = list()
if result.base_path:
    REPORTS_DIRS = [x.path for x in os.scandir(result.base_path) if x.is_dir()]

regexp_ttp = re.compile(r'(^| )(t\d{3,}(\.\d+)?)([., ]|$)', flags=re.IGNORECASE)


def analyse_table(tbl: dict):
    ret = dict()

    table = DataFrame.from_dict(tbl)
    counter_ttp_rows = 0
    for index_row in range(table.shape[0]):
        # Поищем строки с номерами TTP.
        row_strings = list()
        have_ttp = False
        code = ''
        for index_column in range(table.shape[1]):
            cell = table.iloc[index_row, index_column]
            row_strings.append(str(cell).replace('"', '\"'))
            matcher_ttp = regexp_ttp.search(str(cell))
            if matcher_ttp:
                have_ttp = True
                counter_ttp_rows += 1
                code = matcher_ttp.group(2).lower()
        if have_ttp:
            if ret.get(code) is None:
                ret[code] = list()
            ret[code] += row_strings

    if (table.shape[0] != 0) and ((counter_ttp_rows/table.shape[0]) > 0.5):
        print('Found Table with ATTCK')
        return ret

    return ret


def extract(report_dir_path: str):
    print('Try to extract: ' + report_dir_path)
    report_model_path = os.path.join(report_dir_path, 'model.json')
    if not os.path.exists(report_model_path):
        print('model.json not found in '+report_dir_path)
        return {}, {}

    report_model = ReportModel()
    with open(report_model_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
        report_model.from_dict(d)

    ret = dict()
    tables = list()
    # Найдем талицу.
    for chapter in report_model:
        for sentence in chapter:
            if sentence.have_tags:
                for name, content in sentence.get_tags().items():
                    if not name.startswith('<table_'): continue
                    print('Table Found')
                    ttps = analyse_table(content)
                    if len(ttps) == 0:
                        continue
                    for k, v in ttps.items():
                        if ret.get(k) is None:
                            ret[k] = list()
                        for i in v:
                            ret[k].append(i)
                    tables.append(content)
    return ret, tables


ttps_descrs = list()
ttps_tables = list()
try:
    for r in REPORTS_DIRS:
        ttps, tables = extract(r)
        if len(tables) != 0:
            ttps_tables.append({'report_path': r, 'tables': tables})
        for ttp_code, ttp_descrs in ttps.items():
            for i in ttp_descrs:
                ttps_descrs.append('{}|"{}"|"{}"\n'.format(ttp_code, i, r))

    with open(os.path.join(result.outdir_path, 'attck_descrs.txt'), 'w', encoding='utf-8') as f:
        f.writelines(ttps_descrs)
    with open(os.path.join(result.outdir_path, 'attck_tables.json'), 'w', encoding='utf-8') as f:
        json.dump(ttps_tables, f)

except Exception as ex:
    print('module=UtilsATTCKAnalyser, msg="{}"'.format(ex))
    traceback.print_tb(ex.__traceback__, limit=None, file=sys.stderr)
    sys.stderr.write(str(ex))
    sys.exit(1)
