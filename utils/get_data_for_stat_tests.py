import matplotlib as plt
import sys
import glob
import errno
import os
import json
import numpy as np
from plot_curves import plot_curves


def get_data_for_stat_tests(input_path, output_path):
    name_mapper = []
    for filename in glob.glob(os.path.join(input_path, '*.json')):
        try:
            data = np.array(1)
            with open(filename) as f:  # No need to specify 'r': this is the default.
                json_content = json.loads(f.read())
                # pretty print and see what's in the json file
                # print json.dumps(json_content, indent=4, sort_keys=True)
                file_name = json_content['metadata']['filepath'].split("/")[-1]
                for cnt, curve in enumerate(json_content['data']):
                    yruns = curve['ys']
                    x = curve['xs']
                    label = curve['name']

                    row_info = np.array(yruns)
                    row_info = row_info.conj().transpose()
                    # one curve
                    x = np.array(x).conj().transpose()
                    x = np.tile(np.array(x), (row_info.shape[0], 1))
                    # row_info = np.hstack((row_info, x))
                    row_info = np.hstack((cnt * np.ones((row_info.shape[0], 1)), row_info))
                    row_info = np.hstack((np.tile(np.array([file_name, label]), (row_info.shape[0], 1)), row_info))

                    if data.size == 1:
                        data = row_info
                    else:
                        data = np.vstack((data, row_info))

                import pandas as pd
                df = pd.DataFrame(data)
                excel_filename = file_name + ".xlsx"
                print filename
                pickle_filename = file_name + ".pkl"
                excelobj = pd.ExcelWriter(output_path + excel_filename, engine="xlsxwriter")
                df.to_excel(excelobj)
                excelobj.save()
                excelobj.close()
                df.to_pickle(output_path + pickle_filename)


        except IOError as exc:
            if exc.errno != errno.EISDIR:  # Do not fail if a directory is found, just ignore it.
                raise  # Propagate other kinds of IOError.


if __name__ == '__main__':
    input_path = '../src/experiments/results/'
    output_path = '../src/experiments/results/raw_data_runs/'
    get_data_for_stat_tests(input_path, output_path)
