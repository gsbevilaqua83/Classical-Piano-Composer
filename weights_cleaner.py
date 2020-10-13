import os
import time
import fire

def remove_files(files_list):
    for fn in files_list:
        os.remove(fn)
    print('removed: ', files_list)

def run_cleaner(sleep_time=5):
    not_cleaned_for_n_times = 0
    while(1):
        weights_list = []
        for fn in (os.listdir()):
            if os.path.basename(fn).split('.')[-1] == 'hdf5':
                weights_list.append(fn)

        sorted_weights = sorted(weights_list, key=lambda x: float(x.split('-')[3]))

        if len(sorted_weights) > 2:
            not_cleaned_for_n_times = 0
            remove_files(sorted_weights[2:])
        else:
            print('--- no file to remove')
            not_cleaned_for_n_times += 1

        if not_cleaned_for_n_times == 20:
            print('--- giving up now... bye')
            break
        time.sleep(sleep_time)

if __name__ == '__main__':
    fire.Fire(run_cleaner)