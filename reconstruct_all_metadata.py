import json
import subprocess #Need to clean up GPU memory after execution so forced to restart interpreter
import numpy as np
import argparse

def find_omegas(dat_lines, start_idx, end_idx, entry_line):
    METADATA_LINE_LEN = 51 
    SCAN_ID_IDX = 7
    OMEGA_IDX = 31

    omegas = np.zeros(end_idx - start_idx + 1)
    found_omegas = np.zeros(end_idx - start_idx + 1)

    for i in range(entry_line, 0, -1):
        line = dat_lines[i].split(' ')
        if len(line) == METADATA_LINE_LEN:
            scan_num = int(line[SCAN_ID_IDX])

            if scan_num < start_idx:
                break

            if scan_num <= end_idx:
                store_idx = scan_num - start_idx
                omegas[store_idx] = float(line[OMEGA_IDX])
                found_omegas[store_idx] = True

    if not np.all(found_omegas):
        raise ValueError(f"Unable to find all omegas for given scan range {start_idx}-{end_idx}")    
    
    return omegas

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to course-map voids from a series of scans as defined by a metadata file in 1ID format'
    )
    parser.add_argument(
        'metadata_fp', 
        help='path to the scan metadata to be reconstructed'
    )
    parser.add_argument(
        'output_fp', 
        help='path to place output file'
    )
    return parser.parse_args()

if __name__ == '__main__':
    scans = []
    NUM_FIELD_BEGIN = 10
    NUM_FIELD_END = 20

    args = parse_args()


    with open(args.metadata_fp) as dat_f:
        read_counter = 0
        cur_scan = {}
        dat_lines = dat_f.readlines()

        for i, line in enumerate(dat_lines):
            if line.startswith('End'):
                read_counter = 2
            elif read_counter == 2:
                read_counter -= 1
                cur_scan['start'] = int(line.split(' ')[4].strip()) + NUM_FIELD_BEGIN
            elif read_counter == 1:
                read_counter -= 1
                cur_scan['end'] = int(line.split(' ')[4].strip()) - NUM_FIELD_END
                cur_scan['omega'] = find_omegas(dat_lines, 
                                                cur_scan['start'],
                                                cur_scan['end'],
                                                i)
                scans.append(cur_scan)
                cur_scan = {}
                

    with open('working_dir/config.json', 'r') as config_f:
        config = json.load(config_f)

    for scan in scans:
        print(f"{scan['start'], scan['end']}")

    

    error_scans = []
    for i, scan in enumerate(scans):
        config['img_range'] = [scan['start'], scan['end']]
        config['omega'] = scan['omega'].tolist()
        assert scan['end'] - scan['start'] + 1 == len(config['omega'])

        curr_config_fp = 'working_dir/temp_all_config.json'
        with open(curr_config_fp, 'w') as curr_conf_f:
            json.dump(config, curr_conf_f)
        map_proc = subprocess.run(['python', 'coarse_mesh.py', f'{curr_config_fp}', args.output_fp])
        if map_proc.returncode != 0:
            error_scans.append(f"{scan['start']}-{scan['end']}")

        print(f'Finished scan {i + 1}/{len(scans)}')




