import json
import subprocess #Need to clean up GPU memory after execution so forced to restart interpreter
import numpy as np
import argparse
import os

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
    parser.add_argument(
        '--p',
        help='override image path'
    )
    parser.add_argument(
        '--pfx',
        help='override image prefix'
    )
    parser.add_argument(
        '--i',
        action='store_true',
        help='enable this flag for interactive mode'
    )
    return parser.parse_args()


def find_img_data(dat_lines, 
                  scan_range,
                  entry_line, 
                  override_path=False, 
                  override_pfx=False):
    METADATA_LINE_LEN = 51 
    SCAN_ID_IDX = 7
    OMEGA_IDX = 31
    PATH_LEN = 6
    IMG_PFX_LEN = 14

    start_idx = scan_range[0]
    end_idx = scan_range[1]

    omegas = np.zeros(end_idx - start_idx + 1)
    found_omegas = np.zeros(end_idx - start_idx + 1)

    path = None
    img_pfx = None

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

        elif dat_lines[i].startswith('Path:'):
            path = dat_lines[i][PATH_LEN:].strip()
        
        elif dat_lines[i].startswith('Image prefix:'):
            img_pfx = dat_lines[i][IMG_PFX_LEN:].strip()


    if not np.all(found_omegas):
        raise ValueError(f"Unable to find all omegas for given scan range {start_idx}-{end_idx}")
    
    if path is None and not override_path:
        raise ValueError(f"Unable to find image path. Please check metadata file structure.")
    
    if img_pfx is None and not override_pfx:
        raise ValueError(f"Unable to find image prefixes. Please check metadata file structure.")

    return omegas, path, img_pfx


def extract_scan_data(metadata_fp):
    scans = []
    NUM_FIELD_BEGIN = 10
    NUM_FIELD_END = 20

    with open(metadata_fp) as dat_f:
        read_counter = 0
        cur_scan = {}
        img_range = [0, 0]
        dat_lines = dat_f.readlines()

        override_path = args.p is not None
        override_pfx = args.pfx is not None
        for i, line in enumerate(dat_lines):
            if line.startswith('End'):
                read_counter = 2
            elif read_counter == 2:
                read_counter -= 1
                img_range[0] = int(line.split(' ')[4].strip()) + NUM_FIELD_BEGIN
            elif read_counter == 1:
                read_counter -= 1
                img_range[1] = int(line.split(' ')[4].strip()) - NUM_FIELD_END

                omega, path, img_pfx = find_img_data(dat_lines, 
                                                     img_range, 
                                                     i,
                                                     override_path,
                                                     override_pfx)
                cur_scan['img_range'] = img_range
                cur_scan['omega'] = omega
                if override_path: path = args.p
                cur_scan['img_dir'] = path 
                if override_pfx: img_pfx = args.pfx
                cur_scan['img_prefix'] = img_pfx

                scans.append(cur_scan)
                cur_scan = {}
                img_range = [0, 0]
    return scans


def execute_coarse_map(config):
    config['omega'] = config['omega'].tolist()

    num_omegas = len(config['omega'])
    num_imgs = config['img_range'][1] - config['img_range'][0] + 1 
    if num_imgs != num_omegas:
        raise ValueError(f"Number of detected omegas {num_omegas} and images {num_imgs} not equal!")

    config_fp = os.path.join(args.output_fp, 'cur_config.json')
    with open(config_fp, 'w') as curr_conf_f:
        json.dump(config, curr_conf_f)
    map_proc = subprocess.run(['python', 'coarse_mesh.py', f'{config_fp}', args.output_fp])
    
    return map_proc.returncode


def interactive_mode(scans):
    commands = ('Commands: "[int]" run scan on specified range | "l": list scan ranges'
                '\n          "c": print commands | "q": quit program')
    scan_txt = '\n'.join([f"[{i}]: {scan['img_range']}" for i, scan in enumerate(scans)])
    input_line = 'Enter Command: '

    print(scan_txt)
    print(commands)
    while True:
        user_input = input(input_line)
        if user_input.strip() == 'c':
            print(commands)
        elif user_input.strip() == 'l':
            print(scan_txt)
        elif user_input.strip() == 'q':
            return
        else:
            try:
                scan_idx = int(user_input)
            except ValueError:
                print("Unknown command or int!")
            else:
                if scan_idx < 0 or scan_idx > len(scans) - 1:
                    print('Invalid scan index. Choose one within number of scans.')
                else:
                    print(f"Running Scan {scans[scan_idx]['img_range']}")
                    ret_code = execute_coarse_map(scans[scan_idx])
                    if ret_code != 0:
                        print("Warning: Coarse scan exited with non-zero return code!")
        

if __name__ == '__main__':
    args = parse_args()

    scans = extract_scan_data(args.metadata_fp)
                
    if not os.path.exists(args.output_fp):
        os.mkdir(args.output_fp)

    if not args.i:
        for scan in scans:
            print(f"{scan['img_range']}")

    if args.i:
        interactive_mode(scans)
    else:
        error_scans = []
        for i, config in enumerate(scans):
            ret_code = execute_coarse_map(config)
            if ret_code != 0:
                error_scans.append(str(config['img_range']))

            print(f'Finished scan {i + 1}/{len(scans)}')
        
        print(f'Scans with errors: {str(error_scans)}')