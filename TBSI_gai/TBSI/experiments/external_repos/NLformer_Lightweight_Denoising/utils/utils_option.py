import os
from collections import OrderedDict
from datetime import datetime
import json

def get_timestamp():
    return datetime.now().strftime('_%y%m%d_%H%M%S')

def parse(opt_path):
    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
            
    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['opt_path'] = opt_path
    
    return opt

def save(opt):
    opt_path = opt['opt_path']
    opt_path_copy = opt['save_dir']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    if not os.path.exists(opt_path_copy):
        os.makedirs(opt_path_copy)
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)