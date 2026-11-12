from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pingpong.library import prepare_training_index

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--library',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();print(json.dumps(prepare_training_index(args.library,args.output),indent=2))
