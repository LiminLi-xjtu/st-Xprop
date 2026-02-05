import argparse
import json
import os
import sys

def get_args():
    parser = argparse.ArgumentParser(description='st-Xprop', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:1",
                        help="Device to run the model on, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_spatial', type=int, default=6)
    parser.add_argument('--k_image', type=int, default=6)
    parser.add_argument('--rad_cutoff', type=int, default=150)
    parser.add_argument('--epoch_pre', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--show_training_details', type=bool, default=False)
    parser.add_argument('--emb', action='store_false', help='Enable embedding step')
    parser.add_argument('--refinement', action='store_true', help='Enable refinement step')
    parser.add_argument('--num_clusters', type=int, default=10)

    parser.add_argument('--name', type=str, default="CHD")
    parser.add_argument('--slice', type=str, default="D10")

    parser.add_argument('--lambda_1', type=float, default=10)
    parser.add_argument('--lambda_2', type=float, default=50)
    parser.add_argument('--lambda_3', type=float, default=10)
    parser.add_argument('--lambda_4', type=float, default=5)
    
    parser.add_argument('--vit_type', type=str, default='stMVC') 

    parser.add_argument('--r1', type=float, default=1)
    parser.add_argument('--r2', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_pr', type=float, default=1e-4)

    parser.add_argument('--start', type=float, default=0.01)
    parser.add_argument('--end', type=float, default=4)
    
    args = parser.parse_args()
    args_dict = vars(args)

    opt_to_dest = {opt: action.dest for opt, action in parser._option_string_actions.items()}
    user_specified = set()

    for token in sys.argv[1:]:
        if token.startswith('--') or token.startswith('-'):
            key = token.split('=')[0]  
            dest = opt_to_dest.get(key)
            if dest:
                user_specified.add(dest)

    config_path = 'config_stMVC.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        if args.name in config and args.slice in config[args.name]:
            dataset_params = config[args.name][args.slice]
            for key, value in dataset_params.items():

                if key not in user_specified:
                    args_dict[key] = value


    args = argparse.Namespace(**args_dict)
    return args
