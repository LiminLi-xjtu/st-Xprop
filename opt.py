import argparse
import json
import os
import sys

def get_args(preset=None):
    parser = argparse.ArgumentParser(description='st-Xprop', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--name', type=str, default="BRCA")
    parser.add_argument('--slice', type=str, default="BRCA")
    parser.add_argument('--k_spatial', type=int, default=6)
    parser.add_argument('--k_image', type=int, default=6)
    parser.add_argument('--rad_cutoff', type=int, default=150)
    parser.add_argument('--adj_type', type=str, default='KNN')
    parser.add_argument('--image_type', type=str, default='stMVC')
    parser.add_argument('--image_emb_type', type=str, default='stMVC')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=128)
        
    parser.add_argument('--epoch_pre', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--show_training_details', action='store_true', default=False)
    parser.add_argument('--emb', action='store_true', default=True)
    parser.add_argument('--refinement', action='store_true', default=False)

    parser.add_argument('--lambda_1', type=float, default=2)
    parser.add_argument('--lambda_2', type=float, default=10)
    parser.add_argument('--lambda_3', type=float, default=2)
    parser.add_argument('--lambda_4', type=float, default=1)
    parser.add_argument('--r1', type=float, default=1)
    parser.add_argument('--r2', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_pr', type=float, default=1e-4)



    args = parser.parse_args()
    args_dict = vars(args)

    # Record command-line explicitly specified parameters
    opt_to_dest = {opt: action.dest for opt, action in parser._option_string_actions.items()}
    user_specified = set()
    for token in sys.argv[1:]:
        if token.startswith('--') or token.startswith('-'):
            key = token.split('=')[0]
            dest = opt_to_dest.get(key)
            if dest:
                user_specified.add(dest)

    # Load config (use preset values to determine lookup path, avoiding preset being bypassed by config logic)
    _img_type = preset.get('image_type', args.image_type) if preset else args.image_type
    config_path = 'config_stMVC.json' if _img_type == "stMVC" else 'config_ViT.json'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        _name = preset.get('name', args.name) if preset else args.name
        _slice = preset.get('slice', args.slice) if preset else args.slice
        if _name in config and _slice in config[_name]:
            dataset_params = config[_name][_slice]
            for key, value in dataset_params.items():
                if key not in user_specified:
                    args_dict[key] = value

    # Finally apply preset (highest priority, overrides command-line and config)
    if preset:
        for key, value in preset.items():
            args_dict[key] = value

    return argparse.Namespace(**args_dict)