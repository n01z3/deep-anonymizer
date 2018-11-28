import PGN
import argparse 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_list', type=str, default='', required=True, help='Path to a file with absolute paths to images')
parser.add_argument('--img_size', type=int, default=0,  help='')
parser.add_argument('--tta', type=str, default='0.5,0.75,1.25,1.5,1.75',  help='')

args = parser.parse_args()


model_pgn = PGN.PGN()

model_pgn.load_data(
                   path_data='' 
                    , path_list_image=args.img_list
                    , path_list_image_id=None
                    , size_image=None if args.img_size <= 0 else (args.img_size, args.img_size)
)

tta = [float(x) for x in args.tta.split(',')] if args.tta != '' else []

model_pgn.build_model(n_class=20, path_model_trained='./checkpoint/CIHP_pgn', tta = tta)

model_pgn.predict(flg_debug=False)