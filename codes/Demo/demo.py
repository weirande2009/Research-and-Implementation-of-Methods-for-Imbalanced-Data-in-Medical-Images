import argparse
import datetime
import os

from evaluate import *
from train import train_multi
from train_binary import train_binary
from unet import *
from model.deeplabv3 import *
from predict import *

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main(args):
    root_path = "results"
    save_path = os.path.join(root_path, start_time)
    print_cmd_info(args)
    make_folder(args, root_path)
    if args.multi_class:
        print("Start training multi-class model")
        train_multi(args, os.path.join(save_path, "multi_class"))
        print("Finish training multi-class model")
    if args.binary_class:
        print("Start training binary-class model")
        train_binary(args, os.path.join(save_path, "binary_class"))
        print("Finish training binary-class model")
    if args.combine:
        predict_from_both(args, 
                          os.path.join(save_path, "multi_class", "predictions"), 
                          os.path.join(save_path, "binary_class", "predictions"),
                          os.path.join(save_path, "combine"))
        print("Finish training binary-class model")

def make_folder(args, root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path = os.path.join(root_path, start_time)
    os.mkdir(save_path)
    if args.predict:
        if args.multi_class:
            os.mkdir(os.path.join(save_path, "multi_class"))
            os.mkdir(os.path.join(save_path, "multi_class", "predictions"))
            os.mkdir(os.path.join(save_path, "multi_class", "predictions", "color"))
            os.mkdir(os.path.join(save_path, "multi_class", "predictions", "raw"))
        if args.binary_class:
            os.mkdir(os.path.join(save_path, "binary_class"))
            os.mkdir(os.path.join(save_path, "binary_class", "predictions"))
        if args.combine:
            os.mkdir(os.path.join(save_path, "combine"))
            os.mkdir(os.path.join(save_path, "combine", "predictions"))

def print_cmd_info(args):
    print("Dataset Info:")
    if args.dataset != "":
        print("\tPath: "+args.dataset)
        print(f"\tValidation Ratio: {args.val}")
    else:
        print("\tTrain Path: "+args.train_set)
        print("\tTest Path: "+args.test_set)

    print("Training Info:")
    print(f"\tDevice: {args.device}")
    print(f"\tEpoch: {args.epochs}")
    print(f"\tBatch Size: {args.batch_size}")
    print(f"\tLearning Rate: {args.lr}")
    if args.multi_class:
        print("\tMulti: ")
        print("\t\tLoss: ", end="")
        if args.m_focal_loss:
            print(f"Focal({args.focal_weight})", end="")
        if args.m_dice_loss:
            print("Dice  ", end="")
        if args.m_tversky_loss:
            print("Tversky  ", end="")
        if args.m_ob_loss:
            print(f"OnlineBootstrap({args.hard_num})  ", end="")
        print()
        print("\t\tNet: "+args.m_net)
    if args.binary_class:
        print("\tBinary: ")
        print("\t\tLoss: ", end="")
        if args.b_focal_loss:
            print(f"Focal({args.focal_weight})", end="")
        if args.b_dice_loss:
            print("Dice  ", end="")
        if args.b_tversky_loss:
            print("Tversky  ", end="")
        if args.b_ob_loss:
            print(f"OnlineBootstrap({args.hard_num})  ", end="")
        print()
        print("\t\tNet: "+args.b_net)
    print("\tOptimizer: "+args.optimizer)

    if args.predict:
        print("Predict: Yes")
    else:
        print("Predict: No")

    if args.save:
        print("Save: Yes")
        print("\tSave Path: results/"+start_time)
    else:
        print("Save: No")  

def get_args():
    parser = argparse.ArgumentParser(description='A two-stage method of reusing background for semantic segmentation')

    parser.add_argument('--multi-class', '-mc', type=bool, default=True, help='Train multi-class segmentation')
    parser.add_argument('--binary-class', '-bc', type=bool, default=True, help='Train binary-class segmentation')
    parser.add_argument('--combine', '-cb', type=bool, default=True, help='Apply two-stage method of reusing background')

    parser.add_argument('--predict', '-p', type=bool, default=True, help='Predict or not')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--optimizer', '-op', type=str, default="Adam", help='RMSprop, Adam')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--device', '-d', type=str, default="cuda:0", help='Use which GPU')

    parser.add_argument('--m_net', '-mn', type=str, default="unet", help='Net Kind: unet, unet++, unet+++, attunet, deeplabv3. Multi-class model')
    parser.add_argument('--b_net', '-bn', type=str, default="unet", help='Net Kind: unet, unet++, unet+++, attunet, deeplabv3. Binary-class model')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    
    parser.add_argument('--save',  default=True, help='Save the checkpoint')
    
    parser.add_argument('--dataset', '-ds', type=str, default="MoNuSAC", help='Dataset path which include two folders: images, masks')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1, help='Percent of the data that is used as validation (0-1)')
    
    parser.add_argument('--train-set', type=str, default="", help='Training dataset path which include two folders: images, masks')
    parser.add_argument('--test-set', type=str, default="", help='Testing Dataset path which include two folders: images, masks')

    parser.add_argument('--m_focal_loss', type=bool, default=False, help='Use Focal Loss in multi-class')
    parser.add_argument('--m_dice_loss', type=bool, default=False, help='Use Dice Loss in multi-class')
    parser.add_argument('--m_tversky_loss', type=bool, default=False, help='Use Tversky Loss in multi-class')
    parser.add_argument('--m_ob_loss', type=bool, default=False, help='Use OnlineBootstrap Loss in multi-class')

    parser.add_argument('--b_focal_loss', type=bool, default=False, help='Use Focal Loss in binary-class')
    parser.add_argument('--b_dice_loss', type=bool, default=False, help='Use Dice Loss in binary-class')
    parser.add_argument('--b_tversky_loss', type=bool, default=False, help='Use Tversky Loss in binary-class')
    parser.add_argument('--b_ob_loss', type=bool, default=False, help='Use Online Bootstrapping Loss in binary-class')
    
    parser.add_argument('--focal_weight', '-fw', type=int, default=1, help='Focal weights kind')
    parser.add_argument('--hard_num', '-hn', type=int, default=512, help='Hard number in OB Loss')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)


                
            




