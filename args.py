import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',metavar='string',type=str,required = True,help='Path to extracted kitti data set folder')
parser.add_argument('--output',metavar='String',type=str,required = True,help='The location where the keras model will be saved as .h5 file')
parser.add_argument('--batch',metavar='Integer',type=int,default=4,help='Batch size of dataset for training and evaluation(Note: reduce the batch size if your PC resources are limited)')
parser.add_argument('--epoch',metavar='Integer',type=int,default=15,help='Epoch for training and evaluation')

args = parser.parse_args()
print(args.dataset)
print(args.output)
print(args.batch)
print(args.epoch)

label_file=open(args.dataset+"/label_colors.txt",'r')

content= label_file.readlines()
print(len(content))
