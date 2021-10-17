import os
import argparse
import imageio

parser = argparse.ArgumentParser(description='Genetic optimization of route in graph. ')

parser.add_argument('--source_dir', type=str, default='experiments/frames', help='where the frames are stored')
parser.add_argument('--filename', type=str, default='out.gif', help='where to save frames')
parser.add_argument('--fps', type=int, default=2, help='frames per second')

args = parser.parse_args()


def main():
    frames = [imageio.imread(os.path.join(args.source_dir, filename)) for filename in os.listdir(args.source_dir)]
    imageio.mimsave(args.filename, frames, fps=args.fps)


if __name__ == '__main__':
    main()
