import os
import imageio

NAME = 'movie'
SOURCE_DIR = 'plots'
SAVE_DIR = 'out'
FPS = 5

images = os.listdir('plots')
frames = []
for filename in images:
    frames.append(imageio.imread(os.path.join(SOURCE_DIR, filename)))
imageio.mimsave(os.path.join(SAVE_DIR, NAME + '.mp4'), frames, fps=FPS)