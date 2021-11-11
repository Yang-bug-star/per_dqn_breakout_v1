import base64
from collections import deque
import os
import pathlib
import shutil

from IPython import display as ipydisplay
import torch

from collections import deque
from utils_env import MyEnv
from utils_drl import Agent

model = f"./models/model_perv1"

device = torch.device("cpu")
env = MyEnv(device)

agent = Agent(env.get_action_dim(), device, 0.99, 0.4, 0.001, 0, 0, 0, 1, model)

obs_queue = deque(maxlen=5)


TEST_TIMES = 100 
total = 0.0

for i in range(0, TEST_TIMES + 1):
    reward, frames= env.evaluate(obs_queue, agent, render=True)
    total += reward
            
with open("test.txt", "a") as fp:
         fp.write(f"Avg: {total / TEST_TIMES:.2f}\n")

target_dir = f"eval"
    
path_to_mp4 = os.path.join(target_dir, "movie.mp4")
if not os.path.exists(path_to_mp4):
    shutil.move(target_dir, "tmp_eval_frames")
    # Generate an mp4 video from the frames
    os.system('ffmpeg -i "./tmp_eval_frames/%06d.png" -pix_fmt yuv420p -y ./tmp_eval_movie.mp4 > /dev/null 2>&1')
    os.system('rm -r tmp_eval_frames')
    os.mkdir(target_dir)
    shutil.move("tmp_eval_movie.mp4", path_to_mp4)
    
HTML_TEMPLATE = """<video alt="{alt}" autoplay loop controls style="height: 400px;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4" />
</video>"""

def show_video(path_to_mp4: str) -> None:
    """show_video creates an HTML element to display the given mp4 video in IPython."""
    mp4 = pathlib.Path(path_to_mp4)
    video_b64 = base64.b64encode(mp4.read_bytes())
    html = HTML_TEMPLATE.format(alt=mp4, data=video_b64.decode('ascii'))
    ipydisplay.display(ipydisplay.HTML(data=html))

show_video(path_to_mp4)