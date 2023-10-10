import gizeh as gz
import moviepy.editor as mpy
import gymnasium as gym
import pickle 
import neat 
import os 

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)
    c= c.genotype.neatGenome



# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

print('Loaded genome:')
print(c)
env = gym.make('Pendulum-v1', g=9.81, render_mode = 'rgb_array')



class SimulationRenderer:
    def __init__(self, env, net):
        self.env = env
        self.observation, self.observation_init_info = env.reset()
        self.terminated = False

    def step(self, t):
        output = net.activate(self.observation)
        self.observation, reward, terminated, done, info = self.env.step(output)
        if terminated or self.terminated:
            self.terminated = terminated 
        ret = self.env.render()
        #print(t, terminated, ret is None)
        return ret


def make_movie(net, duration_seconds, output_filename):
    w, h = 300, 100
    scale = 300 / 6
    renderer = SimulationRenderer(env, net)

    clip = mpy.VideoClip(renderer.step, duration=duration_seconds)
    clip.write_videofile(output_filename, codec="mpeg4", fps=50)


if __name__ == "__main__":
    net = neat.nn.FeedForwardNetwork.create(c, config)
    make_movie(net, 10, 'neat-bipedal-walker.mp4')