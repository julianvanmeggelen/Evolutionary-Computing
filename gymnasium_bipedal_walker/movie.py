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
env = gym.make("BipedalWalker-v3", hardcore=False, max_episode_steps=1000, render_mode = 'rgb_array')


def make_movie(net, duration_seconds, output_filename):
    w, h = 300, 100
    scale = 300 / 6

    
    observation, observation_init_info = env.reset()

    def make_frame(t, observation):
        output = net.activate(observation)
        observation, reward, terminated, done, info = env.step(output)
        return env.render()

    clip = mpy.VideoClip(lambda t: make_frame(t,observation), duration=duration_seconds)
    clip.write_videofile(output_filename, codec="mpeg4", fps=50)


if __name__ == "__main__":
    net = neat.nn.FeedForwardNetwork.create(c, config)
    make_movie(net, 20, 'neat-bipedal-walker.mp4')