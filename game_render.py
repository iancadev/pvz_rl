from agents import ReinforceAgentV2, PolicyNetV2, PlayerV2
from agents import KeyboardAgent
from agents import PlayerQ, QNetwork, QNetwork_DQN
from agents import ACAgent3, TrainerAC3
from pvz import config
import gym
import torch
import pygame
import numpy as np
np.bool8 = np.bool

class PVZ():
    def __init__(self,render=True, max_frames = 1000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        
    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * (config.LANE_LENGTH + 2)

    def num_actions(self):
        return self.env.action_space.n

    def play(self,agent):
        """ Play one episode and collect observations and rewards """
        observation = self.env.reset()
        t = 0

        for t in range(self.max_frames):
            if(self.render):
                self.env.render()

            action = agent.decide_action(observation)
            observation, reward, done, info = self.env.step(action)

            if done:
                break

    def get_render_info(self):
        return self.env.get_render_info()

def render(render_info):
    pygame.init()
    pygame.font.init() # you have to call this at the start, 
                    # if you want to use this module.
    myfont = pygame.font.SysFont('calibri', 30)


    screen = pygame.display.set_mode((1450, 650))
    zombie_sprite = {"zombie": pygame.image.load("assets/zombie_scaled.png").convert_alpha(),
     "zombie_cone": pygame.image.load("assets/zombie_cone_scaled.png").convert_alpha(),
     "zombie_bucket": pygame.image.load("assets/zombie_bucket_scaled.png").convert_alpha(),
     "zombie_flag" :pygame.image.load("assets/zombie_flag_scaled.png").convert_alpha(),  }
    plant_sprite = {"peashooter": pygame.image.load("assets/peashooter_scaled.png").convert_alpha(),
                    "sunflower": pygame.image.load("assets/sunflower_scaled.png").convert_alpha(),
                    "wallnut": pygame.image.load("assets/wallnut_scaled.png").convert_alpha(),
                    "potatomine":pygame.image.load("assets/potatomine_scaled.png").convert_alpha()}
    projectile_sprite = {"pea": pygame.image.load("assets/pea.png").convert_alpha()}
    clock = pygame.time.Clock()
    cell_size = 75
    offset_border = 100
    offset_y = int(0.8 * cell_size)
    cumulated_score=0

    while render_info:
        clock.tick(config.FPS*5)
        screen.fill((130, 200, 100))
        frame_info = render_info.pop(0)
        
        # The grid
        for i in range(config.LANE_LENGTH+1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border + i * cell_size, offset_border), 
                (offset_border + i * cell_size, offset_border + cell_size * (config.N_LANES)), 1)
        for j in range(config.N_LANES+1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border, offset_border + j * cell_size), 
                (offset_border + cell_size * (config.LANE_LENGTH), offset_border + j * cell_size), 1)
        
        
        # The objects
        for lane in range(config.N_LANES):
            for zombie_name, pos, offset in frame_info["zombies"][lane]:
                zombie_name = zombie_name.lower()
                screen.blit(zombie_sprite[zombie_name], (offset_border + cell_size * (pos + offset) - zombie_sprite[zombie_name].get_width(),
                    offset_border + lane * cell_size + offset_y - zombie_sprite[zombie_name].get_height()))
            for plant_name, pos in frame_info["plants"][lane]:
                plant_name = plant_name.lower()
                screen.blit(plant_sprite[plant_name], (offset_border + cell_size * pos, 
                    offset_border + lane * cell_size + offset_y - plant_sprite[plant_name].get_height()))
            for projectile_name, pos, offset in frame_info["projectiles"][lane]:
                projectile_name = projectile_name.lower()
                screen.blit(projectile_sprite[projectile_name], (offset_border + cell_size * (pos+offset) - projectile_sprite[projectile_name].get_width(), 
                    offset_border + lane * cell_size))
        
        #Text
        sun_text = myfont.render('Sun: '+ str(frame_info["sun"]), False, (0, 0, 0))
        screen.blit(sun_text, (50, 600))
        cumulated_score += frame_info["score"]
        score_text = myfont.render('Score: '+ str(cumulated_score), False, (0, 0, 0))
        screen.blit(score_text, (200, 600))
        cooldowns_text = myfont.render('Cooldowns: '+ str(frame_info["cooldowns"]), False, (0, 0, 0))
        screen.blit(cooldowns_text, (350, 600))
        time = myfont.render('Time: '+ str(frame_info["time"]), False, (0, 0, 0))
        screen.blit(time, (900, 100))
        
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render_info = []
        
        
        
        pygame.display.flip()

    pygame.quit()


agent_type = input("Agent type [Reinforce|DDQN|AC]: ")
nn_name = input("nn_name: ")


if __name__ == "__main__":

    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames = 500 * config.FPS)
        agent = ReinforceAgentV2(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        nn_name = nn_name or "agents/agent_zoo/dfp5"
        agent.load(nn_name)
        
    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        nn_name = nn_name or "agents/agent_zoo/dfq5_epsexp"
        agent = torch.load(nn_name, weights_only=False)
        
    if agent_type == "AC":
        env = TrainerAC3(render=False, max_frames = 500*config.FPS)
        agent = ACAgent3(
                input_size=env.num_observations(),
                possible_actions=env.get_actions()
        )
        nn_name = nn_name or "agents/agent_zoo/ac_v1"
        agent.load(f"{nn_name}(policy)", f"{nn_name}(valuen)")
    
    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames = 500*config.FPS)
        agent = KeyboardAgent()
    env.play(agent)
    render_info = env.get_render_info()
    render(render_info)
