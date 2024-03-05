import sys
from src.parsing import Parse
from src.game import Game
from src.Custom_Environment.custom_envinronment.env.custom_env import CustomEnvironment

file_path = 'model.xml'
system_obj = Parse.from_file(file_path)

system_obj.initialize_system()
game = Game(system_obj, 1000)
red_agent=game.create_player("red_agent",50)
blue_agent=game.create_player("blue_agent",100)
env = CustomEnvironment(system_obj,game,{"red_agent":30,"blue_agent":20})
print(env.resources)
# for _ in range(0,10):
#     print(env.game.choose_action(red_agent))
for _ in range(0,30):
    print(env.step())
    env.agent_selection = env._agent_selector.next()