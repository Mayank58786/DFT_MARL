import sys
from src.parsing import Parse
from src.game import Game
from src.Custom_Environment.custom_envinronment.env.custom_env import CustomEnvironment

file_path = 'model.xml'
system_obj = Parse.from_file(file_path)

system_obj.initialize_system()
game = Game(system_obj, 1000)
