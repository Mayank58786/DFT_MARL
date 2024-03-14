import sys
from src.parsing import Parse
from src.game import Game
from src.Custom_Environment.custom_envinronment.env.custom_env import CustomEnvironment

file_path = 'model.xml'

class env_creator():

    def create_env():
        system_obj = Parse.from_file(file_path)
        system_obj.initialize_system()
        game = Game(system_obj, 100)
        red_agent=game.create_player("red_agent",20)
        blue_agent=game.create_player("blue_agent",1000)
        env = CustomEnvironment(system_obj,game)
        return env
