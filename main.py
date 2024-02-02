from src.parsing import Parse
file_path='model.xml'
system_obj=Parse.from_file(file_path)
for obj in system_obj.events:
    print(obj.name,obj.state)
system_obj.update_states()
for obj in system_obj.events:
    print(obj.name,obj.state)