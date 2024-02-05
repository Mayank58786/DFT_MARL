from src.parsing import Parse
file_path='model.xml'
system_obj=Parse.from_file(file_path)
# for obj in system_obj.events:
#     print(obj.name,obj.state)
system_obj.initialize_tree()
for obj in system_obj.events:
    print(obj.name,obj.state)
for obj in system_obj.events:
    if obj.name=="K":
        system_obj.apply_action(obj,0)
for obj in system_obj.events:
    print(obj.name,obj.state)
