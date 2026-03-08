import reproducible
import inspect

# Get functions in a dictionary
available_functions = {name: obj for name, obj in inspect.getmembers(reproducible) 
                       if inspect.isfunction(obj) and obj.__module__ == reproducible.__name__}

print("Available functions:")
for name in available_functions.keys():
    print(f"- {name}")

choice = input("Enter the name of the function to run: ")

if choice in available_functions:
    available_functions[choice]()
else:
    print(f"Function {choice} not found.")
