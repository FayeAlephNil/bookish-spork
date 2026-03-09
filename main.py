import reproducible
import inspect

# Get functions in a dictionary
available_functions = {name: obj for name, obj in inspect.getmembers(reproducible) 
                       if inspect.isfunction(obj) and obj.__module__ == reproducible.__name__}

print("Available functions:")
for name in available_functions.keys():
    print(f"- {name}")
print('- all')

choice = input("Enter the name of the function to run: ")
show = input("Show data in viewer y/N? ")
seeded = input("Use pregenerated seeds Y/n? ")
print("\n")
if show == 'y' or show == 'Y':
    show_it = True
else:
    show_it = False

if seeded == 'n' or seeded == 'N':
    seeded = False
else:
    seeded = True


if choice in available_functions:
    available_functions[choice](show_it,seeded)
elif choice == 'all':
    for x in available_functions.values():
        x(show_it,seeded)
else:
    print(f"Function {choice} not found.")
