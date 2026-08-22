import random

# ============================================================
# SAFE ELEVATOR DISPATCH SIMULATOR
# ============================================================

print("=" * 60)
print("        SAFE ELEVATOR DISPATCH SIMULATOR")
print("=" * 60)


# ============================================================
# BUILDING SETUP
# ============================================================

while True:
    try:
        floors = int(input("\nEnter number of floors: "))

        if floors >= 2:
            break

        print("Enter at least 2 floors.")

    except ValueError:
        print("Please enter a number.")


# ============================================================
# ELEVATOR SETUP
# ============================================================

while True:
    try:
        number_of_elevators = int(
            input("Enter number of elevators: ")
        )

        if number_of_elevators >= 1:
            break

        print("Enter at least 1 elevator.")

    except ValueError:
        print("Please enter a number.")


elevators = {}

for i in range(1, number_of_elevators + 1):

    while True:
        try:
            capacity = float(
                input(
                    f"Enter capacity of Elevator {i} (kg): "
                )
            )

            if capacity > 0:
                break

            print("Capacity must be greater than 0.")

        except ValueError:
            print("Please enter a valid number.")

    elevators[f"Elevator {i}"] = {
        "floor": 1,
        "capacity": capacity,
        "load": 0,
        "passengers": [],
        "distance": 0
    }


# ============================================================
# PASSENGER LIST
# ============================================================

people = []


# ============================================================
# STATISTICS
# ============================================================

total_served = 0
total_waiting_time = 0
total_distance = 0
overload_attempts = 0


# ============================================================
# GENERATE RANDOM PEOPLE
# ============================================================

def generate_people():

    global people

    while True:
        try:
            number = int(
                input(
                    "\nEnter number of people to generate: "
                )
            )

            if number > 0:
                break

            print("Enter a number greater than 0.")

        except ValueError:
            print("Please enter a valid number.")

    people = []

    for i in range(1, number + 1):

        start_floor = random.randint(1, floors)

        destination_floor = random.randint(1, floors)

        while destination_floor == start_floor:
            destination_floor = random.randint(1, floors)

        weight = random.randint(40, 100)

        person = {
            "id": i,
            "start": start_floor,
            "destination": destination_floor,
            "weight": weight,
            "status": "Waiting",
            "waiting_time": 0
        }

        people.append(person)

    print(
        f"\n✓ {number} people generated successfully."
    )


# ============================================================
# PEAK DEMAND SIMULATION
# ============================================================

def peak_demand():

    print("\n" + "=" * 60)
    print("PEAK DEMAND SIMULATION")
    print("=" * 60)

    print("\nDemand Levels:")
    print("1. Normal Demand")
    print("2. Medium Demand")
    print("3. Peak Demand")

    while True:

        choice = input(
            "\nSelect demand level (1-3): "
        ).strip()

        if choice in ["1", "2", "3"]:
            break

        print("Please enter 1, 2 or 3.")

    if choice == "1":

        number = random.randint(3, 5)
        demand = "Normal"

    elif choice == "2":

        number = random.randint(6, 10)
        demand = "Medium"

    else:

        number = random.randint(11, 20)
        demand = "Peak"

    # Clear previous passengers
    people.clear()

    for i in range(1, number + 1):

        start_floor = random.randint(1, floors)

        destination_floor = random.randint(1, floors)

        while destination_floor == start_floor:
            destination_floor = random.randint(1, floors)

        weight = random.randint(40, 100)

        person = {
            "id": i,
            "start": start_floor,
            "destination": destination_floor,
            "weight": weight,
            "status": "Waiting",
            "waiting_time": 0
        }

        people.append(person)

    print("\n✓ Demand level:", demand)
    print("✓ People generated:", number)

    print("\nPeak demand requests:")
    print("-" * 60)

    for person in people:

        print(
            f"P{person['id']} | "
            f"Floor {person['start']} -> "
            f"Floor {person['destination']} | "
            f"{person['weight']} kg"
        )


# ============================================================
# SHOW BUILDING
# ============================================================

def show_building():

    print("\n" + "=" * 60)
    print("BUILDING")
    print("=" * 60)

    for floor in range(floors, 0, -1):

        waiting = []

        for person in people:

            if (
                person["start"] == floor
                and person["status"] == "Waiting"
            ):

                waiting.append(
                    f"P{person['id']}"
                )

        if waiting:

            print(
                f"Floor {floor}: "
                + ", ".join(waiting)
            )

        else:

            print(
                f"Floor {floor}: Empty"
            )


# ============================================================
# SHOW ELEVATORS
# ============================================================

def show_elevators():

    print("\n" + "=" * 60)
    print("ELEVATOR STATUS")
    print("=" * 60)

    for name, elevator in elevators.items():

        print(
            f"{name} | "
            f"Floor: {elevator['floor']} | "
            f"Load: {elevator['load']}/"
            f"{elevator['capacity']} kg | "
            f"Passengers: {elevator['passengers']}"
        )


# ============================================================
# SHOW PASSENGER QUEUE
# ============================================================

def show_people():

    print("\n" + "=" * 60)
    print("PASSENGER QUEUE")
    print("=" * 60)

    if not people:

        print("No people generated yet.")
        return

    for person in people:

        print(
            f"P{person['id']} | "
            f"Floor {person['start']} -> "
            f"Floor {person['destination']} | "
            f"Weight: {person['weight']} kg | "
            f"Status: {person['status']} | "
            f"Waiting: {person['waiting_time']} sec"
        )


# ============================================================
# SELECT SAFE ELEVATOR
# ============================================================

def select_elevator(person):

    global overload_attempts

    selected = None
    shortest_distance = float("inf")

    for name, elevator in elevators.items():

        distance = abs(
            elevator["floor"] - person["start"]
        )

        new_load = (
            elevator["load"] + person["weight"]
        )

        if new_load <= elevator["capacity"]:

            if distance < shortest_distance:

                shortest_distance = distance
                selected = name

        else:

            overload_attempts += 1

    return selected


# ============================================================
# MOVE ELEVATOR
# ============================================================

def move_elevator(name, target):

    global total_distance

    elevator = elevators[name]

    current = elevator["floor"]

    distance = abs(current - target)

    if distance == 0:

        print(
            f"{name} is already at Floor {target}."
        )

        return

    print(
        f"\n{name}: Floor {current} -> Floor {target}"
    )

    while current != target:

        if current < target:

            current += 1

        else:

            current -= 1

        print(
            f"   {name} reached Floor {current}"
        )

    elevator["floor"] = target

    elevator["distance"] += distance

    total_distance += distance


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation():

    global total_served
    global total_waiting_time
    global total_distance
    global overload_attempts

    if not people:

        print(
            "\n❌ No people available."
        )

        print(
            "First select option 3 or option 7."
        )

        return

    # Reset statistics
    total_served = 0
    total_waiting_time = 0
    total_distance = 0
    overload_attempts = 0

    for elevator in elevators.values():

        elevator["load"] = 0
        elevator["passengers"] = []
        elevator["distance"] = 0

    print("\n" + "=" * 60)
    print("SIMULATION STARTED")
    print("=" * 60)

    # Process passengers
    for person in people:

        if person["status"] != "Waiting":
            continue

        print("\n" + "-" * 60)

        print(
            f"P{person['id']} is waiting at "
            f"Floor {person['start']}"
        )

        print(
            f"Destination: Floor {person['destination']}"
        )

        print(
            f"Weight: {person['weight']} kg"
        )

        # Select safe elevator
        selected = select_elevator(person)

        if selected is None:

            print(
                "❌ No safe elevator available."
            )

            continue

        elevator = elevators[selected]

        print(
            f"✓ Selected: {selected}"
        )

        # Calculate waiting time
        waiting_time = abs(
            elevator["floor"] - person["start"]
        )

        person["waiting_time"] = waiting_time

        total_waiting_time += waiting_time

        print(
            f"Waiting time: {waiting_time} sec"
        )

        # Move to passenger
        move_elevator(
            selected,
            person["start"]
        )

        # Passenger enters
        elevator["load"] += person["weight"]

        elevator["passengers"].append(
            person["id"]
        )

        person["status"] = "Inside Elevator"

        print(
            f"✓ P{person['id']} entered {selected}"
        )

        print(
            f"Current load: "
            f"{elevator['load']} / "
            f"{elevator['capacity']} kg"
        )

        # Move to destination
        move_elevator(
            selected,
            person["destination"]
        )

        # Passenger exits
        elevator["load"] -= person["weight"]

        elevator["passengers"].remove(
            person["id"]
        )

        person["status"] = "Completed"

        total_served += 1

        print(
            f"✓ P{person['id']} reached "
            f"Floor {person['destination']}"
        )

        print(
            f"✓ P{person['id']} exited {selected}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)

    print(
        f"People generated : {len(people)}"
    )

    print(
        f"People served    : {total_served}"
    )

    if total_served > 0:

        average_waiting = (
            total_waiting_time / total_served
        )

    else:

        average_waiting = 0

    print(
        f"Average waiting time : "
        f"{average_waiting:.2f} sec"
    )

    print(
        f"Total elevator distance : "
        f"{total_distance} floors"
    )

    print(
        f"Overload attempts : "
        f"{overload_attempts}"
    )


# ============================================================
# STATISTICS
# ============================================================

def show_statistics():

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(
        f"Total people : {len(people)}"
    )

    print(
        f"People served : {total_served}"
    )

    if total_served > 0:

        average = (
            total_waiting_time / total_served
        )

    else:

        average = 0

    print(
        f"Average waiting time : "
        f"{average:.2f} sec"
    )

    print(
        f"Total distance : "
        f"{total_distance} floors"
    )

    print(
        f"Overload attempts : "
        f"{overload_attempts}"
    )


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n")
    print("=" * 60)
    print("MAIN MENU")
    print("=" * 60)

    print("1. Show Building")
    print("2. Show Elevators")
    print("3. Generate People")
    print("4. Show Passenger Queue")
    print("5. Run Elevator Simulation")
    print("6. Show Statistics")
    print("7. Peak Demand Simulation")
    print("8. Exit")

    print("=" * 60)

    choice = input(
        "Enter your choice (1-8): "
    ).strip()

    if choice == "1":

        show_building()

    elif choice == "2":

        show_elevators()

    elif choice == "3":

        generate_people()

    elif choice == "4":

        show_people()

    elif choice == "5":

        run_simulation()

    elif choice == "6":

        show_statistics()

    elif choice == "7":

        peak_demand()

    elif choice == "8":

        print(
            "\nSafe Elevator Simulator closed."
        )

        break

    else:

        print(
            "\n❌ Invalid choice."
        )

        print(
            "Please enter 1, 2, 3, 4, 5, 6, 7 or 8."
        )