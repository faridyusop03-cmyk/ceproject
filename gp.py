import pandas as pd
import numpy as np
import random
import streamlit as st

# Load dataset (gunakan dataset yang telah dimuat naik)
df = pd.read_csv('/mnt/data/project_benchmark_data_ce.csv')

# Set up Streamlit layout
st.title("Evolution Strategies for Energy Scheduling")

# Display the dataset
st.subheader("Device Information")
st.write(df)

# Hyperparameters (adjustable sliders in Streamlit)
st.sidebar.header("ES Hyperparameters")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
generations = st.sidebar.slider("Generations", 10, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)
learning_rate = st.sidebar.slider("Learning Rate", 0.0, 1.0, 0.5)

# Constraints and Weights
st.sidebar.header("Constraints & Weights")
max_peak_power = st.sidebar.slider("Max Peak Power (kW)", 1.0, 10.0, 5.0)
discomfort_weight = st.sidebar.slider("Discomfort Weight (Cost of 1 hr Delay)", 0.0, 2.0, 0.1)

# Baseline (without optimization)
baseline_cost = df['Power (kW)'].sum() * 0.1  # Example cost per kWh
baseline_discomfort = 0  # Example placeholder for discomfort calculation
baseline_peak_power = df['Power (kW)'].max()

st.write(f"Baseline Cost (No Optimization): RM {baseline_cost:.2f}")
st.write(f"Baseline Discomfort: {baseline_discomfort} hrs")
st.write(f"Baseline Peak Power: {baseline_peak_power:.2f} kW")

# Fitness function to evaluate cost and discomfort
def fitness(schedule):
    total_cost = 0
    total_discomfort = 0
    
    for _, row in df.iterrows():
        power = row['Power (kW)']
        duration = row['Duration']
        
        # Calculate cost based on time of use (assuming a simple cost model for demo)
        cost = power * 0.1 * duration  # Cost = power * rate * duration (example calculation)
        total_cost += cost
        
        # Discomfort is calculated based on delay from preferred time (simplified)
        delay = row['Delay']  # Using 'Delay' as a placeholder
        discomfort = delay * discomfort_weight
        total_discomfort += discomfort
        
    return total_cost, total_discomfort

# Initialize population: each individual is a set of start times for each device
def init_population(pop_size, num_devices):
    population = []
    for _ in range(pop_size):
        schedule = np.random.randint(0, 24, num_devices)  # Random start times between 0 and 23
        population.append(schedule)
    return population

# Mutation function to modify the schedule (start times)
def mutate(schedule, mutation_rate):
    new_schedule = schedule.copy()
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            new_schedule[i] = random.randint(0, 24)  # Randomly change the start time
    return new_schedule

# ES Execution Button
if st.button("Start Optimization"):
    # Initialize population (each individual is a schedule with random start times)
    num_devices = len(df)
    population = init_population(pop_size, num_devices)

    # Evolve the population over several generations
    for gen in range(generations):
        # Evaluate fitness for the current population
        fitness_scores = []
        for individual in population:
            cost, discomfort = fitness(individual)
            fitness_scores.append((cost, discomfort))
        
        # Select the best individuals based on the lowest cost and discomfort
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
        sorted_fitness_scores = sorted(fitness_scores)
        
        # Select top half of the population (elitism)
        population = sorted_population[:pop_size // 2]
        
        # Create new population using mutated individuals (elitism + mutation)
        new_population = population.copy()
        while len(new_population) < pop_size:
            parent = random.choice(population)
            child = mutate(parent, mutation_rate)
            new_population.append(child)
        
        # Evaluate the best solution found so far
        best_solution = new_population[0]
        best_cost, best_discomfort = fitness(best_solution)
        
        st.write(f"Generation {gen + 1} - Best Cost: RM {best_cost:.2f}, Best Discomfort: {best_discomfort:.2f} hrs")

    # Output the best solution found
    st.write("Optimized Scheduling Result:")
    st.write(f"Best Schedule: {best_solution}")
    st.write(f"Best Cost: RM {best_cost:.2f}")
    st.write(f"Best Discomfort: {best_discomfort:.2f} hrs")
