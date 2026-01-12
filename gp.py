import pandas as pd
import numpy as np
import random
import streamlit as st
from deap import base, creator, tools, gp
import operator

# Load your dataset here
df = pd.read_csv('/mnt/data/project_benchmark_data_ce.csv')

# Set up Streamlit layout
st.title("Genetic Programming for Energy Optimization")

# Quick Preset Selection (Conservative, Balanced, Aggressive)
preset = st.selectbox("Select Preset", ["Conservative", "Balanced", "Aggressive"])

# ACO Hyperparameters (adjustable sliders)
st.sidebar.header("GP Hyperparameters")
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
generations = st.sidebar.slider("Generations", 10, 100, 50)
crossover_prob = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.7)
mutation_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.2)

# Constraints and Weights
st.sidebar.header("Constraints & Weights")
max_peak_power = st.sidebar.slider("Max Peak Power (kW)", 1.0, 10.0, 5.0)
discomfort_weight = st.sidebar.slider("Discomfort Weight (Cost of 1 hr Delay)", 0.0, 2.0, 0.1)

# Display devices and their attributes
st.subheader("Device Information")
st.write(df)

# Baseline (without optimization)
baseline_cost = df['Power (kW)'].sum() * 0.1  # Example cost per kWh
baseline_discomfort = 0  # Example placeholder for discomfort calculation
baseline_peak_power = df['Power (kW)'].max()

st.write(f"Baseline Cost (No Optimization): RM {baseline_cost:.2f}")
st.write(f"Baseline Discomfort: {baseline_discomfort} hrs")
st.write(f"Baseline Peak Power: {baseline_peak_power:.2f} kW")

# GP Setup using DEAP library

# Define the problem and the fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize cost and discomfort
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the set of possible functions for the GP (program)
pset = gp.PrimitiveSet("MAIN", 1)  # 1 input: appliance info (power, time, etc.)
pset.addPrimitive(operator.add, 2)  # + operator
pset.addPrimitive(operator.sub, 2)  # - operator
pset.addPrimitive(operator.mul, 2)  # * operator
pset.addPrimitive(operator.truediv, 2)  # / operator
pset.addTerminal(1)  # Adding terminal values (constants)

# Fitness function to evaluate the programs
def evaluate(individual):
    # The individual represents a program (a function)
    # We'll execute the program to compute the cost and discomfort
    
    # Convert the individual (tree) to a function
    func = gp.compile(expr=individual, pset=pset)
    
    # Evaluate the function's performance on the appliances
    cost = 0
    discomfort = 0
    
    for _, row in df.iterrows():
        power = row['Power (kW)']
        duration = row['Duration']
        # Assume the function calculates cost (e.g., based on power usage)
        appliance_cost = func(power, duration)  # Using GP-generated program to calculate cost
        cost += appliance_cost

        # Discomfort (could be a function of delay)
        delay = row['Delay']  # Placeholder for delay info
        discomfort += func(power, delay)  # Using GP-generated program for discomfort
    
    return cost, discomfort

# Create the population
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
toolbox.register("evaluate", evaluate)

# GP Execution Button
if st.button("Start Optimization"):
    population = toolbox.population(n=pop_size)

    # Run the GP algorithm
    for gen in range(generations):
        # Evaluate the fitness of the population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate the fitness
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new one
        population[:] = offspring

    # Get the best solution
    best_individual = tools.selBest(population, 1)[0]
    st.write(f"Optimized Total Cost: RM {best_individual.fitness.values[0]:.2f}")
    st.write(f"Optimized Discomfort: {best_individual.fitness.values[1]:.2f} hrs")

    # Peak Power Constraints
    st.write(f"Peak Power: {max_peak_power} kW")
