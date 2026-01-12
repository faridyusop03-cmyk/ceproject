import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
from deap import algorithms
import random
import operator

# Load and preprocess dataset
@st.cache
def load_data():
    # Replace this with your file path
    df = pd.read_csv('/mnt/data/time_of_use_tariff.csv')  # Example CSV
    return df

df = load_data()

# Assume the dataset contains columns: 'time_slot', 'tariff', 'consumption'
# time_slot: e.g., '6:00 AM - 8:00 AM'
# tariff: cost per kWh during the time slot
# consumption: energy consumption during the time slot

# Clean the dataset (remove unnecessary spaces, etc.)
df.columns = df.columns.str.strip()

# Feature matrix and target
X = df[['tariff', 'consumption']].values  # Tariff and consumption
y = df['tariff'] * df['consumption']  # Target: Cost (RM) for each time period

# Define the problem (Minimize cost)
def eval_func(individual):
    func = toolbox.compile(expr=individual)
    # Apply the GP function to each time slot and compute total cost
    predictions = [func(tariff, consumption) for tariff, consumption in X]
    total_cost = np.sum(np.array(predictions))  # Total cost is the sum of costs
    return total_cost,

# Create a primitive set for genetic programming (modeling cost calculation)
pset = gp.PrimitiveSet("MAIN", arity=2)  # Two inputs: tariff, consumption
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.div, 2)
pset.addTerminal(1)  # Add constant terminals

# Define the fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize cost
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Set up the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", tools.initPrimitive, pset=pset, min_=2, max_=5)  # Create individuals with 2 to 5 levels
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)  # Uniform mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
toolbox.register("evaluate", eval_func)  # Evaluation function

# Run GP Algorithm to optimize the cost
def run_gp():
    population = toolbox.population(n=300)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolutionary algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, stats=stats, halloffame=None)

    return population, stats

# Streamlit User Interface
st.title("Genetic Programming for Cost Minimization (RM)")

# Button to start the algorithm
if st.button("Run Genetic Programming"):
    st.write("Running GP algorithm to minimize cost...")
    population, stats = run_gp()

    # Get the best individual and display results
    best_individual = tools.selBest(population, 1)[0]
    st.write("Best Individual: ", best_individual)
    st.write("Total Cost (RM) for Best Individual: ", best_individual.fitness.values)

    # Display convergence plot
    gen, avg, min, max = stats.compile(population)
    plt.plot(gen, avg, label="Average Fitness")
    plt.plot(gen, min, label="Min Fitness")
    plt.plot(gen, max, label="Max Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    st.pyplot(plt)
