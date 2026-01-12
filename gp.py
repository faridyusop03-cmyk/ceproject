import streamlit as st
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
from deap import algorithms
import os

# Load and preprocess dataset
@st.cache
def load_data():
    # Replace this with your file path
    df = pd.read_csv('/mnt/data/time_of_use_tariff.csv')  # Example CSV file path
    return df

df = load_data()

# Check if data is loaded
if df is not None:
    # Print column names for debugging in Streamlit
    st.write("Columns in the dataset:", df.columns)

    # Clean up column names by stripping extra spaces (if any)
    df.columns = df.columns.str.strip()

    # Inspect columns to identify the correct target column
    st.write("Columns in the dataset:", df.columns)

    # Ensure we are using the correct target column
    # Assuming 'kw' is the target column (you can replace it with 'cost' or other relevant column names)
    if 'kw' in df.columns:
        target_column = 'kw'
    else:
        st.error("Target column 'kw' not found. Please check the dataset.")
        target_column = None

    if target_column is not None:
        # Feature matrix (tariff, consumption) and target (cost or kw)
        X = df[['tariff', 'consumption']].values  # Tariff and consumption
        y = df[target_column].values  # Target: kw or cost

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
