import streamlit as st
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
from deap import algorithms
import os

# Load the dataset
@st.cache
def load_data():
    # Ensure the file path is correct if using Streamlit's file uploader
    file_path = 'project_benchmark_data_ce.csv'  # Replace with the correct path if needed
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Check if data is loaded
if df is not None:
    # Print column names for debugging in Streamlit
    st.write("Columns in the dataset:", df.columns)

    # Clean up column names by stripping extra spaces (if any)
    df.columns = df.columns.str.strip()

    # Ensure the 'target' column exists
    if 'target' not in df.columns:
        st.error("The dataset does not contain a column named 'target'. Please check the column name.")
    else:
        # Assume 'target' is the dependent variable and the rest are features
        X = df.drop('target', axis=1).values
        y = df['target'].values

        # Define the problem (simple regression)
        def eval_func(individual):
            func = toolbox.compile(expr=individual)
            predictions = [func(*x) for x in X]  # Apply the GP function to each feature vector
            mse = np.mean((np.array(predictions) - y) ** 2)
            return mse,

        # Create a primitive set for genetic programming (basic math operations)
        pset = gp.PrimitiveSet("MAIN", arity=X.shape[1])  # arity is the number of features in X
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.div, 2)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)
        pset.addTerminal(1)  # Terminal node with a constant value

        # Define the fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize error (MSE)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", tools.initPrimitive, pset=pset, min_=2, max_=5)  # Create individuals with 2 to 5 levels
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("mate", gp.cxTwoPoint)  # Two-point crossover
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)  # Uniform mutation
        toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection
        toolbox.register("evaluate", eval_func)  # Evaluation function

        # Function to run GP algorithm
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
        st.title("Genetic Programming for Regression")

        # Button to start the algorithm
        if st.button("Run Genetic Programming"):
            st.write("Running GP algorithm...")
            population, stats = run_gp()

            # Get the best individual and display results
            best_individual = tools.selBest(population, 1)[0]
            st.write("Best Individual: ", best_individual)
            st.write("Fitness Value (MSE): ", best_individual.fitness.values)

            # Display convergence plot
            gen, avg, min, max = stats.compile(population)
            plt.plot(gen, avg, label="Average Fitness")
            plt.plot(gen, min, label="Min Fitness")
            plt.plot(gen, max, label="Max Fitness")
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            st.pyplot(plt)

