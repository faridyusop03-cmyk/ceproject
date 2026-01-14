import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="HEMS Optimization – Firefly Algorithm",
    layout="wide"
)

st.title("HEMS Optimization using Firefly Algorithm (FFA)")
st.markdown("""
**Algorithm:** Firefly Algorithm (FFA)  
**Objective:** Minimize electricity cost and user discomfort under power constraints  
""")

# 2. DATA LOADING

@st.cache_data
def load_dataset():
    filename = "project_benchmark_data_ce.csv"
    if not os.path.exists(filename):
        return None

    df = pd.read_csv(filename)
    df['Is_Shiftable'] = df['Is_Shiftable'].astype(str).str.upper() == 'TRUE'
    return df

dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found. Please upload `project_benchmark_data_ce.csv`.")
    st.stop()

# 3. TARIFF CONFIGURATION (MALAYSIA TOU)

RATE_PEAK = 0.4592
RATE_OFF_PEAK = 0.4183
PEAK_START = 8
PEAK_END = 22

def electricity_rate(hour):
    return RATE_PEAK if PEAK_START <= hour < PEAK_END else RATE_OFF_PEAK

# 4. FIREFLY ALGORITHM CLASS
class FireflyHEMS:
    def __init__(self, shiftable_df, base_profile, max_power,
                 population_size, generations, alpha, beta0, gamma, seed=None):

        # Set the random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.shiftable_df = shiftable_df
        self.base_profile = base_profile
        self.max_power = max_power
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.bounds = [
            24 - int(row['Duration_Hours'])
            for _, row in shiftable_df.iterrows()
        ]

    def initialize_population(self):
        return [
            [random.randint(0, b) for b in self.bounds]
            for _ in range(self.population_size)
        ]

    def evaluate(self, schedule):
        profile = self.base_profile.copy()

        for i, start in enumerate(schedule):
            duration = int(self.shiftable_df.iloc[i]['Duration_Hours'])
            power = self.shiftable_df.iloc[i]['Avg_Power_kW']
            for h in range(start, start + duration):
                if h < 24:
                    profile[h] += power

        peak_power = np.max(profile)

        penalty = 0
        if peak_power > self.max_power:
            penalty = 1000 + (peak_power - self.max_power) * 100

        cost = sum(profile[h] * electricity_rate(h) for h in range(24))

        discomfort = sum(
            abs(schedule[i] - self.shiftable_df.iloc[i]['Preferred_Start_Hour'])
            for i in range(len(schedule))
        )

        fitness = cost + 0.1 * discomfort + penalty
        return fitness, cost, discomfort, peak_power

    @staticmethod
    def distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def run(self, progress, status):
        population = self.initialize_population()
        evaluated = [list(self.evaluate(ind)) + [ind] for ind in population]
        evaluated.sort(key=lambda x: x[0])

        best_solution = evaluated[0]
        convergence = []

        for gen in range(self.generations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if evaluated[j][0] < evaluated[i][0]:
                        r = self.distance(evaluated[i][4], evaluated[j][4])
                        beta = self.beta0 * math.exp(-self.gamma * r ** 2)

                        new_schedule = []
                        for k in range(len(self.bounds)):
                            move = (
                                evaluated[i][4][k]
                                + beta * (evaluated[j][4][k] - evaluated[i][4][k])
                                + self.alpha * random.uniform(-0.5, 0.5)
                            )
                            move = int(round(move))
                            move = max(0, min(move, self.bounds[k]))
                            new_schedule.append(move)

                        evaluated[i] = list(self.evaluate(new_schedule)) + [new_schedule]

            evaluated.sort(key=lambda x: x[0])

            if evaluated[0][0] < best_solution[0]:
                best_solution = evaluated[0]

            convergence.append(best_solution[1])

            if gen % 10 == 0:
                progress.progress((gen + 1) / self.generations)
                status.text(f"Generation {gen} | Best Cost RM {best_solution[1]:.2f}")

        progress.progress(100)
        return best_solution, convergence

# 5. SIDEBAR PARAMETERS

st.sidebar.header("Algorithm Parameters")

population_size = st.sidebar.slider("Population Size", 10, 100, 20)
generations = st.sidebar.slider("Generations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha (Randomness)", 0.0, 1.0, 0.5)
beta0 = st.sidebar.slider("Beta₀ (Attractiveness)", 0.1, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma (Absorption)", 0.01, 1.0, 0.1)
max_power_limit = st.sidebar.number_input("Maximum Power Limit (kW)", value=5.0)

# Removed the random_seed slider from the sidebar

# 6. EXECUTION

shiftable = dataset[dataset['Is_Shiftable']].reset_index(drop=True)
non_shiftable = dataset[~dataset['Is_Shiftable']]

base_profile = np.zeros(24)
for _, row in non_shiftable.iterrows():
    for h in range(row['Preferred_Start_Hour'],
                   row['Preferred_Start_Hour'] + row['Duration_Hours']):
        if h < 24:
            base_profile[h] += row['Avg_Power_kW']

st.subheader("Input Dataset")
st.dataframe(dataset)

if st.button("Run Optimization", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    # Use a fixed random seed for reproducibility (you can set it to any value you prefer)
    random_seed = 42

    optimizer = FireflyHEMS(
        shiftable, base_profile, max_power_limit,
        population_size, generations,
        alpha, beta0, gamma,
        seed=random_seed
    )

    best, convergence = optimizer.run(progress, status)
    _, best_cost, best_disc, best_peak, best_schedule = best

    st.success("Optimization Completed Successfully")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cost", f"RM {best_cost:.2f}")
    c2.metric("Peak Power", f"{best_peak:.2f} kW")
    c3.metric("Total Discomfort", f"{best_disc} hrs")

    # CONVERGENCE GRAPH

    st.subheader("Convergence Curve (Best Cost vs Generation)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(convergence, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Cost (RM)")
    ax.set_title("Firefly Algorithm Convergence")
    ax.grid(True)
    st.pyplot(fig)
