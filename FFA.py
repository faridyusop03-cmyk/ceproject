# =========================================================
# HEMS OPTIMIZATION USING FIREFLY ALGORITHM (FFA)
# Course : JIE42903 – Evolutionary Computing
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HEMS Optimization – Firefly Algorithm",
    layout="wide"
)

st.title("⚡ HEMS Optimization using Firefly Algorithm (FFA)")
st.markdown("""
**Course:** JIE42903 – Evolutionary Computing  
**Objective:** Minimize electricity cost & user discomfort  
""")

# =========================================================
# 2. LOAD DATASET
# =========================================================
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
    st.error("Dataset not found. Please upload project_benchmark_data_ce.csv")
    st.stop()

# =========================================================
# 3. MALAYSIA TOU TARIFF
# =========================================================
RATE_PEAK = 0.570
RATE_OFF_PEAK = 0.290
PEAK_START = 14
PEAK_END = 22

def electricity_rate(hour):
    return RATE_PEAK if PEAK_START <= hour < PEAK_END else RATE_OFF_PEAK

# =========================================================
# 4. FIREFLY ALGORITHM CLASS
# =========================================================
class FireflyHEMS:
    def __init__(self, shiftable_df, base_profile, max_power,
                 pop_size, generations, alpha, beta0, gamma):

        self.shiftable_df = shiftable_df
        self.base_profile = base_profile
        self.max_power = max_power
        self.pop_size = pop_size
        self.generations = generations

        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.bounds = [
            24 - int(row['Duration_Hours'])
            for _, row in shiftable_df.iterrows()
        ]

        self.convergence = []

    def initialize_population(self):
        return [
            [random.randint(0, b) for b in self.bounds]
            for _ in range(self.pop_size)
        ]

    def evaluate(self, schedule):
        profile = self.base_profile.copy()

        for i, start in enumerate(schedule):
            dur = int(self.shiftable_df.iloc[i]['Duration_Hours'])
            power = self.shiftable_df.iloc[i]['Avg_Power_kW']
            for h in range(start, start + dur):
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
        evaluated = [(self.evaluate(ind), ind) for ind in population]
        evaluated.sort(key=lambda x: x[0][0])

        best = evaluated[0]

        for gen in range(self.generations):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if evaluated[j][0][0] < evaluated[i][0][0]:

                        r = self.distance(evaluated[i][1], evaluated[j][1])
                        beta = self.beta0 * math.exp(-self.gamma * r ** 2)

                        new_schedule = []
                        for k in range(len(self.bounds)):
                            move = (
                                evaluated[i][1][k]
                                + beta * (evaluated[j][1][k] - evaluated[i][1][k])
                                + self.alpha * random.uniform(-0.5, 0.5)
                            )
                            move = int(round(move))
                            move = max(0, min(move, self.bounds[k]))
                            new_schedule.append(move)

                        new_eval = (self.evaluate(new_schedule), new_schedule)

                        # Accept only better solution
                        if new_eval[0][0] < evaluated[i][0][0]:
                            evaluated[i] = new_eval

            evaluated.sort(key=lambda x: x[0][0])
            if evaluated[0][0][0] < best[0][0]:
                best = evaluated[0]

            self.convergence.append(best[0][0])

            # Alpha decay (critical)
            self.alpha *= 0.95

            if gen % 5 == 0:
                progress.progress((gen + 1) / self.generations)
                status.text(f"Generation {gen} | Best Cost RM {best[0][1]:.2f}")

        progress.progress(100)
        return best

# =========================================================
# 5. SIDEBAR PARAMETERS
# =========================================================
st.sidebar.header("⚙️ Algorithm Parameters")

pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
generations = st.sidebar.slider("Generations", 10, 200, 50)

alpha = st.sidebar.slider("Alpha (Randomness)", 0.0, 1.0, 0.5)
beta0 = st.sidebar.slider("Beta₀ (Attractiveness)", 0.1, 2.0, 1.0)
gamma = st.sidebar.slider("Gamma (Absorption)", 0.01, 1.0, 0.1)

max_power_limit = st.sidebar.number_input("Max Power Limit (kW)", value=5.0)

# =========================================================
# 6. RUN OPTIMIZATION
# =========================================================
shiftable = dataset[dataset['Is_Shiftable']].reset_index(drop=True)
non_shiftable = dataset[~dataset['Is_Shiftable']]

base_profile = np.zeros(24)
for _, row in non_shiftable.iterrows():
    for h in range(row['Preferred_Start_Hour'],
                   row['Preferred_Start_Hour'] + row['Duration_Hours']):
        if h < 24:
            base_profile[h] += row['Avg_Power_kW']

st.subheader("📋 Input Dataset")
st.dataframe(dataset)

if st.button("🚀 Run Optimization", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    optimizer = FireflyHEMS(
        shiftable, base_profile, max_power_limit,
        pop_size, generations,
        alpha, beta0, gamma
    )

    best = optimizer.run(progress, status)
    (fitness, cost, discomfort, peak), schedule = best

    st.success("Optimization Completed Successfully")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cost", f"RM {cost:.2f}")
    c2.metric("Peak Power", f"{peak:.2f} kW")
    c3.metric("Total Discomfort", f"{discomfort} hrs")

    # =====================================================
    # CONVERGENCE GRAPH
    # =====================================================
    st.subheader("📈 Convergence Curve (Best Fitness)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(optimizer.convergence, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness Value")
    ax.set_title("Firefly Algorithm Convergence")
    ax.grid(True)

    st.pyplot(fig)
