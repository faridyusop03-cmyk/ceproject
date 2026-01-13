import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="HEMS Optimization (FFA)", layout="wide")

st.title("⚡ HEMS Optimization using Firefly Algorithm (FFA)")
st.markdown("""
**Project:** JIE42903 - Evolutionary Computing
**Algorithm:** Firefly Algorithm (FFA)
**Description:** Inspired by the flashing behavior of fireflies. Brighter fireflies attract less bright ones.
**Dataset:** `project_benchmark_data.csv`
""")

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    filename = 'project_benchmark_data_ce.csv'
    if not os.path.exists(filename):
        return None
    
    df = pd.read_csv(filename)
    if df['Is_Shiftable'].dtype == 'object':
         df['Is_Shiftable'] = df['Is_Shiftable'].astype(str).str.upper() == 'TRUE'
    return df

base_df = load_data()

# ==========================================
# 3. FIREFLY ALGORITHM (FFA) LOGIC
# ==========================================
RATE_PEAK = 0.570
RATE_OFF_PEAK = 0.290
PEAK_START = 14
PEAK_END = 22

def get_rate(hour):
    if PEAK_START <= hour < PEAK_END:
        return RATE_PEAK
    else:
        return RATE_OFF_PEAK

class HEMS_FireflyAlgorithm:
    def __init__(self, shiftable_df, base_profile, max_power, pop_size, max_generation, alpha, beta0, gamma):
        self.shiftable_df = shiftable_df
        self.base_profile = base_profile
        self.max_power = max_power
        self.pop_size = pop_size
        self.max_generation = max_generation
        
        # FFA Parameters
        self.alpha = alpha   # Randomness parameter
        self.beta0 = beta0   # Attractiveness at r=0
        self.gamma = gamma   # Light absorption coefficient
        
        self.bounds = []
        for _, row in shiftable_df.iterrows():
            max_start = 24 - int(row['Duration_Hours'])
            self.bounds.append(max_start)
            
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            # Firefly position (Schedule)
            ind = [random.randint(0, b) for b in self.bounds]
            population.append(ind)
        return population
            
    def calculate_fitness(self, individual):
        # Calculate Objective Function (Minimize Cost + Penalty)
        # In FFA, Intensity (I) is inversely proportional to Objective Value f(x)
        # But we will work directly with minimizing f(x)
        
        profile = self.base_profile.copy()
        
        for i, start_time in enumerate(individual):
            dur = int(self.shiftable_df.iloc[i]['Duration_Hours'])
            power = self.shiftable_df.iloc[i]['Avg_Power_kW']
            for h in range(start_time, start_time + dur):
                if h < 24: profile[h] += power
                    
        max_p = np.max(profile)
        penalty = 0
        if max_p > self.max_power:
            penalty = 1000 + (max_p - self.max_power) * 100 
            
        total_cost = sum(profile[h] * get_rate(h) for h in range(24))
        
        total_discomfort = 0
        for i, start_time in enumerate(individual):
            pref = self.shiftable_df.iloc[i]['Preferred_Start_Hour']
            total_discomfort += abs(start_time - pref)
            
        fitness = total_cost + (total_discomfort * 0.1) + penalty
        return fitness, total_cost, total_discomfort, max_p, profile

    def distance(self, f1, f2):
        # Euclidean distance between two fireflies (schedules)
        dist_sq = sum((f1[k] - f2[k]) ** 2 for k in range(len(f1)))
        return math.sqrt(dist_sq)

    def run(self, progress_bar, status_text):
        # 1. Initialize Fireflies
        fireflies = self.initialize_population()
        
        # Evaluate Initial Intensity (Fitness)
        # We store tuples: [fitness, firefly_list, cost, disc, max_p]
        scored_fireflies = []
        for f in fireflies:
            fit, c, d, mp, _ = self.calculate_fitness(f)
            scored_fireflies.append([fit, f, c, d, mp])
            
        # Best Solution Tracking
        best_overall = None
        best_fitness_global = float('inf')

        # Find initial best
        scored_fireflies.sort(key=lambda x: x[0])
        best_overall = (scored_fireflies[0][1], scored_fireflies[0][2], scored_fireflies[0][3], scored_fireflies[0][4])
        best_fitness_global = scored_fireflies[0][0]

        # 2. Main Loop
        for gen in range(self.max_generation):
            
            # For each firefly i
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    
                    # If firefly j is brighter (better fitness) than firefly i
                    if scored_fireflies[j][0] < scored_fireflies[i][0]:
                        
                        # Calculate Distance r_ij
                        r = self.distance(scored_fireflies[i][1], scored_fireflies[j][1])
                        
                        # Calculate Attractiveness: beta = beta0 * exp(-gamma * r^2)
                        beta = self.beta0 * math.exp(-self.gamma * (r ** 2))
                        
                        # Move firefly i towards j
                        new_pos = []
                        for k in range(len(self.bounds)):
                            current_x = scored_fireflies[i][1][k]
                            target_x = scored_fireflies[j][1][k]
                            
                            # Random factor
                            rand_epsilon = random.uniform(-0.5, 0.5)
                            
                            # Movement Equation
                            move = current_x + beta * (target_x - current_x) + self.alpha * rand_epsilon
                            
                            # Discretize (Round) and Clamp
                            move = int(round(move))
                            move = max(0, min(move, self.bounds[k]))
                            new_pos.append(move)
                        
                        # Evaluate New Position
                        new_fit, nc, nd, nmp, _ = self.calculate_fitness(new_pos)
                        
                        # Update firefly i if new position is better (Optional elitism check, typical in discrete FFA)
                        # Standard FFA just moves. But to ensure convergence in discrete space, we often accept if better.
                        # Let's use direct update to follow standard FFA logic strictly.
                        scored_fireflies[i] = [new_fit, new_pos, nc, nd, nmp]
            
            # Rank fireflies and update Global Best
            scored_fireflies.sort(key=lambda x: x[0])
            
            if scored_fireflies[0][0] < best_fitness_global:
                best_fitness_global = scored_fireflies[0][0]
                best_ind = scored_fireflies[0]
                best_overall = (best_ind[1], best_ind[2], best_ind[3], best_ind[4])
                
            # Optional: Reduce Alpha over time (Simulated Annealing approach)
            # self.alpha = self.alpha * 0.98 
            
            if gen % 10 == 0:
                progress_bar.progress((gen + 1) / self.max_generation)
                status_text.text(f"Generation {gen} | Best Cost: RM {best_overall[1]:.2f}")

        progress_bar.progress(100)
        return best_overall

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ FFA Configuration")

if base_df is None:
    st.sidebar.error("Dataset `project_benchmark_data.csv` not found!")
    st.stop()
else:
    st.sidebar.success("Loaded: `project_benchmark_data.csv`")

sim_mode = st.sidebar.checkbox("🎲 Simulation Mode", value=False)

st.sidebar.subheader("FFA Parameters")
pop_size = st.sidebar.slider("Number of Fireflies", 10, 100, 20)
max_gen = st.sidebar.slider("Generations", 10, 200, 50)

st.sidebar.markdown("---")
# Core FFA Parameters
alpha = st.sidebar.slider("Alpha (Randomness)", 0.0, 1.0, 0.5, help="Control randomness of movement.")
beta0 = st.sidebar.slider("Beta0 (Attractiveness)", 0.1, 2.0, 1.0, help="Attractiveness at distance=0.")
gamma = st.sidebar.slider("Gamma (Absorption)", 0.01, 1.0, 0.1, help="Light absorption coefficient. Controls convergence speed.")

max_power_limit = st.sidebar.number_input("Max Power Limit (kW)", value=5.0, step=0.5)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
df_run = base_df.copy()

# Logic Display Input
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Input Data")
    if sim_mode:
        df_run['Original_Start'] = df_run['Preferred_Start_Hour']
        shifts = np.random.randint(-3, 4, size=len(df_run))
        df_run['Random_Shift'] = shifts
        df_run['Preferred_Start_Hour'] = (df_run['Preferred_Start_Hour'] + shifts).clip(0, 23)
        st.info("⚠️ Simulation Mode Active: Start times randomized.")
        st.dataframe(df_run[['Appliance', 'Original_Start', 'Random_Shift', 'Preferred_Start_Hour', 'Is_Shiftable']])
    else:
        st.dataframe(df_run[['Appliance', 'Avg_Power_kW', 'Preferred_Start_Hour', 'Duration_Hours', 'Is_Shiftable']])

with col2:
    if st.button("🚀 Run FFA Optimization", type="primary"):
        prog_bar = st.progress(0)
        stat_txt = st.empty()
        
        # Prepare Data
        non_shiftable = df_run[df_run['Is_Shiftable'] == False].copy()
        shiftable = df_run[df_run['Is_Shiftable'] == True].copy().reset_index(drop=True)
        
        # Base Profile
        base_profile = np.zeros(24)
        for _, row in non_shiftable.iterrows():
            start, dur = int(row['Preferred_Start_Hour']), int(row['Duration_Hours'])
            power = row['Avg_Power_kW']
            for h in range(start, start + dur):
                if h < 24: base_profile[h] += power
        
        # Initialize FFA
        ffa = HEMS_FireflyAlgorithm(
            shiftable, base_profile, max_power_limit,
            pop_size=pop_size, max_generation=max_gen, 
            alpha=alpha, beta0=beta0, gamma=gamma
        )
        
        # Run FFA
        best_sol = ffa.run(prog_bar, stat_txt)
        best_sched, best_cost, best_disc, best_mp = best_sol
        stat_txt.text("Optimization Finished!")
        
        # Stats
        orig_sched = shiftable['Preferred_Start_Hour'].tolist()
        _, orig_cost, orig_disc, orig_mp, orig_prof = ffa.calculate_fitness(orig_sched)
        _, _, _, _, opt_prof = ffa.calculate_fitness(best_sched)
        
        # Results Dashboard
        st.divider()
        st.header("🏆 FFA Results")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost", f"RM {best_cost:.2f}", f"{best_cost - orig_cost:.2f}", delta_color="inverse")
        c2.metric("Peak Power", f"{best_mp:.2f} kW", f"{best_mp - orig_mp:.2f}", delta_color="inverse")
        c3.metric("Discomfort", f"{best_disc} hrs", "shifted")
        
        if best_mp > max_power_limit:
            st.error(f"⚠️ Limit Exceeded: {best_mp:.2f} kW")
        else:
            st.success("✅ Power Limit Safe")
            
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        hours = np.arange(24)
        ax.axvspan(14, 22, color='orange', alpha=0.15, label='Peak Tariff')
        ax.step(hours, orig_prof, '--', label='Target Profile', color='red')
        ax.step(hours, opt_prof, '-', label='FFA Optimized', color='green', lw=2)
        ax.axhline(max_power_limit, color='black', ls=':', label='Limit')
        ax.legend()
        ax.set_ylabel("Power (kW)")
        st.pyplot(fig)
        
        # Table
        res = shiftable[['Appliance', 'Preferred_Start_Hour']].copy()
        res['Optimized_Start'] = best_sched
        res['Shift_Amount'] = res['Optimized_Start'] - res['Preferred_Start_Hour']
        st.table(res)
