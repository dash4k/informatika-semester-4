import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gwo import GWO
from functions import functions, formulas

st.sidebar.title("GWO Optimization Settings")

st.write("")
st.sidebar.subheader("Function")
selected_function_name = st.sidebar.selectbox("Select function", list(functions.keys()))

st.sidebar.text("Formula Preview")
st.sidebar.latex(formulas[selected_function_name])

st.write("")
st.sidebar.subheader("Hyperparameter Tuning")
search_agents = st.sidebar.slider("Search agents", 10, 100, 20, step=10)
max_iter = st.sidebar.slider("Max iterations", 100, 2000, 500, step=100)
dimentions = st.sidebar.slider("Dimentions", 10, 100, 20, step=10)

st.write("")
run_button = st.sidebar.button("Run Optimization")

st.write("")
st.sidebar.title("Author")
st.sidebar.markdown("[@dash4k](https://github.com/dash4k)")

st.title("Grey Wolf Optimizer (GWO)")
st.write(f"### Selected Function: `{selected_function_name}`")

if run_button:
    selected_func = functions[selected_function_name]

    with st.spinner("Running GWO..."):
        gwo = GWO(func=selected_func.func,
                  search_agents=search_agents,
                  max_iter=max_iter,
                  dim=dimentions,
                  lb=selected_func.lower_bound,
                  ub=selected_func.upper_bound)
        best_fitness, best_position, curve = gwo.fit()

    st.success("Optimization complete!")
    st.write(f"**Best Fitness:** `{best_fitness:.6f}`")
    st.write(f"**Used Config:** {search_agents} agents, {max_iter} iterations")

    st.write("### Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(curve)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"{selected_function_name} Convergence Curve")
    ax.grid(True)
    st.pyplot(fig)