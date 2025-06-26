import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import functions, formulas
from gwo import GWO
from sklearn.decomposition import PCA
import time

st.set_page_config(layout="wide")

st.sidebar.title("GWO Optimization Settings")

selected_function_name = st.sidebar.selectbox("Select function", list(functions.keys()))
st.sidebar.text("Formula Preview")
st.sidebar.latex(formulas[selected_function_name])

search_agents = st.sidebar.slider("Search agents", 10, 100, 30, step=10)
max_iter = st.sidebar.slider("Max iterations", 50, 2000, 100, step=50)
dimensions = st.sidebar.slider("Dimensions", 2, 100, 10, step=2)

run_button = st.sidebar.button("Run Optimization")

st.sidebar.title("Author")
st.sidebar.markdown("[@dash4k](https://github.com/dash4k)")

st.title("🐺 Grey Wolf Optimizer (GWO)")
st.markdown(f"### Selected Function: `{selected_function_name}`")

if run_button:
    selected_func = functions[selected_function_name]
    with st.spinner("Running GWO..."):
        gwo = GWO(
            func=selected_func.func,
            search_agents=search_agents,
            max_iter=max_iter,
            dim=dimensions,
            lb=selected_func.lower_bound,
            ub=selected_func.upper_bound
        )
        best_fitness, best_position, convergence_curve, position_history = gwo.fit()

    st.success("Optimization complete!")

    st.metric(label="Best Fitness", value=f"{best_fitness:.6f}")
    st.write(f"**Used Config:** `{search_agents}` agents | `{max_iter}` iterations | `{dimensions}` dimensions")

    st.subheader("📉 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(convergence_curve, label="Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{selected_function_name} - Convergence")
    ax.grid(True)
    st.pyplot(fig)

    if position_history:
        st.subheader("🎯 Wolf Position Evolution (PCA)")

        all_positions = np.concatenate(position_history, axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_positions)
        projected_history = [pca.transform(pos) for pos in position_history]
        all_proj = np.vstack(projected_history)

        fig_anim, ax_anim = plt.subplots()
        scat = ax_anim.scatter([], [], c='blue', s=40)
        ax_anim.set_xlim(all_proj[:, 0].min(), all_proj[:, 0].max())
        ax_anim.set_ylim(all_proj[:, 1].min(), all_proj[:, 1].max())
        ax_anim.set_title("Wolf Positions - PCA Projection")

        plot_placeholder = st.empty()

        for frame, pos in enumerate(projected_history):
            scat.set_offsets(pos)
            ax_anim.set_title(f"Iteration {frame+1}")
            plot_placeholder.pyplot(fig_anim)
            time.sleep(0.05)

        plt.close(fig_anim)