import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pso import PSO
from gwo import GWO
from sklearn.decomposition import PCA
import time


def sphere(x):
    return sum(xi ** 2 for xi in x)


st.set_page_config(layout="wide")

st.sidebar.title("Optimization Settings")

st.sidebar.text("Formula Preview")
st.sidebar.latex(r"f(x) = \sum_{i=1}^{n} x_i^2")

selected_optimizer = st.sidebar.selectbox("Select Optimizer", ["Particle Swarm Optimization (PSO)", "Grey Wolf Optimizer (GWO)"])

if selected_optimizer == "Particle Swarm Optimization (PSO)":
    search_agents = st.sidebar.slider("Search agents", 5, 30, 5, step=5)
    max_iter = st.sidebar.slider("Max iterations", 5, 100, 5, step=5)
    dimensions = st.sidebar.slider("Dimensions", 2, 10, 2, step=2)

    run_button = st.sidebar.button("Run Optimization")

    st.title("🦋 Particle Swarm Optimization (PSO)")
    st.markdown("### Sphere Function")

    if run_button:
        with st.spinner("Running PSO..."):
            pso = PSO(
                func=sphere,
                dim=dimensions,
                lb=-5.12,
                ub=5.12,
            )
            
            pso.search_agents = search_agents
            pso.max_iter = max_iter

            best_fitness, best_position, convergence_curve, position_history = pso.fit()

        st.success("Optimization complete!")

        st.metric(label="Best Fitness", value=f"{best_fitness:.6f}")
        st.write(f"**Used Config:** `{search_agents}` agents | `{max_iter}` iterations | `{dimensions}` dimensions")

        st.subheader("📉 Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(convergence_curve, label="Best Fitness", color="green")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Sphere - Convergence")
        ax.grid(True)
        st.pyplot(fig)

        if position_history:
            st.subheader("🎯 Particle Position Evolution (PCA)")

            all_positions = np.concatenate(position_history, axis=0)
            pca = PCA(n_components=2)
            pca.fit(all_positions)
            projected_history = [pca.transform(pos) for pos in position_history]
            all_proj = np.vstack(projected_history)

            fig_anim, ax_anim = plt.subplots()
            scat = ax_anim.scatter([], [], c='green', s=40)
            ax_anim.set_xlim(all_proj[:, 0].min(), all_proj[:, 0].max())
            ax_anim.set_ylim(all_proj[:, 1].min(), all_proj[:, 1].max())
            ax_anim.set_title("Particle Positions - PCA Projection")

            plot_placeholder = st.empty()

            for frame, pos in enumerate(projected_history):
                scat.set_offsets(pos)
                ax_anim.set_title(f"Iteration {frame+1}")
                plot_placeholder.pyplot(fig_anim)
                time.sleep(0.05)

            plt.close(fig_anim)

else:
    search_agents = st.sidebar.slider("Search agents", 10, 100, 5, step=10)
    max_iter = st.sidebar.slider("Max iterations", 50, 500, 5, step=50)
    dimensions = st.sidebar.slider("Dimensions", 2, 20, 2, step=2)

    run_button = st.sidebar.button("Run Optimization")

    st.title("🐺 Grey Wolf Optimizer (GWO)")
    st.markdown(f"### Sphere Function")
     
    if run_button:
        with st.spinner("Running GWO..."):
            gwo = GWO(
                func=sphere,
                search_agents=search_agents,
                max_iter=max_iter,
                dim=dimensions,
                lb=-5.12,
                ub=5.12,
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
        ax.set_title(f"Sphere - Convergence")
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


st.sidebar.title("Authors")
st.sidebar.markdown("[@dash4k](https://github.com/dash4k)")
st.sidebar.markdown("[@DewaMahattama](https://github.com/DewaMahattama)")
st.sidebar.markdown("[@KrisnaUdayana](https://github.com/KrisnaUdayana)")
st.sidebar.markdown("[@Maliqytritata](https://github.com/Maliqytritata)")