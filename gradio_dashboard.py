import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import load_csv
from src.demand_forecasting import DemandForecaster
from src.spatial_analysis import SpatialAnalyzer
from src.decision_engine import DecisionEngine

# ===============================
# Load Data
# ===============================
spatial_df = load_csv("data/processed/ev_spatial_preprocessed.csv.gz", compression="gzip")
charging_df = load_csv("data/processed/cleaned_charging_patterns.csv")

# ===============================
# Pre-compute Intelligence
# ===============================
spatial_analyzer = SpatialAnalyzer(spatial_df)
city_ev_df = spatial_analyzer.ev_count_by_city()

forecaster = DemandForecaster(charging_df)
demand_df = forecaster.compute_demand_index()
peak_hour_df = forecaster.peak_charging_hour()

decision_engine = DecisionEngine(demand_df, city_ev_df)

# ===============================
# Plot Functions (Styled)
# ===============================
sns.set_style("darkgrid")
sns.set_palette("viridis")

def plot_top_cities():
    top = city_ev_df.head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(y="city", x="ev_count", data=top, palette="mako")
    plt.xlabel("Number of EVs")
    plt.ylabel("")
    plt.title("Top 10 Cities by EV Adoption", fontsize=14, weight="bold")
    for index, value in enumerate(top["ev_count"]):
        plt.text(value + 5, index, str(value), va='center')
    plt.tight_layout()
    plt.savefig("top_cities.png", transparent=True)
    plt.close()
    return "top_cities.png"

def plot_peak_demand():
    plt.figure(figsize=(8, 5))
    sns.barplot(x="start_hour", y="avg_energy_kwh", data=peak_hour_df, palette="rocket")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Energy Consumed (kWh)")
    plt.title("Average Charging Demand by Hour", fontsize=14, weight="bold")
    # Removed numbers floating above bars to keep it clean
    plt.tight_layout()
    plt.savefig("peak_demand.png", transparent=True)
    plt.close()
    return "peak_demand.png"

# ===============================
# Insight Generator
# ===============================
def generate_insights():
    infra = decision_engine.infrastructure_recommendation()
    policy = decision_engine.policy_recommendation()
    high_risk = decision_engine.high_risk_cities(top_n=5)

    # Metrics
    total_ev = city_ev_df["ev_count"].sum()
    peak_energy = peak_hour_df["avg_energy_kwh"].max()
    top_city = city_ev_df.iloc[0]["city"]

    explanation = (
        f"This dashboard summarizes EV adoption and charging behavior across cities.\n\n"
        f"Key Intelligence:\n"
        f"- Cities with highest EV concentration may face charging congestion.\n"
        f"- Peak charging hours indicate grid stress periods.\n"
        f"- Total EVs monitored: {total_ev}\n"
        f"- Top EV city: {top_city} with {city_ev_df.iloc[0]['ev_count']} EVs\n\n"
        f"Purpose:\n"
        "Support infrastructure planning and policy decision-making."
    )

    # Generate plots
    top_cities_plot = plot_top_cities()
    peak_demand_plot = plot_peak_demand()

    return (
        total_ev,
        peak_energy,
        top_city,
        top_cities_plot,
        peak_demand_plot,
        high_risk,
        infra,
        policy,
        explanation
    )

# ===============================
# Gradio Dashboard Layout
# ===============================
with gr.Blocks(title="EV Adoption & Charging Dashboard") as dashboard:
    gr.Markdown("## ⚡ EV Adoption & Charging Demand Intelligence", elem_id="dashboard-title")

    # Button at the top
    btn = gr.Button("Generate Insights")

    # Metrics Row
    with gr.Row():
        total_ev_metric = gr.Number(label="Total EVs", value=0, interactive=False)
        peak_energy_metric = gr.Number(label="Peak Hour Demand (kWh)", value=0, interactive=False)
        top_city_metric = gr.Textbox(label="Top EV City", value="N/A", interactive=False)

    # Plots Row
    with gr.Row():
        ev_plot = gr.Image(label="EV Adoption Hotspots (Top Cities)")
        demand_plot = gr.Image(label="Charging Demand Pattern")

    # Insights / Tables
    with gr.Row():
        high_risk_df = gr.Dataframe(value=pd.DataFrame(), label="High-Risk Cities (EV Concentration)")
        infra_box = gr.Textbox(value="", label="Infrastructure Recommendation")
        policy_box = gr.Textbox(value="", label="Policy Recommendation")
        explanation_box = gr.Textbox(value="", label="Decision Explanation")

    # Connect Button
    btn.click(
        generate_insights,
        outputs=[
            total_ev_metric,
            peak_energy_metric,
            top_city_metric,
            ev_plot,
            demand_plot,
            high_risk_df,
            infra_box,
            policy_box,
            explanation_box
        ]
    )

if __name__ == "__main__":
    dashboard.launch()
