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
raw_df = pd.read_csv("data/raw/EV_Population_2.csv")

# Check actual city column name in raw CSV
raw_city_col = "City" if "City" in raw_df.columns else raw_df.columns[0]

code_to_name = dict(zip(spatial_df["city"], raw_df[raw_city_col].str.strip()))
name_to_code = {v: k for k, v in code_to_name.items()}
all_cities = sorted(set(code_to_name.values()))

# Check if charging_df has a city column at all
CHARGING_HAS_CITY = "city" in charging_df.columns

print("charging_df columns:", charging_df.columns.tolist())
print("demand_df columns  :", demand_df.columns.tolist())
print("city_ev_df columns :", city_ev_df.columns.tolist())
print("City mapping sample:", list(code_to_name.items())[:3])

# ===============================
# Plot Functions
# ===============================
def plot_top_cities(top_n):
    top = city_ev_df.head(int(top_n)).copy()
    top["city_label"] = top["city"].map(code_to_name).fillna(top["city"].astype(str))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        y="city_label", x="ev_count", data=top,
        hue="city_label", palette="mako",    
        ax=ax, legend=False
    )
    ax.set_xlabel("Number of EVs")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} Cities by EV Adoption", fontsize=14, weight="bold")
    for index, value in enumerate(top["ev_count"]):
        ax.text(value + 5, index, str(value), va="center")
    plt.tight_layout()
    return fig

def plot_peak_demand(selected_city):
    # ✅ Only filter charging_df if it actually has a city column
    if CHARGING_HAS_CITY and selected_city != "All Cities":
        city_code = name_to_code.get(selected_city)
        filtered = charging_df[charging_df["city"] == city_code]
    else:
        # charging_df has no city column — use full dataset
        filtered = charging_df

    if filtered.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available for selected city",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    local_forecaster = DemandForecaster(filtered)
    local_peak = local_forecaster.peak_charging_hour()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="start_hour", y="avg_energy_kwh", data=local_peak,
        hue="start_hour", palette="rocket",    
        ax=ax, legend=False
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Energy Consumed (kWh)")
    ax.set_title(f"Charging Demand by Hour — {selected_city}", fontsize=14, weight="bold")
    plt.tight_layout()
    return fig

# ===============================
# Insight Generator
# ===============================
def generate_insights(selected_city, top_n, risk_top_n):
    try:
        if selected_city != "All Cities":
            city_code = name_to_code.get(selected_city)

            if city_code is None:
                return (0, 0.0, "Not Found", None, None,
                        pd.DataFrame(),
                        "City not found in mapping.",
                        "City not found in mapping.",
                        f"Could not map '{selected_city}' to a city code.")

            filtered_city_df = city_ev_df[city_ev_df["city"] == city_code]

            # ✅ Only filter demand_df by city if it has a city column
            if "city" in demand_df.columns:
                filtered_demand = demand_df[demand_df["city"] == city_code]
                if filtered_demand.empty:
                    filtered_demand = demand_df   
            else:
                filtered_demand = demand_df

        else:
            filtered_city_df = city_ev_df
            filtered_demand  = demand_df

        if filtered_city_df.empty:
            return (0, 0.0, "No Data", None, None,
                    pd.DataFrame(),
                    "No EV data for this city.",
                    "No EV data for this city.",
                    f"No records found for '{selected_city}'.")

        local_engine = DecisionEngine(filtered_demand, filtered_city_df)

        infra     = local_engine.infrastructure_recommendation()
        policy    = local_engine.policy_recommendation()
        high_risk = local_engine.high_risk_cities(top_n=int(risk_top_n))

        total_ev     = filtered_city_df["ev_count"].sum()
        peak_energy  = round(peak_hour_df["avg_energy_kwh"].max(), 2)
        top_code     = filtered_city_df.iloc[0]["city"]
        top_city     = code_to_name.get(top_code, str(top_code))
        top_ev_count = filtered_city_df.iloc[0]["ev_count"]

        # Decode city names in high_risk table for display
        if not high_risk.empty and "city" in high_risk.columns:
            high_risk = high_risk.copy()
            high_risk["city"] = high_risk["city"].map(code_to_name).fillna(
                                 high_risk["city"].astype(str))

        explanation = (
            f"Showing data for: {selected_city}\n\n"
            f"Key Intelligence:\n"
            f"- Cities with high EV concentration may face charging congestion.\n"
            f"- Peak charging hours indicate grid stress periods.\n"
            f"- Total EVs monitored: {total_ev}\n"
            f"- Top EV city: {top_city} with {top_ev_count} EVs\n\n"
            f"Purpose:\n"
            f"Support infrastructure planning and policy decision-making."
        )

        top_cities_plot  = plot_top_cities(int(top_n))
        peak_demand_plot = plot_peak_demand(selected_city)

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

    except Exception as e:
        import traceback
        print("FULL ERROR:\n", traceback.format_exc())
        return (0, 0.0, "Error", None, None,
                pd.DataFrame(), str(e), str(e),
                f"Error occurred:\n{traceback.format_exc()}")

# ===============================
# Gradio Dashboard Layout
# ===============================
with gr.Blocks(title="EV Adoption & Charging Dashboard") as dashboard:
    gr.Markdown("## ⚡ EV Adoption & Charging Demand Intelligence")

    gr.Markdown("### 🎛 Filters")
    with gr.Row():
        city_dropdown = gr.Dropdown(
            choices=["All Cities"] + all_cities,
            value="All Cities",
            label="Select City",
            interactive=True
        )
        top_n_slider = gr.Slider(
            minimum=5, maximum=20, value=10, step=1,
            label="Top N Cities to Display",
            interactive=True
        )
        risk_n_slider = gr.Slider(
            minimum=3, maximum=10, value=5, step=1,
            label="Number of High-Risk Cities",
            interactive=True
        )

    btn = gr.Button("🔍 Generate Insights", variant="primary")

    gr.Markdown("### 📊 Key Metrics")
    with gr.Row():
        total_ev_metric    = gr.Number(label="Total EVs", value=0, interactive=False)
        peak_energy_metric = gr.Number(label="Peak Hour Demand (kWh)", value=0, interactive=False)
        top_city_metric    = gr.Textbox(label="Top EV City", value="—", interactive=False)

    gr.Markdown("### 📈 Visualisations")
    with gr.Row():
        ev_plot     = gr.Plot(label="EV Adoption Hotspots")
        demand_plot = gr.Plot(label="Charging Demand Pattern")

    gr.Markdown("### 🧠 Decision Intelligence")
    with gr.Row():
        high_risk_df    = gr.Dataframe(label="High-Risk Cities")
        explanation_box = gr.Textbox(label="Decision Explanation", lines=8, interactive=False)
    with gr.Row():
        infra_box  = gr.Textbox(label="🏗 Infrastructure Recommendation", lines=4, interactive=False)
        policy_box = gr.Textbox(label="📜 Policy Recommendation", lines=4, interactive=False)

    btn.click(
        fn=generate_insights,
        inputs=[city_dropdown, top_n_slider, risk_n_slider],
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
    dashboard.launch(inbrowser=True)