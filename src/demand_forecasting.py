# src/demand_forecasting.py
class DemandForecaster:
    def __init__(self, charging_df):
        self.df = charging_df

    def compute_demand_index(self):
        """
        Creating a simple charging demand score
        """
        self.df["demand_index"] = (self.df["energy_consumed_(kwh)"] + self.df["charging_duration_(hours)"])
        
        return self.df

    def peak_charging_hour(self):
        """
        Find hour with highest average energy usage
        """
        peak = (self.df.groupby("start_hour")["energy_consumed_(kwh)"].mean().sort_values(ascending=False))

        return peak.reset_index(name="avg_energy_kwh")
