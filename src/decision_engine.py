# src/decision_engine.py

class DecisionEngine:
    def __init__(self, demand_df, city_ev_df):
        self.demand_df = demand_df
        self.city_ev_df = city_ev_df

    def high_risk_cities(self, top_n=5):
        """
        Cities with highest EV concentration
        """
        return self.city_ev_df.head(top_n)

    def infrastructure_recommendation(self):
        """
        Suggest charging infrastructure action
        """
        avg_demand = self.demand_df["demand_index"].mean()

        if avg_demand > 15:
            return "High charging demand detected: install fast-charging stations."
        else:
            return "Moderate demand: expand standard public chargers."

    def policy_recommendation(self):
        """
        Suggest policy-level action
        """
        peak_hour = (self.demand_df.groupby("start_hour")["demand_index"].mean().idxmax())

        return f"Peak charging demand occurs around hour {peak_hour}. Introduce time-based incentives."
