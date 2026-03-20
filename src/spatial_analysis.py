# src/spatial_analysis.py
class SpatialAnalyzer:
    def __init__(self, spatial_df):
        self.df = spatial_df

    def ev_count_by_city(self):
        """
        Count number of EVs per city
        """
        city_counts = (self.df.groupby("city").size().reset_index(name="ev_count"))

        return city_counts.sort_values(
            "ev_count", ascending=False
        )

    def top_states(self):
        """
        Count EVs per state using one-hot encoded state columns
        """
        state_cols = [c for c in self.df.columns if c.startswith("state_")]

        state_summary = (self.df[state_cols].sum().reset_index())

        state_summary.columns = ["state", "ev_count"]

        return state_summary.sort_values(
            "ev_count", ascending=False
        )
