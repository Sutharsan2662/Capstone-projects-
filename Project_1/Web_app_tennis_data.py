import streamlit as st
import pandas as pd

def separate_apps():
    st.image("tennis.png")
    # Sidebar for file uploads
    st.sidebar.header("Tennis data")

    try:
        df1 = pd.read_csv("competitor.csv")
        df2 = pd.read_csv("complexes.csv")
        df3 = pd.read_csv("doubles_data.csv")
    except FileNotFoundError:
        st.error("Error: One or more CSV files not found in the specified directory.")
        return

    dfs = [df1, df2, df3]
    file_names = ["Competitor Data", "Complexes Data", "Doubles Data"]

    # File selection
    st.sidebar.header("Select File to Analyze")
    selected_file_index = st.sidebar.selectbox("Choose File", file_names)
    selected_df = None

    if selected_file_index == "Competitor Data":
        selected_df = dfs[0]
        if selected_df is not None:
            st.sidebar.header("Filters")
            if "category_name" in selected_df.columns:
                categories = ["All"] + sorted(selected_df["category_name"].unique().astype(str))
                selected_category = st.sidebar.selectbox("Category", categories)
                if selected_category != "All":
                    selected_df = selected_df[selected_df["category_name"] == selected_category]
            if "type" in selected_df.columns:
                types = ["All"] + sorted(selected_df["type"].unique().astype(str))
                selected_type = st.sidebar.selectbox("Type", types)
                if selected_type != "All":
                    selected_df = selected_df[selected_df["type"] == selected_type]
            if "gender" in selected_df.columns:
                genders = ["All"] + sorted(selected_df["gender"].unique().astype(str))
                selected_gender = st.sidebar.selectbox("Gender", genders)
                if selected_gender != "All":
                    selected_df = selected_df[selected_df["gender"] == selected_gender]

    elif selected_file_index == "Complexes Data":
        selected_df = dfs[1]
        if selected_df is not None:
            st.sidebar.header("Filters")
            if "venue_nmae" in selected_df.columns:
                venues = ["All"] + sorted(selected_df["venue_name"].unique().astype(str))
                selected_venue = st.sidebar.selectbox("Venue Name", venues)
                if selected_venue != "All":
                    selected_df = selected_df[selected_df["venue_name"] == selected_venue]
            if "country_name" in selected_df.columns:
                countries = ["All"] + sorted(selected_df["country_name"].unique().astype(str))
                selected_country = st.sidebar.selectbox("Country", countries)
                if selected_country != "All":
                    selected_df = selected_df[selected_df["country_name"] == selected_country]
            if "timezone" in selected_df.columns:
                timezones = ["All"] + sorted(selected_df["timezone"].unique().astype(str))
                selected_timezone = st.sidebar.selectbox("Timezone", timezones)
                if selected_timezone != "All":
                    selected_df = selected_df[selected_df["timezone"] == selected_timezone]
            if "complex_name" in selected_df.columns:
                complex_names = ["All"] + sorted(selected_df["complex_name"].unique().astype(str))
                selected_complex = st.sidebar.selectbox("Complex Name", complex_names)
                if selected_complex != "All":
                    selected_df = selected_df[selected_df["complex_name"] == selected_complex]

    elif selected_file_index == "Doubles Data":
        selected_df = dfs[2]
        if selected_df is not None:
            st.sidebar.header("Filters")
            if "ranks" in selected_df.columns:
                min_rank = int(selected_df["ranks"].min())
                max_rank = int(selected_df["ranks"].max())
                rank_range = st.sidebar.slider("Rank Range", min_rank, max_rank, (min_rank, max_rank))
                selected_df = selected_df[
                    (selected_df["ranks"] >= rank_range[0]) & (selected_df["ranks"] <= rank_range[1])
                ]
            if "competitor_name" in selected_df.columns:
                names = ["All"] + sorted(selected_df["competitor_name"].unique().astype(str))
                selected_name = st.sidebar.selectbox("Name", names)
                if selected_name != "All":
                    selected_df = selected_df[selected_df["competitor_name"] == selected_name]
            if "country" in selected_df.columns:
                countries = ["All"] + sorted(selected_df["country"].unique().astype(str))
                selected_country = st.sidebar.selectbox("Country", countries)
                if selected_country != "All":
                    selected_df = selected_df[selected_df["country"] == selected_country]
            if "competitions_played" in selected_df.columns:
                competitions = ["All"] + sorted(selected_df["competitions_played"].unique().astype(str))
                selected_competition = st.sidebar.selectbox("Competitions", competitions)
                if selected_competition != "All":
                    selected_df = selected_df[selected_df["competitions_playes"] == selected_competition]
            if "points" in selected_df.columns:
                min_points = int(selected_df["points"].min())
                max_points = int(selected_df["points"].max())
                points_range = st.sidebar.slider("Points Range", min_points, max_points, (min_points, max_points))
                selected_df = selected_df[
                    (selected_df["points"] >= points_range[0]) & (selected_df["points"] <= points_range[1])
                ]

    if selected_df is not None:
        st.subheader(f"Filtered Data ({selected_file_index})")
        st.dataframe(selected_df)
        st.write(f"Number of Rows: {selected_df.shape[0]}")
        st.write(f"Number of Columns: {selected_df.shape[1]}")
    # Add more analysis and visualization as needed

if __name__ == "__main__":
    separate_apps()