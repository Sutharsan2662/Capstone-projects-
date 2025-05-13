import streamlit as st
import pandas as pd
import pymysql
import numpy as np

# --- MySQL Database Configuration ---
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DATABASE = "tennis_data"


def create_connection():
    """Creates a connection to the MySQL database."""
    conn = None
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor,  # Fetch results as dictionaries
        )
    except pymysql.MySQLError as e:
        st.error(f"Error connecting to MySQL: {e}")
    return conn


def execute_query(conn, query, params=None):
    """Executes an SQL query and returns the results as a Pandas DataFrame."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()  # Fetch all results
        df = pd.DataFrame(results)  # Convert to DataFrame
        return df
    except pymysql.MySQLError as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()

# --- Streamlit Application ---
st.set_page_config(layout="wide")
def main():
    # --- Centering Image and Title ---
    col1, col2, col3 = st.columns([1, 3, 1])  # Creates three columns with relative widths
    with col2:
        st.image("tennis.png")
        st.title("🎾 Tennis Data Analytics Dashboard")
        st.subheader("Explore and Analyze Tennis Competition Data")

    conn = create_connection()
    if not conn:
        return

    # Load dataframes.  Added loading of dataframes.
    try:
        df_competitor = pd.read_csv("competitor.csv")
        df_complexes = pd.read_csv("complexes.csv")
        df_doubles = pd.read_csv("doubles_data.csv")
    except FileNotFoundError:
        st.error("Error: One or more CSV files not found.")
        return

    # --- Data Selection ---
    st.header("Select Data to Analyze")
    data_selection = st.radio("Choose a dataset:",
                                     ["Competitor Data", "Complexes Data", "Doubles Data"], horizontal=True) # Data Selection in single row

    # --- Analysis based on Data Selection ---
    if data_selection == "Competitor Data":
        selected_df = df_competitor
        st.header("Competitor Data Analysis")
        # st.sidebar.header("Competitor Data Filters")  # Removed sidebar
        st.subheader("Competitor Data Filters") # Added Subheader
    

        # --- Summary Statistics for Competitor Data ---
        st.subheader("Competitor Data Summary")
        col1, col2 = st.columns(2)
        total_competitions = selected_df["comp_id"].nunique()
        col1.metric("Total Competitions", total_competitions)
        total_categories = selected_df["category_name"].nunique()
        col2.metric("Total Categories", total_categories)

        # --- Filters for Competitor Data ---
        if "category_name" in selected_df.columns:
            categories = ["All"] + sorted(selected_df["category_name"].unique().astype(str))
            selected_category = st.selectbox("Category", categories)
            if selected_category != "All":
                selected_df = selected_df[selected_df["category_name"] == selected_category]
        if "type" in selected_df.columns:
                types = ["All"] + sorted(selected_df["type"].unique().astype(str))
                selected_type = st.selectbox("Type", types)
                if selected_type != "All":
                    selected_df = selected_df[selected_df["type"] == selected_type]
        if "gender" in selected_df.columns:
                genders = ["All"] + sorted(selected_df["gender"].unique().astype(str))
                selected_gender = st.selectbox("Gender", genders)
                if selected_gender != "All":
                    selected_df = selected_df[selected_df["gender"] == selected_gender]

        # --- Display Filtered Data ---
        st.subheader("Filtered Competitor Data")
        if not selected_df.empty:
            selected_df = selected_df.reset_index(drop=True) #reset index
            selected_df.insert(0, 'S.No', selected_df.index + 1) #insert S.No
            st.dataframe(selected_df, column_config={col: st.column_config.Column(width="auto") for col in selected_df.columns})  # Adjust column width
        else:
            st.write("No data matches the selected filters.")

    elif data_selection == "Complexes Data":
        selected_df = df_complexes
        st.header("Complexes Data Analysis")
        # st.sidebar.header("Complexes Data Filters") # Removed Sidebar
        st.subheader("Complexes Data Filters")


        # --- Summary Statistics for Complexes Data ---
        st.subheader("Complexes Data Summary")
        col1, col2 = st.columns(2)
        total_venues = selected_df["venue_id"].nunique()  # Assuming venue_id is unique for each venue
        col1.metric("Total Number of Venues", total_venues)
        total_countries_complexes = selected_df["country_name"].nunique()
        col2.metric("Total Number of Countries", total_countries_complexes)

        # --- Filters for Complexes Data ---
        if "venue_name" in selected_df.columns:
            venues = ["All"] + sorted(selected_df["venue_name"].unique().astype(str))
            selected_venue = st.selectbox("Venue Name", venues)
            if selected_venue != "All":
                selected_df = selected_df[selected_df["venue_name"] == selected_venue]
        if "country_name" in selected_df.columns:
            countries = ["All"] + sorted(selected_df["country_name"].unique().astype(str))
            selected_country = st.selectbox("Country", countries)
            if selected_country != "All":
                selected_df = selected_df[selected_df["country_name"] == selected_country]
        if "timezone" in selected_df.columns:
            timezones = ["All"] + sorted(selected_df["timezone"].unique().astype(str))
            selected_timezone = st.selectbox("Timezone", timezones)
            if selected_timezone != "All":
                selected_df = selected_df[selected_df["timezone"] == selected_timezone]
        if "complex_name" in selected_df.columns:
            complex_names = ["All"] + sorted(selected_df["complex_name"].unique().astype(str))
            selected_complex = st.selectbox("Complex Name", complex_names)
            if selected_complex != "All":
                selected_df = selected_df[selected_df["complex_name"] == selected_complex]

        # --- Display Filtered Complexes Data ---
        st.subheader("Filtered Complexes Data")
        if not selected_df.empty:
            selected_df = selected_df.reset_index(drop=True)
            selected_df.insert(0, 'S.No', selected_df.index + 1)  # Insert S.No. column
            st.dataframe(selected_df, column_config={col: st.column_config.Column(width="auto") for col in selected_df.columns})
        else:
            st.write("No data matches the selected filters.")

    elif data_selection == "Doubles Data":
        selected_df = df_doubles
        st.header("Doubles Data Analysis")
        # st.sidebar.header("Doubles Data Filters") # Removed Sidebar
        st.subheader("Doubles Data Filters")
    

        # --- Summary Statistics ---
        st.subheader("Doubles Data Summary")
        col1, col2, col3 = st.columns(3)
        total_competitors = selected_df["competitor_name"].nunique()
        col1.metric("Total Competitors", total_competitors)
        total_countries = selected_df["country"].nunique()
        col2.metric("Number of Countries", total_countries)
        highest_points = selected_df["points"].max()
        col3.metric("Highest Points", highest_points)

        # --- Search and Filter Competitors ---
        st.subheader("Search and Filter Competitors")
        competitor_search_term = st.text_input("Search Competitor by Name:", "")

        st.subheader("Filter Competitors")
        col_filter1, col_filter2, col_filter3 = st.columns(3)

        min_rank, max_rank = int(selected_df["ranks"].min()), int(selected_df["ranks"].max())
        rank_range = col_filter1.slider("Rank Range", min_rank, max_rank, (min_rank, max_rank))

        countries_list = ["All"] + sorted(selected_df["country"].unique().astype(str))
        selected_country = col_filter2.selectbox("Country", countries_list)

        min_points_filter, max_points_filter = int(selected_df["points"].min()), int(selected_df["points"].max())
        points_threshold = col_filter3.slider("Points Threshold", min_points_filter, max_points_filter,
                                            (min_points_filter, max_points_filter))

        filtered_df = selected_df[
            selected_df["competitor_name"].str.contains(competitor_search_term, case=False, na=False)
            & (selected_df["ranks"] >= rank_range[0])
            & (selected_df["ranks"] <= rank_range[1])
            & (selected_df["points"] >= points_threshold[0])
            & (selected_df["points"] <= points_threshold[1])
            ]
        if selected_country != "All":
            filtered_df = filtered_df[filtered_df["country"] == selected_country]

        st.subheader("Filtered Competitors")
        if not filtered_df.empty:
            filtered_df = filtered_df.reset_index(drop=True)
            filtered_df.insert(0, 'S.No', filtered_df.index + 1)  # Insert S.No. column
            st.dataframe(filtered_df, column_config={col: st.column_config.Column(width="auto") for col in filtered_df.columns})
        else:
             st.write("No data matches the selected filters.")

        # --- Competitor Details Viewer ---
        st.subheader("Competitor Details Viewer")
        competitor_names = ["Select Competitor"] + sorted(selected_df["competitor_name"].unique().tolist())
        selected_competitor = st.selectbox("Select a Competitor", competitor_names)

        if selected_competitor != "Select Competitor":
            competitor_details = selected_df[selected_df["competitor_name"] == selected_competitor]
            if not competitor_details.empty:
                st.subheader(f"Details for {selected_competitor}")
                competitor_details = competitor_details.reset_index(drop=True)
                competitor_details.insert(0, 'S.No', competitor_details.index + 1)
                st.table(competitor_details[["S.No","ranks", "movement", "competitions_played", "country"]].iloc[:, 1:],
                column_config={col: st.column_config.Column(width="auto") for col in competitor_details.columns})
            else:
                st.warning(f"No details found for competitor: {selected_competitor}")

        # --- Country-Wise Analysis ---
        st.subheader("Country-Wise Analysis")
        country_analysis = selected_df.groupby("country").agg(
            num_competitors=("competitor_name", "nunique"),
            average_points=("points", "mean")
        ).reset_index()
        country_analysis.index = country_analysis.index + 1
        st.subheader("Number of Competitors and Average Points by Country")
        country_analysis = country_analysis.reset_index(drop=True)
        country_analysis.insert(0, 'S.No', country_analysis.index + 1)
        st.dataframe(country_analysis, column_config={col: st.column_config.Column(width="auto") for col in country_analysis.columns})

        # --- Leaderboards ---
        st.subheader("Leaderboards")
        col_leaderboard1, col_leaderboard2 = st.columns(2)

        top_ranked = selected_df.sort_values("ranks").head(10)
        top_ranked.index = top_ranked.index + 1
        top_ranked = top_ranked.reset_index(drop=True)
        top_ranked.insert(0, 'S.No', top_ranked.index + 1)
        col_leaderboard1.subheader("Top-Ranked Competitors")
        col_leaderboard1.dataframe(top_ranked[["S.No","competitor_name", "ranks"]].iloc[:, 1:],
        column_config={col: st.column_config.Column(width="auto") for col in top_ranked.columns})

        highest_points_df = selected_df.sort_values("points", ascending=False).head(10)
        highest_points_df.index = highest_points_df.index + 1
        highest_points_df = highest_points_df.reset_index(drop=True)
        highest_points_df.insert(0, 'S.No', highest_points_df.index + 1)
        col_leaderboard2.subheader("Competitors with Highest Points")
        col_leaderboard2.dataframe(highest_points_df[["S.No","competitor_name", "points"]].iloc[:, 1:],
        column_config={col: st.column_config.Column(width="auto") for col in highest_points_df.columns})

    conn.close()


if __name__ == "__main__":
    main()
