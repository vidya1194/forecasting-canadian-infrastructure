import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json



# Set a title and theme for the Streamlit app
image_path = "static/favicon.ico"

st.set_page_config(
    page_title="Canadian Infrastructure Prediction",
    page_icon=Image.open(image_path),
    layout="wide",
)

# Load dataset
def load_data():
    population_df = pd.read_csv('dataset/healthcare_education_df.csv')
    return population_df

# Define a function to create the sidebar content
def create_sidebar_content(population_df):
    with st.sidebar:
        st.image('static/Citizenship_and_Immigration_Canada_Logo_white.png', width=250)
        st.subheader('Canadian Infrastructure Projections')
        with st.expander("Input Parameters"):
            #view_selection = st.radio("View", ("Canada", "All Province", "Choose a Province"))
            view_selection = st.radio("View", ("Canada", "Choose a Province"))

            year = st.number_input("Year:", min_value=2009, max_value=2030, key="year")

            if view_selection == "Choose a Province":
                province = st.selectbox("Province:", options=sorted(population_df['Province'].unique()))
            elif view_selection == "All Province":
                province = "NoCanada"
            else:
                province = "Canada"
                
            col1, col2 = st.columns(2)
            with col1:
                temporary_residents = st.number_input("TR:", min_value=0, key="temporary_residents")
            with col2:
                permanent_residents = st.number_input("PR:", min_value=0, key="permanent_residents")

            initial_population = st.number_input("Population:", min_value=0, key="population")
            submit_button = st.button("Submit")

    return year, initial_population, submit_button, province, view_selection

# Main Streamlit app logic
def main():
    # Load data
    population_df = load_data()

    # Sidebar content and data input
    year, initial_population, submit_button, province, view_selection = create_sidebar_content(population_df)

    # Handle data submission and visualization
    if submit_button:
        new_data = {
            "Year": year,
            "Population": initial_population,
            "Province": province
        }

        # Clear existing content
        #st.subheader("Updated Data:")
        #st.empty()

        # Display new data
        #st.write(pd.DataFrame([new_data]))

        # Filter data for the selected province or display data for all of Canada
        
        if view_selection == "Choose a Province":
            province_data = population_df[population_df['Province'] == province]
        elif view_selection == "All Province":
            province_data = population_df[population_df['Province'] != 'Canada']
        else:
            province_data = population_df[population_df['Province'] == 'Canada']

        # Display information for the selected view
        load_informations(year,province_data, province,view_selection)

def load_informations1(input_df, province,view_selection):
    st.markdown(f'#### Total Population - {province}')
    
    
    col = st.columns((3.5, 4.5), gap='medium')
    
    with col[0]:        
        # Display line charts for selected province
        st.subheader(f'Trends for {province} Population')
        if view_selection == "All Province":
            chart_data = input_df.pivot(index='Year', columns='Province', values='Population')
            st.line_chart(chart_data)
        else:
            st.line_chart(input_df[['Year', 'Births', 'Deaths','Temporary Residents','Permanent Residents','Population']].set_index('Year'))
        
    with col[1]:
        # Display line charts for selected province
        st.subheader(f'Trends for {province}')
        st.line_chart(input_df[['Year', 'Full-time educators', 'Part-time educators']].set_index('Year'))
    
    col = st.columns((3.5, 4.5), gap='medium')
    
    with col[0]:        
        # Display the chart in Streamlit
        st.subheader(f'Students Count in {province}')
        st.bar_chart(input_df[['Year', 'Students']].set_index('Year'))
        
    with col[1]:
        # Display line charts for selected province
        st.subheader(f'Trends for {province}')
        st.line_chart(input_df[['Year', 'Full-time educators', 'Part-time educators']].set_index('Year'))
    
    # with col[2]:
        # Display line charts for selected province
        st.subheader(f'Trends for {province}')
        if view_selection == "All Province":
            st.markdown(f'### Total Population Heatmap {province}')
            st.altair_chart(make_heatmap(input_df, 'Year', 'Province', 'Population'), use_container_width=True)
        else:
            st.line_chart(input_df[['Year', 'Beds', 'Physician']].set_index('Year'))
            

def load_informations(sel_year,input_df, province, view_selection):
    st.markdown(f'#### Total Population - {province}')
    
    col1, col2 = st.columns((3.5, 4.5), gap='medium')
    
    with col1:        
        # Display line charts for selected province
        st.subheader(f'Trends for {province} Population')
        if view_selection == "All Province":
            chart_data = input_df.pivot(index='Year', columns='Province', values='Population')
            st.line_chart(chart_data)
        else:
            st.line_chart(input_df[['Year', 'Births', 'Deaths','Temporary Residents','Permanent Residents','Population']].set_index('Year'))
        
    with col2:
        # Display line charts for educators
        st.subheader(f'Number of Educators in {province}')
        st.line_chart(input_df[['Year', 'Full-time educators', 'Part-time educators']].set_index('Year'))
    
    col3, col4 = st.columns((3.5, 4.5), gap='medium')
    
    with col3:        
        # Display bar chart for students
        st.subheader(f'Students Count in {province}')
        st.bar_chart(input_df[['Year', 'Students']].set_index('Year'))
        
    with col4:
        # Display line chart for healthcare facilities
        st.subheader(f'Healthcare Facilities in {province}')
        st.line_chart(input_df[['Year', 'Beds', 'Physician']].set_index('Year'))
    
    col5, col6 = st.columns((3.5, 4.5), gap='medium')
    
    with col5:
        # Display pie chart for gender distribution
        st.subheader(f'Gender Distribution in {province}')
        selected_year_data = input_df[input_df['Year'] == input_df['Year'].max()]
        gender_data = selected_year_data[['Men+', 'Women+']].sum()
        fig = px.pie(values=gender_data.values, names=gender_data.index, title='Gender Distribution')
        st.plotly_chart(fig)
        
    with col6:
        # Display pie chart for gender distribution
        st.subheader(f'Population Age Distribution in {province}')
        age_data = selected_year_data[['Population_age_0 to 3 years', 'Population_age_4 to 17 years', 'Population_age_18 to 50 years', 'Population_age_50 years and above']].sum()
        fig = px.pie(values=age_data.values, names=age_data.index, title='Population Age Distribution')
        st.plotly_chart(fig)
    
    # Display heatmap for All Province view
    if view_selection == "All Province":
        st.subheader(f'Population Heatmap for All Provinces')
        st.altair_chart(make_heatmap(input_df, 'Year', 'Province', 'Population'), use_container_width=True)
        
    # Display map for total population
    if province == 'Canada':
        st.subheader('Population Map')
        display_population_map(input_df[input_df['Year'] == sel_year])


# Function to display population map
def display_population_map(data):
    # Load GeoJSON file for Canada's provinces
    with urlopen('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson') as response:
        provinces = json.load(response)
    
    
    choropleth = px.choropleth(data, locations='Province',
                                geojson=provinces, 
                                featureidkey="properties.name", 
                                color='Population',
                                color_continuous_scale="Viridis",
                                scope="north america",
                                labels={'population':'Population'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth



# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color):
    
    input_df = input_df.sort_values(by=['Province'], ascending=True)
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'{input_color}:Q',
                        legend=alt.Legend(title="Population", labelFontSize=12, titleFontSize=14),
                        scale=alt.Scale(scheme='viridis')),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12,
    )
    return heatmap

# Entry point of the app
if __name__ == "__main__":
    main()
