import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import json

# Set a title and theme for the Streamlit app
image_path = "static/favicon.ico"

st.set_page_config(
    page_title="Canadian Infrastructure Prediction",
    page_icon=Image.open(image_path),
    layout="wide",
)

#######################
# CSS styling
st.markdown("""
    <style>
    [data-testid="stHeader"] {
        display: none;
    }
    .st-emotion-cache-1jicfl2 {
        padding: 3rem 1rem 10rem !important;
    }
    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }
    [data-testid="stVerticalBlock"] {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    [data-testid="stMetric"] {
        background-color: #1f1f1f;
        text-align: center;
        padding: 5px 0;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stMetricDeltaIcon-Up"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }
    [data-testid="stMetricDeltaIcon-Down"] {
        position: relative;
        left: 38%;
        -webkit-transform: translateX(-50%);
        -ms-transform: translateX(-50%);
        transform: translateX(-50%);
    }
    
    </style>
    """, unsafe_allow_html=True)

# Load dataset
def load_data():
    population_df = pd.read_csv('dataset/healthcare_education_df.csv')
    return population_df

# Define a function to create the sidebar content
def create_sidebar_content(population_df):
    with st.sidebar:
        st.image('static/Citizenship_and_Immigration_Canada_Logo_white.png', width=250)
        st.subheader('Infrastructure Projections')
        
        view_selection = st.radio("View", ("Canada", "Choose a Province"), horizontal=True)
        province = "Canada"  # Default value if "Canada" is selected
        if view_selection == "Choose a Province":
            filtered_provinces = sorted(population_df['Province'].unique())
            filtered_provinces.remove('Canada')
            province = st.selectbox("Province:", options=filtered_provinces)
     
        with st.form("sidebar_form"):        
                
            year = st.number_input("Year:", min_value=2024, max_value=2030, key="year")
                
            col1, col2 = st.columns(2)
            with col1:
                tr = st.number_input("TR:", min_value=0, key="temporary_residents")
            with col2:
                pr = st.number_input("PR:", min_value=0, key="permanent_residents")

            initial_population = st.number_input("Population:", min_value=1, key="population")
            
            # Submit button for the form
            submitted = st.form_submit_button("Submit")

    return submitted, year, initial_population, province, tr, pr,  view_selection

# Main Streamlit app logic
def main():
    # Load data
    population_df = load_data()
    
    # Sidebar content and data input
    submitted, year, initial_population, province, tr, pr, view_selection = create_sidebar_content(population_df)
           
    final_predictions_df = pd.DataFrame()
    
    if submitted:
        # CSS to center the spinner
        st.markdown(
            """
            <style>
            [data-testid="stSpinner"] {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 70vh; /* Adjust as needed */
                padding-left: 70vh; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        with st.spinner('Processing...'):
            # Filter data for the selected province or display data for all of Canada
            if view_selection == "Choose a Province":
                input_province_data = population_df[population_df['Province'] == province]
                input_data = ''
                final_predictions_df = predict_features(input_province_data, province, year+1, pr, tr, initial_population)
                province_data = pd.concat([input_province_data, final_predictions_df], ignore_index=True)
            else:
                provinces = population_df['Province'].unique()
                #provinces = provinces[:2]    # Will hide this later, control how much province needs to be looped
                prov = provinces[~pd.isna(provinces)]
                for prov in provinces:
                    input_province_data = population_df[population_df['Province'] == prov]
                    predictions_df = predict_features(input_province_data, prov, year + 1, pr, tr, initial_population)
                    final_predictions_df = pd.concat([final_predictions_df, predictions_df], ignore_index=True)
                            
                filter_province_data = pd.concat([population_df, final_predictions_df], ignore_index=True)
                province_data = filter_province_data[filter_province_data['Province'] == 'Canada']
                input_data = filter_province_data[filter_province_data['Province'] != 'Canada']
                
            
        # Display information for the selected view
        tabs = st.tabs(["Population", "Healthcare", "Education", "Housing"])
        
        with tabs[0]:
            # Display predictions
            load_population_informations(year, province_data, input_data, province, view_selection)
        
        with tabs[1]:
            load_healthcare_informations(year, province_data, province, view_selection)
        
        with tabs[2]:
            load_education_informations(year, province_data, province, view_selection)
        
        with tabs[3]:
            load_housing_informations(year, province_data, province, view_selection)

def load_population_informations(sel_year, province_data, input_data, province, view_selection):
    
    col1, col2, col3, col4, col5 = st.columns(5, gap='medium')
    
    year_filtered = province_data[province_data['Year'] == sel_year]
    men_count = int(province_data[province_data['Year'] == sel_year]['Men+'].values[0])
    women_count = int(province_data[province_data['Year'] == sel_year]['Women+'].values[0])
    tr_count = int(province_data[province_data['Year'] == sel_year]['Temporary Residents'].values[0])
    pr_count = int(province_data[province_data['Year'] == sel_year]['Permanent Residents'].values[0])
    population_count = int(province_data[province_data['Year'] == sel_year]['Population'].values[0])
    men_growth = 0
    women_growth = 0
    tr_growth = 0
    pr_growth = 0

    previous_year_data = province_data[province_data['Year'] == sel_year - 1]
    if not previous_year_data.empty:
        men_growth = round((men_count - previous_year_data['Men+'].values[0]) / 1_000_000, 2)  # in millions
        women_growth = round((women_count - previous_year_data['Women+'].values[0]) / 1_000_000, 2)  # in millions
        population_growth = round((population_count - previous_year_data['Population'].values[0]) / 1_000_000, 2)  # in millions
        tr_growth = round((tr_count - previous_year_data['Temporary Residents'].values[0]) / 1_000_000, 2)  # in millions
        pr_growth = round((pr_count - previous_year_data['Permanent Residents'].values[0]) / 1_000_000, 2)  # in millions
    with col1: 
        st.metric(label="Population", value=f"{round(population_count / 1_000_000, 2)}M", delta=f"{population_growth}M")   
    with col2: 
        st.metric(label="Temporary Residents", value=f"{round(tr_count / 1_000_000, 2)}M", delta=f"{tr_growth}M")
    with col3:
        st.metric(label="Permanent Residents", value=f"{round(pr_count / 1_000_000, 2)}M", delta=f"{pr_growth}M")
    with col4:
        st.metric(label="Men", value=f"{round(men_count / 1_000_000, 2)}M", delta=f"{men_growth}M")
    with col5:
        st.metric(label="Women", value=f"{round(women_count / 1_000_000, 2)}M", delta=f"{women_growth}M")
    
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True) 
    st.markdown(f'Population Over Time In {province}')
    population_trends(province_data)    
    
    col6, col7 = st.columns((4.5, 3.5), gap='medium')
    with col6:
        st.markdown(f'Population Heatmap in {province}')
        if view_selection == "Choose a Province":
            data = province_data
        else:
            data = input_data
        st.altair_chart(make_heatmap(data, 'Year', 'Province', 'Population'), use_container_width=True)    
    with col7:
        age_dist = year_filtered[['Population_age_0 to 3 years', 'Population_age_4 to 17 years', 'Population_age_18 to 50 years', 'Population_age_50 years and above']].sum().reset_index()
        age_dist.columns = ['Age Group', 'Population']
        age_dist['Population'] = age_dist['Population'].apply(lambda x: round(x / 1_000_000, 2))
        fig = px.pie(age_dist, names='Age Group', values='Population', title='Age Group Distribution')
        st.plotly_chart(fig)
        
    #st.markdown(f'Population Map in {province}')
    if view_selection == "Choose a Province":
        data = province_data
    else:
        data = input_data
                
    display_map(data,sel_year,province)

def load_healthcare_informations(sel_year, province_data, province, view_selection):
    col1, col2, col3 = st.columns(3, gap='medium')
    
    beds_count = int(province_data[province_data['Year'] == sel_year]['Beds'].values[0])
    physicians_count = int(province_data[province_data['Year'] == sel_year]['Physician'].values[0])
    population_count = int(province_data[province_data['Year'] == sel_year]['Population'].values[0])
    beds_growth = 0
    physicians_growth = 0
 

    previous_year_data = province_data[province_data['Year'] == sel_year - 1]
    if not previous_year_data.empty:
        beds_growth = round((beds_count - previous_year_data['Beds'].values[0]) / 10_000, 2)  # in millions
        physicians_growth = round((physicians_count - previous_year_data['Physician'].values[0]) / 10_000, 2)  # in millions
        population_growth = round((population_count - previous_year_data['Population'].values[0]) / 1_000_000, 2)  # in millions
    with col1: 
        st.metric(label="Population", value=f"{round(population_count / 1_000_000, 2)}M", delta=f"{population_growth}M")   
    with col2:
        st.metric(label="Physicians", value=f"{round(physicians_count / 10_000, 2)}K", delta=f"{physicians_growth}K")
    with col3:
        st.metric(label="Beds", value=f"{round(beds_count / 10_000, 2)}K", delta=f"{beds_growth}K")
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True) 
    st.markdown(f'Population Over Time in {province}')
    healthcare_trends(province_data)

def load_education_informations(sel_year, province_data, province, view_selection):
    col1, col2, col3, col4 = st.columns(4, gap='medium')
    
    feducators_count = int(province_data[province_data['Year'] == sel_year]['Full-time educators'].values[0])
    peducators_count = int(province_data[province_data['Year'] == sel_year]['Part-time educators'].values[0])
    student_count = int(province_data[province_data['Year'] == sel_year]['Students'].values[0])
    population_count = int(province_data[province_data['Year'] == sel_year]['Population'].values[0])
    peducator_growth = 0
    feducator_growth = 0
    student_growth = 0
 

    previous_year_data = province_data[province_data['Year'] == sel_year - 1]
    if not previous_year_data.empty:
        feducator_growth = round((feducators_count - previous_year_data['Full-time educators'].values[0]) / 10_000, 2)  # in millions
        peducator_growth = round((peducators_count - previous_year_data['Part-time educators'].values[0]) / 10_000, 2)  # in millions
        student_growth = round((student_count - previous_year_data['Students'].values[0]) / 10_000, 2)  # in millions
        population_growth = round((population_count - previous_year_data['Population'].values[0]) / 1_000_000, 2)  # in millions
    with col1: 
        st.metric(label="Population", value=f"{round(population_count / 1_000_000, 2)}M", delta=f"{population_growth}M")   
    with col2:
        st.metric(label="Students", value=f"{round(student_count / 10_000, 2)}K", delta=f"{student_growth}K")
    with col3:
        st.metric(label="Full-time educators", value=f"{round(feducators_count / 10_000, 2)}K", delta=f"{feducator_growth}K")
    with col4:
        st.metric(label="Part-time educators", value=f"{round(peducators_count / 10_000, 2)}K", delta=f"{peducator_growth}K")
        
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True) 
    st.markdown(f"Education Information for {province} in {sel_year}")
    education_trends(province_data)
    #fig = student_to_educator_ratio(province_data)
    #st.pyplot(fig)

def load_housing_informations(sel_year, province_data, province, view_selection):
    st.markdown(f"## Housing Information for {province} in {sel_year}")
    # Add Housing visualizations here

def make_heatmap(input_df, input_y, input_x, input_color):
    #input_df = input_df.sort_values(by=['Province'], ascending=True)
    input_df[input_color] = input_df[input_color].apply(lambda x: round(x / 1_000_000, 2))
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'{input_color}:Q',
                        legend=alt.Legend(title="Population (in millions)", labelFontSize=12, titleFontSize=14),
                        scale=alt.Scale(scheme='viridis')),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12,
    )
    return heatmap

def display_map(df, year,prov):
    #st.write(df)

    with open('dataset/Canada_ShapeFile.geojson', 'r') as geo:
        mp = json.load(geo)
     
    # Create the choropleth map
    fig = px.choropleth_mapbox(
        data_frame=df,
        geojson=mp,
        featureidkey="properties.prov_name_en",
        locations='Province',
        color='Population',
        color_continuous_scale="Viridis",
        mapbox_style='carto-positron',  # Use a more visually appealing map style
        center=dict(lat=59.959354, lon=-101.990312),
        zoom=2.00,
        opacity=0.6,  # Slightly adjust opacity for better visibility
        width=800,  # Adjusted width for better alignment
        height=600  # Adjusted height for better alignment
    )

    # Update the layout for better styling and black font
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},  # Adjust top margin for title
        font=dict(color='black'),  # Set font color to black
        coloraxis_colorbar=dict(
            x=0.9,  # Adjust the x-coordinate (0.0 to 1.0) for horizontal position
            y=0.5,  # Adjust the y-coordinate (0.0 to 1.0) for vertical position
            len=0.5,  # Adjust the length of the color bar
            title="Population",
            tickfont=dict(color='black'),  # Set tick font color to black
            titlefont=dict(color='black')  # Set title font color to black
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def population_trends(df):
   # Copy the DataFrame
    df_millions = df.copy()

    # Convert columns to desired units
    cols_to_convert_k = ['Births', 'Deaths', 'Temporary Residents', 'Permanent Residents']
    df_millions[cols_to_convert_k] = df_millions[cols_to_convert_k] / 100_000
    df_millions['Population'] = df_millions['Population'] / 1_000_000

    # Create traces with custom colors and rounded tooltips
    trace1 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Births'], 
        mode='lines+markers',
        name='Births',
        yaxis='y1',
        line=dict(color='blue'),  # Custom color for Births
        marker=dict(color='blue'),
        hovertemplate='Year: %{x}<br>Births: %{y:.2f}K'
    )
    trace2 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Deaths'], 
        mode='lines+markers',
        name='Deaths',
        yaxis='y1',
        line=dict(color='#d10202'),  # Custom color for Deaths
        marker=dict(color='#d10202'),
        hovertemplate='Year: %{x}<br>Deaths: %{y:.2f}K'
    )
    trace3 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Temporary Residents'], 
        mode='lines+markers',
        name='Temporary Residents',
        yaxis='y1',
        line=dict(color='#16f516'),  # Custom color for Temporary Residents
        marker=dict(color='#16f516'),
        hovertemplate='Year: %{x}<br>Temporary Residents: %{y:.2f}K'
    )
    trace4 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Permanent Residents'], 
        mode='lines+markers',
        name='Permanent Residents',
        yaxis='y1',
        line=dict(color='#eece24'),  # Custom color for Permanent Residents
        marker=dict(color='#eece24'),
        hovertemplate='Year: %{x}<br>Permanent Residents: %{y:.2f}K'
    )
    trace5 = go.Bar(
        x=df_millions['Year'], 
        y=df_millions['Population'], 
        name='Population',
        yaxis='y2',
        opacity=0.3,
        marker=dict(color='grey'),  # Custom color for Population
        hovertemplate='Year: %{x}<br>Population: %{y:.2f}M'
    )

    data = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        xaxis=dict(
            title='Year', 
            title_font=dict(color='white'), 
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on x-axis
        ),
        yaxis=dict(
            title='Population Classifications (in K)',
            side='left',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on y-axis
        ),
        yaxis2=dict(
            title='Population (in M)',
            side='right',
            overlaying='y',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on secondary y-axis
        ),
        legend=dict(
            x=0, 
            y=1.1, 
            font=dict(color='white'), 
            orientation='h',
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant'
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        margin=dict(t=10)  # Reduce the top margin
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
    
def healthcare_trends(df):
   # Copy the DataFrame
    df_millions = df.copy()

    # Convert columns to desired units
    cols_to_convert_k = ['Physician', 'Beds']
    df_millions[cols_to_convert_k] = df_millions[cols_to_convert_k] / 100_000
    df_millions['Population'] = df_millions['Population'] / 1_000_000

    # Create traces with custom colors and rounded tooltips
    trace1 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Physician'], 
        mode='lines+markers',
        name='Physician',
        yaxis='y1',
        line=dict(color='blue'),  # Custom color for Physician
        marker=dict(color='blue'),
        hovertemplate='Year: %{x}<br>Physician: %{y:.2f}K'
    )
    trace2 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Beds'], 
        mode='lines+markers',
        name='Beds',
        yaxis='y1',
        line=dict(color='#16f516'),  # Custom color for Beds
        marker=dict(color='#16f516'),
        hovertemplate='Year: %{x}<br>Beds: %{y:.2f}K'
    )
    trace3 = go.Bar(
        x=df_millions['Year'], 
        y=df_millions['Population'], 
        name='Population',
        yaxis='y2',
        opacity=0.3,
        marker=dict(color='grey'),  # Custom color for Population
        hovertemplate='Year: %{x}<br>Population: %{y:.2f}M'
    )

    data = [trace1, trace2, trace3]

    layout = go.Layout(
        xaxis=dict(
            title='Year', 
            title_font=dict(color='white'), 
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on x-axis
        ),
        yaxis=dict(
            title='Physicans & Beds (in K)',
            side='left',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on y-axis
        ),
        yaxis2=dict(
            title='Population (in M)',
            side='right',
            overlaying='y',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on secondary y-axis
        ),
        legend=dict(
            x=0, 
            y=1.1, 
            font=dict(color='white'), 
            orientation='h',
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant'
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        margin=dict(t=10)  # Reduce the top margin
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
   
def education_trends(df):
    # Copy the DataFrame
    df_millions = df.copy()

    # Convert columns to desired units
    cols_to_convert_k = ['Full-time educators', 'Part-time educators', 'Students']
    df_millions[cols_to_convert_k] = df_millions[cols_to_convert_k] / 100_000
    df_millions['Population'] = df_millions['Population'] / 1_000_000

    # Create traces with custom colors and rounded tooltips
    trace1 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Full-time educators'], 
        mode='lines+markers',
        name='Full-time educators',
        yaxis='y1',
        line=dict(color='blue'),  # Custom color for Full-time educators
        marker=dict(color='blue'),
        hovertemplate='Year: %{x}<br>Full-time educators: %{y:.2f}K'
    )
    trace2 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Part-time educators'], 
        mode='lines+markers',
        name='Part-time educators',
        yaxis='y1',
        line=dict(color='#16f516'),  # Custom color for Part-time educators
        marker=dict(color='#16f516'),
        hovertemplate='Year: %{x}<br>Part-time educators: %{y:.2f}K'
    )
    trace3 = go.Scatter(
        x=df_millions['Year'], 
        y=df_millions['Students'], 
        mode='lines+markers',
        name='Students',
        yaxis='y1',
        line=dict(color='#eece24'),  # Custom color for Students
        marker=dict(color='#eece24'),
        hovertemplate='Year: %{x}<br>Students: %{y:.2f}K'
    )
    trace4 = go.Bar(
        x=df_millions['Year'], 
        y=df_millions['Population'], 
        name='Population',
        yaxis='y2',
        opacity=0.3,
        marker=dict(color='grey'),  # Custom color for Population
        hovertemplate='Year: %{x}<br>Population: %{y:.2f}M'
    )

    data = [trace1, trace2, trace3, trace4]

    layout = go.Layout(
        xaxis=dict(
            title='Year', 
            title_font=dict(color='white'), 
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on x-axis
        ),
        yaxis=dict(
            title='Educators & Students (in K)',
            side='left',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on y-axis
        ),
        yaxis2=dict(
            title='Population (in M)',
            side='right',
            overlaying='y',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=False  # Hide grid lines on secondary y-axis
        ),
        legend=dict(
            x=0, 
            y=1.1, 
            font=dict(color='white'), 
            orientation='h',
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant'
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        margin=dict(t=10)  # Reduce the top margin
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
   
def student_to_educator_ratio(data):
    data['Educators'] = data['Full-time educators'] + data['Part-time educators'] 
    # Convert counts to millions and round to 2 decimal places
    data['Educators'] = [round(x / 1_000_000, 2) for x in data['Educators']]
    data['Students'] = [round(x / 1_000_000, 2) for x in data['Students']]
    
    # Calculate ratio
    data['Ratio'] = [round(e / s, 2) if s != 0 else 0 for e, s in zip(data['Educators'], data['Students'])]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['Year'],
        y=data['Educators'],
        name='Educators (in millions)',
        marker_color='blue',
        hovertemplate='<b>Year: %{x}</b><br>Educators: %{y}M<br>Ratio: %{customdata}<extra></extra>',
        customdata=data['Ratio']
    ))
 
    # Update layout for better styling
    fig.update_layout(
        title='Students Over Years (in millions)',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Count (in millions)'),
        barmode='group',
        bargap=0.15,  # Gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # Gap between bars of the same location coordinates.
        legend=dict(
            x=0.1,
            y=1.1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(0, 0, 0, 0)'
        ),
        font=dict(color='black')
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Function to predict features using LSTM and Linear Regression
def predict_features(df, province, input_year, input_pr="", input_tr="", input_population=""):
    
    df = df.copy()   
    
    # Drop the 'Province' column as it is not relevant for the model
    data = df.drop(columns=['Province'])

   # Extract necessary columns
    population_data = data[['Year', 'Population', 'Births', 'Deaths', 'Temporary Residents', 
                            'Permanent Residents', 'Population_age_0 to 3 years', 
                            'Population_age_18 to 50 years', 'Population_age_4 to 17 years', 
                            'Population_age_50 years and above', 'Men+', 'Women+']]

    # Convert Year to datetime
    population_data['Year'] = pd.to_datetime(population_data['Year'], format='%Y')

    # Scale the population data
    scaler = MinMaxScaler(feature_range=(0, 1))
    population_scaled = scaler.fit_transform(population_data['Population'].values.reshape(-1, 1))

    # Prepare the data for LSTM
    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 3
    X, y = create_dataset(population_scaled, look_back)

    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], look_back, 1))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)

    # Make predictions
    def predict_population(model, data, look_back, steps):
        predictions = []
        current_step = data[-look_back:]
        for _ in range(steps):
            pred = model.predict(current_step.reshape(1, look_back, 1))
            predictions.append(pred[0, 0])
            current_step = np.append(current_step[1:], pred)[-look_back:]
        return np.array(predictions)

    # Forecast future population
    steps = input_year - 2022 
    predicted_population = predict_population(model, population_scaled, look_back, steps)
    predicted_population = scaler.inverse_transform(predicted_population.reshape(-1, 1)).flatten()

    # Prepare the forecast DataFrame
    future_years = pd.date_range(start=population_data['Year'].iloc[-1] + pd.DateOffset(years=1), periods=steps, freq='YS')
    forecast_df = pd.DataFrame({'Year': future_years, 'Population': predicted_population})

    # Calculate proportional changes for other columns
    proportional_changes = {}
    for col in population_data.columns[1:]:
        proportional_changes[col] = (population_data[col].values[-1] - population_data[col].values[-2]) / (population_data['Population'].values[-1] - population_data['Population'].values[-2])

    # Predict other columns based on the proportional changes
    predicted_inputs = {}
    for col in population_data.columns[2:]:
        predicted_values = []
        for i in range(len(predicted_population)):
            if i == 0:
                # Initialize with the last known value
                predicted_values.append(population_data[col].values[-1])
            else:
                change = proportional_changes[col] * (predicted_population[i] - predicted_population[i-1])
                predicted_values.append(predicted_values[-1] + change)
        predicted_inputs[col] = np.array(predicted_values)

    # Adjust the calculation for Births to avoid negative values
    last_known_births = population_data['Births'].values[-1]
    births_growth_rate = last_known_births / population_data['Population'].values[-1]
    predicted_births = [last_known_births]
    for i in range(1, len(predicted_population)):
        predicted_birth = births_growth_rate * predicted_population[i]
        predicted_births.append(predicted_birth)
    predicted_inputs['Births'] = np.array(predicted_births)

    # Combine predictions with forecasted population
    for col, values in predicted_inputs.items():
        forecast_df[col] = values

    # Prepare data for linear regression models
    metrics = ['Beds', 'Physician', 'Full-time educators', 'Part-time educators', 'Students']
    X = population_data['Population'].values.reshape(-1, 1)

    # Initialize dictionaries to store models and predictions
    models = {}
    predictions = {}

    # Fit linear regression models and make predictions
    for metric in metrics:
        y = data[metric].values
        model = LinearRegression()
        model.fit(X, y)
        models[metric] = model
        
        # Make predictions for 2022 to 2029
        population_forecast = forecast_df['Population'].values.reshape(-1, 1)
        predictions[metric] = model.predict(population_forecast)

    # Combine predictions with forecasted population
    for metric in metrics:
        forecast_df[metric] = predictions[metric]
  
    # Convert the predicted values to integers by truncating the decimal part, excluding the 'Year' column
    forecast_df.iloc[:, 1:] = forecast_df.iloc[:, 1:].apply(np.floor).astype(int)

    # Print the final combined forecast
    forecast_df['Year'] = forecast_df['Year'].dt.year

    forecast_df['Province'] = province

    return forecast_df
   

# Entry point of the app
if __name__ == "__main__":
    main()
