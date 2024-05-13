import streamsync as ss
import pandas as pd
import plotly.express as px
import requests
from credentials import config
import json
from pyspark.sql import SparkSession
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


weeks = 10
# Get token
def get_token():

  if not config['client_id']:
    raise ValueError('client_id must be set in credentials.py')

  if not config['client_secret']:
    raise ValueError('client_secret must be set in credentials.py')

  req = requests.post(config['token_url'],
    data={
        'grant_type': 'client_credentials',
        'client_id': config['client_id'],
        'client_secret': config['client_secret'],
        'scope': 'api'
    },
    headers={'content-type': 'application/x-www-form-urlencoded'})

  req.raise_for_status()
  print('Token request successful')
  return req.json()



# Function to download the data for all loctions for a year
def year_data(token, year, week):
    api_base_url = config['api_base_url']
    endpoint = f"/v1/geodata/fishhealth/locality/{year}/{week}"
    headers = {
        'Authorization': f'Bearer {token["access_token"]}',
        'Content-Type': 'application/json',
    }
    response = requests.get(api_base_url + endpoint, headers=headers) 
    response.raise_for_status()
    return response.json()

def get_year_data(token, year):
    year_list = []
    for i in range(weeks):
        weeksummary = year_data(token, year, str(i+1)) 
        year_list.append(weeksummary)
    return year_list

def _get_main_df(year):
    token = get_token()

    year_data = get_year_data(token, year)
    week_lice_data = []
    for week in year_data:
        week_num = week['week']
        for locality in week['localities']:
            lat = locality['lat']
            if lat < 63.81:
                region = "South"
            elif lat <= 68.85:
                region = "Middle"
            else:
                region = "North"
            # Add the region to the locality.
            locality['region'] = region
            week_lice_data.append({
                'week': week_num,
                'localityName': locality['name'],
                'localityNo': locality['localityNo'],
                'region': region,
                'lat': locality['lat'],
                'lon': locality['lon'],
                'hasPd': locality['hasPd'],
                'hasIla': locality['hasIla'],
                'avgAdultFemaleLice': locality['avgAdultFemaleLice']
    })
    df_week_lice = pd.DataFrame(week_lice_data)
    return df_week_lice

# Lice Count
def get_lice_count_data(token, state):
    localityNo = state["localityNo"]
    year = state["year"]

    year_list = []
    api_base_url = config['api_base_url']
    for week in range(1, weeks):  
        endpoint = f"/v1/geodata/fishhealth/locality/{localityNo}/{year}/{week}"
        url = f"{api_base_url}{endpoint}"
        headers = {
            'Authorization': 'Bearer ' + token['access_token'],
            'Content-Type': 'application/json',
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        weeksummary = response.json()
        year_list.append(weeksummary)

    avgAdultFemaleLice = []
    avgMobileLice = []
    avgStationaryLice = []
    time = []

    for week in year_list:
        #getting the coordinates of the locality that will be used later for weather data
        lat = week['aquaCultureRegister']['lat']
        lon = week['aquaCultureRegister']['lon']
        locality_week = week.get('localityWeek')
        
        if locality_week is not None:
            time.append(week.get('aquaCultureRegister').get('aquaCultureUpdateTime'))
            avgAdultFemaleLice.append(locality_week.get('avgAdultFemaleLice', None))
            avgMobileLice.append(locality_week.get('avgMobileLice', None))
            avgStationaryLice.append(locality_week.get('avgStationaryLice', None))

            
    # create a dataframe for all licetype
    df_licetype = pd.DataFrame({
        'time': time,
        'avgAdultFemaleLice': avgAdultFemaleLice,
        'avgMobileLice': avgMobileLice,
        'avgStationaryLice': avgStationaryLice,
    })  

    df_licetype['time'] = pd.to_datetime(df_licetype['time'], utc=True)
    df_licetype.set_index('time', inplace=True)
    df_licetype.index = df_licetype.index.normalize()

    #getting the weather data for the locality

    print(df_licetype)

    date_strings = df_licetype.index.strftime('%Y-%m-%d').tolist()

    # Assuming the first and last dates define the range for weather data
    date_range = f'{date_strings[0]}/{date_strings[-1]}'
    # Get weather data for the defined date range
    #weather_df = get_weather_data(date_range, lat, lon)
    #state["weather_df"] = weather_df

    #print(weather_df)
    return df_licetype

def get_weather_data(time, lat, lon):
    endpoint = 'https://frost.met.no/sources/v0.jsonld'
    client_id = '735f154b-d80e-4d7a-a7c3-6227a81ba1db'

    parameters = {
        'geometry': f'nearest(POINT({lon} {lat}))',

    }

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()
    
    # Check if the request worked, print out any errors
    if r.status_code == 200:
        location_id = json['data'][0]['id']
        print('Data retrieved from frost.met.no!')
        print(location_id)
        endpoint2 = 'https://frost.met.no/observations/v0.jsonld'
        client_id = '735f154b-d80e-4d7a-a7c3-6227a81ba1db'

        parameters2 = {
            'sources': location_id,
            'elements': 'mean(air_temperature P1D) , sum(precipitation_amount P1D), mean(wind_speed P1D), mean(relative_humidity P1D)',
            'referencetime': time
        }

        # Issue an HTTP GET request
        r2 = requests.get(endpoint2, parameters2, auth=(client_id,''))
        # Extract JSON data
        json2 = r2.json()

        # Check if the request worked, print out any errors
        if r2.status_code == 200:
            data = json2['data']
            print('Data retrieved from frost.met.no!')
        else:
            print('Error! Returned status code %s' % r.status_code)
            print('Message: %s' % json2['error']['message'])
            print('Reason: %s' % json2['error']['reason'])

                # Initialize an empty DataFrame
        weather_data_list = []

        # Process each JSON object in the list
        for json_data in data:
            row_data = {'referenceTime': json_data['referenceTime']}
            for observation in json_data['observations']:
                row_data[observation['elementId']] = observation['value']
            weather_data_list.append(row_data)

        # Create DataFrame
        weather_df = pd.DataFrame(weather_data_list)
        weather_df['referenceTime'] = pd.to_datetime(weather_df['referenceTime'])
        weather_df.set_index('referenceTime', inplace=True)

        weather_df = weather_df.resample('W').mean()

        return weather_df
    
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])


# Plot restaurants
def _update_plotly_localities(state):
    localities = state["localities_df"]
    selected_num = state["selected_num"]
    sizes = [10]*len(localities)
    if selected_num != -1:
        sizes[selected_num] = 20
    fig_localities = px.scatter_mapbox(
        localities,
        lat="lat",
        lon="lon",
        hover_name="localityName",
        hover_data=["lat","lon"],
        color_discrete_sequence=["darkgreen"],
        zoom=5,
        height=750,
        width=550,
    )
    overlay = fig_localities['data'][0]
    overlay['marker']['size'] = sizes
    fig_localities.update_layout(mapbox_style="open-street-map")
    fig_localities.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Create the pie chart figure
    pie_chart = go.Figure(go.Pie(
        labels=['Has Piscine Dermatitis (PD)', 'Does Not Have PD'],
        values=[
            localities['hasPd'].sum(),
            len(localities) - localities['hasPd'].sum(),
        ],
        hole=0.3,
        marker_colors=['gold', 'lightcoral']
    ))

    # Update layout for the pie chart figure
    pie_chart.update_layout(
        title_text="Distribution of Piscine Dermatitis (PD) in Localities",
        height=350,
        showlegend=False
    )

    #bar plot for the selected parameter for localities
    if state["selected_localities_column"] in localities.columns:
        
        # Get the total count of localities per week
        total_count_per_week = localities.groupby('week')['localityName'].nunique()

        # Count the number of localities with the parameter as True per week
        true_count_per_week = localities[localities[state["selected_localities_column"]] == True].groupby('week')['localityName'].nunique()

        merged = pd.DataFrame({'Total Localities': total_count_per_week, 
                               f'True {state["selected_localities_column"]}': true_count_per_week}).reset_index() 

        # Plotting using Plotly
        fig_bar_plot = px.bar(merged, x='week', y=['Total Localities', f'True {state["selected_localities_column"]}'],
                            title=f'Localities with {state["selected_localities_column"]} per Week',
                            barmode='group')
        fig_bar_plot.update_layout(yaxis_title='Number of Localities')
        # fig.show()
        state["plotly_bar_plot"] = fig_bar_plot
    else:
        print(f'Parameter {state["selected_localities_column"]} not found in DataFrame.')


    state["plotly_localities"] = fig_localities
    state["plotly_pie_chart"] = pie_chart
    #state["loading"] = False
    #state["isLoaded"] = True


#settung up dropdown
def _update_localities_column_choice(state):
    localities = state["localities_df"]
    # column selector for locality data
    parameter_columns = [col for col in localities.columns if col.startswith(('has', 'is'))]
    # Creating the dictionary with keys and values being the same
    state['parameter_dict_localities'] = {col: col for col in parameter_columns}

def handle_localities_column_choice(state, payload):
    state["loading"] = True
    state["isLoaded"] = False
    localities = state["localities_df"]
    state["selected_localities_column"] = payload
    print("Selected column", payload, "for localities")
    # Call the function to update the plot
    if state["selected_localities_column"] in localities.columns:
        
        # Get the total count of localities per week
        total_count_per_week = localities.groupby('week')['localityName'].nunique()

        # Count the number of localities with the parameter as True per week
        true_count_per_week = localities[localities[state["selected_localities_column"]] == True].groupby('week')['localityName'].nunique()

        merged = pd.DataFrame({'Total Localities': total_count_per_week, 
                               f'True {state["selected_localities_column"]}': true_count_per_week}).reset_index() 

        # Plotting using Plotly
        fig_bar_plot = px.bar(merged, x='week', y=['Total Localities', f'True {state["selected_localities_column"]}'],
                            title=f'Count of Localities with {state["selected_localities_column"]} as True per Week',
                            barmode='group')
        fig_bar_plot.update_layout(yaxis_title='Number of Localities')
        # fig.show()
        state["plotly_bar_plot"] = fig_bar_plot
    else:
        print(f'Parameter {state["selected_localities_column"]} not found in DataFrame.')


def _update_plotly_locality(state):
    # Extract the DataFrame and selected lice type from the state
    df_licetype = state["df_licetype"]
    selected_lice_type = state["selected_lice_column"]
    threshold = 0.09
    #Plot the selected lice type
    trace = go.Scatter(x=df_licetype.index, y=df_licetype[selected_lice_type], mode='lines', name=selected_lice_type)

    # Trace for the threshold line
    threshold_line = go.Scatter(x=df_licetype.index, y=[threshold]*len(df_licetype.index), mode='lines', name='Threshold', line=dict(color='red', dash='dash'))

    # Layout of the plot
    layout = go.Layout(
        title=f'{selected_lice_type} Over Time',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Count')
        # legend=dict(title='Lice Types')
    )

    if max(df_licetype[selected_lice_type]) > threshold:
        state["warning"] = True

    print("Check ####: ", selected_lice_type)
    # Create the figure with the trace and threshold line
    fig = go.Figure(data=[trace, threshold_line], layout=layout)
    state["plotly_locality"] = fig
    print(f"Updated plotly locality for {selected_lice_type}")

def _update_plotly_weather(state):
    # Extract the DataFrame and selected lice type from the state
    weather_df = state["weather_df"]
    selected_weather_column = state["selected_weather_column"]
    
    # Plot the selected lice type
    trace = go.Scatter(x=weather_df.index, y=weather_df[selected_weather_column], mode='lines', name=selected_weather_column)

    # Layout of the plot
    layout = go.Layout(
        title=f'{selected_weather_column} Over Time',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Count'),
        legend=dict(title='Weather')
    )

    # Create the figure with the trace and threshold line
    fig = go.Figure(data=[trace], layout=layout)
    state["plotly_weather"] = fig
    print(f"Updated plotly weather")

def handle_click(state, payload):
    localities = state["localities_df"]
    state["selected"] = localities["localityName"].values[payload[0]["pointNumber"]]
    print(state["selected"])
    state["selected_num"] = payload[0]["pointNumber"]
    localityNo = localities["localityNo"].values[payload[0]["pointNumber"]]
    state["localityNo"] = int(localityNo)
    print("Selected locality", localityNo)
    _update_plotly_localities(state)

    # Dictionary for the lice columns
    state["df_licetype"] = get_lice_count_data(get_token() ,state)
    lice_df = state["df_licetype"]
    state["lice_columns"] = {col: col for col in lice_df.columns}
    state['selected_lice_column'] = lice_df.columns[0]
    _update_plotly_locality(state)
    
    #Dictionary for the weather columns
    weather_df = state["weather_df"]
    state["weather_columns"] = {col: col for col in weather_df.columns}
    state['selected_weather_column'] = weather_df.columns[0]
    _update_plotly_weather(state)

def slider_handle(state, payload):
    state['year'] = int(payload)
    state['localities_df'] = _get_main_df(state['year'])
    _update_plotly_localities(state)

def handle_lice_type(state, payload):
    state["warning"] = False
    # Update the selected lice type in the state
    state["selected_lice_column"] = payload
    print("selected_lice_column", payload, "for locality", state["localityNo"])
    # Call the function to update the plot
    _update_plotly_locality(state)

def handle_weather_column_choice(state, payload):
    state['selected_weather_column'] = payload
    print("Selected weather column", payload)
    # Call the function to update the plot
    _update_plotly_weather(state)



def arimax_model(state):
    lice_column = state["selected_lice_column"]
    weather_column = state["selected_weather_column"]

    weather_df = state["weather_df"]
    merged_df = pd.merge(state["df_licetype"], weather_df, left_index=True, right_index=True, how='inner')

    # Define ARIMAX model
    model = SARIMAX(merged_df[lice_column],
                    order=(1, 0, 1),  # Example: ARIMA(1,0,1) model, adjust based on your data
                    exog=merged_df[weather_column])

    # Fit the model
    results = model.fit()

    # Display the summary
    summary = results.summary()

    # Convert the summary table to a DataFrame
    combined_df = pd.DataFrame(summary.tables[1].data)

    # Rename columns and remove the first row (header row in summary table)
    combined_df.columns = combined_df.iloc[0]
    combined_df = combined_df.drop(0)

    # Reset index
    combined_df = combined_df.reset_index(drop=True)
    
    state["combined_df"] = combined_df

    
# Initialise the state

# "_my_private_element" won't be serialised or sent to the frontend,
# because it starts with an underscore (not used here)

initial_state = ss.init_state({
    "my_app": {
        "title": "Localities selection"
    },
    "selected":"Click to select",
    "selected_num":-1,
    'year':2022,
    "localityNo":24175,
    "localities_df":_get_main_df(2020),
    "selected_localities_column": 'hasPd',
    "warning":False,
    "selected_lice_column": 'avgAdultFemaleLice',
    "combined_df":None,

})

_update_plotly_localities(initial_state)
_update_localities_column_choice(initial_state)