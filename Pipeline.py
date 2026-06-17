# %%
"""
Title: Pipeline.py
Description: Spanish Electricity analysis pipeline.
Author: Mario Díez Fernández
Institution: Universidad de Cantabria
Version: 1.0
Date: 2026-06-17
Python Version: 3.11
License: MIT License
"""


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy import stats
import seaborn as sns
from pathlib import Path

# %%
def widget_generation_mix(data_mix, output_html='Output/generation_mix.html'):

    # Convert the Hour column to datetime format
    data_mix['Hour'] = pd.to_datetime(data_mix['Hour'])

    # Create a string version of the hour for display on the x-axis (HH:MM)
    data_mix['Hour_str'] = data_mix['Hour'].dt.strftime('%H:%M')

    # Get all available days and select the first one by default
    days = sorted(data_mix['Day'].unique())
    selected_day = days[0]

    # Get generation source columns
    # Excludes 'Day', 'Hour', and the last two columns ('Hour' and 'Hour_str')
    sources = data_mix.columns[1:-2].tolist()

    # Filter data for the selected day
    df_day = data_mix[data_mix['Day'] == selected_day]

    # Create the Plotly figure
    fig = go.Figure()

    # Add the initial trace using the first generation source
    fig.add_trace(
        go.Scatter(
            x=df_day['Hour_str'],
            y=df_day[sources[0]],
            mode='lines',
            name=sources[0]
        )
    )

    # Create dropdown menu options for generation sources
    buttons = []

    for source in sources:

        buttons.append(
            dict(
                label=source,          # Displayed option name
                method='update',       # Update the existing figure
                args=[
                    {
                        'y': [df_day[source]],  # Update y-axis data
                        'name': [source]        # Update trace name
                    },
                    {
                        'title': f'{selected_day} - Generation Mix'
                    }
                ]
            )
        )

    # Configure chart layout
    fig.update_layout(
        title=f'{selected_day} - Generation Mix',
        xaxis_title='Hour',
        yaxis_title='Power (MW)',
        width=900,
        height=500,
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',  # Dropdown menu orientation
                showactive=True,   # Highlight selected option
                x=0.1,
                y=1.15
            )
        ]
    )

    # Save the figure as an interactive HTML file
    fig.write_html(output_html)

    # Return nothing (prevents automatic display in Jupyter notebooks)
    return None


def compare_demand(day_selected, df_demand, root_output='Output'):

    # Convert Hour columns to datetime format
    day_selected['Hour'] = pd.to_datetime(day_selected['Hour'])
    df_demand['Hour'] = pd.to_datetime(df_demand['Hour'])

    # Create a decimal representation of the hour
    # Example: 14:30 becomes 14.5
    day_selected['Hour_decimal'] = (
        day_selected['Hour'].dt.hour +
        day_selected['Hour'].dt.minute / 60
    )

    df_demand['Hour_decimal'] = (
        df_demand['Hour'].dt.hour +
        df_demand['Hour'].dt.minute / 60
    )

    # Demand variables to compare
    variables = ['Real', 'Predicted', 'Programmed']

    # Format the selected date for plot labels
    selected_date = pd.to_datetime(
        day_selected['Day'].iloc[0]
    ).strftime('%d/%m/%Y')


    # 1. NORMALIZED DEMAND

    # Create one subplot per demand variable
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, var in zip(axes, variables):

        # Plot all historical days with high transparency
        # This creates the background reference distribution
        for day in df_demand['Day'].unique():

            df_day_demand = df_demand[df_demand['Day'] == day]

            ax.plot(
                df_day_demand['Hour_decimal'],
                df_day_demand[var],
                alpha=0.1
            )

        # Highlight the selected day in red
        ax.plot(
            day_selected['Hour_decimal'],
            day_selected[var],
            color='red',
            linewidth=2,
            label=selected_date
        )

        ax.set_title(f'{var} Demand (MW)')
        ax.set_ylabel('MW')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Hour of day')

    plt.tight_layout()

    # Save the normalized demand chart as a PNG file
    plt.savefig(
        root_output + '/demand_normalized.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # 2. Z-SCORE

    # Create one subplot per demand variable
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, var in zip(axes, variables):

        # Calculate mean and standard deviation for the variable
        mean = df_demand[var].mean()
        std = df_demand[var].std()

        # Plot the z-score of each historical day
        for day in df_demand['Day'].unique():

            df_day_demand = df_demand[df_demand['Day'] == day]

            z = (df_day_demand[var] - mean) / std

            ax.plot(
                df_day_demand['Hour_decimal'],
                z,
                alpha=0.1
            )

        # Calculate and plot the z-score of the selected day
        z_selected = (day_selected[var] - mean) / std

        ax.plot(
            day_selected['Hour_decimal'],
            z_selected,
            color='red',
            linewidth=2,
            label=selected_date
        )

        # Add a horizontal reference line at z-score = 0
        ax.axhline(
            0,
            color='black',
            linestyle='--',
            linewidth=1
        )

        ax.set_title(f'{var} (Z-score)')
        ax.set_ylabel('Z-score')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Hour of day')

    plt.tight_layout()

    # Save the z-score chart as a PNG file
    plt.savefig(
        root_output + '/demand_z-score.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # 3. P10-P90

    # Create one subplot per demand variable
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for ax, var in zip(axes, variables):

        # Calculate the average demand by hour of day
        mean = df_demand.groupby('Hour_decimal')[var].mean()

        # Calculate the 10th percentile by hour of day
        p10 = df_demand.groupby('Hour_decimal')[var].quantile(0.1)

        # Calculate the 90th percentile by hour of day
        p90 = df_demand.groupby('Hour_decimal')[var].quantile(0.9)

        # Fill the area between P10 and P90
        # This shows the normal demand range for each hour
        ax.fill_between(
            p10.index,
            p10.values,
            p90.values,
            alpha=0.3,
            label='10-90 percentile'
        )

        # Plot the hourly mean demand
        ax.plot(
            mean.index,
            mean.values,
            color='black',
            linewidth=2,
            label='Mean'
        )

        # Highlight the selected day in red
        ax.plot(
            day_selected['Hour_decimal'],
            day_selected[var],
            color='red',
            linewidth=2,
            label=selected_date
        )

        ax.set_title(f'{var} Demand (MW)')
        ax.set_ylabel('MW')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Hour of day')

    plt.tight_layout()

    # Save the P10-P90 chart as a PNG file
    plt.savefig(
        root_output + '/demand_p10-90.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # Return nothing
    return None

def RoCoF(day_selected, root_output='Output'):

    # Create a copy to avoid modifying the original DataFrame
    day_selected = day_selected.copy()

    # Rename columns to meaningful names
    day_selected.columns = [
        "Hour",
        "Frequency min",
        "Frequency max",
        "Frequency average"
    ]

    # =========================================================
    # DATA CLEANING AND TIME CONVERSION
    # =========================================================

    # Convert the Hour column to string and remove extra spaces
    day_selected["Hour"] = day_selected["Hour"].astype(str).str.strip()

    # Replace the last colon before milliseconds with a dot
    # Example:
    # 28/04/2025 12:34:56:789 → 28/04/2025 12:34:56.789
    day_selected["Hour"] = day_selected["Hour"].str.replace(
        r"(\d{2}:\d{2}:\d{2}):(\d+)$",
        r"\1.\2",
        regex=True
    )

    # Convert the cleaned timestamp string to datetime format
    day_selected["Hour"] = pd.to_datetime(
        day_selected["Hour"],
        format="%d/%m/%Y %H:%M:%S.%f"
    )

    # Round timestamps down to the nearest second
    day_selected['Hour_sec'] = day_selected['Hour'].dt.floor('s')

    # =========================================================
    # FREQUENCY AGGREGATION
    # =========================================================

    # Calculate the average frequency for each second
    day_selected_sec = (
        day_selected
        .groupby('Hour_sec')['Frequency average']
        .mean()
        .reset_index()
        .sort_values('Hour_sec')
    )

    # Remove the last 20 samples
    # (typically done to avoid incomplete or noisy data at the end)
    day_selected_sec = day_selected_sec.iloc[:-20].copy()

    # =========================================================
    # ROCOF CALCULATION
    # =========================================================

    # Calculate time difference between consecutive samples (seconds)
    day_selected_sec['dt'] = (
        day_selected_sec['Hour_sec']
        .diff()
        .dt.total_seconds()
    )

    # Calculate frequency difference between consecutive samples
    day_selected_sec['df'] = (
        day_selected_sec['Frequency average']
        .diff()
    )

    # Calculate Rate of Change of Frequency (RoCoF)
    # RoCoF = ΔFrequency / ΔTime
    day_selected_sec['RoCoF'] = (
        day_selected_sec['df'] /
        day_selected_sec['dt']
    )

    # PLOT ROCOF


    fig = go.Figure()

    # Add RoCoF time series
    fig.add_trace(
        go.Scatter(
            x=day_selected_sec['Hour_sec'],
            y=day_selected_sec['RoCoF'],
            mode='lines',
            name='RoCoF (Hz/s)'
        )
    )

    # Add reference line at zero
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='black'
    )

    # Configure chart layout
    fig.update_layout(
        title='RoCoF',
        xaxis_title='Time',
        yaxis_title='RoCoF (Hz/s)',
        width=900,
        height=400,
        template='plotly_white'
    )

    # Save the interactive chart as an HTML file
    fig.write_html(root_output + '/RoCoF.html')

    # Return nothing (prevents automatic display in Jupyter notebooks)
    return None

def metric(day_selected, root_output='Output'):

    # PREPARATION


    # Convert the Hour column to datetime format
    day_selected['Hour'] = pd.to_datetime(day_selected['Hour'])

    # Sort the data by time to ensure correct sequential differences
    day_selected = day_selected.sort_values('Hour')

    # Use maximum frequency and remove missing values
    df_var = day_selected[['Hour', 'Frequency max']].dropna().copy()


    # ROW-TO-ROW DIFFERENCE

    # Calculate the difference in maximum frequency between consecutive rows
    df_var['diff'] = df_var['Frequency max'].diff()

    # Take the absolute value of the frequency difference
    df_var['abs_diff'] = df_var['diff'].abs()


    # GROUP INTO 20-SECOND WINDOWS

    # Resample the absolute differences into 20-second intervals
    # and calculate the accumulated variation in each window
    df_var_20s = (
        df_var
        .set_index('Hour')
        .resample('20S')['abs_diff']
        .sum()
        .reset_index()
    )

    # VISUALIZATION

    plt.figure(figsize=(12, 5))

    # Plot accumulated frequency variation over time
    plt.plot(
        df_var_20s['Hour'],
        df_var_20s['abs_diff']
    )

    plt.title("Frequency variation (20-second windows)")
    plt.xlabel("Time")
    plt.ylabel("Accumulated variation")

    plt.tight_layout()

    # Save the chart as a PNG file
    plt.savefig(
        root_output + '/frequency-variation-20sec-windows.png',
        dpi=300,
        bbox_inches='tight'
    )

    # Close the figure to prevent automatic display
    plt.close()

    # Return nothing
    return None

def frequency(frequency_data,root_output='Output'):

    # Rename columns
    frequency_data.columns = [
        'Hour',
        'Frequency min',
        'Frequency max',
        'Frequency average'
    ]

    # Replace last ":" before microseconds with "."
    frequency_data['Hour'] = frequency_data['Hour'].str.replace(
        r'(\d{2}:\d{2}:\d{2}):(\d+)$',
        r'\1.\2',
        regex=True
    )

    # Convert to datetime
    frequency_data['Hour'] = pd.to_datetime(
        frequency_data['Hour'],
        format='%d/%m/%Y %H:%M:%S.%f'
    )

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frequency_data['Hour'],
        y=frequency_data['Frequency average'],
        mode='lines',
        name='ETSIIT',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Frequency Comparison',
        xaxis_title='Time',
        yaxis_title='Frequency (Hz)',
        width=900,
        height=500,
        template='plotly_white'
    )
    fig.write_html(root_output + '/frequency_per_hour.html')
    return None

def demand_day(demand_data, root_output='Output'):

    fig = go.Figure()

    # Plot actual demand
    fig.add_trace(
        go.Scatter(
            x=demand_data['Hour'],
            y=demand_data['Real'],
            mode='lines',
            name='Real',
            line=dict(color='blue')
        )
    )

    # Plot forecasted demand
    fig.add_trace(
        go.Scatter(
            x=demand_data['Hour'],
            y=demand_data['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='red')
        )
    )

    # Plot scheduled/programmed demand
    fig.add_trace(
        go.Scatter(
            x=demand_data['Hour'],
            y=demand_data['Programmed'],
            mode='lines',
            name='Programmed',
            line=dict(color='green')
        )
    )


    # LAYOUT CONFIGURATION


    fig.update_layout(
        title='Demand per Day',
        xaxis_title='Hour',
        yaxis_title='Demand (MW)',
        width=900,
        height=500,
        template='plotly_white'
    )


    # EXPORT HTML


    # Save the interactive chart as an HTML file
    fig.write_html(
        root_output + '/demand_per_day.html'
    )

    # Return nothing
    return None

def renewables_total(data_mix,mix_day,root_output='Output'):
    # Separating renewable and non-renewable generation for analysis
    renewables = [
        'Wind', 'Solar Photovoltaic', 'Solar Thermal',
        'Hydropower', 'Renewable Thermal'
    ]

    non_renewables = [
        'Nuclear', 'Coal', 'Combined Cycle',
        'Diesel Engines', 'Gas Turbine', 'Steam Turbine'
    ]

    data_mix['Renewables'] = data_mix[renewables].sum(axis=1)
    data_mix['Non_Renewables'] = data_mix[non_renewables].sum(axis=1)

    mix_day['Renewables'] = mix_day[renewables].sum(axis=1)
    mix_day['Non_Renewables'] = mix_day[non_renewables].sum(axis=1)
    variables = ['Renewables', 'Non_Renewables']

    mix_day['Hour_decimal'] = mix_day['Hour'].dt.hour + mix_day['Hour'].dt.minute / 60
    data_mix['Hour_decimal'] = data_mix['Hour'].dt.hour + data_mix['Hour'].dt.minute / 60

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for ax, var in zip(axes, variables):

        mean = data_mix.groupby('Hour_decimal')[var].mean()
        p10 = data_mix.groupby('Hour_decimal')[var].quantile(0.1)
        p90 = data_mix.groupby('Hour_decimal')[var].quantile(0.9)

        ax.fill_between(p10.index, p10.values, p90.values, alpha=0.3, label='10-90 percentile')
        ax.plot(mean.index, mean.values, color='black', label='Mean')

        ax.plot(
            mix_day['Hour_decimal'],
            mix_day[var],
            color='red',
            linewidth=2,
            label='Blackout day'
        )

        ax.set_title(var)
        ax.set_ylabel('MW')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Hour of day')
    plt.tight_layout()
    plt.savefig(
        root_output + '/renewables_non-renewables.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    return None


def renewables_pct(data_mix, mix_day, root_output='Output'):
    # CALCULATE RENEWABLE SHARE

    # Calculate total generation for all historical data
    data_mix['Total'] = (
        data_mix['Renewables'] +
        data_mix['Non_Renewables']
    )

    # Calculate renewable generation share
    data_mix['Renewable_share'] = (
        data_mix['Renewables'] /
        data_mix['Total']
    )

    # Calculate total generation for the selected day
    mix_day['Total'] = (
        mix_day['Renewables'] +
        mix_day['Non_Renewables']
    )

    # Calculate renewable generation share for the selected day
    mix_day['Renewable_share'] = (
        mix_day['Renewables'] /
        mix_day['Total']
    )

    # CREATE FIGURE

    fig, ax = plt.subplots(figsize=(8, 4))

    # Calculate hourly statistics from historical data

    # Mean renewable share for each hour of the day
    mean = (
        data_mix
        .groupby('Hour_decimal')['Renewable_share']
        .mean()
    )

    # 10th percentile renewable share
    p10 = (
        data_mix
        .groupby('Hour_decimal')['Renewable_share']
        .quantile(0.1)
    )

    # 90th percentile renewable share
    p90 = (
        data_mix
        .groupby('Hour_decimal')['Renewable_share']
        .quantile(0.9)
    )

    # PLOT HISTORICAL RANGE

    # Shade the area between the 10th and 90th percentiles
    ax.fill_between(
        p10.index,
        p10.values,
        p90.values,
        alpha=0.3,
        label='10-90 percentile'
    )

    # Plot the historical mean
    ax.plot(
        mean.index,
        mean.values,
        color='black',
        label='Mean'
    )


    # PLOT SELECTED DAY

    # Highlight the selected day in red
    ax.plot(
        mix_day['Hour_decimal'],
        mix_day['Renewable_share'],
        color='red',
        linewidth=2,
        label='Selected day'
    )

    # CHART FORMATTING

    ax.set_title('Renewable Share')
    ax.set_ylabel('Fraction')
    ax.set_xlabel('Hour of day')
    ax.grid(True)
    ax.legend()


    # SAVE FIGURE


    plt.savefig(
        root_output + '/renewable_share.png',
        dpi=300,
        bbox_inches='tight'
    )

    # Close the figure to prevent automatic display
    plt.close()

    # Return nothing
    return None

def demand_percentiles(data_mix, blackout_day, target_hour=12.5,root_output='Output'):
    data_mix['Day'] = data_mix['Hour'].dt.date
    # Copy dataframe
    df = data_mix.copy()

    # Drop unnecessary columns if they exist
    df = df.drop(columns=['Day', 'Hour_str'], errors='ignore')

    # Separate normal and blackout days
    df_normal = data_mix[data_mix['Day'] != blackout_day].copy()
    df_blackout = data_mix[data_mix['Day'] == blackout_day].copy()

    # Find closest blackout row to target hour
    row_blackout = df_blackout.iloc[
        (df_blackout['Hour_decimal'] - target_hour).abs().argsort()[:1]
    ]

    # Energy source columns
    sources = [
        col for col in df.columns
        if col not in ['Hour_decimal']
    ]

    # Store results
    results = []

    for source in sources:

        # Historical distribution around target hour
        df_hour = df_normal.iloc[
            (df_normal['Hour_decimal'] - target_hour).abs().argsort()[:len(df_normal)//96]
        ]

        # Blackout value
        value_blackout = row_blackout[source].values[0]

        # Percentile
        percentile = stats.percentileofscore(
            df_hour[source].dropna(),
            value_blackout
        )

        results.append({
            'Source': source,
            'Value_blackout': value_blackout,
            'Percentile': percentile
        })

    # Final dataframe
    df_percentiles = pd.DataFrame(results).sort_values(
        by='Percentile',
        ascending=False
    )

    # Filter only share variables
    df_share = df_percentiles[
        df_percentiles['Source'].str.contains('share', case=False, na=False)
    ]

    return df_percentiles, df_share

def demand_percentiles(data_mix, blackout_day, target_hour=12.5,root_output='Output'):
    # Copy dataframe
    df = data_mix.copy()

    # Convert Day column to datetime
    df['Day'] = pd.to_datetime(df['Day'])

    # Convert blackout_day to datetime
    blackout_day = pd.to_datetime(blackout_day)

    # Remove unnecessary columns
    df_clean = df.drop(columns=['Day', 'Hour_str'], errors='ignore')

    # Separate normal and blackout days
    df_normal = df[df['Day'] != blackout_day].copy()
    df_blackout = df[df['Day'] == blackout_day].copy()

    # Safety check
    if df_blackout.empty:
        raise ValueError(
            f"No data found for blackout_day = {blackout_day}"
        )

    # Closest blackout row to target hour
    row_blackout = df_blackout.iloc[
        (df_blackout['Hour_decimal'] - target_hour).abs().argsort()[:1]
    ]

    # Source columns
    sources = [
        col for col in df_clean.columns
        if col != 'Hour_decimal'
    ]

    results = []

    for source in sources:

        # Historical distribution near target hour
        df_hour = df_normal.iloc[
            (df_normal['Hour_decimal'] - target_hour)
            .abs()
            .argsort()[:len(df_normal)//96]
        ]

        # Skip empty values
        if row_blackout[source].empty:
            continue

        value_blackout = row_blackout[source].values[0]

        percentile = stats.percentileofscore(
            df_hour[source].dropna(),
            value_blackout
        )

        results.append({
            'Source': source,
            'Value_blackout': value_blackout,
            'Percentile': percentile
        })

    # Create dataframe
    df_percentiles = pd.DataFrame(results)


        # Share dataframe
    df_share = df_percentiles[
        df_percentiles['Source']
        .str.contains('share', case=False, na=False)
    ].copy()

    # Add non_renewable_share row
    renewable_row = df_percentiles[
        df_percentiles['Source'] == 'Renewable_share'
    ]

    if not renewable_row.empty:

        renewable_value = renewable_row['Value_blackout'].values[0]
        renewable_percentile = renewable_row['Percentile'].values[0]

        non_renewable_row = pd.DataFrame([{
            'Source': 'non_renewable_share',
            'Value_blackout': 1 - renewable_value,
            'Percentile': 1 - renewable_percentile
        }])

        df_share = pd.concat(
            [df_share, non_renewable_row],
            ignore_index=True
        )
    # Remove unwanted rows
    df_percentiles = df_percentiles[
        ~df_percentiles['Source'].isin([
            'Renewable_share',
            'Hour'
        ])
    ]

    # Sort
    df_percentiles = df_percentiles.sort_values(
        by='Percentile',
        ascending=False
    )


    df_percentiles.to_csv(root_output + '/percentiles.csv', index=False)
    df_share.to_csv(root_output + '/percentiles_share.csv', index=False)
    return None

def renewable_share_percentile(data_mix, blackout_day, root_output='Output'):
    data_mix['Day'] = data_mix['Hour'].dt.date
    blackout_day = pd.to_datetime(blackout_day)
    # Copy dataframe
    df = data_mix.copy()

    # Datetime conversion
    df['Day'] = pd.to_datetime(df['Day'])
    blackout_day = pd.to_datetime(blackout_day)

    # Split datasets
    df_blackout = df[df['Day'] == blackout_day].copy()
    df_normal = df[df['Day'] != blackout_day].copy()

    # Safety check
    if df_blackout.empty:
        raise ValueError(
            f'No data found for blackout day: {blackout_day}'
        )

    percentiles = []

    # Iterate over blackout rows
    for _, row in df_blackout.iterrows():

        target_hour = row['Hour_decimal']

        # Historical distribution at same hour
        df_hour = df_normal.iloc[
            (df_normal['Hour_decimal'] - target_hour)
            .abs()
            .argsort()[:len(df_normal)//96]
        ]

        # Blackout renewable share
        blackout_value = row['Renewable_share']

        # Percentile
        percentile = stats.percentileofscore(
            df_hour['Renewable_share'].dropna(),
            blackout_value
        )

        percentiles.append({
            'Hour_decimal': target_hour,
            'Percentile': percentile
        })

    # Result dataframe
    df_percentile = pd.DataFrame(percentiles)

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_percentile['Hour_decimal'],
        y=df_percentile['Percentile'],
        mode='lines',
        name='Renewable Share Percentile',
        line=dict(color='green')
    ))

    # Layout
    fig.update_layout(
        title='Renewable Share Percentile During Blackout Day',
        xaxis_title='Hour',
        yaxis_title='Percentile',
        width=900,
        height=500,
        template='plotly_white'
    )

    # Export HTML
    fig.write_html(
        "renewable_share_percentile_blackout.html"
    )

    # Show figure

    fig.write_html(root_output + '/renewable_share_percentile.html')
    return df_percentile



def create_folder_structure(test_data, base_dir="."):
    """
    Creates the following directory structure:

    Output/
        <central_day_value>/
            Own/
            Statistics/
            Comparisons/
                Mix/
                Demand/

    Parameters
    ----------
    test_data : pandas.DataFrame
        DataFrame containing a 'Day' column.

    base_dir : str or Path, optional
        Base directory where the Output folder will be created.
        Default is the current working directory.

    Returns
    -------
    pathlib.Path
        Path to the created base folder.
    """

    # =========================
    # GET THE CENTRAL DAY VALUE
    # =========================

    # Select the middle value from the Day column
    central_value = test_data["Day"].iloc[len(test_data) // 2]

    # Convert to string in case it is a date, number, etc.
    folder_name = str(central_value)

    # =========================
    # BUILD BASE PATH
    # =========================

    base_path = (
        Path(base_dir) /
        "Output" /
        folder_name
    )

    # =========================
    # CREATE DIRECTORY STRUCTURE
    # =========================

    # Create:
    # Output/<day>/Own
    (base_path / "Own").mkdir(
        parents=True,
        exist_ok=True
    )

    # Create:
    # Output/<day>/Statistics
    (base_path / "Statistics").mkdir(
        parents=True,
        exist_ok=True
    )

    # Create:
    # Output/<day>/Comparisons/Mix
    (base_path / "Comparisons" / "Mix").mkdir(
        parents=True,
        exist_ok=True
    )

    # Create:
    # Output/<day>/Comparisons/Demand
    (base_path / "Comparisons" / "Demand").mkdir(
        parents=True,
        exist_ok=True
    )

    # Return the root folder path
    return base_path

# %%
# IMPORT DATASETS

# Load demand dataset
demand = pd.read_csv('Data/2_Demanda_2024_01_01_2025_06_30.csv')

# Translate column names to English
demand.columns = ['Hour', 'Real', 'Predicted', 'Programmed']


# Load generation mix dataset
data_mix = pd.read_csv('Data/2_Generacion_2024_01_01_2025_09_30.csv')

# Translate column names to English
data_mix.columns = [
    'Hour',
    'Wind',
    'Nuclear',
    'Coal',
    'Combined Cycle',
    'International Exchanges',
    'Solar Photovoltaic',
    'Solar Thermal',
    'Renewable Thermal',
    'Diesel Engines',
    'Gas Turbine',
    'Steam Turbine',
    'Auxiliary Generation',
    'Cogeneration and Waste',
    'Export to Andorra',
    'Export to Morocco',
    'Export to Portugal',
    'Export to France',
    'Import from France',
    'Import from Portugal',
    'Import from Morocco',
    'Import from Andorra',
    'Hydropower'
]


# Load frequency datasets
esquileo = pd.read_csv(
    'Data/freq_black_02_COMP_SET ESQUILEO_0bdc_28_04_25_00_00_00_High resolution parameters Cycle.csv',
    sep=';'
)

esquileo.columns = [
    'Hour',
    'Frequency min',
    'Frequency max',
    'Frequency average'
]


etsiit = pd.read_csv(
    'Data/freq_black_02_COMP_ETSIIT_S255_0881_28_04_25_00_00_00_High resolution parameters Cycle.csv',
    sep=';'
)

etsiit.columns = [
    'Hour',
    'Frequency min',
    'Frequency max',
    'Frequency average'
]



# DATE AND TIME CLEANING

# Convert Hour columns to datetime format
demand['Hour'] = pd.to_datetime(demand['Hour'])
data_mix['Hour'] = pd.to_datetime(data_mix['Hour'])

# The frequency datasets contain milliseconds separated by a colon,
# which is not a standard datetime format.


etsiit['Hour'] = etsiit['Hour'].str.replace(
    r'(\d{2}:\d{2}:\d{2}):(\d+)$',
    r'\1.\2',
    regex=True
)

etsiit['Hour'] = pd.to_datetime(
    etsiit['Hour'],
    format='%d/%m/%Y %H:%M:%S.%f'
)

esquileo['Hour'] = esquileo['Hour'].str.replace(
    r'(\d{2}:\d{2}:\d{2}):(\d+)$',
    r'\1.\2',
    regex=True
)

esquileo['Hour'] = pd.to_datetime(
    esquileo['Hour'],
    format='%d/%m/%Y %H:%M:%S.%f'
)

# REMOVE MISSING FREQUENCY VALUES

# Drop rows with missing frequency values to avoid errors in the analysis
etsiit = etsiit.dropna(
    subset=[
        'Frequency min',
        'Frequency max',
        'Frequency average'
    ]
)

esquileo = esquileo.dropna(
    subset=[
        'Frequency min',
        'Frequency max',
        'Frequency average'
    ]
)

# DAILY DATA PREPARATION


# Extract the date from the demand dataset
demand['Day'] = demand['Hour'].dt.date

# Load selected generation mix day
data_prueba = pd.read_csv('Input/data_mix.csv')



# CREATE OUTPUT FOLDER STRUCTURE


# Create the main output folder structure
created_path = create_folder_structure(data_prueba)

# Get the most common day in the selected data
most_common_day = str(data_prueba["Day"].mode().iloc[0])

# Define base output path for the selected day
base_output = Path("Output") / most_common_day

# Define output subfolders
own_dir = base_output / "Own"
statistics_dir = base_output / "Statistics"
comparisons_dir = base_output / "Comparisons"
mix_dir = comparisons_dir / "Mix"
demand_dir = comparisons_dir / "Demand"

# Ensure all required folders exist
for folder in [own_dir, statistics_dir, mix_dir, demand_dir]:
    folder.mkdir(parents=True, exist_ok=True)


# OWN DAY ANALYSIS

# Generate interactive generation mix chart
widget_generation_mix(
    data_prueba,
    output_html=str(own_dir / "generation_mix.html")
)

# Load selected demand day
demand_prueba = pd.read_csv('Input/demand.csv')

# Compare selected demand day against historical demand
compare_demand(
    day_selected=demand_prueba,
    df_demand=demand,
    root_output=str(demand_dir)
)

# Calculate and export RoCoF chart
RoCoF(
    day_selected=pd.read_csv('Input/frequency.csv', sep=';'),
    root_output=str(own_dir)
)

# Calculate frequency variation metric
metric(
    etsiit,
    root_output=str(own_dir)
)

# Generate frequency chart
frequency(
    pd.read_csv('Input/frequency.csv', sep=';'),
    root_output=str(own_dir)
)

# Generate demand chart for the selected day
demand_day(
    demand_prueba,
    root_output=str(own_dir)
)



# GENERATION MIX COMPARISONS


# Compare total renewable and non-renewable generation
renewables_total(
    data_mix,
    data_prueba,
    root_output=str(mix_dir)
)

# Compare renewable share against historical values
renewables_pct(
    data_mix,
    data_prueba,
    root_output=str(mix_dir)
)



# STATISTICAL ANALYSIS


# Calculate renewable share percentile for the blackout day
df_percentile = renewable_share_percentile(
    data_mix,
    blackout_day='2025-04-28',
    root_output=str(statistics_dir)
)

# Calculate demand percentiles for a user-selected hour
demand_percentiles(
    data_mix,
    '2025-04-28',
    target_hour=float(input("Hour to analyze: ")),
    root_output=str(statistics_dir)
)


