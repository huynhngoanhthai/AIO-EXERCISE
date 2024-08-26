import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = 'data/opsd_germany_daily.csv'
# opsd_daily = pd.read_csv(dataset_path)

# # print(opsd_daily.shape)

# # print(opsd_daily.dtypes)

# # print(opsd_daily.head(3))

# opsd_daily = opsd_daily.set_index('Date')
# opsd_daily.head(3)

# print(opsd_daily)


opsd_daily = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

# Add columns with year, month, and weekday name
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.day_name()

# Display a random sampling of 5 rows
data = opsd_daily.sample(5, random_state=0)

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})

# Plot the 'Consumption' column with a linewidth of 0.5
opsd_daily['Consumption'].plot(linewidth=0.5)

opsd_daily['Consumption'].plot(linewidth=0.5)

# Save the figure to a file


# Define the columns to plot
cols_plot = ['Consumption', 'Solar', 'Wind']

# Plot each column in separate subplots
axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None',
                                  figsize=(11, 9), subplots=True)

# Set ylabel for each subplot
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')


fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# Plot a boxplot for each column ('Consumption', 'Solar', 'Wind')
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
    sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)

    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')


# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind + Solar']

# Resample the data to weekly frequency and calculate the mean for each week
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()

# Display the first 3 rows of the resampled data
opsd_weekly_mean.head(3)

opsd_7d = opsd_daily[data_columns]. rolling(7, center=True) . mean()
opsd_7d . head(10)

# Compute the annual sums, setting values to NaN for any year with fewer than 360 days of data
opsd_annual = opsd_daily[data_columns].resample('Y').sum(min_count=360)

# Set the index to the year component for easier handling
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'


# Display the last 3 rows of the DataFrame
opsd_annual.tail(3)

# Create a bar plot for 'Wind + Solar / Consumption' from 2012 onward
ax = opsd_annual.loc[2012:, 'Wind + Solar / Consumption'].plot.bar(color='C0')

# Set the y-axis label
ax.set_ylabel('Fraction')

# Set the y-axis limits
ax.set_ylim(0, 0.3)

# Set the plot title
ax.set_title('Wind + Solar Share of Annual Electricity Consumption')

# Rotate the x-axis labels to 0 degrees
plt.xticks(rotation=0)


# Compute the 365-day rolling mean, handling missing days
opsd_365d = opsd_daily[data_columns].rolling(
    window=365, center=True, min_periods=360).mean()

# Plot daily, 7-day rolling mean, and 365-day rolling mean time series
fig, ax = plt.subplots()

# Plot daily data
ax.plot(opsd_daily['Consumption'], marker='.', markersize=2,
        color='0.6', linestyle='None', label='Daily')

# Plot 7-day rolling mean (assumed to be defined previously as opsd_7d)
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-day Rolling Mean')

# Plot 365-day rolling mean
ax.plot(opsd_365d['Consumption'], color='0.2',
        linewidth=3, label='Trend (365-day Rolling Mean)')

# Set x-ticks to yearly intervals and add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')


plt.savefig('consumption_plot.png')

# Show the plot
plt.show()
