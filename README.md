Streamsync

Create a new Streamsync project that we will use to follow up on the first three parts of the project work.
Pages (we will fill the contents gradually):
Let the first page be reserved for BarentsWatch plots for the collection of all locations (/v1/geodata/fishhealth/locality/{year}/{week}).
Let the second page be reserved for BarentsWatch plots for a single location (/v1/geodata/fishhealth/locality/{localityId}/{year}/{week}).
Let the third page be a data description and instruction page.
A graphical depiction of the three pages is now available on pdf file.
Contents for the first page:
A map of Norway with all the aquaculture sites of a chosen year (sites come and go) plotted using Plotly Express' scatter mapbox.
When a site is clicked on, the site should be saved in the state dictionary and a message box should update to show the site name.
A slider should be used to select a year.
A dropdown box should select a column from the weekly data of the chosen year.
The pie chart of PD from the first compulsory should be adapted and included.
A histogram showing the data of the chosen column.
Content for the second page:
A lice type selector (dropdown menu).
A plot of the chosen lice as a function of weeks.
Include a threshold line (threshold chosen by the student).
When a value passes the threshold line at any timepoint in the chosen year/location some graphical symbol should appear to warn.
A weather type selector (dropdown menu) according to the second compulsory.
A plot of the chosen weather as a function of weeks.
For weather stations not containing the selected weather type, the plot should show a text telling the user that data is not available.
Bonus: Let the choices in the selector be dependent on available data.
An ARIMAX model with chosen lice type as autoregressive element and chosen weather type as exogenous variable.
Show a table of coefficients and P-values (at least).
Bonus: Plot the one-step-ahead predictions for the chosen year.
Content for the third page:
Short description of how things are connected behind the scenes.

Data acquisition

Revisiting the BarentsWatch API data download.

In the first part of the project work, you downloaded data for one year for all locations (/v1/geodata/fishhealth/locality/{year}/{week}). Wrap this code in a function with the input parameter 'year', such that corresponding data for any year can be downloaded, formatted and injected into your database.
Bonus: Preferably, data should only be downloaded if not already in your database to reduce overhead, and the database should expand with each download.
Repeat the function wrapping procedure for the extended data for a single location (/v1/geodata/fishhealth/locality/{localityId}/{year}/{week}). Both 'year' and 'location' should be selectable.
Bonus: Same as the previous bonus, but for the chosen location.
Create a function that downloads weekly weather data from the station closest to a set of coordinates (lat and lon) similar to what we have done previously (then we had daily data). We will not mind if only a subset of the measurements specified in the second part of the project work are available from the nearest station, as we will add logic to the plots to compensate.
