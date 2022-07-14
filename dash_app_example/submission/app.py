from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import plotly.figure_factory as ff
import numpy as np

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Data:
df = pd.read_csv("songs_normalize_norm.csv").drop(["Explicit_f","Popularity_f"], axis=1)
df = df.drop(df[df["Year"] == 1998].index)
df = df.drop(df[df["Year"] == 2020].index)
df = df.drop(df[df["Genre"] == "set()"].index)

attributes = df.columns[2:-1]
attribute_without_year = list(attributes)
attribute_without_year.remove("Year")

# Scale the data:
def min_max_scale_columns(dataframe, columns):
    for col in columns:
        dataframe[col] = pd.DataFrame(MinMaxScaler().fit_transform(df[[col]].values), columns=[col])
    return
#
# min_max_scale_columns(df, attribute_without_year)

# Correlation scatter plot - arrtibutes correlation:
fig = go.Figure(data=go.Splom(
                dimensions=[dict(label=att,values=df[att]) for att in attributes],
                showlowerhalf=False,
                marker=dict(color="black",
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5)))
fig.update_layout(title='Iris Data set', width=1000, height=1000)
fig.update_yaxes(visible=False, showticklabels=False)
fig_scatter_all_attributes = px.scatter_matrix(df, dimensions=attributes, title='Iris Data set', width=1000, height=1000)

# Arrtibute by year line:
group_by_year = df.groupby(['Year']).mean().reset_index(level=0)
fig_line_arrtibute_by_year = px.line(group_by_year, x="Year", y="Popularity", template="simple_white")

# Seperate genres dataframe:
data = []
genre_dict = {'blues':'Blues', 'classical':'Classical', 'hip hop':'Hip Hop', 'pop':'Pop',
              'country':'Country', 'metal':'Metal', 'R&B':'R&B', 'rock':'Rock',
              'Dance/Electronic':'Electronic', 'easy listening':'Easy Listening',
              'Folk/Acoustic':'Folk/Acoustic', 'jazz':'Jazz','latin':'Latin', 'World/Traditional':'World'}

# Creating df_separate_genres:

df_separate_genres = pd.DataFrame(columns=df.columns)
i = 0
for index, row in df.iterrows():
    for g in row['Genre'].split(', '):
        r = row
        #if g == 'World/Traditional': print(genre_dict[g])
        r['Genre'] = genre_dict[g]
        df_separate_genres.loc[i] = r
        i += 1
#df_separate_genres.to_csv("df_separate_genres.csv")
#df_separate_genres = pd.read_csv("df_separate_genres.csv")

# Group by Genre - mean:
group_by_genre_no_year = df_separate_genres.copy(deep=True).drop(["Year"], axis=1).drop(["Artist"], axis=1).drop(["Song"], axis=1)
group_by_genre_no_year = group_by_genre_no_year.groupby(['Genre']).mean()
group_by_genre_pop = group_by_genre_no_year.query("Genre == 'Pop'").T#.drop(["Unnamed: 0"], axis=0)#.drop("Genre", axis=0).reset_index(level=0)

# Genre attributes bar
fig_genre_att = px.bar(group_by_genre_pop, template="simple_white")
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

# Genre pie:
genres_count = dict()
for index, row in df.iterrows():
    for g in row['Genre'].split(', '):
        genres_count[genre_dict[g]] = genres_count.get(genre_dict[g], 0) + 1
genres_count_df = pd.DataFrame(data={"Genre": genres_count.keys(), "Count" :genres_count.values()})
fig_pie = px.pie(genres_count_df, values='Count', names='Genre', title='Genre Distribution:', color_discrete_sequence=px.colors.qualitative.Dark24[::-1])
fig_pie.update_traces(textinfo='none')


# Scatter artists:
df_group_artists_mean = df.groupby(["Artist"]).mean()
df_group_artists_count = df.groupby(["Artist"]).count()

sotr_by_pupolarity_func = lambda a: -df_group_artists_mean["Popularity"].loc[a]*df_group_artists_count["Popularity"].loc[a]

artists = ["Rihanna"]
df_by_artist = df[df['Artist'].isin(artists)]
fig_scatter_by_artist = px.scatter(df_by_artist, x="Year", y="Valence",
                 size="Popularity", color="Artist", hover_name="Song",
                 log_x=True, size_max=60)

# # Lots of pies:
fig_pies_by_genre = make_subplots(rows=1, cols=len(group_by_genre_no_year.index), specs=[[{'type':'domain'} for i in range(len(group_by_genre_no_year.index))]], subplot_titles=group_by_genre_no_year.index)
index = 1
for g in group_by_genre_no_year.index:
    x = round(group_by_genre_no_year["Popularity"].loc[[g]][0],3)
    fig_pies_by_genre.add_trace(go.Pie(labels=None, values=[x, 1-x], name=g, marker_colors=['rgb(33, 75, 99)', 'rgb(129, 180, 179)']), 1, index)
    index += 1
fig_pies_by_genre.update_traces(hole=.4, hoverinfo="name")


# Density:
x1 = np.random.randn(200) - 1
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 1

hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']
colors = ['#333F44', '#37AA9C', '#94F3E4']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

# Add title
fig.update_layout(title_text='Curve and Rug Plot')



# App:
app.layout = \
    html.Div(children = [html.H1(id = "7", children='Spotify Hits Statistics', style={'textAlign': 'center','color': 'rgb(0,0,0)'}),
        html.H3(children='2000 Hits from 1999-2019', style={'textAlign': 'center','color': 'rgb(0,0,0)'}),
        html.Div([
            html.Div([dcc.Graph(id="2", figure = fig_pie)], style={'width': '25%', 'float': 'right'}),
            html.Div([dcc.Dropdown(attribute_without_year, "Popularity", id="3"), dcc.Graph(id="1", figure = fig_line_arrtibute_by_year)],style={'width': '30%', 'height':'10%', 'float': 'left'}),
            ], style={'padding': 10,'flex-direction': 'row'}),
        html.Div([dcc.Dropdown(list(genres_count.keys()), "Rock", id="5"), dcc.Graph(id="4", figure = fig_genre_att)], style={'display': 'flex', 'flex-direction': 'column'}),
        html.Div(children='Popularity Presense In Genre:', style={'textAlign': 'center','color': 'rgb(0,0,0)'}, id="11"),
        html.Div([dcc.Graph(id="6", figure = fig_pies_by_genre)], style={'display': 'flex', 'flex-direction': 'column'}),
        html.Div([dcc.Dropdown(sorted(list(set(df["Artist"])), key = sotr_by_pupolarity_func), ['Rihanna'], multi=True, id ="9")]),
        html.Div([dcc.Graph(id="8", figure = fig_scatter_by_artist)], style={'display': 'flex', 'flex-direction': 'column'})
])

color_bar = px.colors.qualitative.Antique
color_att = dict([(group_by_genre_pop.index[i], color_bar[i]) for i in range(len(group_by_genre_pop.index))])


@app.callback(
    Output(component_id='11', component_property='children'),
    Input(component_id="3", component_property='value')
)
def update_line(input_value):
    s = input_value+' Presenae In Genre:'
    return s

@app.callback(
    Output(component_id='1', component_property='figure'),
    Input(component_id="3", component_property='value')
)
def update_line(input_value):
    f = px.line(group_by_year, x="Year", y=input_value, template="simple_white", color_discrete_map={input_value:  color_att[input_value]})
    f.update_xaxes(tickvals=[str(i) for i in range(1999, 2020, 4)], tickangle = 90)
    f.update_layout(title = "Avrage Hits " + input_value + " Over The Years:")
    return f

@app.callback(
    Output(component_id='6', component_property='figure'),
    Input(component_id="3", component_property='value'),
    Input(component_id="5", component_property='value')
)
def update_lots_pies(input_value, input_genre):
    fig_pies_by_genre = make_subplots(rows=2, cols=7,specs=[[{'type': 'domain'} for i in range(7)],[{'type': 'domain'} for i in range(7)]],
                                      subplot_titles=group_by_genre_no_year.index)
    index = 0
    for g in group_by_genre_no_year.index:
        x = round(group_by_genre_no_year[input_value].loc[[g]][0], 3)
        if g == input_genre:
            c = "#000000"
        else:
            c = color_att[input_value]
        fig_pies_by_genre.add_trace(go.Pie(labels=["P", "N"], values=[x, 1 - x], name=None,
                                           marker_colors=[c, '#DEDEDE']), index//7+1, index%7+1)
        index += 1
    fig_pies_by_genre.update_traces(hole=.4, textinfo='none', hoverinfo="name")
    return fig_pies_by_genre



@app.callback(
    Output(component_id='4', component_property='figure'),
    Input(component_id="5", component_property='value')
)
def update_genre_bar(input_value):
    gbg = group_by_genre_no_year.query("Genre == '" + str(input_value)+"'").T#.drop(["Unnamed: 0"], axis=0)
    f = px.bar(gbg, template="simple_white", color= gbg.index, color_discrete_sequence = color_bar) #color_discrete_sequence=[(0.0, "green"), (0.125, "green"), (0.125, "red"),  (0.25, "red"), (0.125, "green"), (0.125, "green"),
                                                     #(0.125, "blue"),  (0.125, "blue"),(0.125, "blue"),  (0.125, "blue")])
    f.update_layout(yaxis_range=[0, 1], title_text=input_value+' Statistics:', showlegend=False)
    #dict(title=None, orientation=None,  yanchor="bottom", x=0.5, xanchor="center"))
    # f.update_traces(marker_color='rgb(69,139,116)', marker_line_color='rgb(8,48,107)',
    #                   marker_line_width=1.5, opacity=0.6)
    f.update_traces(marker_line_width=1.5, opacity=0.6)
    f.update_xaxes(title_text = "")
    f.update_yaxes(title_text="")
    return f

@app.callback(
    Output(component_id='8', component_property='figure'),
    Input(component_id="9", component_property='value'),
    Input(component_id="3", component_property='value')
)
def update_artists_scatter(input_value, input):
    artists = input_value
    df_by_artist = df[df['Artist'].isin(artists)].iloc[::-1]
    fig_scatter_by_artist = px.scatter(df_by_artist, x="Year", y=input,
                                       size="Popularity", color="Artist", color_discrete_sequence=px.colors.qualitative.Dark24[::-1], hover_name="Song",
                                       log_x=True, size_max=40, template="simple_white")
    fig_scatter_by_artist.update_layout(title_text="Artists Compering, height = "+input+":")
    return fig_scatter_by_artist


if __name__ == '__main__':
    app.run_server(debug=True)