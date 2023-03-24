import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

import datetime

from highlight_text import  ax_text, fig_text

from soccerplots.utils import add_image
from soccerplots.radar_chart import Radar

from mplsoccer import Pitch, VerticalPitch, PyPizza

import math

from scipy.stats import stats

import matplotlib as mpl
import matplotlib.pyplot as plt

# FONT FAMILY
# Set the Lato font file path
lato_path = '../Fonts/Lato-Black.ttf'

# Register the Lato font with Matplotlib
custom_font = FontProperties(fname=lato_path)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.header('Player Recruitment')

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def load_data(filePath):
    data = pd.read_csv(filePath)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data['Age']  = data['Age'].astype(int)

    return data

data = load_data('./Data/data.csv')

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def load_wyscout(filePath):
    wyscout = pd.read_csv(filePath)
    wyscout.drop(['Unnamed: 0'], axis=1, inplace=True)
    wyscout['Age']  = wyscout['Age'].astype(int)
    
    return wyscout

wyscout = load_wyscout('./Data/wyscout.csv')

#st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
#def loadS3_Data(fileName):
#    s3 = boto3.resource('s3')
#    S3_BUCKET_NAME = 'gpsdataceara'
#    folder_name = 'python/gps/'
#    file_name = fileName
#    key_name = f'{folder_name}{file_name}'

#    s3_object = s3.Object(S3_BUCKET_NAME, key_name)
#    body = s3_object.get()['Body']
#    wyscout = pd.read_csv(body)
#    return wyscout

#wyscout = loadS3_Data('wyscout.csv')

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def load_dataOPTA(filePath):
    return pd.read_csv(filePath)
opta = load_dataOPTA('./Data/opta.csv')

center_Back = ['Non-penalty goals/90', 'Offensive duels %', 'Progressive runs/90',
                'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
               'PAdj Interceptions', 'PAdj Sliding tackles', 'Defensive duels/90', 'Defensive duels %',
               'Aerial duels/90', 'Aerial duels won, %', 'Shots blocked/90']

full_Back = ['Aerial duels won, %', 'Touches in box/90', 'Offensive duels %', 'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
            'Passes %', 'Deep completions/90', 'Progressive passes/90', 'Key passes/90', 'Third assists/90',
             'PAdj Interceptions', 'Aerial duels/90', 'Aerial duels won, %']

defensive_Midfield  = ['xG/90', 'Shots', 'Progressive runs/90', 'Aerial duels won, %',
                       'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90','PAdj Sliding tackles',
                       'PAdj Interceptions', 'Aerial duels won, %', 'Defensive duels %', 'Offensive duels %']

Midfield  = ['xG/90', 'Shots', 'Progressive runs/90', 'Aerial duels won, %',
             'Passes %', 'Forward passes %', 'Forward passes/90', 'Progressive passes/90',
             'Key passes/90', 'Second assists/90', 'Assists', 'xA',
             'PAdj Interceptions', 'Aerial duels won, %', 'Defensive duels %']

offensive_Midfield = ['xG/90', 'Goals/90', 'Progressive runs/90', 'Aerial duels won, %',
                      'xA/90', 'Deep completions/90', 'Passes to penalty area/90',
                      'Touches in box/90', 'Key passes/90', 'Passes final 1/3 %',
                      'Passes penalty area %', 'Progressive passes/90',
                      'Succ defensive actions/90', 'PAdj Interceptions', 'Aerial duels won, %', 'Defensive duels %']

offensive_Midfield_BS = ['Aerial duels won, %', 'xA/90', 'Deep completions/90', 'Passes to penalty area/90',
                      'Key passes/90', 'Passes final 1/3 %']

Winger = ['Aerial duels won, %', 'Goals', 'xG/90',
          'xA/90', 'Touches in box/90', 'Dribbles/90', 'Passes to penalty area/90', 'Key passes/90',
          'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
          'Offensive duels/90', 'PAdj Interceptions']

Forward = ['Goals', 'xG/90', 'Shots on target, %', 'Goal conversion, %',
           'Aerial duels won, %', 'xA/90', 'Touches in box/90', 'Dribbles/90',
           'Offensive duels/90', 'PAdj Interceptions', 'Aerial duels/90', 'Aerial duels won, %']

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def bars(data, playerName, league, metrics, ax):

    Creation = ['Sight play', 'Create Chances Ability', 'KeyPass Ability', 'decisionMake']

    WithOut_Ball = ['Concentration Ability', 'Aerial Ability', 'Interception Ability', 'Tackle Ability', 'positioning Defence']

    finalThird = ['Dribbling Ability', 'Finishing Ability', 'decisionMake', 'Heading Ability']

    buildUp = ['ballPlaying Deep', 'Pass Ability', 'Sight play', 'progressiveRuns']

    setPieces = ['SetPieces Ability']

    playerDF = data.loc[data.Player == playerName].reset_index(drop=True)
    position = playerDF['Main Pos'].unique()[0]

    df = data.loc[(data.Comp == league) & (data['Minutes played'] >= 1500) & (data['Main Pos'] == position)].reset_index(drop=True)
    
    #fig.set_facecolor('#181818')
    #ax.set_facecolor('#181818')

    # Data
    x = metrics
    max_values = []

    # Iterate through the metrics
    for metric in metrics:
        # Find the maximum value of the current metric
        max_value = df[metric].max()
        # Append the maximum value to the list
        max_values.append(max_value)
        
    current_value = []
    
    player = df.loc[(df['Player'] == playerName)][metrics].reset_index(drop=True)
    player = list(player.loc[0])
    player = player[:]
    
    # Set the maximum limit of each bar using the max_values list
    for i in range(len(metrics)):
        ax.barh(x[i], max_values[i], color='#1e1e1e')
        current_value.append(player[i])
        ax.barh(x[i], current_value[i], color='#fcac14')
        ax.text(current_value[i] + 2, x[i], str(int(current_value[i])), ha='left', va='center', fontsize=17, color='#E8E8E8')

    # Add labels and title
    if metrics == Creation:
        ax_text(x=50, y=4, s='Creation', va='center', ha='center',
                size=25, color='#E8E8E8', ax=ax)
        
    elif metrics == WithOut_Ball:
        ax_text(x=50, y=5, s='With Out', va='center', ha='center',
                size=25, color='#E8E8E8', ax=ax)
   
    elif metrics == finalThird:
        ax_text(x=50, y=4, s='Final Third', va='center', ha='center',
                size=25, color='#E8E8E8', ax=ax)
   
    elif metrics == buildUp:
        ax_text(x=50, y=4, s='BuildUp', va='center', ha='center',
                size=25, color='#E8E8E8', ax=ax)

    elif metrics == setPieces:
        ax_text(x=50, y=4, s='Set Pieces', va='center', ha='center',
                size=25, color='#E8E8E8', ax=ax)

    ax.tick_params(axis='both', colors='#E8E8E8')
    for tick in ax.get_xticklabels():
        tick.set_color('#E8E8E8')
    for tick in ax.get_yticklabels():
        tick.set_color('#E8E8E8')

    ax.spines['bottom'].set_color('#E8E8E8')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#E8E8E8')
    ax.spines['right'].set_visible(False)
 
    # Show chart 
    #return plt.show()

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def barsAbility(df, playerName):

    Creation = ['Sight play', 'Create Chances Ability', 'KeyPass Ability', 'decisionMake']

    WithOut_Ball = ['Concentration Ability', 'Aerial Ability', 'Interception Ability', 'Tackle Ability', 'positioning Defence']

    finalThird = ['Dribbling Ability', 'Finishing Ability', 'decisionMake', 'Heading Ability']

    buildUp = ['ballPlaying Deep', 'Pass Ability', 'Sight play', 'progressiveRuns']

    setPieces = ['SetPieces Ability']

    metrics_list = [Creation, WithOut_Ball, finalThird, buildUp]

    dataDash = df.loc[df.Player == playerName].reset_index(drop=True)
    
    league = dataDash.Comp.unique()[0]
    
    #club = dataDash.Team.unique()
    #club = club[0]
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.8)
    fig.set_facecolor('#181818')
    axs = axs.ravel()

    for i, metrics in enumerate(metrics_list):
        ax = axs[i]
        ax.set_facecolor('#181818')
        bars(df, playerName, league, metrics, ax)

    #fig = add_image(image='../Images/Players/' + league + '/' + club + '/' + playerName + '.png', fig=fig, left=-0.001, bottom=0.85, width=0.08, height=0.23)

    #fig = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Country/' + country + '.png', fig=fig, left=0.08, bottom=0.775, width=0.1, height=0.07)

    return plt.show()

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def Scatters(data, playerName, mode=None):
    
    colorScatter = '#3D3D3D'
    if mode == None:
        color = '#E8E8E8'
        background = '#181818'
    elif mode != None:
        color = '#181818'
        background = '#E8E8E8'
        
    player = data.loc[data.Player == playerName].reset_index(drop=True)
    position = player['Main Pos'].unique()[0]

    df = data.loc[(data.Tier >= 1) & (data['Minutes played'] >= 2000) & (data['Main Pos'] == position)].reset_index(drop=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 15), dpi=500, facecolor = background)

    ##############################################################################################
    ax[0, 0].set_facecolor(background)
    ax[0, 0].scatter(x=df['Sight play'], y=df['Create Chances Ability'], alpha=.1, lw=1, color='#E8E8E8', hatch='///////')
    ax[0, 0].set_title('Sight play vs Chances', color=color, size=18)
    ax[0, 0].axvline(df['Sight play'].mean(), c=color)
    ax[0, 0].axhline(df['Create Chances Ability'].mean(), c=color)

    ax[0, 0].tick_params(axis='x', colors=color, labelsize=12)
    ax[0, 0].tick_params(axis='y', colors=color, labelsize=12)

    ax[0, 0].spines['bottom'].set_color(color)
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].spines['left'].set_color(color)
    ax[0, 0].spines['right'].set_visible(False)

    for i in range(len(player)):
        ax[0, 0].scatter(x=player['Sight play'].values[i], y=player['Create Chances Ability'].values[i], s=400, c='#FCAC14', lw=3, edgecolor=background, label=playerName, zorder=3)

    ##############################################################################################

    ax[0, 1].set_facecolor(background)
    ax[0, 1].scatter(x=df['KeyPass Ability'], y=df['decisionMake'], alpha=.1, lw=1, color='#E8E8E8', hatch='///////')
    ax[0, 1].set_title('KeyPass vs Decision Make', color=color, size=18)
    ax[0, 1].axvline(df['KeyPass Ability'].mean(), c=color)
    ax[0, 1].axhline(df['decisionMake'].mean(), c=color)

    ax[0, 1].tick_params(axis='x', colors=color, labelsize=12)
    ax[0, 1].tick_params(axis='y', colors=color, labelsize=12)

    ax[0, 1].spines['bottom'].set_color(color)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['left'].set_color(color)
    ax[0, 1].spines['right'].set_visible(False)

    for i in range(len(player)):
        ax[0, 1].scatter(x=player['KeyPass Ability'].values[i], y=player['decisionMake'].values[i], s=400, c='#FCAC14', lw=2, edgecolor=background, label=playerName, zorder=3)

    ##############################################################################################

    ax[1, 0].set_facecolor(background)
    ax[1, 0].scatter(x=df['Dribbling Ability'], y=df['decisionMake'], alpha=.1, lw=1, color='#E8E8E8', hatch='///////')
    ax[1, 0].set_title('Dribbles vs Decision', color=color, size=18)
    ax[1, 0].axvline(df['Dribbling Ability'].mean(), c=color)
    ax[1, 0].axhline(df['decisionMake'].mean(), c=color)

    ax[1, 0].tick_params(axis='x', colors=color, labelsize=12)
    ax[1, 0].tick_params(axis='y', colors=color, labelsize=12)

    ax[1, 0].spines['bottom'].set_color(color)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['left'].set_color(color)
    ax[1, 0].spines['right'].set_visible(False)

    for i in range(len(player)):
        ax[1, 0].scatter(x=player['Dribbling Ability'].values[i], y=player['decisionMake'].values[i], s=400, c='#FCAC14', lw=2, edgecolor=background, label=playerName, zorder=3)

    ##############################################################################################

    ax[1, 1].set_facecolor(background)
    ax[1, 1].scatter(x=df['Finishing Ability'], y=df['Heading Ability'], alpha=.1, lw=1, color='#E8E8E8', hatch='///////')
    ax[1, 1].set_title('Finishing vs Heading', color=color, size=18)
    ax[1, 1].axvline(df['Finishing Ability'].mean(), c=color)
    ax[1, 1].axhline(df['Heading Ability'].mean(), c=color)

    ax[1, 1].tick_params(axis='x', colors=color, labelsize=12)
    ax[1, 1].tick_params(axis='y', colors=color, labelsize=12)

    ax[1, 1].spines['bottom'].set_color(color)
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['left'].set_color(color)
    ax[1, 1].spines['right'].set_visible(False)

    for i in range(len(player)):
        ax[1, 1].scatter(x=player['Finishing Ability'].values[i], y=player['Heading Ability'].values[i], s=400, c='#FCAC14', lw=2, edgecolor=background, label=playerName, zorder=3)

##############################################################################################################################################################################

    #Criação da legenda
    l = plt.legend(facecolor=background, framealpha=.05, labelspacing=.9, prop={'size': 10})
    #Ciclo FOR para atribuir a white color na legend
    for text in l.get_texts():
        text.set_color(color)

    fig.suptitle('Offensive Aspects', color=color, size=35)
    return plt.show()

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def radar_chart_compare(df, player, player2, cols):

    #Obtenção dos dois jogadores que pretendemos
    pl1 = df[(df['Player'] == player)]

    position = pl1['Position'].unique()
    position = position.tolist()
    position = position[0]
    if ', ' in position:
        position = position.split(', ')[0]

    val1 = pl1[cols].values[0]

    club = pl1['Team'].values[0]
    league = pl1['Comp'].values[0]

    #Obtenção dos dois jogadores que pretendemos
    pl2 = df[(df['Player'] == player2)]
    val2 = pl2[cols].values[0]

    position2 = pl2['Position'].unique()
    position2 = position2.tolist()
    position2 = position2[0]
    if ', ' in position2:
        position2 = position2.split(', ')[0]

    club2 = pl2['Team'].values[0]
    league2 = pl2['Comp'].values[0]

    #Obtenção dos valores das colunas que pretendemos colocar no radar chart, não precisamos aceder ao index porque só iriamos aceder aos valores de um dos jogadores
    values = [val1, val2]

    rango = df.loc[(df['Comp'] == league) & (df.Position.str.contains(position))].reset_index(drop=True)

    #Obtençaõ dos valores min e max das colunas selecionadas
    ranges = [(rango[col].min(), rango[col].max()) for col in cols] 

    #Atribuição dos valores aos titulos e respetivos tamanhos e cores
    title = dict(
        #Jogador 1
        title_name = player,
        title_color = '#548135',
        
        #Jogador 2
        title_name_2 = player2,
        title_color_2 = '#fb8c04',

        #Tamnhos gerais do radar chart
        title_fontsize = 20,
        subtitle_fontsize = 15,

        subtitle_name=club,
        subtitle_color='#181818',
        subtitle_name_2=club2

    )

    #team_player = df[col_name_team].to_list()

    #dict_team ={'Dortmund':['#ffe011', '#000000'],
                #'Nice':['#cc0000', '#000000'],
                #'Nice':['#cc0000', '#000000']}

    #color = dict_team.get(team_player[0])

    ## endnote 
    endnote = "Visualization made by: Pedro Meneses(@menesesp20)"

    #Criação do radar chart
    fig, ax = plt.subplots(figsize=(18,15), dpi=500)
    radar = Radar(background_color="#E8E8E8", patch_color="#181818", range_color="#181818", label_color="#181818", label_fontsize=10, range_fontsize=11)
    fig, ax = radar.plot_radar(ranges=ranges, 
                                params=cols, 
                                values=values, 
                                radar_color=['#548135','#fb8c04'], 
                                figax=(fig, ax),
                                title=title,
                                endnote=endnote, end_size=0, end_color="#1b1b1b",
                                compare=True)

    fig.set_facecolor('#E8E8E8')

    return plt.show()

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def radar_chart(df, player, cols, player2=None):

    league = df.loc[df.Player == player]['Comp'].unique()[0]

    leagueDF = df.loc[df.Comp == league].reset_index(drop=True)

    tier = leagueDF.Tier.unique()
    tier = tier.tolist()
    tier = tier[0]

    if player2 == None:
        #Atribuição do jogador a colocar no gráfico
        players = df.loc[(df['Player'] == player)].reset_index(drop=True)

        club = players.Team.unique()
        club = club.tolist()
        club = club[0]

        tierPlayer = players.Tier.unique()
        tierPlayer = tierPlayer.tolist()
        tierPlayer = tierPlayer[0]

        position = players['Position'].unique()
        position = position.tolist()
        position = position[0]
        if ', ' in position:
            position = position.split(', ')[0]

        #####################################################################################################################

        #Valores que pretendemos visualizar no radar chart, acedemos ao index 0 para obtermos os valores dentro da lista correta
        values = players[cols].values[0]
        #Obtenção do alcance minimo e máximo dos valores

        rango = df.loc[(df['Comp'] == league) & (df.Position.str.contains(position)) | (df.Player == player)].reset_index(drop=True)

        ranges = [(rango[col].min(), rango[col].max()) for col in cols]

        color = ['#FF0000','#ffffff']
        #Atribuição dos valores aos titulos e respetivos tamanhos e cores
        title = dict(
            title_name = player,
            title_color = color[0],
            title_fontsize = 25,
            subtitle_fontsize = 15,

            subtitle_name=club,
            subtitle_color='#181818',
        )

        #team_player = df[col_name_team].to_list()

        #dict_team ={'Dortmund':['#ffe011', '#000000'],
                    #'Nice':['#cc0000', '#000000'],}

        #color = dict_team.get(team_player[0])

        ## endnote 
        endnote = "Visualization made by: Pedro Meneses(@menesesp20)"

        #Criação do radar chart
        fig, ax = plt.subplots(figsize=(18,15))
        radar = Radar(background_color="#E8E8E8", patch_color="#181818", range_color="#181818", label_color="#181818", label_fontsize=10, range_fontsize=11)
        fig, ax = radar.plot_radar(ranges=ranges, 
                                    params=cols, 
                                    values=values, 
                                    radar_color=color,
                                    figax=(fig, ax),
                                    image_coord=[0.464, 0.81, 0.1, 0.075],
                                    title=title,
                                    endnote=endnote)

        fig.set_facecolor('#E8E8E8')

    else:
        radar_chart_compare(df, player, player2, cols)
        
    return plt.show()

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def PizzaChart(df, cols, playerName):
    # parameter list
    params = cols

    league = df.loc[df.Player == playerName]['Comp'].unique()[0]

    playerDF = df.loc[(df.Player == playerName) & (df.Comp == league)]

    league = playerDF.Comp.unique()

    league = league.tolist()

    league = league[0]

    position = playerDF['Position'].unique()

    position = position.tolist()

    position = position[0]
    if ', ' in position:
        position = position.split(', ')[0]

    marketValue = playerDF['Market value'].unique()

    marketValue = marketValue.tolist()
    
    marketValue = marketValue[0]

    df = df.loc[(df['Comp'] == league) & (df['Position'].str.contains(position))].reset_index(drop=True)

    player = df.loc[(df['Player'] == playerName) & (df['Comp'] == league)][cols].reset_index()
    player = list(player.loc[0])
    player = player[1:]

    values = []
    for x in range(len(params)):   
        values.append(math.floor(stats.percentileofscore(df[params[x]], player[x])))

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    if cols == Forward:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 4 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 12

    elif cols == Winger:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 3 + ["#fb8c04"] * 8 + ["#eb04e3"] * 2
        text_colors = ["#F2F2F2"] * 13

    elif cols == defensive_Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 4 + ["#eb04e3"] * 5
        text_colors = ["#F2F2F2"] * 13
        
    elif cols == Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 8 + ["#eb04e3"] * 3
        text_colors = ["#F2F2F2"] * 15

    elif cols == full_Back:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 6 + ["#fb8c04"] * 4 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 14

    elif cols == center_Back:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 3 + ["#fb8c04"] * 4 + ["#eb04e3"] * 7
        text_colors = ["#F2F2F2"] * 14

    elif cols == offensive_Midfield:
        # color for the slices and text
        slice_colors = ["#2d92df"] * 4 + ["#fb8c04"] * 8 + ["#eb04e3"] * 4
        text_colors = ["#F2F2F2"] * 16

    # instantiate PyPizza class
    baker = PyPizza(
        params=params,                  # list of parameters
        background_color="#1b1b1b",     # background color
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_color="#000000",    # color for last line
        last_circle_lw=1,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=20            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(15, 10),                # adjust the figsize according to your need
        color_blank_space="same",        # use the same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,        # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="#000000", zorder=2, linewidth=1
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="#F2F2F2", fontsize=10,
            va="center"
        ),                               # values to be used when adding parameter labels
        kwargs_values=dict(
            color="#F2F2F2", fontsize=11,
            zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )                                # values to be used when adding parameter-values labels
    )

    if cols == Forward:

        fig_text(s =  'Forward Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == Winger:

        fig_text(s =  'Winger Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == defensive_Midfield:

        fig_text(s =  'Defensive Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == Midfield:

        fig_text(s =  'Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == full_Back:

        fig_text(s =  'Full Back Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)
    elif cols == center_Back:

        fig_text(s =  'Center Back Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    elif cols == offensive_Midfield:

        fig_text(s =  'Offensive Midfield Template',
             x = 0.253, y = 0.035,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8)

    ###########################################################################################################

    fig_text(s =  playerName,
             x = 0.5, y = 1.12,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=50);

    if playerName != 'David Neres':

        fig_text(s =  'Percentile Rank | ' + league + ' | Pizza Chart | Season 2021-22',
                x = 0.5, y = 1.03,
                color='#F2F2F2',
                fontweight='bold', ha='center',
                fontsize=14);

    elif playerName == 'David Neres':

        fig_text(s =  'Percentile Rank | ' + league + ' | Pizza Chart | Calendar Year 2021',
                x = 0.5, y = 1.03,
                color='#F2F2F2',
                fontweight='bold', ha='center',
                fontsize=14);

    #fig_text(s =  str(marketValue),
    #         x = 0.5, y = 1.02,
    #         color='#F2F2F2',
    #         fontweight='bold', ha='center',
    #         fontsize=18);

    # add credits
    CREDIT_1 = "data: WyScout"
    CREDIT_2 = "made by: @menesesp20"
    CREDIT_3 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"


    # CREDITS
    fig_text(s =  f"{CREDIT_1}\n{CREDIT_2}\n{CREDIT_3}",
             x = 0.35, y = 0.02,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=8);

    # Attacking
    fig_text(s =  'Attacking',
             x = 0.41, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # Possession
    fig_text(s =  'Possession',
             x = 0.535, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # Defending
    fig_text(s =  'Defending',
             x = 0.665, y = 0.988,
             color='#F2F2F2',
             fontweight='bold', ha='center',
             fontsize=16);

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.34, 0.97), 0.025, 0.021, fill=True, color="#2d92df",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.47, 0.97), 0.025, 0.021, fill=True, color="#fb8c04",
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.60, 0.97), 0.025, 0.021, fill=True, color="#eb04e3",
            transform=fig.transFigure, figure=fig
        ),
    ])

    # add image
    add_image('C:/Users/menes/Documents/Data Hub/Images/SWL LOGO.png', fig, left=0.462, bottom=0.436, width=0.10, height=0.132)

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def plotMaps(df, playerName, mode=None):
    if mode == None:
        color = '#E8E8E8'
        background = '#181818'
    elif mode != None:
        color = '#181818'
        background = '#E8E8E8'

    # SUBPLOTS DEFINE ROWS AND COLS
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 15), dpi=500, facecolor = background)

    # FOOTBALL PITCH
    pitch = Pitch(pitch_type='opta',
                  pitch_color=background, line_color=color,
                  line_zorder=1, linewidth=2, spot_scale=0.002)

    # DRAW THE PITCH IN THE FIGURE AXIS
    pitch.draw(ax=ax[0, 0])

    # AXIS TITLE
    ax[0, 0].set_title('Position With Ball', color=color, size=18)

    # DATAFRAME ONLY WITH THE TOUCHES OF THE PLAYER
    player = df.loc[(df['name'] == playerName) & ((df['isTouch'] == True) | (df['typedisplayName'] == 'Pass'))].reset_index(drop=True)

    #####################################################################################################################################

    # GRADIENT COLOR
    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        [background, '#3d0000', '#ff0000'], N=10)

    # PATH EFFECTS OF THE HEAT MAP
    path_eff = [path_effects.Stroke(linewidth=3, foreground=color),
                path_effects.Normal()]

    # BINS FOR THE HEAT MAP
    bs1 = pitch.bin_statistic_positional(player['x'], player['y'],  statistic='count', positional='full', normalize=True)

    # HEAT MAP POSITIONAL PITCH (WAY OF PLAY FOOTBALL)
    pitch.heatmap_positional(bs1, edgecolors=color, ax=ax[0, 0], cmap=pearl_earring_cmap, alpha=0.6)

    # LABEL HEATMAP
    pitch.label_heatmap(bs1, color=background, fontsize=16,
                                ax=ax[0, 0], ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff, zorder=5)

    #filter that dataframe to exclude outliers. Anything over a z score of 1 will be excluded for the data points
    convex = player[(np.abs(stats.zscore(player[['x','y']])) < 1).all(axis=1)]

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=ax[0, 0], edgecolor=background, facecolor=color, alpha=0.2, linestyle='--', linewidth=3, zorder=2)

    pitch.scatter(player['x'], player['y'], ax=ax[0, 0], s=25, edgecolor=color, facecolor=background, linewidth=1.5, alpha=0.3, zorder=2)

    pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=ax[0, 0], c='#FF0000', edgecolor=color, lw=2, s=400, zorder=3)
    
    #####################################################################################################################################
    
    # FOOTBALL PITCH
    pitch = Pitch(pitch_type='opta',
                  pitch_color=background, line_color=color,
                  line_zorder=1, linewidth=2, spot_scale=0.002)

    # DRAW THE PITCH IN THE FIGURE AXIS
    pitch.draw(ax=ax[0, 1])

    # AXIS TITLE
    ax[0, 1].set_title('Position WithOut Ball', color=color, size=18)

    defensiveActions = ['Clearance', 'Interception', 'Aerial', 'BlockedPass', 'Foul', 'Card', 'Challenge', 'Tackle']

    # DATAFRAME ONLY WITH THE TOUCHES OF THE PLAYER
    player = df.loc[(df['name'] == playerName) &
                    ((df['typedisplayName'] == defensiveActions[0]) | (df['typedisplayName'] == defensiveActions[1]) |
                     (df['typedisplayName'] == defensiveActions[2]) | (df['typedisplayName'] == defensiveActions[3]) |
                     (df['typedisplayName'] == defensiveActions[4]) | (df['typedisplayName'] == defensiveActions[5]) |
                     (df['typedisplayName'] == defensiveActions[6]) | (df['typedisplayName'] == defensiveActions[7]))].reset_index(drop=True)

    #####################################################################################################################################

    # GRADIENT COLOR
    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        [background, '#3d0000', '#ff0000'], N=10)

    # PATH EFFECTS OF THE HEAT MAP
    path_eff = [path_effects.Stroke(linewidth=3, foreground=color),
                path_effects.Normal()]

    # BINS FOR THE HEAT MAP
    bs2 = pitch.bin_statistic_positional(player['x'], player['y'],  statistic='count', positional='full', normalize=True)

    # HEAT MAP POSITIONAL PITCH (WAY OF PLAY FOOTBALL)
    pitch.heatmap_positional(bs2, edgecolors=color, ax=ax[0, 1], cmap=pearl_earring_cmap, alpha=0.6)

    # LABEL HEATMAP
    pitch.label_heatmap(bs2, color=background, fontsize=16,
                                ax=ax[0, 1], ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff, zorder=5)

    #filter that dataframe to exclude outliers. Anything over a z score of 1 will be excluded for the data points
    convex = player[(np.abs(stats.zscore(player[['x','y']])) < 1).all(axis=1)]

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=ax[0, 1], edgecolor=background, facecolor=color, alpha=0.2, linestyle='--', linewidth=3, zorder=2)

    pitch.scatter(player['x'], player['y'], ax=ax[0, 1], s=25, edgecolor=color, facecolor=background, linewidth=1.5, alpha=0.3, zorder=2)

    pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=ax[0, 1], c='#FF0000', edgecolor=color, lw=2, s=400, zorder=3)
    
    #####################################################################################################################################

    # FOOTBALL PITCH
    pitch = VerticalPitch(pitch_type='opta',
                pitch_color=background, line_color=color,
                line_zorder=1, linewidth=2, spot_scale=0.002)

    # DRAW THE PITCH IN THE FIGURE AXIS
    pitch.draw(ax=ax[1, 0])

    # AXIS TITLE
    ax[1, 0].set_title('Chance Creation', color=color, size=18)

    # DATAFRAME WITH THE CHANCES CREATED BY THE PLAYER
    chances = df.loc[(df['name'] == playerName) & (df['qualifiers'].str.contains('KeyPass') == True)].reset_index(drop=True)

    # GRADIENT COLOR
    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        [background, '#3d0000', '#ff0000'], N=10)

    # PATH EFFECTS OF THE HEAT MAP
    path_eff = [path_effects.Stroke(linewidth=3, foreground=color),
                path_effects.Normal()]

    # BINS FOR THE HEAT MAP
    bs3 = pitch.bin_statistic_positional(chances['y'], chances['x'],  statistic='count', positional='full', normalize=True)

    # HEAT MAP POSITIONAL PITCH (WAY OF PLAY FOOTBALL)
    pitch.heatmap_positional(bs3, edgecolors=color, ax=ax[1, 0], cmap=pearl_earring_cmap, alpha=0.6)

    # LABEL HEATMAP
    pitch.label_heatmap(bs3, color=background, fontsize=18,
                                ax=ax[1, 0], ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff)

    #####################################################################################################################################

    # FOOTBALL PITCH
    pitch = VerticalPitch(pitch_type='opta', half=True,
                pitch_color=background, line_color=color,
                line_zorder=1, linewidth=2, spot_scale=0.002)

    # DRAW PITCH IN THE FIGURE AXIS
    pitch.draw(ax=ax[1, 1])

    # AXIS TITLE
    ax[1, 1].set_title('Key Passes', color=color, size=18)

    # DATAFRAME ONLY WITH THE PASSES OF THE PLAYER
    passes = df.loc[(df['name'] == playerName) & (df['typedisplayName'] == 'Pass') & (df['x'] >= 55)].reset_index(drop=True)

    # DATAFRAME WITHT THE KEY PASSES OF THE PLAYER
    keyPass = passes.loc[(passes['qualifiers'].str.contains('KeyPass') == True)].reset_index(drop=True)

    # DATAFRAME WITHT THE KEY PASSES OF THE PLAYER
    penBox = passes.loc[(passes['endX'] >= 94.2) & ((passes['endY'] >= 36.8) | (passes['endY'] <= 63.2))].reset_index(drop=True)

    # DATAFRAME WITHT THE CROSSES OF THE PLAYER
    cross = passes.loc[(passes['qualifiers'].str.contains('Cross') == True)].reset_index(drop=True)

    ####################################################################################################

    # PASSES INTO PENALTY BOX
    pitch.lines(cross['x'], cross['y'],
                cross['endX'], cross['endY'],
                color = '#EA04DC', alpha=0.5,
                lw=4, transparent=True, comet=True,
                zorder=5, ax=ax[1, 1])

    # FINAL 1/3 SCATTER
    pitch.scatter(cross['x'], cross['y'], s=150,
                marker='o', edgecolors=background, lw=2, c='#EA04DC',
                zorder=5, ax=ax[1, 1], label='Cross')

    ####################################################################################################

    # PASSES INTO PENALTY BOX
    pitch.lines(penBox['x'], penBox['y'],
                penBox['endX'], penBox['endY'],
                color = '#2D92DF', alpha=0.5,
                lw=4, transparent=True, comet=True,
                zorder=5, ax=ax[1, 1])

    # FINAL 1/3 SCATTER
    pitch.scatter(penBox['x'], penBox['y'], s=150,
                marker='o', edgecolors=background, lw=2, c='#2D92DF',
                zorder=5, ax=ax[1, 1], label='Pen box')

    ####################################################################################################

    # KEY PASSES
    pitch.lines(keyPass['x'], keyPass['y'],
                keyPass['endX'], keyPass['endY'],
                color = '#FFBA08', alpha=0.5,
                lw=4, transparent=True, comet=True,
                zorder=5, ax=ax[1, 1])

    # KEY PASS SCATTER
    pitch.scatter(keyPass['x'], keyPass['y'], s=150,
                marker='o', edgecolors=background, lw=2, c='#FFBA08',
                zorder=5, ax=ax[1, 1], label='Key Pass')

    ####################################################################################################

    # PASSES
    pitch.lines(passes['x'], passes['y'],
                passes['endX'], passes['endY'],
                color = color, alpha=0.2,
                lw=4, transparent=True, comet=True, ax=ax[1, 1])

    ####################################################################################################

    # LEGEND
    l = plt.legend(facecolor=background, framealpha=.05, labelspacing=.9, bbox_to_anchor =(0.29, 0.955), ncol = 3, prop={'size': 8})
    # FOR LOOP TO ATTRIBUTE WHITE COLOR TO LEGEND TEXT
    for text in l.get_texts():
        text.set_color(color)
        
    return plt.show()

with st.form("select-buttons"):
    options_PlayerOPTA = st.sidebar.selectbox(
        'Choose Player you want analyse (Event Data)',
        sorted(opta.name.unique()))

    options_Player = st.sidebar.selectbox(
        'Choose Player you want analyse',
        sorted(wyscout.loc[wyscout['Minutes played'] >= 1500]['Player'].unique()))

    options_Player_Compare = st.sidebar.selectbox(
        'Choose Player you want to compare',
        wyscout.Player.unique())
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    btn1 = col1.form_submit_button(label='Bars')
    btn2 = col2.form_submit_button(label='Scatters')
    btn3 = col3.form_submit_button(label='Radar')
    btn4 = col4.form_submit_button(label='Radar Compare')
    btn5 = col5.form_submit_button(label='Percentile')
    btn6 = col6.form_submit_button(label='Pitch')

#col1, col2, col3, col4, col5, col6 = st.columns(6)

if btn1:
        figBars = barsAbility(data, options_Player)
        st.pyplot(figBars)

if btn2:
    figScatters = Scatters(data, options_Player)
    
    st.pyplot(figScatters)

if btn3:

    cols = ['Aerial duels won, %', 'Touches in box/90', 'Offensive duels %', 'Progressive runs/90', 'Crosses/90', 'Deep completed crosses/90',
                'Passes %', 'Deep completions/90', 'Progressive passes/90', 'Key passes/90', 'Third assists/90',
                'PAdj Interceptions', 'Aerial duels/90', 'Aerial duels won, %']

    figRadar = radar_chart(wyscout, options_Player, cols)
    
    st.pyplot(figRadar)

if btn4:
    figRadarCompare = radar_chart(wyscout, options_Player, full_Back, options_Player_Compare)
    
    st.pyplot(figRadarCompare)

if btn5:

    figPercentile = PizzaChart(wyscout, full_Back, options_Player)
    
    st.pyplot(figPercentile)

if btn6:
    
    figPositionPitch = plotMaps(opta, options_PlayerOPTA)
    
    st.pyplot(figPositionPitch)
