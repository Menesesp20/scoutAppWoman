import pandas as pd
import numpy as np

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
import matplotlib.patches as patches

from sklearn.cluster import KMeans
import scipy.stats as stats

from mplsoccer import VerticalPitch, Pitch, Radar, FontManager, grid, PyPizza

import sys
sys.path.append('Functions')
sys.path.append('passing_networks_in_python_master/visualization')

from Functions.utils import read_json
from passing_networks_in_python_master.visualization.passing_network import draw_pitch, draw_pass_map

from highlight_text import  ax_text, fig_text
from soccerplots.utils import add_image

# FONT FAMILY
# Set the Lato font file path
lato_path = './Fonts/Lato-Black.ttf'

# Register the Lato font with Matplotlib
custom_font = FontProperties(fname=lato_path)

# Register the Lato font with Matplotlib
custom_font = FontProperties(fname=lato_path)

# DICTIONARY OF COLORS

clubColors = {'Brazil' : ['#fadb04', '#1c3474'],
            'Portugal' : ['#e1231b', '#004595'],
            'Argentina' : ['#52a9dc', '#dbe4ea'],
            'Saudi Arabia' : ['#145735', '#dbe4ea'],
            'Ghana' : ['#145735', '#dbe4ea'],
            'Serbia' : ['#FF0000', '#ffffff'],
            'Spain' : ['#FF0000', '#ffffff'],
            'Germany' : ['#aa9e56', '#FF0000'],
            'France' : ['#202960', '#d10827'],
            'Poland' : ['#d10827', '#ffffff'],
            'Morocco' : ['#db221b', '#044c34'],
            'Croatia' : ['#e71c23', '#3f85c5'],
            'Netherlands' : ['#f46c24', '#dcd9d7'],
            'Senegal' : ['#34964a', '#eedf36'],
            'Denmark' : ['#cb1617', '#ffffff'],
            'Iran' : ['#269b44', '#dd1212'],
            'Belgium' : ['#ff0000', '#e30613'],
            'USA' : ['#ff0000', '#202960'],
            'Switzerland' : ['#ff0000', '#ffffff'],
            'Australia' : ['#202960', '#e30613'],
            'Wales' : ['#ff0000', '#ffffff'],
            'Mexico' : ['#00a94f', '#ff0000'],
            'Uruguay' : ['#52a9dc', '#ffffff'],
            'Canada' : ['#ff0000', '#ff0000'],
            'Costa Rica' : ['#ff0000', '#202960'],
            'Catar' : ['#7f1244', '#ffffff'],
            'Ecuador' : ['#ffce00', '#002255'],
            'South Korea' : ['#021858', '#ffffff'],
            'Real Madrid' : ['#064d93', '#E8E8E8'],
            'Liverpool' : ['#FF0000', '#E8E8E8']}


def carry(df, team, gameDay, carrydf=None, progressive=None):
    def checkCarryPositions(endX, endY, nextX, nextY):
        distance = np.sqrt(np.square(nextX - endX) + np.square(nextY - endY))
        if distance < 3:
            return True
        else:
            return False

    def isProgressiveCarry(x, y, endX, endY):
        distanceInitial = np.sqrt(np.square(105 - x) + np.square(34 - y))
        distanceFinal = np.sqrt(np.square(105 - endX) + np.square(34 - endY))
        if x < 52.5 and endX < 52.5 and distanceInitial - distanceFinal > 12.5:
            return True
        elif x < 52.5 and endX > 52.5 and distanceInitial - distanceFinal > 7.5:
            return True
        elif x > 52.5 and endX > 52.5 and distanceInitial - distanceFinal > 5:
            return True

        return False

    def get_carries(new_df, teamId):
        df = new_df.copy()
        df["recipient"] = df["playerId"].shift(-1)
        df["nextTeamId"] = df["teamId"].shift(-1)

        a = np.array(
            df[(df["typedisplayName"] == "Pass") & (df["outcomeTypedisplayName"] == "Successful") & (df["teamId"] == int(teamId))].index.tolist()
        )
        b = np.array(
            df[
                (
                    (df["typedisplayName"] == "BallRecovery")
                    | (df["typedisplayName"] == "Interception")
                    | (df["typedisplayName"] == "Tackle")
                    | (df["typedisplayName"] == "BlockedPass")
                )
                & (df["outcomeTypedisplayName"] == "Successful")
                & (df["teamId"] == int(teamId))
            ].index.tolist()
        )

        carries_df = pd.DataFrame()

        for value in a:
            carry = pd.Series()
            carry["minute"] = df.iloc[value].minute
            carry["second"] = df.iloc[value].second
            carry["playerId"] = df.iloc[value].recipient
            carry["x"] = df.iloc[value].endX
            carry["y"] = df.iloc[value].endY
            if (
                df.iloc[value + 1].typedisplayName == "OffsideGiven"
                or df.iloc[value + 1].typedisplayName == "End"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
            ):
                continue
            elif (
                df.iloc[value + 1].typedisplayName == "Challenge"
                and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                and df.iloc[value + 1].teamId != teamId
            ):
                carry["playerId"] = df.iloc[value + 2].playerId
                value += 1
                while (df.iloc[value + 1].typedisplayName == "TakeOn" and df.iloc[value + 1].outcomeTypedisplayName == "Successful") or (
                    df.iloc[value + 1].typedisplayName == "Challenge" and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                ):
                    value += 1
                if (
                    df.iloc[value + 1].typedisplayName == "OffsideGiven"
                    or df.iloc[value + 1].typedisplayName == "End"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
                ):
                    continue
            if df.iloc[value + 1].teamId != int(teamId):
                continue
            else:
                carry["endX"] = df.iloc[value + 1].x
                carry["endY"] = df.iloc[value + 1].y
            carries_df = carries_df.append(carry, ignore_index=True)

        for value in b:
            carry = pd.Series()
            carry["playerId"] = df.iloc[value].playerId
            carry["minute"] = df.iloc[value].minute
            carry["second"] = df.iloc[value].second
            carry["x"] = df.iloc[value].x
            carry["y"] = df.iloc[value].y
            if (
                df.iloc[value + 1].typedisplayName == "OffsideGiven"
                or df.iloc[value + 1].typedisplayName == "End"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
            ):
                continue
            elif (
                df.iloc[value + 1].typedisplayName == "Challenge"
                and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                and df.iloc[value + 1].teamId != teamId
            ):
                carry["playerId"] = df.iloc[value + 2].playerId
                value += 1
                while (df.iloc[value + 1].typedisplayName == "TakeOn" and df.iloc[value + 1].outcomeTypedisplayName == "Successful") or (
                    df.iloc[value + 1].typedisplayName == "Challenge" and df.iloc[value + 1].outcomeTypedisplayName == "Unsuccessful"
                ):
                    value += 1
                if (
                    df.iloc[value + 1].typedisplayName == "OffsideGiven"
                    or df.iloc[value + 1].typedisplayName == "End"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOff"
                    or df.iloc[value + 1].typedisplayName == "SubstitutionOn"
                ):
                    continue
            if df.iloc[value + 1].playerId != df.iloc[value].playerId or df.iloc[value + 1].teamId != int(teamId):
                continue
            carry["endX"] = df.iloc[value + 1].x
            carry["endY"] = df.iloc[value + 1].y
            carries_df = carries_df.append(carry, ignore_index=True)

        carries_df["Removable"] = carries_df.apply(
            lambda row: checkCarryPositions(row["x"], row["y"], row["endX"], row["endY"]), axis=1
        )
        carries_df = carries_df[carries_df["Removable"] == False]
        return carries_df

    def isProgressivePass(x, y, endX, endY):
        distanceInitial = np.sqrt(np.square(105 - x) + np.square(34 - y))
        distanceFinal = np.sqrt(np.square(105 - endX) + np.square(34 - endY))
        if x <= 52.5 and endX <= 52.5:
            if distanceInitial - distanceFinal > 30:
                return True
        elif x <= 52.5 and endX > 52.5:
            if distanceInitial - distanceFinal > 15:
                return True
        elif x > 52.5 and endX > 52.5:
            if distanceInitial - distanceFinal > 10:
                return True
        return False

    def clean_df(df, homeTeam, awayTeam, teamId):
        names = df[["name", "playerId"]].dropna().drop_duplicates()
        df["x"] = df["x"] * 1.05
        df["y"] = df["y"] * 0.68
        df["endX"] = df["endX"] * 1.05
        df["endY"] = df["endY"] * 0.68
        df["progressive"] = False
        df["progressive"] = df[df["typedisplayName"] == "Pass"].apply(
            lambda row: isProgressivePass(row.x, row.y, row.endX, row.endY), axis=1
        )
        carries_df = get_carries(df, teamId)
        carries_df["progressiveCarry"] = carries_df.apply(
            lambda row: isProgressiveCarry(row.x, row.y, row.endX, row.endY), axis=1
        )
        carries_df["typedisplayName"] = "Carry"
        carries_df["teamId"] = teamId
        carries_df = carries_df.join(names.set_index("playerId"), on="playerId")
        df = pd.concat(
            [
                df,
                carries_df[
                    [
                        "playerId",
                        "minute",
                        "second",
                        "teamId",
                        "x",
                        "y",
                        "endX",
                        "endY",
                        "progressiveCarry",
                        "typedisplayName",
                        "name",
                    ]
                ],
            ]
        )
        df["homeTeam"] = homeTeam
        df["awayTeam"] = awayTeam
        df = df.sort_values(["minute", "second"], ascending=[True, True])
        return df

    df = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
    homeTeam = df.home_Team.unique()
    homeTeam = homeTeam[0]
    awayTeam = df.away_Team.unique()
    awayTeam = awayTeam[0]

    teamID = df.loc[df.team == team].reset_index(drop=True)
    teamID = teamID.teamId.unique()
    teamID = teamID[0]

    data = clean_df(df, homeTeam, awayTeam, teamID)

    def get_progressive_carries(df, team_id):
        df_copy = df[df["teamId"] == team_id].copy()

        df_copy = df_copy[(df_copy["typedisplayName"] == "Carry") & (df_copy["progressiveCarry"] == True)]

        ret_df = df_copy.groupby(["name", "playerId"]).agg(prog_carries=("progressiveCarry", "count")).reset_index()

        return ret_df
    
    if progressive != None:
        return get_progressive_carries(data, teamID)
    elif carrydf !=None:
        return get_carries(df, teamID)
    else:
        return clean_df(df, homeTeam, awayTeam, teamID)
       
def dataFramexTFlow(df, club, data):

    if data == 'WyScout':
        df_Home = df.loc[(df['team.name'] == club)].reset_index(drop=True)

        df_Away = df.loc[(df['team.name'] != club)].reset_index(drop=True)
        
    elif data == 'WhoScored':
        df_Home = df.loc[(df['team'] == club)].reset_index(drop=True)

        df_Away = df.loc[(df['team'] != club)].reset_index(drop=True)
        
    home_xT = []
    away_xT = []

    #Criação da lista de jogadores
    Minutes = range(df['minute'].min(), df['minute'].max())

    #Ciclo For de atribuição dos valores a cada jogador
    for minute in Minutes:
        home_xT.append(df_Home.loc[df_Home['minute'] == minute, 'xT'].sum())
        away_xT.append(df_Away.loc[df_Away['minute'] == minute, 'xT'].sum())
    data = {
        'Minutes' : Minutes,
        'home_xT' : home_xT,
        'away_xT' : away_xT
        }

    df = pd.DataFrame(data)
    return df

def shotAfterRecover(df, team):
    def recoverShot(df, team, gameDay):
        from datetime import timedelta

        cols = ['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y', 'away_Team', 'home_Team', 'xG', 'xGOT', 'Match_ID']

        teamDF = df[(df['team'] == team) & (df['Match_ID'] ==  gameDay)].reset_index(drop=True)

        recovery_list = pd.DataFrame(columns=cols)

        for idx, row in teamDF.iterrows():
            if 'BallRecovery' in row['typedisplayName']:
                recovery_time = row['matchTimestamp']
                shots_after_recovery = teamDF[(teamDF['matchTimestamp'] > recovery_time) & (teamDF['matchTimestamp'] <= recovery_time + timedelta(seconds=10))]
                goals_after_recovery = shots_after_recovery[shots_after_recovery['typedisplayName'].str.contains('Goal')]
                if not goals_after_recovery.empty:
                    recovery_list = pd.concat([recovery_list, row[cols].to_frame().T, goals_after_recovery[cols]])

        recovery_list.drop_duplicates(inplace=True)

        return recovery_list

    def shotRecover(df, team):
        matchId = df['Match_ID'].unique()
        data = pd.DataFrame(columns=['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y', 'away_Team', 'home_Team', 'xG', 'xGOT', 'Match_ID'])
        for id in matchId:
            data = pd.concat([data, recoverShot(df, team, id)])
        
        data.reset_index(drop=True, inplace=True)
        return data
    
    return shotRecover(df, team)
    
def counterPress(df, team):
    def lost_and_recovered(df, team, game_day):
        cols = ['name', 'matchTimestamp', 'team', 'typedisplayName', 'x', 'y']
        team_df = df[(df['team'] == team) & (df['Match_ID'] == game_day)].reset_index(drop=True)
        recovery_list = pd.DataFrame(columns=cols)
        for i, row in team_df.iterrows():
            if 'Dispossessed' in row['typedisplayName']:
                recoveries = team_df[(team_df['matchTimestamp'] > row['matchTimestamp']) &
                                     (team_df['matchTimestamp'] <= row['matchTimestamp'] + pd.Timedelta(seconds=5)) &
                                     (team_df['typedisplayName'] == 'BallRecovery')].reset_index(drop=True)
                if not recoveries.empty:
                    recovery_list = pd.concat([recovery_list, row.to_frame().T, recoveries], ignore_index=True)
        return recovery_list

    all_games = df.Match_ID.unique()
    data = []
    for game_day in all_games:
        data.append(lost_and_recovered(df, team, game_day))
    return pd.concat(data).reset_index(drop=True)

def defensiveCoverList(df):
    cols = ['name', 'team', 'expandedMinute', 'typedisplayName', 'qualifiers', 'x', 'y']
    defensiveCover_list = pd.DataFrame(columns=cols)
    
    takeOn_indices = df.index[df['typedisplayName'] == 'TakeOn'].tolist()
    for i in takeOn_indices:
        if df.iloc[i+1]['typedisplayName'] == 'BallRecovery':
            defensiveCover_list = defensiveCover_list.append(df.iloc[i][cols], ignore_index=True)
            defensiveCover_list = defensiveCover_list.append(df.iloc[i+1][cols], ignore_index=True)
            
    defensiveCover_list.drop_duplicates(inplace=True)
    return defensiveCover_list

def xT(df):
  eventsPlayers_xT = df

  #Import xT Grid, turn it into an array, and then get how many rows and columns it has
  xT = pd.read_csv('xT/xT_Grid.csv', header=None)
  xT = np.array(xT)
  xT_rows, xT_cols = xT.shape

  eventsPlayers_xT['x1_bin'] = pd.cut(eventsPlayers_xT['x'], bins=xT_cols, labels=False)
  eventsPlayers_xT['y1_bin'] = pd.cut(eventsPlayers_xT['y'], bins=xT_rows, labels=False)
  eventsPlayers_xT['x2_bin'] = pd.cut(eventsPlayers_xT['endX'], bins=xT_cols, labels=False)
  eventsPlayers_xT['y2_bin'] = pd.cut(eventsPlayers_xT['endY'], bins=xT_rows, labels=False)

  eventsPlayers_xT['start_zone_value'] = eventsPlayers_xT[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)
  eventsPlayers_xT['end_zone_value'] = eventsPlayers_xT[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1]][x[0]], axis=1)

  eventsPlayers_xT['xT'] = round(eventsPlayers_xT['end_zone_value'] - eventsPlayers_xT['start_zone_value'], 2)

  eventsPlayers_xT = eventsPlayers_xT.iloc[1:]
  eventsPlayers_xT.reset_index(drop=True, inplace=True)
  #eventsPlayers_xT.drop('Unnamed: 0', axis=1, inplace=True)
    
  return eventsPlayers_xT

def cluster_Event(df, teamName, event_name, n_clusters):
    cols_Cluster = ['team', 'typedisplayName', 'qualifiers', 'x', 'y', 'endX', 'endY']
    cols_coords = ['x', 'y', 'endX', 'endY']

    df_cluster = df[cols_Cluster].copy()
    df_cluster = df_cluster[(df_cluster['team'] == teamName) & (df_cluster['qualifiers'].str.contains(event_name))].reset_index(drop=True)
    
    if df_cluster.empty:
        return pd.DataFrame(columns=['team', 'typedisplayName', 'qualifiers', 'x', 'y', 'endX', 'endY', 'cluster'])

    X = df_cluster[cols_coords].values
    kmeans = KMeans(n_clusters = n_clusters, random_state=100)
    kmeans.fit(X)
    df_cluster['cluster'] = kmeans.predict(X)
    
    return df_cluster

def cluster_Shots(df, teamName, n_clusters):

  cols_Cluster = ['team', 'typedisplayName', 'qualifiers', 'isShot', 'x', 'y', 'endX', 'endY']
  cols_coords = ['x', 'y', 'endX', 'endY']

  df_cluster = df[cols_Cluster].copy()
  df_cluster = df_cluster[(df_cluster['team'] == teamName) & (df_cluster['isShot'] == 'True')].reset_index(drop=True)
  
  if df_cluster.empty:
      return pd.DataFrame(columns=['team', 'typedisplayName', 'qualifiers', 'x', 'y', 'endX', 'endY', 'cluster'])

  X = df_cluster[cols_coords].values
  kmeans = KMeans(n_clusters = n_clusters, random_state=100)
  kmeans.fit(X)
  df_cluster['cluster'] = kmeans.predict(X)
    
  return df_cluster

def sides(xTDF, club):

    xTDF = xTDF.loc[(xTDF['team'] == club)].reset_index(drop=True)

    left_xT = xTDF[(xTDF['y'] >= 67) & (xTDF['x'] >= 55)]
    left_xT['side'] = 'Left'

    center_xT = xTDF[(xTDF['y'] < 67) & (xTDF['y'] > 33) & (xTDF['x'] >= 55)]
    center_xT['side'] = 'Center'

    right_xT = xTDF[(xTDF['y'] <= 33) & (xTDF['x'] >= 55)]
    right_xT['side'] = 'Right'

    sides = pd.concat([left_xT, center_xT, right_xT], axis=0)

    return sides

def dataFrame_xTFlow(df):

    leftfinal3rd = []
    centerfinal3rd = []
    rightfinal3rd = []

    left = df.loc[(df['side'] == 'Left'), 'xT'].sum()
    center = df.loc[(df['side'] == 'Center'), 'xT'].sum()
    right = df.loc[(df['side'] == 'Right'), 'xT'].sum()
    
    leftfinal3rd.append(left)
    centerfinal3rd.append(center)
    rightfinal3rd.append(right)

    data = {
        'left_xT' : leftfinal3rd,
        'center_xT' : centerfinal3rd,
        'right_xT' : rightfinal3rd
    }
    
    df = pd.DataFrame(data)
    
    return df

def search_qualifierOPTA(df, event):
    return df[df['qualifiers'].str.contains(event) == True].reset_index(drop=True)

def heatMapChances(df, league, team, player=None):
    
    # Plotting the pitch
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    pitch = VerticalPitch(pitch_type='opta', half=True,
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=1, linewidth=2, spot_scale=0.005)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    if player == None:
        fig_text(s = 'Where has ' + team + ' created from',
                    x = 0.53, y = 0.94, fontweight='bold',
                    ha='center',fontsize=16, color='#181818');

        pass
    
    else:
        df = df.loc[df.name == player].reset_index(drop=True)

        fig_text(s = 'Where has ' + player + ' created from',
                    x = 0.53, y = 0.94, fontweight='bold',
                    ha='center',fontsize=16, color='#181818');

    fig_text(s = 'All open-play chances created in the ' + league,
                x = 0.53, y = 0.9, fontweight='bold',
                ha='center',fontsize=8, color='#181818', alpha=0.4);

    #fig_text(s = 'Coach: Jorge Jesus',
    #         x = 0.29, y = 0.862, color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=6);

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + team + '.png', fig=fig, left=0.25, bottom=0.885, width=0.08, height=0.07)

    # Opportunity
    opportunity = df.loc[(df['x'] >= 50) & (df['team'] == team) & (df['qualifiers'].str.contains('KeyPass') == True)].reset_index(drop=True)

    #bin_statistic = pitch.bin_statistic_positional(opportunity['x'], opportunity['y'], statistic='count',
    #                                               positional='full', normalize=True)

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]

    bs = pitch.bin_statistic(opportunity['x'], opportunity['y'],  statistic='count', normalize=True, bins=(7, 5))

    pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

    pitch.label_heatmap(bs, color='#E8E8E8', fontsize=18,
                                ax=ax, ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff)

def highTurnovers(df, league, club, player=None):
    
    if player == None:
        pass
    else:
        df = df.loc[df.name == player].reset_index(drop=True)
        
    #Plotting the pitch
    highTurnover = df.loc[(df['typedisplayName'] == 'BallRecovery') & (df.y >= 65) & (df.team == club)].reset_index(drop=True)
    highTurnover.drop_duplicates(['name', 'typedisplayName', 'x', 'y'], keep='first')
    
    dfxG = sides(df, club)

    dfShot = shotAfterRecover(df, club)
    dfShot = dfShot.loc[(dfShot.y >= 50) & (df.team == club)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)

    pitch = VerticalPitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818', half = True,
                            line_zorder=1, linewidth=2, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    #Title of our plot

    fig.suptitle(club, fontsize=50, color='#181818', fontproperties=custom_font, y=1)

    fig_text(s = 'High Turnovers | Made by: @menesesp20',
             fontproperties=custom_font,
             x = 0.5, y = 0.94, color='#181818', ha='center', fontsize=12);

    if dfShot.empty:
        xG = 0
        xGOT = 0
    else:
        xG = round(dfShot.xG.sum(), 2)
        xGOT = round(dfShot.xGOT.sum(), 2)

    fig_text(s = f'xG: {xG} \n xGOT: {xGOT}',
             x = 0.22, y = 0.88,
             fontproperties=custom_font,
             color='#181818', ha='center', fontsize=12);

    ax.axhline(65,c='#ff0000', ls='--', lw=4)

    ax.scatter(highTurnover.x, highTurnover.y, label = 'High Turnovers' + ' ' + '(' + f'{len(highTurnover)}' + ')',
                        c='#ff0000', marker='o', edgecolor='black', s=250, zorder=3)

    ax.scatter(dfShot.x, dfShot.y, label = 'Shot after a turnover within 5 seconds' + ' ' + '(' + f'{len(dfShot)}' + ')',
                        c='#ffba08', marker='*', edgecolor='black', s=500, zorder=3)

    #Criação da legenda
    l = ax.legend(bbox_to_anchor=(0.04, 0.27), loc='upper left', facecolor='#E8E8E8', framealpha=0, labelspacing=.8)

    #Ciclo FOR para atribuir a white color na legend
    for text in l.get_texts():
        text.set_color('#181818')

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.14, bottom=0.91, width=0.2, height=0.09)

    if player != None:
        # Player Image
        fig = add_image(image='Images/Players/' + league + '/' + club + '/' + player + '.png', fig=fig, left=0.68, bottom=0.91, width=0.2, height=0.11)

def dataFramexTFlow(df, matchDay):

        homeTeam = df.loc[(df['Match_ID'] == matchDay)]['home_Team'].unique()[0]
        awayTeam = df.loc[(df['Match_ID'] == matchDay)]['away_Team'].unique()[0]
        
        df_Home = df.loc[(df['team'] == homeTeam)].reset_index(drop=True)

        df_Away = df.loc[(df['team'] == awayTeam)].reset_index(drop=True)
                
        home_xT = []
        away_xT = []

        #Criação da lista de jogadores
        Minutes = range(df['minute'].min(), df['minute'].max())

        #Ciclo For de atribuição dos valores a cada jogador
        for minute in Minutes:
                home_xT.append(df_Home.loc[df_Home['minute'] == minute, 'xT'].sum())
                away_xT.append(df_Away.loc[df_Away['minute'] == minute, 'xT'].sum())
        data = {
                'Minutes' : Minutes,
                'home_xT' : home_xT,
                'away_xT' : away_xT
                }

        df = pd.DataFrame(data)
        return df

def xT_Flow(df, league, club, gameDay, axis=False):

      home = df.loc[df.Match_ID == gameDay]['home_Team'].unique()[0]
      away = df.loc[df.Match_ID == gameDay]['away_Team'].unique()[0]
      
      color = clubColors.get(home)
      color2 = clubColors.get(away)

      dfXT = df.loc[(df['outcomeTypedisplayName'] == 'Successful') & (df['Match_ID'] == gameDay)].reset_index(drop=True)

      xTDF = xT(dfXT)

      df = dataFramexTFlow(xTDF, gameDay)

      df['xTH'] = df['home_xT'].rolling(window=5).mean()

      df['xTH'] = round(df['xTH'], 2)

      df['xTA'] = df['away_xT'].rolling(window=5).mean()

      df['xTA'] = round(df['xTA'], 2)

      #Drop rows with NaN values
      df = df.dropna(axis=0, subset=['xTH', 'xTA'])

      if axis == False:
            fig, ax = plt.subplots(figsize=(20,12))

            #Set color background outside the graph
            fig.set_facecolor('#e8e8e8')

            #Set color background inside the graph
            ax.set_facecolor('#e8e8e8')

      else:
            ax = axis
            ax.set_facecolor('#E8E8E8')
      # Set the number of shots taken by the home team and the opposing team
      home_shots_taken = [abs(i) if i <= 0 else i for i in list(df['xTH'])]
      away_shots_taken = [-abs(i) if i >= 0 else i for i in list(df['xTA'])]

      # Set the position of the bars on the y-axis
      x_pos = list(range(len(home_shots_taken)))

      # Create the bar graph
      ax.bar(x_pos, home_shots_taken, width=0.7, color=color[0], label='Home Team xT')
      ax.bar(x_pos, away_shots_taken, width=0.7, color=color2[0], label='Away Team xT')
      
      x_pos = str(x_pos)

      if axis == False:
            #Params for the text inside the <> this is a function to highlight text
            highlight_textprops =\
                  [{"color": color[0]}
                  ]

            #Title
            Title = fig_text(s = f'<{club}>' + ' ' + 'xT Flow',
                              x = 0.38, y = 1, highlight_textprops = highlight_textprops, fontproperties=custom_font,
                              ha='center',fontsize=40, color='#181818');

            fig_text(s = 'Season 22-23 | xT values based on Karun Singhs model | Created by: @menesesp20',
                     fontproperties=custom_font,
                     x = 0.45, y = 0.95, ha='center', fontsize=12,
                     color='#181818', alpha=0.8);

            # Half Time Line
            halfTime = 45

            ax.axvline(halfTime, color='#181818', ls='--', lw=2.5)

            #ax.axhline(diferencialLine, color='#181818', ls='-', lw=3)

            fig_text(s = 'HALF TIME',
                        x = (halfTime + 6) / 100, y = 0.9, fontweight='bold',
                        fontproperties=custom_font,
                        ha='center', fontsize=16, color='#181818');

            #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
            ax.tick_params(axis='x', colors='#181818', labelsize=16)
            ax.tick_params(axis='y', colors='#181818', labelsize=16, left = False)
            
            #Setg the color of the line in the spines and retire the spines from the top and right sides
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Club Logo
            fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.925, width=0.2, height=0.1)
      else:

            #Atribuição da cor e tamanho das tick labels, the left=False retires the ticks
            ax.tick_params(axis='x', colors='#181818', labelsize=11)
            ax.tick_params(axis='y', colors='#181818', labelsize=11, left = False)
            
            #Setg the color of the line in the spines and retire the spines from the top and right sides
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

def buildUpPasses(df, club):
    from datetime import timedelta

    cols = df.columns
    
    teamDF = df.loc[df['team'] == club].reset_index(drop=True)

    passesBuildUp = pd.DataFrame(columns=cols)

    contador = 0
    
    for idx, row in teamDF.iterrows():
        if (row['qualifiers'].__contains__('GoalKick') == True):
            tempo = row['matchTimestamp']
            jogadas = teamDF.loc[(teamDF['matchTimestamp'] > tempo) & (teamDF['matchTimestamp'] <= timedelta(seconds=15) + tempo)]
            for i in jogadas.index.unique():
                if (df.iloc[i]['typedisplayName'] == 'Pass'):
                    if contador == 0:
                        contador = 1
                        eventsGK = pd.DataFrame([row[cols].values], columns=cols)
                        passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)

                    eventsGK = pd.DataFrame([jogadas.loc[i][cols].values], columns=cols)
                    passesBuildUp = pd.concat([passesBuildUp, eventsGK], ignore_index=True)
                    
            contador = 0        

    return passesBuildUp

def draw_heatmap_construcao(df, league, club):

  passesGk = buildUpPasses(df, club)

  fig, ax = plt.subplots(figsize=(15,10), dpi=300)

  pitch = Pitch(pitch_type='opta',
                pitch_color='#e8e8e8', line_color='#181818',
                line_zorder=3, linewidth=5, spot_scale=0.00)

  pitch.draw(ax=ax)

  fig.set_facecolor('#e8e8e8')

  pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#e8e8e8', '#ff0000'], N=10)
  passesGk['x'] = passesGk['x'].astype(float)
  passesGk['y'] = passesGk['y'].astype(float)

  bs = pitch.bin_statistic(passesGk['x'], passesGk['y'], bins=(12, 8))

  pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap)

  fig.suptitle('How do they come out playing?', fontsize=50, color='#181818',
                fontweight = "bold", x=0.53, y=1.07)

  fig_text(s = "GoalKick | Season 22-23 | Made by: @menesesp20",
          x = 0.5, y = 1,
          color='#181818', fontweight='bold', ha='center', fontsize=16);

  #fig_text(s = "Coach: Roger Schmidt",
  #        x = 0.21, y = 0.88,
  #        color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=14);

  # Club Logo
  fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.05, bottom=0.98, width=0.2, height=0.1)

  fig_text(s = 'Attacking Direction',
           x = 0.5, y = 0.12,
           color='#181818', fontweight='bold',
           ha='center', va='center',
           fontsize=14)

  # ARROW DIRECTION OF PLAY
  ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
            arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

def GoalKick(df, league, club, n_cluster):
        
        #################################################################################################################################################

        goalKick = cluster_Event(df, club, 'GoalKick', n_cluster)

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18, 14), dpi=300)

        pitch = Pitch(pitch_type='opta', pitch_color='#e8e8e8',
                        line_color='#181818', line_zorder=1,
                        linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#e8e8e8')

        #################################################################################################################################################

        # Title of our plot

        fig.suptitle('How do they come out playing?', fontsize=50, color='#181818',
                fontweight = "bold", x=0.53, y=0.95)

        fig_text(s = "GoalKick | Season 22-23 | Made by: @menesesp20",
                x = 0.5, y = 0.9,
                color='#181818', fontweight='bold', ha='center' ,fontsize=16);

        #################################################################################################################################################

        # Key Passes Cluster
        for x in range(len(goalKick['cluster'])):

                # First
                if goalKick['cluster'][x] == 0:
                        pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                color='#ea04dc', alpha=0.8,
                                lw=3, zorder=2,
                                ax=ax)
                        
                # Second
                if goalKick['cluster'][x] == 2:
                        pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                color='#2d92df', alpha=0.8,
                                lw=3, zorder=2,
                                ax=ax)
                
                # Third
                if goalKick['cluster'][x] == 1:
                        pitch.arrows(xstart=goalKick['x'][x], ystart=goalKick['y'][x],
                                xend=goalKick['endX'][x], yend=goalKick['endY'][x],
                                color='#fb8c04', alpha=0.8,
                                lw=3, zorder=2,
                                ax=ax)

        #################################################################################################################################################

        fig_text(s = 'Most frequent zone',
                x = 0.8, y = 0.79,
                color='#ea04dc', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Second most frequent zone',
                x = 0.8, y = 0.76,
                color='#2d92df', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Third most frequent zone',
                x = 0.8, y = 0.73,
                color='#fb8c04', fontweight='bold', ha='center' ,fontsize=12);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.1, bottom=0.865, width=0.2, height=0.1)

        fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.17,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

        # ARROW DIRECTION OF PLAY
        ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

def player_Network(df, playerName, team, gameDay, afterSub=None, axis=False):

        if gameDay != 'All Season':
            dataDF = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            dataDF = df.copy()
        
        homeTeam = dataDF.home_Team.unique()[0]
        if team == 'Real Madrid':
                league = dataDF.League.unique()[0]
        else:
                league = dataDF.League.unique()[1]

        awayTeam = dataDF.away_Team.unique()[0]

        homeName = homeTeam
        color = clubColors.get(homeName)
        #color = [color[0], color[1]]

        awayName = awayTeam
        color2 = clubColors.get(awayName)
        #color2 = [color2c[0], color2c[1]]

        data = xT(dataDF)

        ###########################################################################################################################
        if gameDay != 'All Season':
            network = data.loc[(data['team'] == team) & (data['Match_ID'] == gameDay)].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            network = data.loc[(data['team'] == team)].reset_index(drop=True)
            
        network = network.sort_values(['matchTimestamp'], ascending=True)

        network["newsecond"] = 60 * network["minute"] + network["second"]

        #find time of the team's first substitution and filter the df to only passes before that
        Subs = network.loc[(network['typedisplayName'] == "SubstitutionOff")]
        SubTimes = Subs["newsecond"]
        SubOne = SubTimes.min()

        ###########################################################################################################################
        if afterSub == None:
          network = network.loc[network['newsecond'] < SubOne].reset_index(drop=True)

        elif afterSub != None:
          network = network.loc[network['newsecond'] > SubOne].reset_index(drop=True)

        ###########################################################################################################################

        network['passer'] = network['name']
        network['recipient'] = network['passer'].shift(-1)
        network['passer'] = network['passer'].astype(str)
        network['recipient'] = network['recipient'].astype(str)

        passes = network.loc[(network['typedisplayName'] == "Pass") &
                             (network['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)
        
        passes_Player = passes.loc[(passes['typedisplayName'] == "Pass") &
                                   (passes['outcomeTypedisplayName'] == 'Successful') &
                                   (passes['name'] == playerName)].reset_index(drop=True)

        ###########################################################################################################################

        avg = passes.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})
        avg.columns = ['x_avg', 'y_avg', 'count']
        #avg.reset_index(inplace=True)
        #avg.rename(columns={'index': 'passer'}, inplace=True)

        player_pass_count = passes.groupby("passer").size().to_frame("num_passes")
        player_pass_value = passes.groupby("passer")['xT'].sum().to_frame("pass_value")

        passes["pair_key"] = passes.apply(lambda x: "_".join(sorted([x["passer"], x["recipient"]])), axis=1)
        pair_pass_count = passes.groupby("pair_key").size().to_frame("num_passes")
        pair_pass_value = passes.groupby("pair_key")['xT'].sum().to_frame("pass_value")
        
        pair_pass = pd.merge(pair_pass_count, pair_pass_value, on='pair_key')
        pair_pass.reset_index(inplace=True)
        pair_pass.rename(columns={'index': 'pair_key'}, inplace=True)

        ###########################################################################################################################

        btw = passes.groupby(['passer', 'recipient']).id.count().reset_index()
        btw.rename({'id':'pass_count'}, axis='columns', inplace=True)

        merg1 = btw.merge(avg, left_on='passer', right_index=True)
        pass_btw = merg1.merge(avg, left_on='recipient', right_index=True, suffixes=['', '_end'])
        
        pass_btw = pass_btw.loc[pass_btw['pass_count'] > 5]

        ##################################################################################################################################################################

        if axis == False:
                fig, ax = plt.subplots(figsize=(25, 20))

                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=ax)

                fig.set_facecolor('#E8E8E8')
                
                sizeScatter = 2000
                sizeName = 11

                sizeLine = 25
                
        else:
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=axis)

                ax = axis

                sizeScatter = 450,
                sizeName = 5

                sizeLine = 1

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#e8e8e8', color2[0]], N=10)

        bs = pitch.bin_statistic(passes_Player['endX'], passes_Player['endY'], bins=(6, 3))

        pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

        max_player_count = None
        max_player_value = None
        max_pair_count = None
        max_pair_value = None
        
        max_player_count = player_pass_count.num_passes.max() if max_player_count is None else max_player_count
        max_player_value = player_pass_value.pass_value.max() if max_player_value is None else max_player_value
        max_pair_count = pair_pass_count.num_passes.max() if max_pair_count is None else max_pair_count
        max_pair_value = pair_pass_value.pass_value.max() if max_pair_value is None else max_pair_value

        avg['x_avg'] = round(avg['x_avg'], 2)
        avg['y_avg'] = round(avg['y_avg'], 2)
        pair_stats = pd.merge(pair_pass_count, pair_pass_value, left_index=True, right_index=True)
        pair_stats.reset_index(inplace=True)
        pair_stats.rename(columns={'index': 'pair_key'})
        pair_stats = pair_stats.loc[pair_stats.pair_key.str.contains(playerName)].reset_index(drop=True)
        pair_stats['pair_key'] = pair_stats['pair_key'].astype(str)
        
        for row, pair_key in pair_stats.iterrows():
                player1, player2 = pair_key['pair_key'].split("_")
                
                player1_x = avg.loc[player1]["x_avg"]
                player1_y = avg.loc[player1]["y_avg"]

                player2_x = avg.loc[player2]["x_avg"]
                player2_y = avg.loc[player2]["y_avg"]

                num_passes = pair_key["num_passes"]
                if num_passes > sizeLine:
                        num_passes = sizeLine
                        
                pass_value = pair_key["pass_value"]

                norm = Normalize(vmin=0, vmax=max_pair_value)
                edge_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                        ['#181818', color2[0]], N=10)
                edge_color = edge_cmap(norm(pass_value))

                ax.plot([player1_y, player2_y], [player1_x, player2_x],
                        'w-', linestyle='-', alpha=1, lw=num_passes, zorder=2, color=edge_color)

                ax.plot([player2_y - 1.5, player1_y - 1.5], [player2_x - 1.5, player1_x - 1.5],
                        'w--', linestyle='-', alpha=1, lw=num_passes, zorder=2, color=edge_color)

        cycles = 1
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        #plt.colorbar(ScalarMappable(cmap=pearl_earring_cmap), label='xT', orientation="horizontal", shrink=0.3, pad=0.)

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

        #avg.drop(['Unnamed: 0'], axis=1, inplace=True)

        #Criação da lista de jogadores
        
        xTDF = data.loc[data['team'] == team].reset_index(drop=True)
        
        players = xTDF['name'].unique()


        players_xT = []

        #Ciclo For de atribuição dos valores a cada jogador
        for player in players:
                players_xT.append(xTDF.loc[xTDF['name'] == player, 'xT'].sum())
        data = {
        'passer' : players,
        'xT' : players_xT
        }

        xTDF = pd.DataFrame(data)

        avg = pd.merge(avg, xTDF, on='passer')

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
        
        pass_nodes = pitch.scatter(avg['x_avg'], avg['y_avg'], s=sizeScatter,
                                cmap=pearl_earring_cmap, edgecolors="#010101", c=avg['xT'], linewidth=1.3, ax=ax, zorder=3)


        #Uncomment these next two lines to get each node labeled with the player id. Check to see if anything looks off, and make note of each player if you're going to add labeles later like their numbers

        for index, row in avg.iterrows():
                pitch.annotate(row.passer, xy=(row['x_avg'], row['y_avg']), c='#E8E8E8', va='center', ha='center', fontproperties=custom_font, size=sizeName, ax=ax, bbox=dict(facecolor='#181818', alpha=0.7, edgecolor='none'))

        ##################################################################################################################################################################

        if axis == False:

                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                        [{"color": color[0], "fontweight": 'bold'},
                        {"color": color2[0], "fontweight": 'bold'}]

                fig_text(s = f'<{homeName}>' + ' ' + 'vs' + ' ' + f'<{awayName}>',
                        x = 0.52, y = 0.94,
                        color='#181818', ha='center',
                        fontproperties=custom_font,
                        highlight_textprops = highlight_textprops,
                        fontsize=35);

                matchID = network.Match_ID.unique()
                matchID = matchID[0]

                fig_text(s = 'Player Network' + ' ' + '|' + ' ' + 'MatchDay' + ' ' + str(matchID) + ' | @menesesp20',
                        x = 0.52, y = 0.91,
                        fontproperties=custom_font,
                        color='#181818', ha='center',
                        fontsize=11);

                fig_text(s = 'The color of the nodes is based on xT value',
                        x = 0.417, y = 0.875,
                        fontproperties=custom_font,
                        color='#181818', ha='center',
                        fontsize=11);

                # Club Logo
                fig = add_image(image='Images/Clubs/' + league + '/' + team + '.png', fig=fig, left=0.33, bottom=0.905, width=0.06, height=0.05)
        else:
                pass

def passing_networkWhoScored(df, team, gameDay, afterSub=None, axis=False):

        if gameDay != 'All Season':
            dataDF = df.loc[df.Match_ID == gameDay].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            dataDF = df.copy()

        homeTeam = dataDF.home_Team.unique()[0]
        awayTeam = dataDF.away_Team.unique()[0]
        if homeTeam == team:
                league = dataDF.League.unique()[1]
                colorHeatMap = clubColors.get(homeTeam)
        else:
                league = dataDF.League.unique()[0]
                colorHeatMap = clubColors.get(awayTeam)

        color = clubColors.get(homeTeam)
        color2 = clubColors.get(awayTeam)

        data = xT(dataDF)

        ###########################################################################################################################
        if gameDay != 'All Season':
            network = data.loc[(data['team'] == team) & (data['Match_ID'] == gameDay)].reset_index(drop=True)
            
        elif gameDay == 'All Season':
            network = data.loc[(data['team'] == team)].reset_index(drop=True)
            
        network = network.sort_values(['matchTimestamp'], ascending=True)

        network["newsecond"] = 60 * network["minute"] + network["second"]

        #find time of the team's first substitution and filter the df to only passes before that
        Subs = network.loc[(network['typedisplayName'] == "SubstitutionOff")]
        SubTimes = Subs["newsecond"]
        SubOne = SubTimes.min()

        ###########################################################################################################################
        if afterSub == None:
          network = network.loc[network['newsecond'] < SubOne].reset_index(drop=True)

        elif afterSub != None:
          network = network.loc[network['newsecond'] > SubOne].reset_index(drop=True)

        ###########################################################################################################################

        network['passer'] = network['name']
        network['recipient'] = network['passer'].shift(-1)
        network['passer'] = network['passer'].astype(str)
        network['recipient'] = network['recipient'].astype(str)

        passes = network.loc[(network['typedisplayName'] == "Pass") &
                             (network['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

        ###########################################################################################################################

        avg = passes.groupby('passer').agg({'x':['mean'], 'y':['mean', 'count']})
        avg.columns = ['x_avg', 'y_avg', 'count']
        #avg.reset_index(inplace=True)
        #avg.rename(columns={'index': 'passer'}, inplace=True)

        player_pass_count = passes.groupby("passer").size().to_frame("num_passes")
        player_pass_value = passes.groupby("passer")['xT'].sum().to_frame("pass_value")

        passes["pair_key"] = passes.apply(lambda x: "_".join(sorted([x["passer"], x["recipient"]])), axis=1)
        pair_pass_count = passes.groupby("pair_key").size().to_frame("num_passes")
        pair_pass_value = passes.groupby("pair_key")['xT'].sum().to_frame("pass_value")
        
        pair_pass = pd.merge(pair_pass_count, pair_pass_value, on='pair_key')
        pair_pass.reset_index(inplace=True)
        pair_pass.rename(columns={'index': 'pair_key'}, inplace=True)
        
        pair_pass.to_excel('pair_pass.xlsx')
        
        avg.to_excel('avg.xlsx')

        ###########################################################################################################################

        btw = passes.groupby(['passer', 'recipient']).id.count().reset_index()
        btw.rename({'id':'pass_count'}, axis='columns', inplace=True)

        merg1 = btw.merge(avg, left_on='passer', right_index=True)
        pass_btw = merg1.merge(avg, left_on='recipient', right_index=True, suffixes=['', '_end'])
        
        pass_btw = pass_btw.loc[pass_btw['pass_count'] > 5]

        ##################################################################################################################################################################

        if axis == False:
                fig, ax = plt.subplots(figsize=(25, 20))

                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=ax)

                fig.set_facecolor('#E8E8E8')
                
                sizeScatter = 2000
                sizeName = 11

                sizeLine = 8
                
        else:
                pitch = VerticalPitch(pitch_type='opta',
                                pitch_color='#E8E8E8', line_color='#181818',
                                line_zorder=3, linewidth=0.5, spot_scale=0.00)

                pitch.draw(ax=axis)

                ax = axis

                sizeScatter = 450,
                sizeName = 5

                sizeLine = 1

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#e8e8e8', colorHeatMap[0]], N=10)

        bs = pitch.bin_statistic(passes['endX'], passes['endY'], bins=(6, 3))

        pitch.heatmap(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.5)

        max_player_count = None
        max_player_value = None
        max_pair_count = None
        max_pair_value = None
        
        max_player_count = player_pass_count.num_passes.max() if max_player_count is None else max_player_count
        max_player_value = player_pass_value.pass_value.max() if max_player_value is None else max_player_value
        max_pair_count = pair_pass_count.num_passes.max() if max_pair_count is None else max_pair_count
        max_pair_value = pair_pass_value.pass_value.max() if max_pair_value is None else max_pair_value

        avg['x_avg'] = round(avg['x_avg'], 2)
        avg['y_avg'] = round(avg['y_avg'], 2)
        pair_stats = pd.merge(pair_pass_count, pair_pass_value, left_index=True, right_index=True)

        for pair_key, row in pair_stats.iterrows():
            player1, player2 = pair_key.split("_")
            
            player1_x = avg.loc[player1]["x_avg"]
            player1_y = avg.loc[player1]["y_avg"]

            player2_x = avg.loc[player2]["x_avg"]
            player2_y = avg.loc[player2]["y_avg"]

            num_passes = row["num_passes"]
            if num_passes > sizeLine:
                    num_passes = sizeLine
                    
            pass_value = row["pass_value"]

            norm = Normalize(vmin=0, vmax=max_pair_value)
            edge_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#181818', colorHeatMap[0]], N=10)
            edge_color = edge_cmap(norm(pass_value))

            ax.plot([player1_y, player2_y], [player1_x, player2_x],
                    'w-', linestyle='-', alpha=1, lw=num_passes, zorder=2, color=edge_color)

            ax.plot([player2_y - 1.5, player1_y - 1.5], [player2_x - 1.5, player1_x - 1.5],
                    'w--', linestyle='-', alpha=1, lw=num_passes, zorder=2, color=edge_color)

        cycles = 1
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        #plt.colorbar(ScalarMappable(cmap=pearl_earring_cmap), label='xT', orientation="horizontal", shrink=0.3, pad=0.)

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

        #avg.drop(['Unnamed: 0'], axis=1, inplace=True)

        #Criação da lista de jogadores
        
        xTDF = data.loc[data['team'] == team].reset_index(drop=True)
        
        players = xTDF['name'].unique()


        players_xT = []

        #Ciclo For de atribuição dos valores a cada jogador
        for player in players:
                players_xT.append(xTDF.loc[xTDF['name'] == player, 'xT'].sum())
        data = {
        'passer' : players,
        'xT' : players_xT
        }

        xTDF = pd.DataFrame(data)

        avg = pd.merge(avg, xTDF, on='passer')

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
        
        pass_nodes = pitch.scatter(avg['x_avg'], avg['y_avg'], s=sizeScatter,
                                cmap=pearl_earring_cmap, edgecolors="#010101", c=avg['xT'], linewidth=1.3, ax=ax, zorder=3)


        #Uncomment these next two lines to get each node labeled with the player id. Check to see if anything looks off, and make note of each player if you're going to add labeles later like their numbers

        for index, row in avg.iterrows():
                pitch.annotate(row.passer, xy=(row['x_avg'], row['y_avg']), c='#E8E8E8', va='center', ha='center', fontproperties=custom_font, size=sizeName, ax=ax, bbox=dict(facecolor='#181818', alpha=0.7, edgecolor='none'))

        ##################################################################################################################################################################

        if axis == False:

                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                        [{"color": color[0],"fontweight": 'bold'},
                        {"color": color2[0],"fontweight": 'bold'}]

                fig_text(s = f'<{homeTeam}>' + ' ' + 'vs' + ' ' + f'<{awayTeam}>',
                        x = 0.52, y = 0.94,
                        color='#181818', fontproperties=custom_font, ha='center',
                        highlight_textprops = highlight_textprops,
                        fontsize=35);

                matchID = network.Match_ID.unique()
                matchID = matchID[0]

                fig_text(s = 'Passing Network' + ' ' + '|' + ' ' + 'MatchDay' + ' ' + str(matchID) + ' | @menesesp20',
                        x = 0.52, y = 0.91,
                        color='#181818', fontproperties=custom_font, ha='center',
                        fontsize=11);

                fig_text(s = 'The color of the nodes is based on xT value',
                        x = 0.417, y = 0.875,
                        color='#181818', fontproperties=custom_font, ha='center',
                        fontsize=11);

                # Club Logo
                fig = add_image(image='Images/Clubs/' + league + '/' + team + '.png', fig=fig, left=0.35, bottom=0.905, width=0.06, height=0.05)
        else:
                pass

def plot_Shots(df, axis=None):

    data = df.loc[(df.typedisplayName == 'Goal') | (df.typedisplayName == 'SavedShot') | (df.typedisplayName == 'MissedShots')].reset_index(drop=True)
    data['xGOT'].fillna(0, inplace=True)

    home = data['home_Team'].unique()[0]
    
    away = data['away_Team'].unique()[0]  

    color = clubColors.get(home)
    
    color2 = clubColors.get(away)

    team_Home = df.loc[df.team == home].reset_index(drop=True)
    
    team_Away = df.loc[df.team == away].reset_index(drop=True)
    
    home_Shots = data.loc[data.team == home].reset_index(drop=True)
    
    away_Shots = data.loc[data.team == away].reset_index(drop=True)

    if axis == None:
        fig, axis = plt.subplots(figsize=(15, 10), dpi=300)
    else:
        pass
    
    pitch = Pitch(pitch_type='opta',
                        pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(ax=axis)

    pitch = Pitch(pitch_type='opta',
                        pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=3, linewidth=0.5, spot_scale=0.00)

    pitch.draw(axis)

    #####################################################################################################################

    pitch.annotate(str(round(sum(home_Shots['xG']), 2)), xy=(36, 83), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' Expected Goals (xG): ', xy=(50.5, 83), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)

    pitch.annotate(str(round(sum(away_Shots['xG']), 2)), xy=(65, 83), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)

    #####################################################################################################################

    pitch.annotate(str(round(sum(home_Shots['xGOT']), 2)), xy=(36, 73), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' xG on target (xGOT): ', xy=(50.5, 73), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)

    pitch.annotate(str(round(sum(away_Shots['xGOT']), 2)), xy=(65, 73), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)
    
    #####################################################################################################################

    pitch.annotate(str(len(home_Shots)), xy=(36, 63), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' Shots: ', xy=(50.5, 63), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)

    pitch.annotate(str(len(away_Shots)), xy=(65, 63), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)

    #####################################################################################################################

    pitch.annotate(str(len(team_Home.loc[team_Home['typedisplayName'] == 'Foul'])), xy=(36, 53), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' Fouls: ', xy=(50.5, 53), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)
    
    pitch.annotate(str(len(team_Away.loc[team_Away['typedisplayName'] == 'Foul'])), xy=(65, 53), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)
    
    #####################################################################################################################
    pitch.annotate(str(round((len(team_Home.loc[team_Home['typedisplayName'] == 'Pass']) / len(df.loc[(df['typedisplayName'] == 'Pass')]) * 100), 2)), xy=(36, 43), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' Ball Possession: ', xy=(50.5, 43), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)

    pitch.annotate(str(round((len(team_Away.loc[team_Away['typedisplayName'] == 'Pass']) / len(df.loc[(df['typedisplayName'] == 'Pass')]) * 100), 2)), xy=(65, 43), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)

    #####################################################################################################################

    pitch.annotate(str(len(team_Home.loc[team_Home['typedisplayName'] == 'CornerAwarded'])), xy=(36, 33), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color[0], edgecolor='black', alpha=0.7), ax=axis)

    pitch.annotate(' Corners: ', xy=(50.5, 33), c='#E8E8E8', va='center', ha='center',
                size=12, bbox=dict(facecolor='#181818', edgecolor='black', alpha=0.8), ax=axis)

    pitch.annotate(str(len(team_Away.loc[team_Away['typedisplayName'] == 'CornerAwarded'])), xy=(65, 33), c='#181818', va='center', ha='center',
                size=15, bbox=dict(facecolor=color2[0], edgecolor='black', alpha=0.7), ax=axis)

    #####################################################################################################################

    for i in range(len(home_Shots)):
        if home_Shots['typedisplayName'].values[i] == 'Goal':
            #Criação das setas que simbolizam os passes realizados bem sucedidos
            pitch.scatter(100 - home_Shots['x'].values[i], 100 - home_Shots['y'].values[i],
                            color='#FFBA08', marker='*', edgecolors=color[0], lw=0.8, ax=axis, s=(home_Shots['xG'].values[i] * 1500),
                            zorder=3)
        else:
            #Criação das setas que simbolizam os passes realizados bem sucedidos
            pitch.scatter(100 - home_Shots['x'].values[i], 100 - home_Shots['y'].values[i],
                            color=color[0], marker='h', edgecolors='#181818', lw=1.5, ax=axis, s=(home_Shots['xG'].values[i] * 1500),
                            zorder=3)

    for i in range(len(away_Shots)):
        if away_Shots['typedisplayName'].values[i] == 'Goal':
            #Criação das setas que simbolizam os passes realizados bem sucedidos
            pitch.scatter(away_Shots['x'].values[i], away_Shots['y'].values[i],
                            color='#FFBA08', marker='*', edgecolors=color2[0], lw=0.8, ax=axis, s=(away_Shots['xG'].values[i] * 1500),
                            zorder=3)
        else:
            #Criação das setas que simbolizam os passes realizados bem sucedidos
            pitch.scatter(away_Shots['x'].values[i], away_Shots['y'].values[i],
                            color=color2[0], marker='h', edgecolors='#181818', lw=1.5, ax=axis, s=(away_Shots['xG'].values[i] * 1500),
                            zorder=3)

def heatMap_xT(df, league, club, player=None):
    
    color = clubColors.get(club)

    dfXT = df.loc[(df['typedisplayName'] == 'Pass') & (df['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

    xTDF = xT(dfXT)

    if (player == None):
            xTheatMap = xTDF.loc[(xTDF['team'] == club)]
    else:
            xTheatMap = xTDF.loc[(xTDF['name'] == player)]

    # setup pitch
    pitch = Pitch(pitch_type='opta', line_zorder=2,
                    pitch_color='#E8E8E8', line_color='#181818')

    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor('#E8E8E8')

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#E8E8E8', color[0]], N=10)

    xTheatHeat = xTheatMap.loc[xTheatMap.xT > 0]
    bs = pitch.bin_statistic(xTheatHeat['x'], xTheatHeat['y'], bins=(10, 8))
    pitch.heatmap(bs, edgecolors='#E8E8E8', ax=ax, cmap=pearl_earring_cmap)

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=-0.01, bottom=1, width=0.2, height=0.12)

    # TITLE
    if player == None:
            fig_text(s = 'Where' + ' ' + club + ' ' + 'generate the most xT',
                    x = 0.54, y = 1.1, color='#181818', fontweight='bold', ha='center' ,fontsize=29.5);
    else:
            fig_text(s = 'Where' + ' ' + player + ' ' + 'generate the most xT',
                    x = 0.54, y = 1.1, color='#181818', fontweight='bold', ha='center' ,fontsize=27);     

    #fig_text(s = 'Coach: Jorge Jesus',
    #         x = 0.123, y = 0.97, color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=12);

    # TOTAL xT
    fig_text(s = str(round(sum(xTheatMap.xT), 2)) + ' ' + 'xT Generated', 
            x = 0.51, y = 1.02, color='#181818', fontweight='bold', ha='center' ,fontsize=18);

    fig_text(s = 'Attacking Direction',
                    x = 0.5, y = 0.05,
                    color='#181818', fontweight='bold',
                    ha='center', va='center',
                    fontsize=14)

    # ARROW DIRECTION OF PLAY
    ax.annotate('', xy=(0.3, -0.03), xycoords='axes fraction', xytext=(0.7, -0.03), 
            arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

def counterPressMap(df, league, team, player=None):

    # Plotting the pitch
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=2, spot_scale=0.005)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    fig_text(s = team + ' counter press',
                x = 0.53, y = 0.98, fontweight='bold',
                ha='center',fontsize=16, color='#181818');

    fig_text(s = 'Season 2022-23 ' + league,
                x = 0.53, y = 0.93, fontweight='bold',
                ha='center',fontsize=8, color='#181818', alpha=0.4);

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + team + '.png', fig=fig, left=0.32, bottom=0.93, width=0.08, height=0.07)
    
    # Counter Press DataFrame
    dataCP = counterPress(df, team)
    
    if player == None:
        dataCP = dataCP.loc[dataCP.typedisplayName == 'BallRecovery'].reset_index(drop=True)
    else:
        dataCP = dataCP.loc[(dataCP.typedisplayName == 'BallRecovery') & (dataCP.name == player)].reset_index(drop=True)

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#e8e8e8', '#3d0000', '#ff0000'], N=10)

    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]

    bs = pitch.bin_statistic_positional(dataCP['x'], dataCP['y'],  statistic='count', positional='full', normalize=True)
    
    pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)

    pitch.label_heatmap(bs, color='#E8E8E8', fontsize=12,
                                ax=ax, ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff)

def through_passMap(df, gameID, league, club, playerName=None):

    color = clubColors.get(club)

    if playerName == None:
            df = df.loc[(df['team'] == club) & (df['Match_ID'] == gameID)].reset_index(drop=True)
    else:
            df = df.loc[(df['name'] == playerName) & (df['Match_ID'] == gameID)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(18,14), dpi=300)

    pitch = Pitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=2, linewidth=5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    ###############################################################################################################################################################
    ###############################################################################################################################################################
            
    #identify the passer and then the recipient, who'll be the playerId of the next action
    df['passer'] = df['name']

    df['recipient'] = df['passer'].shift(-1)
    
    through_pass = df.loc[df['qualifiers'].apply(lambda x: 'Throughball' in x)].reset_index(drop=True)

    through_passSucc = through_pass.loc[through_pass['outcomeTypedisplayName'] == 'Successful'].reset_index(drop=True)

    through_passUnsucc = through_pass.loc[through_pass['outcomeTypedisplayName'] == 'Unsuccessful'].reset_index(drop=True)

    through_passKP = through_pass.loc[through_pass['qualifiers'].apply(lambda x: 'KeyPass' in x)].reset_index(drop=True)

    through_passAst = through_pass.loc[through_pass['qualifiers'].apply(lambda x: 'IntentionalGoalAssist' in x)].reset_index(drop=True)

    ###############################################################################################################################################################
    ###############################################################################################################################################################

    # Plot Through Passes Successful
    pitch.lines(through_passSucc['x'], through_passSucc['y'], through_passSucc['endX'], through_passSucc['endY'],
            lw=5, color='#08d311', comet=True,
            label='Through Passes Successful', ax=ax)

    pitch.scatter(through_passSucc['endX'], through_passSucc['endY'], s=100,
            marker='o', edgecolors='#08d311', c="#08d311", zorder=3, ax=ax)

    # Plot Through Passes Unsuccessful
    pitch.lines(through_passUnsucc['x'], through_passUnsucc['y'], through_passUnsucc['endX'], through_passUnsucc['endY'],
            lw=5, color='#ff0000', comet=True,
            label='Through Passes Unsuccessful', ax=ax)

    pitch.scatter(through_passUnsucc['endX'], through_passUnsucc['endY'], s=100,
            marker='o', edgecolors='#ff0000', c='#ff0000', zorder=3, ax=ax)

    for i in range(len(through_pass)):
            plt.text(through_pass['x'].values[i] + 0.7, through_pass['y'].values[i] + 0.7, through_pass['name'].values[i], color=color[0], zorder=5)

    for i in range(len(through_passSucc)):        
            plt.text(through_passSucc['endX'].values[i] + 0.7, through_passSucc['endY'].values[i] + 0.7, through_passSucc['recipient'].values[i], color=color[0], zorder=5)
    
    for i in range(len(through_passKP)):
            plt.text(through_passKP['endX'].values[i] + 0.7, through_passKP['endY'].values[i] + 0.7, through_passKP['recipient'].values[i], color=color[0], zorder=5)

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    
    # Plot Key Passes
    pitch.lines(through_passKP['x'], through_passKP['y'], through_passKP['endX'], through_passKP['endY'],
            lw=5, color='#ffba08', comet=True,
            label='Key Passes', ax=ax)

    # Plot Key Passes
    pitch.scatter(through_passKP['endX'], through_passKP['endY'], s=100,
            marker='o', edgecolors='#ffba08', c='#ffba08', zorder=3, ax=ax)

    ###############################################################################################################################################################
    ###############################################################################################################################################################
    
    # Plot Key Passes
    pitch.lines(through_passAst['x'], through_passAst['y'], through_passAst['endX'], through_passAst['endY'],
            lw=5, color='#fb8c04', comet=True,
            label='Assist', ax=ax)

    # Plot Key Passes
    pitch.scatter(through_passAst['endX'], through_passAst['endY'], s=100,
            marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=ax)

    ###############################################################################################################################################################
    ###############################################################################################################################################################

    #Criação da legenda
    l = ax.legend(bbox_to_anchor=(0.02, 1), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
    #Ciclo FOR para atribuir a color legend
    for text in l.get_texts():
            text.set_color("#181818")

    ###############################################################################################################################################################
    ###############################################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0], "fontweight": 'bold'}]

    if (playerName == None) & (gameID != 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    elif (playerName == None) & (gameID == 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='All Season' + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    if (playerName != None) & (gameID != 'All Season'):
            fig_text(s =f'<{playerName}>' + ' ' + 'Throughballs',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    elif (playerName != None) & (gameID == 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Throughballs',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='All Season' + ' ' +  '| Season 22-23 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    ###############################################################################################################################################################
    ###############################################################################################################################################################


    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.87, width=0.2, height=0.08)

    fig_text(s = 'Attacking Direction',
                    x = 0.5, y = 0.17,
                    color='#181818', fontweight='bold',
                    ha='center', va='center',
                    fontsize=14)

    # ARROW DIRECTION OF PLAY
    ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
            arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

def touch_Map(df, gameID, league, club, Player=None):

    color = clubColors.get(club)

    df = df.loc[(df['name'] == Player) & (df['outcomeTypedisplayName'] == 'Successful')]

    # Plotting the pitch

    fig, ax = plt.subplots(figsize=(18,14), dpi=500)

    pitch = Pitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    #############################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0], "fontweight": 'bold'}]

    if (Player == None) & (gameID != 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Touch Map',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    elif (Player == None) & (gameID == 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Touch Map',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='All Season' + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    if (Player != None) & (gameID != 'All Season'):
            fig_text(s =f'<{Player}>' + ' ' + 'Touch Map',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='MatchDay:' + str(gameID) + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    elif (Player != None) & (gameID == 'All Season'):
            fig_text(s =f'<{club}>' + ' ' + 'Touch Map',
                    x = 0.5, y = 0.91, highlight_textprops = highlight_textprops,
                    color='#181818', fontweight='bold', ha='center', va='center', fontsize=48);
            
            fig_text(s ='All Season' + ' ' +  '| Season 21-22 | @menesesp20',
                    x = 0.5, y = 0.85, color='#181818', fontweight='bold', ha='center', va='center', fontsize=16, alpha=0.7);

    #############################################################################################################################################

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#E8E8E8', color[0]], N=10)
    bs = pitch.bin_statistic(df['x'], df['y'], bins=(8, 8))
    pitch.heatmap(bs, edgecolors='#1b1b1b', ax=ax, cmap=pearl_earring_cmap, zorder=2)

    #filter that dataframe to exclude outliers. Anything over a z score of 1 will be excluded for the data points
    convex = df[(np.abs(stats.zscore(df[['x','y']])) < 1).all(axis=1)]

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=ax, edgecolor='#181818', facecolor='#181818', alpha=0.5, linestyle='--', linewidth=2.5, zorder=2)

    pitch.scatter(df['x'], df['y'], ax=ax, edgecolor='#181818', facecolor='black', alpha=0.5, zorder=2)

    pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=ax, c='#E8E8E8', edgecolor=color[0], s=1500, zorder=3)

    #############################################################################################################################################

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.87, width=0.2, height=0.08)

    fig_text(s = 'Attacking Direction',
                    x = 0.5, y = 0.17,
                    color='#181818', fontweight='bold',
                    ha='center', va='center',
                    fontsize=14)

    # ARROW DIRECTION OF PLAY
    ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
            arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

def dashboardDeffensive(df, league, club, matchDay, playerName):
    
    color = clubColors.get(club)

    fig = plt.figure(figsize=(15,8), dpi = 500)
    grid = plt.GridSpec(6, 6)

    a1 = fig.add_subplot(grid[0:5, 0:2])
    a2 = fig.add_subplot(grid[0:5, 2:4])
    a3 = fig.add_subplot(grid[0:5, 4:9])

    #################################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0],"fontweight": 'bold'},
    {"color": color[0],"fontweight": 'bold'}]

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.98, width=0.2, height=0.1)

    fig.set_facecolor('#E8E8E8')

    fig_text(s =f'<{playerName}>' + "<'s>" + ' ' + 'performance',
            x = 0.42, y = 1.08, highlight_textprops = highlight_textprops,
            color='#181818', fontweight='bold', ha='center' ,fontsize=35);
    
    if matchDay != 'All Season':
            fig_text(s = 'MatchDay:' + ' ' + str(matchDay) + ' ' + '| Season 22-23 | @menesesp20',
                    x = 0.33, y = 1.015 , color='#181818', fontweight='bold', ha='center' ,fontsize=11);

    if matchDay == 'All Season':
            fig_text(s = 'Season 22-23 | @menesesp20',
                    x = 0.3, y = 1.015 , color='#181818', fontweight='bold', ha='center' ,fontsize=11);

    fig_text(s = 'Territory Plot',
            x = 0.25, y = 0.91 , color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    fig_text(s = 'Pass Plot',
            x = 0.513, y = 0.91, color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    fig_text(s = 'Defensive Actions Plot',
            x = 0.78, y = 0.91, color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    #################################################################################################################################################
    # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

    if matchDay == 'All Season':
            df1 = df[(df['name'] == playerName) & (df['outcomeTypedisplayName'] == 'Successful')]
    else:
            df1 = df[(df['name'] == playerName) & (df['outcomeTypedisplayName'] == 'Successful') & (df.Match_ID == matchDay)]

    pitch = VerticalPitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=3, spot_scale=0.00)

    pitch.draw(ax=a1)

    #################################################################################################################################################

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                    ['#E8E8E8', color[0]], N=10)

    bs = pitch.bin_statistic(df1['x'], df1['y'], bins=(10, 6))

    convex = df1[(np.abs(stats.zscore(df1[['x','y']])) < 1).all(axis=1)]

    pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

    pitch.scatter(df1['x'], df1['y'], ax=a1, edgecolor='#181818', facecolor='black', alpha=0.3)

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=2.5)

    pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=a1, c='#E8E8E8', edgecolor=color[0], s=700, zorder=4)


    #################################################################################################################################################
    # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

    #df['beginning'] = np.sqrt(np.square(100 - df['x']) + np.square(100 - df['y']))
    #df['end'] = np.sqrt(np.square(100 - df['endX']) + np.square(100 - df['endY']))

    #df['progressive'] = [(df['end'][x]) / (df['beginning'][x]) < .75 for x in range(len(df.beginning))]

    if matchDay != 'All Season':
            player = df.loc[(df['name'] == playerName) & (df.Match_ID == matchDay)]
    else:
            player = df.loc[(df['name'] == playerName)]
            
    keyPass = player.loc[player['qualifiers'].apply(lambda x: 'KeyPass' in x)]

    Pass = player.loc[(player['typedisplayName'] == 'Pass')]

    sucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Successful')]

    unsucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Unsuccessful')]
    
    #Progressive = Pass.loc[Pass['progressive'] == True]

    Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

    #################################################################################################################################################
    pitch = VerticalPitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=1, linewidth=3, spot_scale=0.00)

    pitch.draw(ax=a2)

    fig.set_facecolor('#E8E8E8')

    #Criação das setas que simbolizam os passes realizados bem sucedidos
    pitch.arrows(sucess['x'], sucess['y'], sucess['endX'], sucess['endY'], color='#181818', ax=a2,
            width=2, headwidth=5, headlength=5, label='Passes' + ':' + ' ' + f'{len(Pass)}' + ' ' + '(' + f'{Pass_percentage}' + '%' + ' ' + 'Completion rate' + ')' )
    
    #Criação das setas que simbolizam os passes realizados bem sucedidos
    pitch.arrows(unsucess['x'], unsucess['y'], unsucess['endX'], unsucess['endY'], color='#181818', alpha=0.4, ax=a2,
            width=2, headwidth=5, headlength=5, label='Passes unsuccessful' + ':' + ' '  + f'{len(unsucess)}')

    #Criação das setas que simbolizam os passes realizados falhados
    #pitch.arrows(Progressive['x'], Progressive['y'], Progressive['endX'], Progressive['endY'], color='#00bbf9', ax=a2,
    #        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}')

    #Criação das setas que simbolizam os passes realizados falhados
    pitch.arrows(keyPass['x'], keyPass['y'], keyPass['endX'], keyPass['endY'], color='#ffba08', ax=a2,
            width=2, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}')
    
    pitch.scatter(keyPass['endX'], keyPass['endY'], s = 150, marker='*', color='#ffba08', ax=a2)

    #################################################################################################################################################

    #Criação da legenda ffba08
    l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
    #Ciclo FOR para atribuir a color legend
    for text in l.get_texts():
            text.set_color("#181818")

    #################################################################################################################################################
    # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE

    #if matchDay != 'All Season':
    #        df3 = df.loc[(df['x'] <= 55) & (df['name'] == playerName) & (df['Match_ID'] == matchDay)]
    #else:
    #        df3 = df.loc[(df['x'] <= 55) & (df['name'] == playerName)]
            
    df3 = df.loc[(df['name'] == playerName) & (df['Match_ID'] == matchDay)]
    

    # Tackle
    tackle = df3.loc[(df3['typedisplayName'] == 'Tackle') & (df3['outcomeTypedisplayName'] == 'Successful')]

    tackleUn = df3.loc[(df3['typedisplayName'] == 'Tackle') & (df3['outcomeTypedisplayName'] == 'Unsuccessful')]

    # Pressures
    #pressure = df3.loc[df3['type.secondary'].apply(lambda x: 'counterpressing_recovery' in x)]

    # Interception
    interception = df3.loc[df3['typedisplayName'] == 'Interception']

    # Aerial
    aerial = df3.loc[(df3['typedisplayName'] == 'Aerial') & (df3['outcomeTypedisplayName'] == 'Successful')]
    
    aerialUn = df3.loc[(df3['typedisplayName'] == 'Aerial') & (df3['outcomeTypedisplayName'] == 'Unsuccessful')]

    # Clearance
    clearance = df3.loc[(df3['typedisplayName'] == 'Clearance') & (df3['outcomeTypedisplayName'] == 'Successful')]

    # Ball Recovery
    ballRecovery = df3.loc[(df3['typedisplayName'] == 'BallRecovery')]
    
    # Plotting the pitch
    pitch = VerticalPitch(pitch_type='opta',
                            pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=1, linewidth=3, spot_scale=0.005)

    pitch.draw(ax=a3)

    fig.set_facecolor('#E8E8E8')

    dfConvex = df3.loc[(df3['typedisplayName'] == 'BallRecovery') | (df3['typedisplayName'] == 'Clearance') |
                        (df3['typedisplayName'] == 'Aerial') | (df3['typedisplayName'] == 'Interception') |
                        (df3['typedisplayName'] == 'Tackle')].reset_index(drop=True)

    convex = dfConvex.loc[(np.abs(stats.zscore(dfConvex[['x','y']])) < 1).all(axis=1)]

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=a3, edgecolor='#181818', facecolor='#181818', alpha=0.3, linestyle='--', linewidth=2.5)

    pitch.scatter(tackle['x'], tackle['y'], ax=a3, marker='s', color='#fac404', edgecolor='#fac404', linestyle='--', s=150, label='Tackle', zorder=2)

    pitch.scatter(ballRecovery['x'], ballRecovery['y'], ax=a3, marker='8', edgecolor='#fac404', facecolor='none', hatch='//////', linestyle='--', s=150, label='Ball Recovery', zorder=2)

    pitch.scatter(aerial['x'], aerial['y'], ax=a3, marker='^', color='#fac404', edgecolor='#fac404', linestyle='--', s=150, label='Aerial', zorder=2)
    
    pitch.scatter(interception['x'], interception['y'], ax=a3, marker='P', color='#fac404', edgecolor='#fac404',  linestyle='--', s=150, label='Interception', zorder=2)

    pitch.scatter(clearance['x'], clearance['y'], ax=a3, marker='*', color='#fac404', edgecolor='#fac404', linestyle='--', s=200, label='Clearance', zorder=2)

    #pitch.scatter(pressure['x'], pressure['y'], ax=a3, marker='.', color='#fac404', edgecolor='#fac404', linestyle='--', s=200, label='Pressure', zorder=2)


    #Criação da legenda
    l = a3.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
    #Ciclo FOR para atribuir a color legend
    for text in l.get_texts():
            text.set_color("#181818")

def dashboardOffensive(events, league, club, playerName, matchDay):

    df = events.copy()

    color = ['#ff0000', '#181818']

    fig = plt.figure(figsize=(18,12), dpi = 500)
    grid = plt.GridSpec(8, 8)

    a1 = fig.add_subplot(grid[0:5, 0:2])
    a2 = fig.add_subplot(grid[0:5, 2:4])
    a3 = fig.add_subplot(grid[0:5, 4:6])
    a4 = fig.add_subplot(grid[0:5, 6:8])

    #################################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0],"fontweight": 'bold'},
    {"color": color[0],"fontweight": 'bold'}]

    # Club Logo
    add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.96, width=0.2, height=0.1)

    fig.set_facecolor('#E8E8E8')


    #fig_text(s='All Pases', color='#e4dst54', highlight_textprops = highlight_textprops)

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0],"fontweight": 'bold'},
    {"color": color[0],"fontweight": 'bold'}]


    fig_text(s =f'<{playerName}>' + "<'s>" + ' ' + 'performance',
                x = 0.5, y = 1.05, color='#181818', highlight_textprops = highlight_textprops, fontweight='bold', ha='center' ,fontsize=45);
    
    if matchDay != 'All Season':
            fig_text(s = 'MatchDay' + ' ' + str(matchDay) + '| Season 2022-23 | @menesesp20',
                    x = 0.35, y = 0.99,
                    color='#181818', fontweight='bold', ha='center' ,fontsize=12, alpha=0.8);

    elif matchDay == 'All Season':
            fig_text(s ='World Cup Catar 2022 | @menesesp20',
                    x = 0.32, y = 0.99,
                    color='#181818', fontweight='bold', ha='center' ,fontsize=12, alpha=0.8);

    fig_text(s = 'Territory Plot',
                x = 0.22, y = 0.88, color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    fig_text(s = 'Pass Plot',
                x = 0.41, y = 0.88, color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    fig_text(s = 'xT Plot',
                x = 0.61, y = 0.88, color='#181818', fontweight='bold', ha='center' ,fontsize=14);

    fig_text(s = 'Offensive Actions',
                x = 0.81, y = 0.88, color='#181818', fontweight='bold', ha='center' ,fontsize=14);
                
    if matchDay != 'All Season':
            events = events.loc[events.Match_ID == matchDay].reset_index(drop=True)
    else:
            events = events.copy()
    #################################################################################################################################################
    # 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE 1 FIGURE

    df1 = events.loc[(events['name'] == playerName) & (events['typedisplayName'] == 'Pass')]

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                            line_zorder=3, linewidth=3, spot_scale=0.00)

    pitch.draw(ax=a1)

    #################################################################################################################################################

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                    ['#E8E8E8', color[0]], N=10)
    bs = pitch.bin_statistic(df1['x'], df1['y'], bins=(12, 8))

    convex = df1[(np.abs(stats.zscore(df1[['x','y']])) < 1).all(axis=1)]

    pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a1, cmap=pearl_earring_cmap)

    pitch.scatter(df1['x'], df1['y'], ax=a1, edgecolor='#181818', facecolor='black', alpha=0.3)

    hull = pitch.convexhull(convex['x'], convex['y'])

    pitch.polygon(hull, ax=a1, edgecolor='#181818', facecolor='#181818', alpha=0.4, linestyle='--', linewidth=2.5)

    pitch.scatter(x=convex['x'].mean(), y=convex['y'].mean(), ax=a1, c='white', edgecolor=color[0], s=700, zorder=2)


    #################################################################################################################################################
    # 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGURE 2 FIGUR

    #df['beginning'] = np.sqrt(np.square(100 - df['x']) + np.square(100 - df['y']))
    #df['end'] = np.sqrt(np.square(100 - df['endX']) + np.square(100 - df['endY']))

    #df['progressive'] = [(df['end'][x]) / (df['beginning'][x]) < .75 for x in range(len(df.beginning))]

    player = events.loc[(events['name'] == playerName)]

    keyPass = player.loc[player['qualifiers'].apply(lambda x: 'KeyPass' in x)]

    Pass = player.loc[(player['typedisplayName'] == 'Pass')]

    sucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Successful')]

    unsucess = Pass.loc[(Pass['outcomeTypedisplayName'] == 'Unsuccessful')]

    #Progressive = Pass.loc[Pass['progressive'] == True]

    Pass_percentage = round((len(sucess) / len(Pass)) * 100, 2)

    #################################################################################################################################################
    pitch = VerticalPitch(pitch_type='opta', pad_top=0.1, pad_bottom=0.5,
            pitch_color='#E8E8E8', line_color='#181818',
            line_zorder=1, linewidth=3, spot_scale=0.00)

    pitch.draw(ax=a2)

    fig.set_facecolor('#E8E8E8')

    #Criação das setas que simbolizam os passes realizados bem sucedidos
    pitch.arrows(sucess['x'], sucess['y'], sucess['endX'], sucess['endY'], color='#181818', ax=a2,
            width=2, headwidth=5, headlength=5, label='Passes' + ':' + ' ' + '76' + ' ' + '(' + '88' + '%' + ' ' + 'Completion rate' + ')' )
    
    #Criação das setas que simbolizam os passes realizados bem sucedidos
    pitch.arrows(unsucess['x'], unsucess['y'], unsucess['endX'], unsucess['endY'], color='#cad2c5', ax=a2,
            width=2, headwidth=5, headlength=5, label='Passes unsuccessful' + ':' + ' '  + '9')

    #Criação das setas que simbolizam os passes realizados falhados
    #pitch.arrows(Progressive['x'], Progressive['y'], Progressive['endX'], Progressive['endY'], color='#00bbf9', ax=a2,
    #        width=2, headwidth=5, headlength=5, label='Progressive passes' + ':' + ' ' + f'{len(Progressive)}')

    #Criação das setas que simbolizam os passes realizados falhados
    pitch.arrows(keyPass['x'], keyPass['y'], keyPass['endX'], keyPass['endY'], color='#ffba08', ax=a2,
            width=2, headwidth=0.1, headlength=0.1, label='Key passes' + ':' + ' ' + f'{len(keyPass)}')
    
    pitch.scatter(keyPass['endX'], keyPass['endY'], s = 150, marker='*', color='#ffba08', ax=a2)

    #################################################################################################################################################

    #Criação da legenda
    l = a2.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
    #Ciclo FOR para atribuir a white color na legend
    for text in l.get_texts():
            text.set_color("#181818")

    #################################################################################################################################################
    # 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE 3 FIGURE

    xTDF = xT(events, data)

    xTheatMap = xTDF.loc[(xTDF.xT > 0) & (xTDF['name'] == playerName)]

    # setup pitch
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=3, linewidth=3, spot_scale=0.00)

    pitch.draw(ax=a3)

    fig.set_facecolor('#E8E8E8')

    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#E8E8E8', color[0]], N=10)

    bs = pitch.bin_statistic(xTheatMap['x'], xTheatMap['y'], bins=(12, 8))

    heatmap = pitch.heatmap(bs, edgecolors='#E8E8E8', ax=a3, cmap=pearl_earring_cmap)

    #################################################################################################################################################
    # 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE 4 FIGURE


    df4 = events.loc[(events['name'] == playerName)].reset_index(drop=True)
    
    # carry
    matchId = df.Match_ID.unique()
    dataAppend = []
    for game in matchId:
            data = carry(events, club, game, carrydf=None, progressive=None)
            dataAppend.append(data)

    carries = pd.concat(dataAppend)
    carries.reset_index(drop=True, inplace=True)
    
    carries = carries.loc[(carries.typedisplayName == 'Carry') & (carries.name == playerName)].reset_index(drop=True)

    #carriesProgressive = carry(events, club, matchDay, carrydf=None, progressive=None)
    #carriesProgressive = carriesProgressive.loc[(carriesProgressive.progressiveCarry == True) & (carries.name == playerName)].reset_index(drop=True)

    # deep_completion
    #deep_completion = df4.loc[df4['type.secondary'].apply(lambda x: 'deep_completion' in x)]

    # smart_pass
    smart_pass = df4.loc[df4['qualifiers'].apply(lambda x: 'KeyPass' in x)].reset_index(drop=True)

    # dribble
    dribble = df4.loc[df4['typedisplayName'] == 'TakeOn'].reset_index(drop=True)

    # Plotting the pitch
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                    line_zorder=1, linewidth=5, spot_scale=0.005)

    pitch.draw(ax=a4)

    fig.set_facecolor('#E8E8E8')

    #pitch.lines(carries['x'], carries['y'],
    #        carries['endX'], carries['endY'],
    #        lw=2, ls='dashed', label='Carry' + ':' + ' ' + f'{len(carries)}',
    #        color='#ffba08', ax=a4 ,zorder=4)

    #pitch.lines(carriesProgressive['x'], carriesProgressive['y'],
    #        carriesProgressive['endX'], carriesProgressive['endY'],
    #        lw=2, ls='dashed', label='Progressive Carry' + ':' + ' ' + f'{len(carriesProgressive)}',
    #        color='#ea04dc', ax=a4 ,zorder=4)

    #pitch.arrows(deep_completion['x'], deep_completion['y'],
    #             deep_completion['endX'], deep_completion['endY'],
    #             color=color[0], ax=a4, width=2, headwidth=5, headlength=5,
    #             label='Deep Completion' + ':' + ' ' + f'{len(deep_completion)}', zorder=4)

    pitch.arrows(smart_pass['x'], smart_pass['y'],
            smart_pass['endX'], smart_pass['endY'],
            color='#ffba08', ax=a4, width=2,headwidth=5, headlength=5,
            label='Key Pass' + ':' + ' ' + f'{len(smart_pass)}', zorder=4)

    pitch.scatter(dribble['x'], dribble['y'],
            s = 200, marker='*', color='#ffba08', ax=a4,
            label='Dribble' + ':' + ' ' + f'{len(dribble)}', zorder=4)


    #Criação da legenda
    l = a4.legend(bbox_to_anchor=(0, 0), loc='upper left', facecolor='#181818', framealpha=0, labelspacing=.7)
    #Ciclo FOR para atribuir a white color na legend
    for text in l.get_texts():
            text.set_color("#181818")

def halfspaces_Zone14(Game, league, club):

    Game = Game.loc[Game['team.name'] == club]

    fig, ax = plt.subplots(figsize=(22,18))

    pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                          line_zorder=1, linewidth=5, spot_scale=0.00)

    pitch.draw(ax=ax)

    fig.set_facecolor('#E8E8E8')

    ###################################################################################################################################

    fig.suptitle(club, fontsize=50, color='#181818', fontweight = "bold", y=0.97)

    Title = fig_text(s = 'Half Spaces Zone 14 passes | Season 21-22 | @menesesp20',
                     x = 0.51, y = 0.92, color='#181818', ha='center',
                     fontweight = "bold", fontsize=12);

    ###################################################################################################################################

    ZONE14 = patches.Rectangle([20.8, 68], width=58, height=15, linewidth = 2, linestyle='-',
                            edgecolor='#181818', facecolor='#ff0000', alpha=0.5, zorder=1 )

    HalfSpaceLeft = patches.Rectangle([67, 67.8], width=20, height=78, linewidth = 2, linestyle='-',
                            edgecolor='#181818', facecolor='#2894e5', alpha=0.5, zorder=1 )

    HalfSpaceRight = patches.Rectangle([13, 67.8], width=20, height=78, linewidth = 2, linestyle='-',
                            edgecolor='#181818', facecolor='#2894e5', alpha=0.5, zorder=1 )

    ###################################################################################################################################

    # HALF SPACE LEFT

    halfspaceleft = Game[(Game['pass.endLocation.y'] <= 83) & (Game['pass.endLocation.y'] >= 65) &
                                  (Game['pass.endLocation.x'] >= 78) &
                                  (Game['pass.accurate'] == True)]

    pitch.arrows(xstart=halfspaceleft['location.x'], ystart=halfspaceleft['location.y'],
                                        xend=halfspaceleft['pass.endLocation.x'], yend=halfspaceleft['pass.endLocation.y'],
                                        color='#2894e5', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    # ZONE14

    zone14 = Game[(Game['pass.endLocation.x'] <= 83) & (Game['pass.endLocation.x'] >= 75) &
                          (Game['pass.endLocation.y'] <= 66) & (Game['pass.endLocation.y'] >= 35) &
                          (Game['pass.accurate'] == True)]

    pitch.arrows(xstart=zone14['location.x'], ystart=zone14['location.y'],
                                        xend=zone14['pass.endLocation.x'], yend=zone14['pass.endLocation.y'],
                                        color='#ff0000', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    # HALF SPACE RIGHT

    halfspaceright = Game[(Game['pass.endLocation.y'] >= 17) & (Game['pass.endLocation.y'] <= 33) &
                          (Game['pass.endLocation.x'] >= 78) &
                          (Game['pass.accurate'] == True)]

    pitch.arrows(xstart=halfspaceright['location.x'], ystart=halfspaceright['location.y'],
                                        xend=halfspaceright['pass.endLocation.x'], yend=halfspaceright['pass.endLocation.y'],
                                        color='#2894e5', alpha=0.8,
                                        lw=3, zorder=3,
                                        ax=ax)

    ###################################################################################################################################

    ax.add_patch(ZONE14)
    ax.add_patch(HalfSpaceLeft)
    ax.add_patch(HalfSpaceRight)

    ###################################################################################################################################

    # Club Logo
    fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.24, bottom=0.885, width=0.2, height=0.1)

def finalThird(df, league, club, matchDay):

        if matchDay != 'All Season':
                # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                final3rd = df.loc[(df['typedisplayName'] == 'Pass') & (df['team'] == club) &
                                        (df['x'] >= 55) & (df['Match_ID'] == matchDay)][['team', 'name', 'x', 'y', 'endX', 'endY', 'typedisplayName', 'outcomeTypedisplayName', 'xG', 'xGOT']]

        elif matchDay == 'All Season':
                # DATAFRAME WITH ALL PASSES IN THE FINAL THIRD
                final3rd = df.loc[(df['qualifiers'].str.contains('KeyPass') == True) &
                                        (df['team'] == club) & (df['x'] >= 55)][['team', 'name', 'x', 'y', 'endX', 'endY', 'typedisplayName', 'outcomeTypedisplayName', 'xG', 'xGOT']]

        # DATAFRAME WITH ALL PASSES IN THE LEFT FINAL THIRD
        #67 LEFT, RIGHT 33, MID BEETWEN THEM
        leftfinal3rd = final3rd[(final3rd['y'] >= 67)]

        # PERCENTAGE OF ATTACKS IN THE LEFT SIDE
        leftfinal3rdTotal = round((len(leftfinal3rd) / len(final3rd)) * 100 ,1)

        # DATAFRAME WITH ALL PASSES IN THE CENTER FINAL THIRD
        centerfinal3rd = final3rd[(final3rd['y'] < 67) & (final3rd['y'] > 33)]

        # PERCENTAGE OF ATTACKS IN THE CENTER SIDE
        centerfinal3rdTotal = round((len(centerfinal3rd) / len(final3rd)) * 100 ,1)

        # DATAFRAME WITH ALL PASSES IN THE RIGHT FINAL THIRD
        rightfinal3rd = final3rd[(final3rd['y'] <= 33)]

        # PERCENTAGE OF ATTACKS IN THE RIGHT SIDE
        rightfinal3rdTotal = round((len(rightfinal3rd) / len(final3rd)) * 100 ,1)

        #################################################################################################################################################

        final3rd_Cluster = cluster_Event(df, club, 'KeyPass', 2)

        final3rd_Cluster0 = final3rd_Cluster.loc[final3rd_Cluster.cluster == 0]
        final3rd_Cluster1 = final3rd_Cluster.loc[final3rd_Cluster.cluster == 1]
        
        x_mean0 = final3rd_Cluster0.x.mean()
        y_mean0 = final3rd_Cluster0.y.mean()

        x_end_mean0 = final3rd_Cluster0.endX.mean()
        y_end__mean0 = final3rd_Cluster0.endY.mean()

        x_mean1 = final3rd_Cluster1.x.mean()
        y_mean1 = final3rd_Cluster1.y.mean()

        x_end_mean1 = final3rd_Cluster1.endX.mean()
        y_end__mean1 = final3rd_Cluster1.endY.mean()

        final3rd_Cluster.loc[len(final3rd_Cluster.index)] = [club, 'Pass', 'Qualifiers', x_mean0, y_mean0, x_end_mean0, y_end__mean0, 'mean0']
        final3rd_Cluster.loc[len(final3rd_Cluster.index)] = [club, 'Pass', 'Qualifiers', x_mean1, y_mean1, x_end_mean1, y_end__mean1, 'mean1']

        #################################################################################################################################################
        
        df2 = df.loc[(df['typedisplayName'] == 'Pass') & (df['outcomeTypedisplayName'] == 'Successful')].reset_index(drop=True)

        xTDF = xT(df2)

        DFSides = sides(xTDF, club)
        
        dfxG = sides(df, club)

        xT_Sides = dataFrame_xTFlow(DFSides)
                
        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

        pitch = VerticalPitch(pitch_type='opta',
                        pitch_color='#E8E8E8', line_color='#181818', half = True,
                        line_zorder=2, linewidth=2,
                        spot_scale=0.0005)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        if matchDay != 'All Season':
                Title = df.loc[df['Match_ID'] == matchDay]
                
                homeName = Title.home_Team.unique()[0]
                color = clubColors.get(homeName)

                awayName = Title.away_Team.unique()[0]
                color2 = clubColors.get(awayName)

        #################################################################################################################################################

                #Params for the text inside the <> this is a function to highlight text
                highlight_textprops =\
                        [{"color": color[0],"fontweight": 'bold'},
                        {"color": color2[0],"fontweight": 'bold'}
                        ]

                fig_text(s =f'<{homeName}>' + ' ' + 'vs' + ' ' + f'<{awayName}>',
                         x = 0.53, y = 0.98, ha='center', va='center',
                         highlight_textprops = highlight_textprops ,
                         color='#181818', fontproperties=custom_font,
                         fontsize=35);
                
                fig_text(s =  'Champions League' + ' ' + '|' + ' '  + str(matchDay) + ' ' + '| Season 2022-23 | @menesesp20',
                         x = 0.51, y = 0.94,
                         color='#181818', fontproperties=custom_font,
                         ha='center', va='center',
                         fontsize=8);

        #################################################################################################################################################

        elif matchDay == 'All Season':
                # Title of our plot
                fig.suptitle(club + ' ' + 'Open Play',
                             fontsize=35, color='#1b1b1b',
                             fontproperties=custom_font,
                             x=0.525, y=1)

                fig_text(s = "Key Passes | Made by: @menesesp20",
                         x = 0.5, y = 0.95,
                         color='#181818', fontproperties=custom_font,
                         ha='center',
                         fontsize=11);

        #################################################################################################################################################
        # RIGHT
        fig_text(s = str(rightfinal3rdTotal) + ' ' + '%',
                x = 0.75, y = 0.5,
                color='black', fontproperties=custom_font, ha='center' ,fontsize=25);

        # xT Right
        ax.scatter( 14 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 10000, zorder=3)

        fig_text(s =str(round(xT_Sides.right_xT[0], 2)),
                x = 0.73, y = 0.37,
                color='black',fontproperties=custom_font, ha='center' ,fontsize=18);

        fig_text(s = str(round(dfxG.loc[dfxG.side == 'Right']['xG'].sum(), 2)),
                x = 0.73, y = 0.25,
                color='black', fontproperties=custom_font, ha='center', fontsize=18);

        #################################################################################################################################################
        # LEFT
        fig_text(s = str(leftfinal3rdTotal) + ' ' + '%',
                x = 0.292, y = 0.5,
                color='black', fontproperties=custom_font, ha='center' ,fontsize=25);

        # xT Left
        ax.scatter( 83 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 10000, zorder=3)

        fig_text(s = str(round(xT_Sides.left_xT[0], 2)),
                x = 0.32, y = 0.37,
                color='black', fontproperties=custom_font, ha='center' ,fontsize=18);

        fig_text(s = str(round(dfxG.loc[dfxG.side == 'Left']['xG'].sum(), 2)),
                x = 0.32, y = 0.25,
                color='black', fontproperties=custom_font, ha='center', fontsize=18);

        #################################################################################################################################################
        # CENTER
        fig_text(s = str(centerfinal3rdTotal) + ' ' + '%',
                x = 0.525, y = 0.5,
                color='black', fontproperties=custom_font, ha='center' ,fontsize=25);

        # xT Center
        ax.scatter( 49.5 , 64.3 , marker ='d', lw=2, edgecolor='black', facecolor='None', s = 10000, zorder=3)

        fig_text(s = str(round(xT_Sides.center_xT[0], 2)),
                x = 0.515, y = 0.37,
                color='black', fontproperties=custom_font, ha='center' ,fontsize=18);

        fig_text(s = str(round(dfxG.loc[dfxG.side == 'Center']['xG'].sum(), 2)),
                x = 0.515, y = 0.25,
                color='black', fontproperties=custom_font, ha='center', fontsize=18);
        
        #################################################################################################################################################

        left =  str(leftfinal3rdTotal)
        center = str(centerfinal3rdTotal)
        right = str(rightfinal3rdTotal)

        if right > left > center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        elif right > center > left:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        ##################################################################################################################

        elif left > right > center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)


        elif left > center > right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1)



        ##################################################################################################################

        elif center > left > right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1)

        elif center > right > left:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.3, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)

        ##################################################################################################################

        elif left == center:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1)

        ##################################################################################################################

        elif left == right:

                # LEFT ZONE
                rectangleLeft = patches.Rectangle([67, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1 )

                # CENTER ZONE
                rectangleCenter = patches.Rectangle([33.1, 50], width=33.8, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.8, zorder=1 )

                # RIGHT ZONE
                rectangleRight = patches.Rectangle([0, 50], width=33, height=50, linewidth = 2, linestyle='-',
                                edgecolor='black', facecolor='#ff0000', alpha=0.5, zorder=1)
                
        # ADD RECTANGLES
        ax.add_patch(rectangleLeft)
        ax.add_patch(rectangleCenter)
        ax.add_patch(rectangleRight)
        
        # Key Passes Cluster
        for x in range(len(final3rd_Cluster['cluster'])):

                if final3rd_Cluster['cluster'][x] == 0:
                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                color='#ea04dc',
                                ax=ax,
                                zorder=5,
                                comet=True,
                                transparent=True,
                                alpha=0.1)

                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                s = 150,
                                c='#ea04dc',
                                edgecolor='#ffffff',
                                ax=ax,
                                zorder=5,
                                alpha=0.1)

                elif final3rd_Cluster['cluster'][x] == 1:
                        pitch.lines(xstart=final3rd_Cluster['x'][x], ystart=final3rd_Cluster['y'][x],
                                xend=final3rd_Cluster['endX'][x], yend=final3rd_Cluster['endY'][x],
                                color='#2d92df',
                                ax=ax,
                                zorder=5,
                                comet=True,
                                transparent=True,
                                alpha=0.1)

                        pitch.scatter(final3rd_Cluster['endX'][x], final3rd_Cluster['endY'][x],
                                s = 150,
                                c='#2d92df',
                                edgecolor='#ffffff',
                                ax=ax,
                                zorder=5,
                                alpha=0.2)
                                
        #################################################################################################################################################

        fig_text(s = 'Most frequent zone',
                 x = 0.34, y = 0.88,
                 color='#ea04dc', fontproperties=custom_font, ha='center' ,fontsize=8);

        fig_text(s = 'Second most frequent zone',
                 x = 0.45, y = 0.88,
                 color='#2d92df', fontproperties=custom_font, ha='center' ,fontsize=8);

        fig_text(s = 'Third most frequent zone',
                 x = 0.57, y = 0.88,
                 color='#fb8c04', fontproperties=custom_font, ha='center' ,fontsize=8);

        # Club Logo
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.1, bottom=0.91, width=0.2, height=0.1)

        # END NOTE
        fig_text(s = 'The values inside the diamond are the xT value for each third',
                 x = 0.5, y = 0.125,
                 color='#181818', fontproperties=custom_font, ha='center' ,fontsize=11);

        fig_text(s = 'xT values based on Karun Singhs model',
                 x = 0.765, y = 0.875,
                 color='#181818', fontproperties=custom_font, ha='center' ,fontsize=8);

def cornersTaken(df, league, club):

        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []
                
        df_Corner = search_qualifierOPTA(df, cornersData, 'CornerTaken')

        right_corner = df_Corner.loc[df_Corner['y'] < 50]

        left_corner = df_Corner.loc[df_Corner['y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18, 14), dpi=300)

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#E8E8E8', line_color='#181818', half = True,
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + 'Corners', fontsize=50, color='#181818', fontweight = "bold", x=0.5, y=0.955, ha='center', va='center')

        Title = fig_text(s = 'Season 22-23 | Made by: @menesesp20',
                         x = 0.5, y = 0.894,
                         color='#181818', fontweight='bold', ha='center', va='center', fontsize=16);

        #################################################################################################################################################

        firstCorner_L_Cluster = cluster_Event(left_corner, club, 'CornerTaken', 3)

        firstCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        firstCorner_R_Cluster = cluster_Event(right_corner, club, 'CornerTaken', 3)

        firstCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)
                
        #################################################################################################################################################

        # RIGHT SIDE CLUSTER
        for x in range(len(firstCorner_R_Cluster['cluster'])):

                if firstCorner_R_Cluster['cluster'][x] == 2:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_R_Cluster['x'][x], firstCorner_R_Cluster['y'][x],
                                firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                color='#ea04dc',
                                ax=ax,
                                zorder=3,
                                comet=True,
                                transparent=True,
                                alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                s = 100,
                                marker='o',
                                c='#1b1b1b',
                                edgecolor='#ea04dc',
                                ax=ax,
                                zorder=4)


        # CIRCLE                            
        ax.scatter( 40 , 95 , s = 20000, color='#eb00e5', alpha=0.5, lw=3)

        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=3))

        fig_text(s = 'Most frequent zone',
                 x = 0.794, y = 0.66,
                 color='#eb00e5', fontweight='bold', ha='center', va='center', fontsize=12);

        #################################################################################################################################################
        # LEFT SIDE CLUSTER
        for x in range(len(firstCorner_L_Cluster['cluster'])):        
                if firstCorner_L_Cluster['cluster'][x] == 1:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_L_Cluster['x'][x], firstCorner_L_Cluster['y'][x],
                                firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                color='#2d92df',
                                ax=ax,
                                zorder=3,
                                comet=True,
                                transparent=True,
                                alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                s = 100,
                                marker='o',
                                c='#1b1b1b',
                                edgecolor='#2d92df',
                                ax=ax,
                                zorder=4)
                
        # CIRCLE                            
        ax.scatter( 60 , 95 , s = 20000, color='#2894e5', alpha=0.5, lw=3)

        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=3))

        fig_text(s = 'Most frequent zone',
                 x = 0.23, y = 0.66,
                 color='#2894e5', fontweight='bold', ha='center', va='center', fontsize=12);

        #################################################################################################################################################

        # PENTAGON RIGHT                          
        ax.scatter( 40 , 65 , marker = 'p', s = 20000, color='#eb00e5', alpha=0.5, lw=3)

        fig_text(s =  str(len(firstCorner_R_Cluster)),
                        x = 0.584, y = 0.378,
                        color='#181818', fontweight='bold', ha='center' ,fontsize=30);

        #################################################################################################################################################

        # PENTAGON LEFT                           
        ax.scatter( 60 , 65 , marker = 'p', s = 20000, color='#2894e5', alpha=0.5, lw=3)

        fig_text(s = str(len(firstCorner_L_Cluster)),
                 x = 0.44, y = 0.378,
                 color='#181818', fontweight='bold', ha='center' ,fontsize=30);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.08, bottom=0.87, width=0.2, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the total of corners made by each side',
                x = 0.338, y = 0.129,
                color='#181818', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Coach: Roger Schmidt',
                x = 0.22, y = 0.863,
                color='#181818', fontweight='bold', ha='center', alpha=0.8, fontsize=14);

def corners1stPostTaken(df, league, club):
        
        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []

        df_Corner = df.loc[df['type.primary'] == 'corner'].reset_index(drop=True)

        right_corner = df_Corner.loc[df_Corner['y'] < 50]

        left_corner = df_Corner.loc[df_Corner['y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18,14))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#1b1b1b', line_color='white', half = True,
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#1b1b1b')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + '1st Post Corners', fontsize=40, color='white',
                      fontweight = "bold", x=0.525, y=0.955)

        Title = fig_text(s = 'Season 22-23 | Made by: @Menesesp20',
                         x = 0.5, y = -0.91,
                         color='white', fontweight='bold', ha='center' ,fontsize=16);

        #################################################################################################################################################

        firstCorner_L = left_corner.loc[(left_corner['endY'] >= 55) & (left_corner['endY'] <= 79)]

        firstCorner_L_Cluster = cluster_Event(firstCorner_L, club, 'corner', 3)

        firstCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        firstCorner_R = right_corner.loc[(right_corner['endY'] <= 45) & (right_corner['endY'] >= 21)]

        firstCorner_R_Cluster = cluster_Event(firstCorner_R, club, 'corner', 3)

        firstCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # RIGHT SIDE CLUSTER
        for x in range(len(firstCorner_R_Cluster['cluster'])):

                if firstCorner_R_Cluster['cluster'][x] == 1:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_R_Cluster['x'][x], firstCorner_R_Cluster['y'][x],
                                    firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                    color='#ea04dc',
                                    ax=ax,
                                    zorder=3,
                                    comet=True,
                                    transparent=True,
                                    alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_R_Cluster['endX'][x], firstCorner_R_Cluster['endY'][x],
                                      s = 100,
                                      marker='o',
                                      c='#1b1b1b',
                                      edgecolor='#ea04dc',
                                      ax=ax,
                                      zorder=4)
        # CIRCLE                            
        ax.scatter( 40 , 95 , s = 20000, color='#eb00e5', alpha=0.5, lw=3)

        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.794, y = 0.66,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # LEFT SIDE CLUSTER
        for x in range(len(firstCorner_L_Cluster['cluster'])):        
                if firstCorner_L_Cluster['cluster'][x] == 0:
                        #Criação das setas que simbolizam os passes realizados falhados
                        pitch.lines(firstCorner_L_Cluster['x'][x], firstCorner_L_Cluster['y'][x],
                                    firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                    color='#2d92df',
                                    ax=ax,
                                    zorder=3,
                                    comet=True,
                                    transparent=True,
                                    alpha_start=0.2,alpha_end=0.8)
                
                        pitch.scatter(firstCorner_L_Cluster['endX'][x], firstCorner_L_Cluster['endY'][x],
                                      s = 100,
                                      marker='o',
                                      c='#1b1b1b',
                                      edgecolor='#2d92df',
                                      ax=ax,
                                      zorder=4)
        # CIRCLE                            
        ax.scatter( 60 , 95 , s = 20000, color='#2894e5', alpha=0.5, lw=3)

        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.23, y = 0.66,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # PENTAGON RIGHT                          
        ax.scatter( 40 , 65 , marker = 'p', s = 20000, color='#eb00e5', alpha=0.5, lw=3)

        # VALUE FIRST CORNER MOST FREQUENT ON RIGHT SIDE

        firstCornerR =  int((len(firstCorner_R) / len(right_corner) * 100))

        fig_text(s =  str(firstCornerR) + '%',
                        x = 0.584, y = 0.378,
                        color='white', fontweight='bold', ha='center' ,fontsize=28);

        #################################################################################################################################################

        # PENTAGON LEFT                           
        ax.scatter( 60 , 65 , marker = 'p', s = 20000, color='#2894e5', alpha=0.5, lw=3)

        # VALUE FIRST CORNER MOST FREQUENT ON LEFT SIDE

        firstCornerL = int((len(firstCorner_L) / len(left_corner) * 100))

        fig_text(s = str(firstCornerL) + '%',
                        x = 0.44, y = 0.378,
                        color='white', fontweight='bold', ha='center' ,fontsize=28);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.12, bottom=0.87, width=0.2, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the percentage of corners made by each side for the circle area',
                x = 0.407, y = 0.14,
                color='white', fontweight='bold', ha='center' ,fontsize=12);

def corners2ndPostTaken(df, league, club):
        
        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass
        
        cornersData = []

        df_Corner = df.loc[df['type.primary'] == 'corner'].reset_index(drop=True)

        right_corner = df_Corner.loc[df_Corner['location.y'] < 50]

        left_corner = df_Corner.loc[df_Corner['location.y'] > 50]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(18,14))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#1b1b1b', line_color='white', half = True,
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#1b1b1b')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + '2nd Post Corners', fontsize=40, color='white',
        fontweight = "bold", x=0.525, y=0.955)

        Title = fig_text(s = 'Season 21-22 | Made by: @Menesesp20',
                        x = 0.5, y = 0.91,
                        color='white', fontweight='bold', ha='center' ,fontsize=16);

        #################################################################################################################################################

        secondCorner_L = left_corner.loc[(left_corner['pass.endLocation.y'] <= 55) & (left_corner['pass.endLocation.y'] >= 21) & (left_corner['pass.endLocation.x'] >= 90)]
        if secondCorner_L.shape[0] == 0:
                pass
        else:
                secondCorner_L_Cluster = cluster_Event(secondCorner_L, club, 'corner', 2)

                secondCorner_L_Cluster['cluster'].value_counts().reset_index(drop=True)

                # LEFT SIDE CLUSTER
                for x in range(len(secondCorner_L_Cluster['cluster'])):        
                        if secondCorner_L_Cluster['cluster'][x] == 0:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(secondCorner_L_Cluster['location.x'][x], secondCorner_L_Cluster['location.y'][x],
                                        secondCorner_L_Cluster['pass.endLocation.x'][x], secondCorner_L_Cluster['pass.endLocation.y'][x],
                                        color='#ea04dc',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(secondCorner_L_Cluster['pass.endLocation.x'][x], secondCorner_L_Cluster['pass.endLocation.y'][x],
                                        s = 100,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#ea04dc',
                                        ax=ax,
                                        zorder=4)
                
                # CIRCLE 2nd Post                           
                ax.scatter( 40 , 95 , s = 20000, color='#2894e5', alpha=0.5, lw=3)

                # PENTAGON LEFT                           
                ax.scatter( 60 , 65 , marker = 'p', s = 20000, color='#2894e5', alpha=0.5, lw=3)

                len2ndCornerL = len(secondCorner_L_Cluster.loc[secondCorner_L_Cluster['cluster']==0])

                secondCornerL = int((len(secondCorner_L) / len(left_corner) * 100))

                fig_text(s = str(secondCornerL) + '%',
                                x = 0.44, y = 0.378,
                                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=28);

        #################################################################################################################################################

        secondCorner_R = right_corner.loc[(right_corner['pass.endLocation.y'] <= 75) & (right_corner['pass.endLocation.y'] >= 55) & (right_corner['pass.endLocation.x'] >= 90)]
        if secondCorner_R.shape[0] == 0:
                pass
        else:
                secondCorner_R_Cluster = cluster_Event(secondCorner_R, club, 'corner', 3)
                
                secondCorner_R_Cluster['cluster'].value_counts().reset_index(drop=True)

                # RIGHT SIDE CLUSTER
                for x in range(len(secondCorner_R_Cluster['cluster'])):

                        if secondCorner_R_Cluster['cluster'][x] == 1:
                                #Criação das setas que simbolizam os passes realizados falhados
                                pitch.lines(secondCorner_R_Cluster['location.x'][x], secondCorner_R_Cluster['location.y'][x],
                                        secondCorner_R_Cluster['pass.endLocation.x'][x], secondCorner_R_Cluster['pass.endLocation.y'][x],
                                        color='#2d92df',
                                        ax=ax,
                                        zorder=3,
                                        comet=True,
                                        transparent=True,
                                        alpha_start=0.2,alpha_end=0.8)
                        
                                pitch.scatter(secondCorner_R_Cluster['pass.endLocation.x'][x], secondCorner_R_Cluster['pass.endLocation.y'][x],
                                        s = 100,
                                        marker='o',
                                        c='#1b1b1b',
                                        edgecolor='#2d92df',
                                        ax=ax,
                                        zorder=4)
                # CIRCLE 1st Post                           
                ax.scatter( 60 , 95 , s = 20000, color='#eb00e5', alpha=0.5, lw=3)            

                # PENTAGON RIGHT                          
                ax.scatter( 40 , 65 , marker = 'p', s = 20000, color='#eb00e5', alpha=0.5, lw=3)

                len2ndCornerR = len(secondCorner_R_Cluster.loc[secondCorner_R_Cluster['cluster']==0])

                secondCornerR = int((len(secondCorner_R) / len(right_corner) * 100))

                fig_text(s =  str(secondCornerR) + '%',
                                x = 0.584, y = 0.378,
                                color='white', fontweight='bold', ha='center' ,fontsize=30);


        #################################################################################################################################################

        # MOST FREQUENT ZONES ARROWS
        ax.annotate('', xy=(18, 84), xytext=(5, 84),
                size=14, color = '#eb00e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#eb00e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.794, y = 0.66,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # MOST FREQUENT ZONES ARROWS
        ax.annotate('', xy=(83, 84), xytext=(95, 84),
                size=14, color = '#2894e5', fontweight = "bold",
                arrowprops=dict(arrowstyle="->", color='#2894e5', lw=3))

        fig_text(s = 'Most frequent zone',
                x = 0.23, y = 0.66,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.12, bottom=0.87, width=0.2, height=0.1)

        #################################################################################################################################################

        # Angle Left Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleLeft.png', fig=fig, left=0.082, bottom=0.842, width=0.2, height=0.1)

        # ANGLE LEFT VALUE
        #fig_text(s = '4.6°',
        #                x = 0.179, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        #################################################################################################################################################

        # Angle Right Logo
        #fig = add_image(image='C:/Users/menes/Documents/Data Hub/angleRight.png', fig=fig, left=0.7425, bottom=0.842, width=0.2, height=0.1)

        # ANGLE RIGHT VALUE
        #fig_text(s = '1.8°',
        #                x = 0.846, y = 0.887,
        #                fontfamily = 'medium', color='white', fontweight='bold', ha='center' ,fontsize=15);

        fig_text(s = 'The values inside pentagon are the percentage of corners made by each side for the circle area',
                x = 0.407, y = 0.129,
                color='white', fontweight='bold', ha='center' ,fontsize=12);

def SetPiece_throwIn(df, league, club, match=None):

        if 'level_0' in df.columns:
                df.drop(['level_0'], axis=1, inplace=True)
        else:
                pass

        throwIn = []

        throwIn = df.loc[df['type.primary'] == 'throw_in'].reset_index(drop=True)

        if match != None:
                match = df.loc[df.Match_ID == match]
        else:
                match = df.copy()

        #################################################################################################################################################

        # DEFEND SIDE
        defendLeft = match.loc[(match['location.x'] < 35) & (match['location.y'] > 50)]

        defendRight = match.loc[(match['location.x'] < 35) & (match['location.y'] < 50)]

        # MIDDLE SIDE
        middleLeft = match.loc[(match['location.x'] > 35) & (match['location.x'] < 65) & (match['location.y'] > 50)]

        middleRight = match.loc[(match['location.x'] > 35) & (match['location.x'] < 65) & (match['location.y'] < 50)]

        # ATTACK SIDE
        attackLeft = match.loc[(match['location.x'] > 65) & (match['location.y'] > 50)]

        attackRight = match.loc[(match['location.x'] > 65) & (match['location.y'] < 50)]

        #################################################################################################################################################

        # Plotting the pitch

        fig, ax = plt.subplots(figsize=(21,15))

        pitch = VerticalPitch(pitch_type='opta',
                              pitch_color='#E8E8E8', line_color='#181818',
                              line_zorder=1, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #################################################################################################################################################

        # Title of our plot - WITH ANGLES BOTTOM: 0.98, 0.93

        fig.suptitle(club + ' ' + "Throw-In's", fontsize=45, color='#181818',
                     fontweight = "bold", x=0.545, y=0.955)

        Title = fig_text(s = 'Season 21-22 | Made by: @Menesesp20',
                         x = 0.54, y = 0.91,
                         color='#181818', fontweight='bold', ha='center' ,fontsize=14);

        #################################################################################################################################################
        # DEFEND SIDE CLUSTER
        defendLeft_Cluster = cluster_Event(defendLeft, club, 'throw_in', 2)

        defendLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        defendRight_Cluster = cluster_Event(defendRight, club, 'throw_in', 3)

        defendRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # MIDDLE SIDE CLUSTER
        middleLeft_Cluster = cluster_Event(middleLeft, club, 'throw_in', 1)

        middleLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        middleRight_Cluster = cluster_Event(middleRight, club, 'throw_in', 3)

        middleRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        #################################################################################################################################################

        # ATTACK SIDE CLUSTER
        attackLeft_Cluster = cluster_Event(attackLeft, club, 'throw_in', 2)

        attackLeft_Cluster['cluster'].value_counts().reset_index(drop=True)

        attackRight_Cluster = cluster_Event(attackRight, club, 'throw_in', 3)

        attackRight_Cluster['cluster'].value_counts().reset_index(drop=True)

        ####################################################################################################################################################
        # DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND
        #################################################################################################################################################
        if defendLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(defendLeft_Cluster['cluster'])):
                        
                        if defendLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=defendLeft_Cluster['location.x'][x], ystart=defendLeft_Cluster['location.y'][x],
                                        xend=defendLeft_Cluster['pass.endLocation.x'][x], yend=defendLeft_Cluster['pass.endLocation.y'][x],
                                        color='#eb00e5',
                                        lw=3, zorder=2,
                                        ax=ax)
        ####################################################################################################################################################
        # DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND DEFEND
        ####################################################################################################################################################

        if defendRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(defendRight_Cluster['cluster'])):
                        
                        if defendRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=defendRight_Cluster['location.x'][x], ystart=defendRight_Cluster['location.y'][x],
                                        xend=defendRight_Cluster['pass.endLocation.x'][x], yend=defendRight_Cluster['pass.endLocation.y'][x],
                                        color='#2894e5',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE
        ####################################################################################################################################################

        if middleLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(middleLeft_Cluster['cluster'])):
                        
                        if middleLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=middleLeft_Cluster['location.x'][x], ystart=middleLeft_Cluster['location.y'][x],
                                        xend=middleLeft_Cluster['pass.endLocation.x'][x], yend=middleLeft_Cluster['pass.endLocation.y'][x],
                                        color='#ffe506',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE MIDDLE
        ####################################################################################################################################################

        if middleRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(middleRight_Cluster['cluster'])):
                        
                        if middleRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=middleRight_Cluster['location.x'][x], ystart=middleRight_Cluster['location.y'][x],
                                        xend=middleRight_Cluster['pass.endLocation.x'][x], yend=middleRight_Cluster['pass.endLocation.y'][x],
                                        color='#ffe506',
                                        lw=3, zorder=2,
                                        ax=ax)

        ####################################################################################################################################################
        # ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK
        ####################################################################################################################################################
        if attackLeft_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(attackLeft_Cluster['cluster'])):
                        
                        if attackLeft_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=attackLeft_Cluster['location.x'][x], ystart=attackLeft_Cluster['location.y'][x],
                                        xend=attackLeft_Cluster['pass.endLocation.x'][x], yend=attackLeft_Cluster['pass.endLocation.y'][x],
                                        color='#eb00e5',
                                        lw=3, zorder=2,
                                        ax=ax)
                                        
        #################################################################################################################################################
        # ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK  ATTACK
        #################################################################################################################################################

        if attackRight_Cluster.shape[0] == 0:
                pass
        else:
                for x in range(len(attackRight_Cluster['cluster'])):
                        
                        if attackRight_Cluster['cluster'][x] == 0:
                                pitch.arrows(xstart=attackRight_Cluster['location.x'][x], ystart=attackRight_Cluster['location.y'][x],
                                        xend=attackRight_Cluster['pass.endLocation.x'][x], yend=attackRight_Cluster['pass.endLocation.y'][x],
                                        color='#2894e5',
                                        lw=3, zorder=2,
                                        ax=ax)

        #################################################################################################################################################

        fig_text(s = 'Blue - Right Side',
                x = 0.648, y = 0.12,
                color='#2894e5', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Purple - Left Side',
                x = 0.38, y = 0.12,
                color='#eb00e5', fontweight='bold', ha='center' ,fontsize=12);

        fig_text(s = 'Yellow - Middle Side',
                x = 0.518, y = 0.12,
                color='#ffe506', fontweight='bold', ha='center' ,fontsize=12);

        #################################################################################################################################################

        ax.axhline(35,c='#181818', ls='--', lw=4)
        ax.axhline(65,c='#181818', ls='--', lw=4)

        #################################################################################################################################################

        # ATTACK
        #fig_text(s = '12',
        #        x = 0.512, y = 0.683,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 27 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # MIDDLE

        #fig_text(s = '12',
        #        x = 0.512, y = 0.518,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 50 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # DEFENSE

        #fig_text(s = '12',
        #        x = 0.512, y = 0.348,
        #        fontfamily = 'medium', color='Black', fontweight='bold', ha='center' ,fontsize=30);

        #ax.scatter( 50 , 72 , marker = 'p', s = 12000, color='#181818', alpha=0.8, lw=3)

        # Club Logo - WITH ANGLES BOTTOM: 0.89, LEFT:0.14
        fig = add_image(image='Images/Clubs/' + league + '/' + club + '.png', fig=fig, left=0.23, bottom=0.90, width=0.2, height=0.07)

def field_Tilt(df, axis=False):
    
    touch = df.loc[(df['isTouch'] == True) & (df['x'] >=75) & (df['endX'] >=75)].reset_index(drop=True)

    #############################################################################################################################################

    league = 'Champions'

    home = touch['home_Team'].unique()[0]
    color = clubColors.get(home)

    away = touch['away_Team'].unique()[0]
    color2 = clubColors.get(away)

    home_Passes = df.loc[(df['typedisplayName'] == 'Pass') & (df['team'] == home)]['typedisplayName'].count()
    away_Passes = df.loc[(df['typedisplayName'] == 'Pass') & (df['team'] == away)]['typedisplayName'].count()

    passes_Total = df.loc[(df['typedisplayName'] == 'Pass')]['typedisplayName'].count()


    home_Passes = int(home_Passes)
    home_Passes = round((home_Passes / int(passes_Total)) * 100, 2)
    
    away_Passes = int(away_Passes)
    away_Passes = round((away_Passes / int(passes_Total)) * 100, 2)

    #############################################################################################################################################


    fieldTilt_Home = touch.loc[touch['team'] == home]

    fieldTilt_Home = round((len(fieldTilt_Home) / len(touch)) * 100, 2)

    fieldTilt_Away = touch.loc[touch['team'] == away]

    fieldTilt_Away = round((len(fieldTilt_Away) / len(touch)) * 100, 2)

    #############################################################################################################################################

    if axis == False:
        # Plotting the pitch
        fig, ax = plt.subplots(figsize=(18,14), dpi=500)

        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=1, linewidth=5, spot_scale=0.005)

        pitch.draw(ax=ax)

        fig.set_facecolor('#E8E8E8')

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
            [{"color": color[0],"fontweight": 'bold'},
            {"color": color2[0],"fontweight": 'bold'}
            ]

        fig_text(s =f'<{home}>' + ' ' + 'vs' + ' ' + f'<{away}>' + ' ' + 'Field Tilt',
                x = 0.55, y = 0.93,
                ha='center', va='center',
                highlight_textprops = highlight_textprops, 
                color='#181818', fontweight='bold',
                fontsize=50);
        
        fig_text(s =  league + ' ' + 'MatchDay:' + ' ' + str(1) + ' ' + '| Season 21-22 | @menesesp20',
                x = 0.53, y = 0.89,
                color='#181818', fontweight='bold',
                ha='center', va='center',
                fontsize=20);

        fig_text(s = str(fieldTilt_Home) + ' ',
                x = 0.474, y = 0.225,
                color=color[0], fontweight='bold',
                ha='center', va='center',
                fontsize=30)

        fig_text(s = ' ' + '   ' + ' ',
                x = 0.512, y = 0.225,
                color=color2[0], fontweight='bold',
                ha='center', va='center',
                fontsize=30)
        
        fig_text(s = ' ' + str(fieldTilt_Away),
                x = 0.55, y = 0.225,
                color=color2[0], fontweight='bold',
                ha='center', va='center',
                fontsize=30)
        
        # Club Logo
        fig = add_image(image='C:/Users/menes/Documents/Data Hub/Images/Clubs/' + league + '/' + home + '.png', fig=fig,
                        left=0.06, bottom=0.88, width=0.2, height=0.09)

    else:
        # Plotting the pitch
        pitch = Pitch(pitch_type='opta',
                      pitch_color='#E8E8E8', line_color='#181818',
                      line_zorder=1, linewidth=1, spot_scale=0.005)

        pitch.draw(ax=axis)   

        ax = axis

    #############################################################################################################################################

    ax.axvspan(75, 100, facecolor='#181818', alpha=0.4)

    ax.axvline(75, c='#181818', ls='--', lw=1.5)


    ax.axvspan(25, 0, facecolor='#181818', alpha=0.4)

    ax.axvline(25, c='#181818', ls='--', lw=1.5)

    #############################################################################################################################################

    for i in range(len(touch)):
        if touch['team'].values[i] == away:
            ax.scatter(touch['x'].values[i] , touch['y'].values[i] , s = 25, color='#181818', edgecolor=color2[0], lw=0.5, alpha=0.9, zorder=4)

        elif touch['team'].values[i] != away:  
            ax.scatter(100 - touch['x'] , 100 - touch['y'] , s = 25, color='#181818', edgecolor=color[0], lw=0.5, alpha=0.9, zorder=4)

    #############################################################################################################################################


    if (home_Passes < 50) & (fieldTilt_Home > 50):

        pitch.annotate('Despite' + ' ' + f'{home}' + ' ' +
                       'had less possession' + ' ' +
                       '(' + f'{str(home_Passes)}%' + ')' + '\n' +
                       'they had greater ease in penetrating' + '\n' +
                       'the final third than' + ' ' + 
                       f'{away}',
                       fontfamily='monospace',
                       xy=(10, 104), c='#181818',
                       size=10, ax=axis)

    elif (away_Passes < 50) & (fieldTilt_Away > 50):

        pitch.annotate('Despite' + ' ' + f'{away}' + ' ' +
                       'had less possession' + ' ' +
                       '(' + f'{str(away_Passes)}%' + ')' + '\n' +
                       'they had greater ease in penetrating' + '\n' +
                       'the final third than' + ' ' + 
                       f'{home}',
                       fontfamily='monospace',
                       xy=(10, 104), c='#181818',
                       size=10, ax=axis)

    elif (home_Passes > 50) & (fieldTilt_Home < 50):
        
        pitch.annotate('Despite' + ' ' + f'{home}' + ' ' +
                       'had more possession' + ' ' +
                       '(' + f'{str(home_Passes)}%' + ')' + '\n' +
                       'they struggled to penetrate' + '\n' +
                       'the last third than' + ' ' + 
                       f'{away}',
                       fontfamily='monospace',
                       xy=(10, 104), c='#181818',
                       size=10, ax=axis)

    elif (away_Passes > 50) & (fieldTilt_Away < 50):

        pitch.annotate('Despite' + ' ' + f'{away}'+ ' ' +
                       'had more possession' + ' ' +
                       '(' + f'<{str(away_Passes)}%>' + ')' + '\n' +
                       'they struggled to penetrate' + '\n' +
                       'the last third than' + ' ' + 
                       f'{home}',
                       xy=(10, 104), c='#181818',
                       fontfamily='monospace',
                       size=10, ax=axis)

    elif (fieldTilt_Home > fieldTilt_Away):

        pitch.annotate(f'{home}' + ' ' +
                       'dominated the game with greater dominance' + '\n' +
                       'of the last third than their opponent' + ' ' +
                       f'{away}.',
                       xy=(2, 104), c='#181818',
                       fontfamily='monospace',
                       size=8, ax=axis)

    elif (fieldTilt_Home < fieldTilt_Away):

        pitch.annotate(f'{away}' + ' ' +
                       'dominated the game with greater dominance' + '\n' +
                       'of the last third than their opponent' + ' ' +
                       f'{home}.',
                       xy=(2, 104), c='#181818',
                       fontfamily='monospace',
                       size=10, ax=axis)

    #############################################################################################################################################

def gameDash(df):

    home = df.home_Team.unique()[0]
    away = df.away_Team.unique()[0]
    
    background = '#E8E8E8'
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 15), dpi=300, facecolor = background, gridspec_kw={'width_ratios': [1, 0.9, 1]})

    fig.subplots_adjust(wspace=0.4, hspace=0.1)

    ######################################################################################################

    passing_networkWhoScored(df, 'Liverpool', '8final_1stLeg', axis=ax[0, 0])

    ax[1, 0].axis('off')
    ax[1, 0].text(0.3, 1, 'MOST XTHREAT VIA PASS', ha='center', va='center',
                bbox=dict(facecolor='#FF0000', edgecolor='black', alpha=0.7))

    ax[1, 0].text(0.1, 0.95, 'TRENT-ALEX-ARNOLD', ha='left', va='center', color='#181818')

    ax[1, 0].text(0.95, 1, 'MOST XTHREAT COMBINATION', ha='center', va='center',
                bbox=dict(facecolor='#FF0000', edgecolor='black', alpha=0.7))

    ax[1, 0].text(0.8, 0.95, 'SALAH-NÚNEZ', ha='left', va='center', color='#181818')

    ax[1, 0].text(0.3, 0.9, 'MOST PASS COMBINATION', ha='center', va='center',
                bbox=dict(facecolor='#FF0000', edgecolor='black', alpha=0.7))

    ax[1, 0].text(0.1, 0.85, 'TRENT-ALEX-ARNOLD - FABINHO', ha='left', va='center', color='#181818')

    ######################################################################################################

    plot_Shots(df, ax[0, 1])

    ######################################################################################################

    #field_Tilt(df, axis=ax[1, 1])
    #ax[1, 1].axis('off')
    xT_Flow(df, 'La Liga', home, '8final_1stLeg', axis=ax[1, 1])

    ######################################################################################################

    passing_networkWhoScored(df, away, '8final_1stLeg', axis=ax[0, 2])

    ######################################################################################################

    ax[1, 2].axis('off')
    ax[1, 2].text(0.3, 1, 'MOST XTHREAT VIA PASS', ha='center', va='center',
                bbox=dict(facecolor='#064d93', edgecolor='black', alpha=0.7))

    ax[1, 2].text(0.1, 0.95, 'MODRIC', ha='left', va='center', color='#181818')

    ax[1, 2].text(0.95, 1, 'MOST XTHREAT COMBINATION', ha='center', va='center',
                bbox=dict(facecolor='#064d93', edgecolor='black', alpha=0.7))

    ax[1, 2].text(0.7, 0.95, 'Karim Benzema-Rodrygo', ha='left', va='center', color='#181818')

    ax[1, 2].text(0.3, 0.9, 'MOST PASS COMBINATION', ha='center', va='center',
                bbox=dict(facecolor='#064d93', edgecolor='black', alpha=0.7))

    ax[1, 2].text(0.1, 0.85, 'Daniel Carvajal - Éder Militão', ha='left', va='center', color='#181818')
    
    add_image(image='Images/Clubs/' + 'Premier League' + '/' + home + '.png', fig=fig, left=0.173, bottom=0.88, width=0.09, height=0.072)
    
    add_image(image='Images/Clubs/' + 'La Liga' + '/' + away + '.png', fig=fig, left=0.748, bottom=0.88, width=0.09, height=0.072)
    
    home_Goals = df.loc[(df.team == 'Liverpool') & (df.typedisplayName == 'Goal')]['typedisplayName'].count()
    home_Goals = str(home_Goals)

    away_Goals = df.loc[(df.team == away) & (df.typedisplayName == 'Goal')]['typedisplayName'].count()
    away_Goals = str(away_Goals)
    
    fig_text(s=f'{home_Goals} : {away_Goals}', fontsize=35, x=0.5, y=0.93, color='#181818', ha='center')
    
    fig_text(s=f'{home} vs {away}', fontsize=20, x=0.5, y=0.89, color='#181818', ha='center')
    
    fig_text(s=f'Champions League 2022/23 | 8 final', fontsize=11, x=0.5, y=0.87, color='#181818', alpha=0.8, ha='center')

def plotCarry(df, player, x=None):
        
        if x == None:
                fig, ax = plt.subplots(figsize=(6, 3), dpi=500)

                pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=2, linewidth=1, spot_scale=0.005)

                pitch.draw(ax=ax)

                fig.set_facecolor('#E8E8E8')

                ###############################################################################################################################################################
                ###############################################################################################################################################################

                carry = df.loc[df['type.secondary'].apply(lambda x: 'carry' in x) & (df['player.name'] == player) & (df['carry.progression'] > 0)].reset_index(drop=True)

                ###############################################################################################################################################################
                ###############################################################################################################################################################

                # Plot Carry
                pitch.lines(carry['location.x'], carry['location.y'], carry['carry.endLocation.x'], carry['carry.endLocation.y'],
                        lw=0.5, color='#fb8c04', zorder=3,
                        label='Through Passes Successful', ax=ax)

                pitch.scatter(carry['carry.endLocation.x'], carry['carry.endLocation.y'], s=0.5,
                                marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=ax)

        else:
                
                pitch = VerticalPitch(pitch_type='opta', pitch_color='#E8E8E8', line_color='#181818',
                        line_zorder=2, linewidth=3, spot_scale=0.005)

                pitch.draw(ax=x)
                
                ###############################################################################################################################################################
                ###############################################################################################################################################################

                carry = df.loc[df['type.secondary'].apply(lambda x: 'carry' in x) & (df['player.name'] == player) & (df['carry.progression'] > 0)].reset_index(drop=True)

                ###############################################################################################################################################################
                ###############################################################################################################################################################

                # Plot Carry
                pitch.lines(carry['location.x'], carry['location.y'], carry['carry.endLocation.x'], carry['carry.endLocation.y'],
                        lw=1.8, color='#ea04dc', zorder=3,
                        label='Through Passes Successful', ax=x)

                pitch.scatter(carry['carry.endLocation.x'], carry['carry.endLocation.y'], s=20,
                                marker='o', edgecolors='#ea04dc', c='#ea04dc', zorder=3, ax=x)

def plotShots(df, team, x=None):
    
    inter = df.loc[(df['team.name'] == team )& (df['shot.xg'] > 0)].reset_index(drop=True)
    
    interGoal = df.loc[(df['team.name'] == team )& (df['shot.isGoal'] == True)].reset_index(drop=True)
    
    if x == None:
        fig, ax = plt.subplots(figsize=(18,14), dpi=500)

        pitch = VerticalPitch(pitch_type='opta', half=True,
                    pitch_color='#e8e8e8', line_color='#181818',
                    line_zorder=2, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#e8e8e8')

        pitch.scatter(inter['location.x'], inter['location.y'], s=inter['shot.xg'] * 1500,
                                marker='o', edgecolors='#fb8c04', c='#E8E8E8', lw=1.5, zorder=3, ax=ax)

        pitch.scatter(interGoal['location.x'], interGoal['location.y'], s=interGoal['shot.xg'] * 1500,
                                marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=ax)
        
        ax.scatter(55, 55, color='#E8E8E8', lw=1.5, edgecolors='#fb8c04', s=500,
                zorder=3)

        ax.text(56.3, 53, 'Shot', color='#181818', size=12,
                zorder=3)

        ax.scatter(45, 55, color='#fb8c04', lw=1.5, edgecolors='#fb8c04', s=500,
                zorder=3)

        ax.text(46.3, 53, 'Goal', color='#181818', size=12,
                zorder=3)

    else:
        pitch = VerticalPitch(pitch_type='opta', half=True,
                    pitch_color='#e8e8e8', line_color='#181818',
                    line_zorder=2, linewidth=2, spot_scale=0.00)

        pitch.draw(ax=x)

        pitch.scatter(inter['location.x'], inter['location.y'], s=inter['shot.xg'] * 500,
                                marker='o', edgecolors='#fb8c04', c='#E8E8E8', lw=1.5, zorder=3, ax=x)

        pitch.scatter(interGoal['location.x'], interGoal['location.y'], s=interGoal['shot.xg'] * 500,
                                marker='o', edgecolors='#fb8c04', c='#fb8c04', zorder=3, ax=x)
        
        x.scatter(55, 55, color='#E8E8E8', lw=1.5, edgecolors='#fb8c04', s=70,
                zorder=3)

        x.text(56.3, 52, 'Shot', color='#181818', size=5,
                zorder=3)

        x.scatter(45, 55, color='#fb8c04', lw=1.5, edgecolors='#fb8c04', s=70,
                zorder=3)

        x.text(46.3, 52, 'Goal', color='#181818', size=5,
                zorder=3)

def plotAssist(df, team, x=None):
    
    keyPass = df.loc[df['type.secondary'].apply(lambda x: 'key_pass' in x) & (df['team.name'] == team)].reset_index(drop=True)
    
    assist = df.loc[df['type.secondary'].apply(lambda x: 'assist' in x) & (df['team.name'] == team)].reset_index(drop=True)

    if x == None:
        fig, ax = plt.subplots(figsize=(18,14), dpi=500)

        pitch = VerticalPitch(pitch_type='opta', half=True,
                    pitch_color='#e8e8e8', line_color='#181818',
                    line_zorder=2, linewidth=5, spot_scale=0.00)

        pitch.draw(ax=ax)

        fig.set_facecolor('#e8e8e8')

        pitch.scatter(keyPass['location.x'], keyPass['location.y'], s=800,
                                marker='o', edgecolors='#2d92df', c='#E8E8E8', lw=1.5, zorder=3, ax=ax)

        pitch.scatter(assist['location.x'], assist['location.y'], s=800,
                                marker='o', edgecolors='#2d92df', c='#2d92df', zorder=3, ax=ax)
        
        ax.scatter(55, 55, color='#E8E8E8', lw=1.5, edgecolors='#2d92df', s=500,
                zorder=3)

        ax.text(56.3, 53, 'Key pass', color='#181818', size=12,
                zorder=3)

        ax.scatter(45, 55, color='#2d92df', lw=1.5, edgecolors='#2d92df', s=500,
                zorder=3)

        ax.text(46.3, 53, 'Assist', color='#181818', size=12,
                zorder=3)

    else:
        
        pitch = VerticalPitch(pitch_type='opta', half=True,
                    pitch_color='#e8e8e8', line_color='#181818',
                    line_zorder=2, linewidth=2, spot_scale=0.00)

        pitch.draw(ax=x)

        pitch.scatter(keyPass['location.x'], keyPass['location.y'], s=150,
                                marker='o', edgecolors='#2d92df', c='#E8E8E8', lw=1.5, zorder=3, ax=x)

        pitch.scatter(assist['location.x'], assist['location.y'], s=150,
                                marker='o', edgecolors='#2d92df', c='#2d92df', zorder=3, ax=x)
        
        x.scatter(55, 55, color='#E8E8E8', lw=1.5, edgecolors='#2d92df', s=70,
                zorder=3)

        x.text(58.3, 52, 'Key pass', color='#181818', size=4.8,
                zorder=3)

        x.scatter(45, 55, color='#2d92df', lw=1.5, edgecolors='#2d92df', s=70,
                zorder=3)

        x.text(46.3, 52, 'Assist', color='#181818', size=5,
                zorder=3)

def dashboardPlayerOffensive(df, league, team, player):

    color = ['#041ca3', '#181818']

    fig = plt.figure(figsize=(18,12), dpi = 500)
    grid = plt.GridSpec(8, 8)

    a1 = fig.add_subplot(grid[0:6, 0:3])
    a2 = fig.add_subplot(grid[0:3, 3:5])
    a3 = fig.add_subplot(grid[3:6, 3:5])

    #################################################################################################################################################

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
    [{"color": color[0],"fontweight": 'bold'},
    {"color": color[0],"fontweight": 'bold'}]

    # Club Logo
    add_image(image='C:/Users/menes/Documents/Data Hub/Images/Clubs/' + league + '/' + team + '.png', fig=fig, left=0.12, bottom=0.9, width=0.08, height=0.05)

    fig_text(s = player,
                x = 0.24, y = 0.93,
                color='#181818', fontweight='bold',
                ha='center', va='center',
                fontsize=25)

    fig_text(s = team + ' |' + league + ' |' + 'Season 2021-22',
                x = 0.255, y = 0.91,
                color='#181818', fontweight='bold',
                ha='center', va='center',
                fontsize=8, alpha=0.5)

    carry = df.loc[df['type.secondary'].apply(lambda x: 'carry' in x) & (df['player.name'] == player) & (df['carry.progression'] > 0)].reset_index(drop=True)
    
    interShots = df.loc[(df['team.name'] == team )& (df['shot.xg'] > 0)].reset_index(drop=True)
    
    interGoals = df.loc[(df['team.name'] == team )& (df['shot.isGoal'] == True)].reset_index(drop=True)
    
    keyPass = df.loc[df['type.secondary'].apply(lambda x: 'key_pass' in x)].reset_index(drop=True)
    
    assist = df.loc[df['type.secondary'].apply(lambda x: 'assist' in x)].reset_index(drop=True)

    highlight_textprops =\
        [{"color": '#ea04dc',"fontweight": 'bold'},
         {"color": '#fb8c04',"fontweight": 'bold'},
         {"color": '#fb8c04',"fontweight": 'bold'},
         {"color": '#2d92df',"fontweight": 'bold'},
         {"color": '#2d92df',"fontweight": 'bold'}]

    fig_text(s = player + ' completed ' + str(len(carry)) + ' <progressive carries>' + '\n' + ' resulting in ' +
             str(len(interShots)) + ' <shots>, ' + str(len(interGoals)) + ' <goals>, ' + str(len(keyPass)) + ' <key passes>' + '\n' + ' and ' + str(len(assist)) + ' <assist>',
             highlight_textprops = highlight_textprops,
             x = 0.52, y = 0.88,
             color='#181818', fontweight='bold',
             ha='center', va='center',
             fontsize=8)

    fig.set_facecolor('#E8E8E8')

    plotCarry(df, player, a1)

    plotShots(df, team, a2)

    plotAssist(df, team, a3)






















