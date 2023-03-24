import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import datetime

from matplotlib import font_manager

font_path = './Fonts/Gagalin-Regular.otf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# Courier New
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Scout App",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded")

st.title('Data Hub')

st.markdown("""
This app performs visualization from sports data to improve the club decision make!
* **Data source:** WyScout & OPTA.
""")

st.sidebar.header('Scouting Hub')

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def load_data(filePath):
    wyscout = pd.read_csv(filePath)
    wyscout.drop(['Unnamed: 0'], axis=1, inplace=True)
    wyscout['Age']  = wyscout['Age'].astype(int)
    
    return wyscout

wyscout = load_data('./Data/data.csv')

options_Tiers = st.sidebar.multiselect(
    'Choose the tiers you want',
    wyscout.Tier.unique(), wyscout.Tier.unique()[0])

# Common filtering criteria
common_filterLeagues = (wyscout['Tier'].isin(options_Tiers))

if len(common_filterLeagues) == 0:
    # Add only the common filtering criteria when options_Leagues is empty
    leagues = wyscout.Comp.unique()
else:
    # Add the common filtering criteria and the league filter when options_Leagues is not empty
    leagues = wyscout.loc[common_filterLeagues].Comp.unique()

options_Leagues = st.sidebar.multiselect(
    'Choose the leagues you want',
    leagues, leagues[0])

options_Roles = st.sidebar.multiselect(
    'Choose the roles you want',
    wyscout.Role.unique(), wyscout.Role.unique()[0])

options_UnderAge = st.sidebar.selectbox(
    'Choose Age (Under)',
    sorted(wyscout.Age.unique(), reverse = True))

# Common filtering criteria
common_filter = (wyscout['Tier'].isin(options_Tiers)) & \
                (wyscout['Age'] <= options_UnderAge) & \
                ((wyscout['Role'].isin(options_Roles)) | (wyscout['Role2'].isin(options_Roles))) 

if len(options_Leagues) == 0:
    # Add only the common filtering criteria when options_Leagues is empty
    data = wyscout.loc[common_filter][['Player', 'Age', 'Minutes played', 'Team', 'Comp', 'Main Pos', 'Role', 'Role2', 'Score']].sort_values('Score', ascending=False).reset_index(drop=True)
else:
    # Add the common filtering criteria and the league filter when options_Leagues is not empty
    data = wyscout.loc[common_filter & (wyscout['Comp'].isin(options_Leagues))][['Player', 'Age', 'Minutes played', 'Team', 'Comp', 'Main Pos', 'Role', 'Role2', 'Score']].sort_values('Score', ascending=False).reset_index(drop=True)

st.dataframe(data, height=500, use_container_width=True)