import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import datetime

from matplotlib import font_manager
import scipy.stats as stats

from highlight_text import  ax_text

from soccerplots.utils import add_image

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

font_path = './Fonts/Gagalin-Regular.otf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# Courier New
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.header('Data Report')

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def load_data(filePath):
    return pd.read_csv(filePath)
wyscout = load_data('./Data/data.csv')
wyscout.drop(['Unnamed: 0'], axis=1, inplace=True)
wyscout['Age']  = wyscout['Age'].astype(int)

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def traditionalReport(data, league, playerName, mode=None):
        ############################################################################# PLAYER'S ABILITY PERCENTILE #####################################################################################################################################################################################################################################

        data['Defensive Ability'] = data['Defensive Ability'].apply(lambda x: math.floor(stats.percentileofscore(data['Defensive Ability'], x)))

        data['crossing'] = data['crossing'].apply(lambda x: math.floor(stats.percentileofscore(data['crossing'], x)))

        data['defending1v1'] = data['defending1v1'].apply(lambda x: math.floor(stats.percentileofscore(data['defending1v1'], x)))

        data['touchQuality'] = data['touchQuality'].apply(lambda x: math.floor(stats.percentileofscore(data['touchQuality'], x)))

        data['positioning Midfield'] = data['positioning Midfield'].apply(lambda x: math.floor(stats.percentileofscore(data['positioning Midfield'], x)))

        data['runs'] = data['runs'].apply(lambda x: math.floor(stats.percentileofscore(data['runs'], x)))

        ############################################################################# PLAYER'S PROFILES #####################################################################################################################################################################################################################################

        stopper = ['A central defender with a great ability to contest duels and a very strong physical presence.\nAccordingly disciplined, strong defensive positioning,\nshortens the space for the forwards well, blocking their attempts to finish.']

        aerialCB = ['Dominant central defender in the air,\ndominates completely the space inside the area,\nstrong defensive positioning, tactically disciplined']

        ballPlayingCB = ["'Central defender with the quality to take responsibility for building the team's game from the back,\nable to break through the opponent's 1/2 line of pressure,\nstrong defensively in duels in anticipation with his vision of the game'"]

        ballCarryingCB = ['A central defender with quality with the ball at his feet,\nfair, the combination of his two main characteristics makes his strongest point the ability to progress with the ball under control, breaking the 1/2 pressure lines of the opponent through his ball conduction']

        ##################################################################################################################################################################################################################################################################################################################

        fullBackCB = ['Versatile in the functions he performs on the field,\ncapable of playing as a central defender in a line of 3 in which he plays the role of a loose center, strong in the vertical pass getting to burn the 1/2 pressure lines of the opponent, also fulfills with quality the role of a more defensive side in a line of 4']

        DefensiveFB = ['Lateral less risen, good positioning in the coverage to his colleagues stopping attacks in transition, strong reaction to the loss of the ball']

        AttackingFB = ['Fast player with the ability to attack the back line,tough player who can play the whole game with great physical ability,very strong dribbler who outmaneuvers his direct opponents,good cross placement']

        WingBack = ['Versatile full-back can play in a line of 3 or 4, physical ability to cover the whole lateral zone, physically strong, excellent crossing ability, good space attack']

        InvertedWingBack = ['Lateral with dominant presence in interior zones, excellent technical and passing ability,\consists of overcoming the 1/2 pressure lines of the adv through his conduction with the outside-in movements,\nforte in defensive coverage']

        ##################################################################################################################################################################################################################################################################################################################

        ballWinner = ['A midfielder who is very strong in ball recovery,strong in defensive coverage of his teammates,good ability to press immediately after losing the ball,strong in duels where he takes advantage of his physical ability very well']

        deepPlaymaker = ["He is the team's builder at the back, plays the main role in the transition from defense to attack, has excellent vision and is able to accelerate the game with forward passing or to clear the side of the ball using flank variations"]

        attackPlaymaker = ['Reference of creation in the final third,\ntakes responsibility for finding or creating the spaces for the team to create danger,\nntechnicalist, excellent command of cake and first touch']

        boxToBox = ['Physically strong, great ability to cover the central terrain,\nstrong in defensive coverage of his teammates,\ngood positioning allowing him to get first\nintercept opponents, strong in the moment of pressure']

        medioGoleador = ['Ease of getting to the finishing zone, dynamic player, moves very well in advanced zones, strong in attacking space and behind the defensive line, strong in finishing']

        ##################################################################################################################################################################################################################################################################################################################

        banda = ['Runner, strong in one-on-one duels, with his dribbling ability he comes out of those situations very well, good decision maker in the final 1/3']

        porDentro = ['Occupies interior spaces,\nDynamic player does a lot of outside-in movement with the ball controlled,\ncoming to frontal finishing zones, good shot placement,\nmoves very well without the ball as well,\nmakes room for the full-back to attack the back line']

        defensivo = ['Performs more defensive tasks giving the necessary balance to the more offensive players to execute their tasks, good knowledge of the game, good reading of the game anticipating and disarming the opponents, fast in transition moments, strong in anticipating crosses']

        falso = ['His positioning is more interior with mobility through the interline zone but also with inside-outside movements with the lateral, arriving at the finishing zones, attacking the space with good finishing quality']

        ##################################################################################################################################################################################################################################################################################################################

        boxForward = ['Pure finisher, area player,\nmovements very well inside the area hiding in the blind side of the defender attacking the space to finish, strong in response to crosses']

        false9 = ['Dynamic forward, strong play with his back to the goal looking for 1-2 combinations with his colleagues,\narrasta with you marking defenders to open space in the back for the extremes/medios to attack the depth']

        targetMan = ['A reference player inside the area, physically strong and using it to win the duels, very strong in the air, excellent header, ability to win the duel, to time and touch the teammate']

        advancedForward = ["Mobile forward,\whether playing in interior spaces with his back to the goal looking for combinations between lines or making inside-outside movements to appear more in corridor zones,\making room for his colleagues to move up the field through interior space."]

        ############################################################################# PLAYER'S ABILITY ANALYSIS #####################################################################################################################################################################################################################################

        passing = ['• Player with a very good passing quality and a wide range of passing options',
                '• Player presents a very good passing quality',
                '• Player has a good passing quality',
                '• In the passing chapter the player needs to improve a little bit the rightness\nin these decisions as well as the variability in passing (short, medium, long)']

        keyPassing = ['• Excellent decision making, calm when deciding',
                        '• Very good on the decision making chapter',
                        '• At times he has moments of clairvoyance when making decisions']
        
        setPieces = ['• Set-piece specialist',
                        '• Good quality in the execution of set pieces']

        dribbling = ['• Ability to get out of duels through dribble,\ngreat technical quality',
                        '• Good ability to use technique\nto dribble your adervaries',
                        '• Good dribbling ability',
                        '• Dribbling is not his strongest point']

        createChances = ['• An important player in creating opportunities thanks to his good vision,\nhe is able to find space for his teammates',
                        '• Very strong player in creating opportunities\nand easily finds his colleagues in the last 1/3',
                        '• Needs improvement but has the quality\nto become a good player in the final 1/3']

        visaoJogo = ['• Excellent vision of the game through his ability to create the necessary space\nfor his team to progress on the field and create goal situations',
                        '• Easiness to place his team in more advanced zones of the field due to his good vision capacity\nin finding the spaces and placing the ball in the right timmig',
                        '• Good vision of the game but\nneeds to take a little more risk in his decisions',
                        '• Have to improve your ability to see the game from the front\nin order to improve your decisions during the game']
        
        concentration = ['• Very focused during the whole game,\nhe can easily anticipate his direct opponents through his concentration\nand is always well positioned to get to the duels first',
                        '• A sense of the space around him allows him to win most of his direct duels\nby disarming and disabling his opponents',
                        '• Has moments during the game that he loses a bit of concentration,\nleading him to arrive late to some duels',
                        '• Needs to improve his focus on the game during the 90 minutes\nso that he can be a more consistent player']

        finishing = ['• Very strong finisher, does not need many opportunities to score a goal',
                        "• A player with a coolness when it comes to finding\nthe bottom of the nets doesn't need much to finish",
                        '• Good quality in finishing but sometimes does not show coolness\nin the face-to-face moment with the goalkeeper',
                        "• Finishing ability is not the player's strong\nsuit he needs to evolve at this moment of the game"]
        
        heading = ['• Responds well to crosses, excelent header',
                '• Strong at aerial duels, good header']

        aerial = ['• Very strong in aerial duels, very good impulsion ability',
                '• Aggressive in aerial duels, good impulsion abilityo',
                '• Needs to improve a bit the duels especially the aerial duels',
                '• Has some difficulties in aerial duels']

        defensive = ['• In the defensive moment he is a very good player, strong in disputes, strong in the ability to read the game and anticipate/disarm his opponents',
                        '• Very strong in the defensive moment with good reading of the game, allowing him to get to the duels first',
                        '• Needs to improve a bit his defensive ability']

        crossing = ['• A lot of quality when it comes to crossing',
                        '• Player with good crossing quality',
                        '• Have to improve the quality at the crossroads']

        defending1v1 = ['• Very strong in 1x1 defensive moves, strong in holding position at the right time to attack the ball',
                        '• Good defensive quality in 1x1 duels, holds the position well',
                        '• Needs to improve in the 1x1 with his opponents and better understands the moment to attack the ball']

        positioningDefensive = ['• Irreproachable defensive positioning, in the moment without the ball he is a very strong player',
                                '• Feels comfortable in the moment without the ball, very good defensive positioning in marking his opponents individually',
                                '• Has to improve at the moment of the game without the ball']

        positioningMidfield = ['• Irreproachable defensive positioning, in the moment without the ball he is a very strong player, excellent defensive coverage to his colleagues, strong in marking his opponents individually and in blocking pass lines',
                                '• Comfortable in the moment without the ball, very good defensive positioning in marking his opponents individually and in blocking pass lines',
                                '• Has to improve at the moment of the game without the ball']
        
        progressiveRuns = ["• Excellent progressive ability with controlled ball\nburns opponent's defensive lines through his quality of progression with controlled ball",
                        '• Good progression with the ball in the foot\nhe overtakes several players using his physique and speed to overcome them in progression with the ball',
                        '• Aspect to be improved in his game is the progression with the ball']

        runsOffBall = ['• Very strong player in the attacking space, moves very well on the field',
                        "• Moves well, attacks the space well but could do it more often and doesn't",
                        '• Somewhat static player on the field needs to move more, attack the space more']
        
        decisionMake = ['• Very strong decision maker when he is in the final 1/3 is a very dangerous player, he can create a scoring chance in any moment',
                        '• Very good player to decide in the last action',
                        "• At times in the final 1/3 he doesn't make the best decisions",
                        '• Need to improve your decision making']
        
        touchQual = ['• Excellent first touch',
                        '• Good first touch',
                        '• Need to improve your first touch']

        ############################################################################# ASSIGN PROFILE'S #####################################################################################################################################################################################################################################

        def assign_profile(role):
                if role == 'Stopper':
                        return stopper
                elif role == 'Aerial CB':
                        return aerialCB
                elif role == 'Ball Playing CB':
                        return ballPlayingCB
                elif role == 'Ball Carrying CB':
                        return ballCarryingCB
                
                elif role == 'Full Back CB':
                        return fullBackCB
                elif role == 'Defensive FB':
                        return DefensiveFB
                elif role == 'Attacking FB':
                        return AttackingFB
                elif role == 'Wing Back':
                        return WingBack
                elif role == 'Inverted Wing Back':
                        return InvertedWingBack
                
                elif role == 'Ball Winner':
                        return ballWinner
                elif role == 'Box-to-box':
                        return boxToBox
                elif role == 'Deep Lying Playmaker':
                                return deepPlaymaker
                elif role == 'Attacking Playmaker':
                        return attackPlaymaker
                elif role == 'Media Punta llegador':
                        return medioGoleador
                
                elif role == 'Banda':
                        return banda
                elif role == 'Por dentro':
                        return porDentro
                elif role == 'Falso':
                        return falso
                elif role == 'Defensivo':
                        return banda
                
                elif role == 'Box Forward':
                        return boxForward
                elif role == 'Advanced Forward':
                        return advancedForward
                elif role == 'False 9':
                        return false9
                elif role == 'Target Man':
                        return targetMan
                
                else:
                        return 'Unknown role'
                
        def assign4(value, profiles):
                if value >= 90:
                        return profiles[0]
                elif (value >= 80) & (value < 90):
                        return profiles[1]
                elif (value >=70) & (value < 80):
                        return profiles[2]
                elif (value >= 40) & (value < 70):
                        return profiles[3]
                elif value < 40:
                        return 'Nothing to report'

        def assign3(value, profiles):
                if value >= 85:
                        return profiles[0]
                elif (value >= 75) & (value < 85):
                        return profiles[1]
                elif (value >=60) & (value < 75):
                        return profiles[2]
                elif value < 60:
                        return 'Nothing to report'

        def assign2(value, profiles):
                if value >= 85:
                        return profiles[0]
                elif (value >= 75) & (value < 85):
                        return profiles[1]
                elif value < 75:
                        return 'Nothing to report'

        def createSkillProfiles(data):
                data['Chances Profile'] = data['Create Chances Ability'].apply(lambda x: assign3(x, createChances))
                data['Touch Profile'] = data['touchQuality'].apply(lambda x: assign3(x, touchQual))
                data['SetPieces Profile'] = data['SetPieces Ability'].apply(lambda x: assign2(x, setPieces))
                data['Decision Profile'] = data['decisionMake'].apply(lambda x: assign4(x, decisionMake))
                data['KeyPass Profile'] = data['KeyPass Ability'].apply(lambda x: assign3(x, keyPassing))
                data['OffBall Profile'] = data['runs'].apply(lambda x: assign3(x, runsOffBall))
                data['SightPlay Profile'] = data['Sight play'].apply(lambda x: assign4(x, visaoJogo))
                data['dribbling Profile'] = data['Dribbling Ability'].apply(lambda x: assign3(x, dribbling))
                data['Pass Profile'] = data['Pass Ability'].apply(lambda x: assign4(x, passing))
                data['Player Profile'] = data['Role'].apply(lambda x: assign_profile(x))
                data['Concentration Profile'] = data['Concentration Ability'].apply(lambda x: assign4(x, concentration))

                # Drop columns that contains certain types in his column names
                data.drop(list(data.filter(regex='Unnamed: 0')), axis=1, inplace=True)
                data.drop(list(data.filter(regex='rank_rank')), axis=1, inplace=True)
                
                return data

        data2 = createSkillProfiles(data)

        def report(playerName, league, mode=None):
                
                if mode == None:
                        color = '#E8E8E8'
                        background = '#181818'
                elif mode != None:
                        color = '#181818'
                        background = '#E8E8E8'
                        
                df = data2.loc[(data2.Comp == league) & (data2.Player == playerName)]
                
                country = df['Birth country'].unique()[0]
                if country == '0':
                        country = df['Passport country'].unique()
                        country = country.tolist()
                        country = country[0]
                if ',' in country:
                        country = country.split(', ')[0]

                Market = df['Market value'].unique()[0]

                if len(str(Market)) == 6:
                        Market = str(Market)[:3]
                                
                elif len(str(Market)) == 7:
                        if str(Market)[:2][1] != 0:
                                Market = str(Market)[:2][0] + '.' + str(Market)[:2][1] + 'M'
                        
                elif len(str(Market)) == 8:
                        Market = str(Market)[:2] + 'M'

                elif len(str(Market)) == 9:
                        Market = str(Market)[:3] + 'M'
                
                elif Market == 0:
                        Market = 'Unknown data'

                position = df['Position'].unique()[0]
                if ', ' in position:
                        position = position.split(', ')[0]

                Contract = df['Contract expires'].unique()[0]

                Height = df['Height'].unique()[0]

                Foot = df['Foot'].unique()[0]

                mainPos = df['Main Pos'].unique()[0]

                age = df['Age'].unique()[0]

                team = df['Team'].unique()[0]
        
                role = df['Role'].unique()[0]

                #######################################################################################################################################

                fig = plt.figure(figsize=(15, 10), dpi=1000, facecolor = background)
                ax = fig.subplots()
                gspec = gridspec.GridSpec(
                ncols=2, nrows=2, wspace = 0.5
                )

                ########################################################################################################################################################

                ax1 = plt.subplot(
                                gspec[0, 0],
                        )
                
                ax1.set_facecolor(background)
                ax1.axis('off')

                ax_text(x=1, y=2.3,  s=playerName, va='center', ha='center',
                        size=35, color=color, ax=ax1)

                profile = df['Player Profile'].explode().unique()[0]

                ax_text(x=0.25, y=0.4,  s='Player Profile', va='center', ha='center',
                        size=18, color='#FCAC14', ax=ax1)
                
                ax_text(x=0.28, y=0.25,  s=profile, va='center', ha='center',
                        size=9, color=color, ax=ax1)
                
                fig = add_image(image='./Images/profile.png', fig=fig, left=0.113, bottom=0.6595, width=0.03, height=0.03)
                
                #######################################################################################################################################
                
                setPieces = df['SetPieces Profile'].explode().unique()[0]
                
                decision = df['KeyPass Profile'].explode().unique()[0]
                
                SightPlay = df['SightPlay Profile'].explode().unique()[0]
                
                OffBall = df['OffBall Profile'].explode().unique()[0]
                
                Pass = df['Pass Profile'].explode().unique()[0]
                
                Concentration = df['Concentration Profile'].explode().unique()[0]
                
                Touch = df['Touch Profile'].explode().unique()[0]
                
                Chances = df['Chances Profile'].explode().unique()[0]
                
                #dribbling = data['dribbling Profile'].explode().unique()[0]
                
                ax_text(x=1, y=2,  s='On Ball', va='center', ha='center',
                        size=18, color='#FCAC14', ax=ax1)

                ax_text(x=1.15, y=1.7,  s= SightPlay + '\n''\n' + Pass + '\n''\n' + Chances, va='center', ha='center',
                                           size=9, color=color, ax=ax1)
                
                fig = add_image(image='./Images/soccer-ball.png', fig=fig, left=0.379, bottom=1.21, width=0.023, height=0.023)

                #######################################################################################################################################
                        
                ax_text(x=1, y=1.28,  s='Off Ball', va='center', ha='center',
                        size=18, color='#FCAC14', ax=ax1)

                #ax_text(x=1, y=1.15,  s='• Jogador muito forte no ataque ao espaço\nmovimenta-se muito bem dentro do campo', va='center', ha='center',
                #        size=9, color=color, ax=ax1)
                
                fig = add_image(image='./Images/lupa.png', fig=fig, left=0.373, bottom=0.962, width=0.03, height=0.03)

                #######################################################################################################################################

                ax_text(x=1, y=0.9,  s='Physical', va='center', ha='center',
                        size=18, color='#FCAC14', ax=ax1)

                #(x=1.15, y=0.7,  s='• Rápido na pressão ao portador da bola juntamente com o seu porte fisico\nconsegue recuperar várias vezes a posse.\n\n• Veloz com bola, boa agilidade consegue facilmente mudar de sentido,\ndemonstra fragildidade nos duelos de corpo como também nos duelos aéreos.', va='center', ha='center',
                #        size=9, color=color, ax=ax1)
                
                fig = add_image(image='./Images/user.png', fig=fig, left=0.37, bottom=0.82, width=0.032, height=0.05)

                #######################################################################################################################################

                values = []
                params = ['Pass Ability', 'KeyPass Ability', 'SetPieces Ability', 'Dribbling Ability', 'Create Chances Ability', 'Sight play',
                        'Concentration Ability', 'Finishing Ability', 'Heading Ability', 'Interception Ability', 'Tackle Ability', 'Aerial Ability', 'Defensive Ability']
                
                for x in range(len(params)):
                        values.append(df[params[x]].values)

                for n,i in enumerate(values):
                        if i == 100:
                                values[n] = 99
                
                elite = []
                veryStrong = []
                strong = []
                improve = []
                weak = []
                veryWeak = []
                for i in range(len(values)):
                        if values[i] >= 95:
                                elite.append(params[i])

                        elif (values[i] >= 80) & (values[i] < 95):
                                veryStrong.append(params[i])

                        elif (values[i] >= 70) & (values[i] < 80):
                                strong.append(params[i])

                        elif (values[i] >= 40) & (values[i] < 70):
                                improve.append(params[i])

                        elif (values[i] < 40) & (values[i] >= 25):
                                weak.append(params[i])

                        elif values[i] < 25:
                                veryWeak.append(params[i])
                        
                #######################################################################################################################################
                
                highlight_textpropsElite =\
                [{"color": '#2ae102', "fontweight": 'bold'}]

                highlight_textpropsVeryStrong =\
                [{"color": '#2cb812', "fontweight": 'bold'}]
                
                highlight_textpropsStrong =\
                [{"color": '#2a9017', "fontweight": 'bold'}]

                highlight_textpropsImprovment =\
                [{"color": '#f48515', "fontweight": 'bold'}]

                ax_text(x=2, y=2, s='<Key Attributes>', va='center', ha='center',
                                        highlight_textprops = highlight_textpropsElite, size=18, color=color, ax=ax1)

                h=1.95
                for i in elite:
                        ax_text(x=2, y=h - 0.07, s='<Elite:>' + ' ' + i, va='center', ha='center',
                                highlight_textprops = highlight_textpropsElite, size=11, color=color, ax=ax1)
                        h=h-0.07

                h=h
                for i in veryStrong:
                        ax_text(x=2, y=h - 0.07, s='<Very Strong:>' + ' ' + i, va='center', ha='center',
                                highlight_textprops = highlight_textpropsVeryStrong, size=11, color=color, ax=ax1)
                        h=h-0.07

                h=h
                for i in strong:
                        ax_text(x=2, y=h - 0.07, s='<Strong:>' + ' ' + i, va='center', ha='center',
                                highlight_textprops = highlight_textpropsStrong, size=11, color=color, ax=ax1)
                        h=h-0.07

                ax_text(x=2, y=h - 0.25, s='<Improvement Points>', va='center', ha='center',
                                        highlight_textprops = highlight_textpropsImprovment, size=18, color=color, ax=ax1)

                h=h - 0.3
                for i in improve:
                        ax_text(x=2, y=h - 0.07, s='<Improve:>' + ' ' + i, va='center', ha='center',
                                highlight_textprops = highlight_textpropsImprovment, size=11, color=color, ax=ax1)
                        h=h-0.07

                h=h
                for i in weak:
                        ax_text(x=2, y=h - 0.07, s='<Improve:>' + ' ' + i, va='center', ha='center',
                                highlight_textprops = highlight_textpropsImprovment, size=11, color=color, ax=ax1)
                        h=h-0.07


                highlight_textpropsInfo =\
                [{"color": '#FCAC14', "fontweight": 'bold'}]

                #######################################################################################################################################

                ax_text(x=0.04, y=1.7,  s='<Age:> ' + str(age), va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.6,  s='<Team:> ' + team, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.5,  s='<HEIGHT:> ' + str(Height) + ' cm', va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.4,  s='<CONTRACT:> ' + Contract, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.3,  s='<VALUE:> ' + Market, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.2,  s='<FOOT:> ' + Foot, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1.1,  s='<POSITION:> ' + mainPos, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)

                #######################################################################################################################################

                ax_text(x=0.04, y=1,  s='<ROLE:> ' + role, va='center', ha='center',
                        size=14, highlight_textprops = highlight_textpropsInfo, color=color, ax=ax1)
                
                #######################################################################################################################################

                highlight_textpropsNotBest =\
                [{"color": '#FF0000', "fontweight": 'bold'}]

                bestRole = df['Role'].values[0]
                roleValue = round(df[bestRole].values[0], 2)
                rolePercentile = math.floor(stats.percentileofscore(df[bestRole], roleValue))
                if rolePercentile >= 90:
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<A>',
                                va='center', ha='center', size=40,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)
                
                elif (rolePercentile < 90) & (rolePercentile >= 75):
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<B>',
                                va='center', ha='center', size=35,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)

                elif (rolePercentile < 75) & (rolePercentile >= 60):
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<C>',
                                va='center', ha='center', size=35,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)

                elif (rolePercentile < 60) & (rolePercentile >= 50):
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<D>',
                                va='center', ha='center', size=35,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)

                elif (rolePercentile < 50) & (rolePercentile >= 25):
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<E>',
                                va='center', ha='center', size=35,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)

                elif (rolePercentile < 25) & (rolePercentile >= 0):
                        ax_text(x=1.2, y=0.39,  s='NOTE',
                                va='center', ha='center', size=28,
                                color=color, ax=ax1)
                
                        ax_text(x=1.213, y=0.24,  s='<F>',
                                va='center', ha='center', size=35,
                                highlight_textprops = highlight_textpropsElite, color=color, ax=ax1)

                #######################################################################################################################################

                import datetime

                current_datetime = datetime.datetime.now()
                today = current_datetime.date()

                ax_text(x=1.5, y=0.39,  s='DATE',
                        va='center', ha='center', size=28,
                        color=color, ax=ax1)

                ax_text(x=1.5, y=0.24,  s=str(today),
                        va='center', ha='center', size=14,
                        color=color, ax=ax1)

                #######################################################################################################################################

                #fig = add_image(image='./Images/Players/' + league + '/' + team + '/' + playerName + '.png', fig=fig, left=0.1, bottom=1.14, width=0.08, height=0.23)

                fig = add_image(image='./Images/Country/' + country + '.png', fig=fig, left=0.185, bottom=1.2, width=0.07, height=0.06)
                
                return plt.show()

        return report(playerName, league)

options_Player = st.sidebar.selectbox(
    'Choose Player you want analyse',
    sorted(wyscout.Player.unique()))

league = wyscout.loc[wyscout.Player == options_Player]['Comp'].unique()[0]

figTraditionReport = traditionalReport(wyscout, league, options_Player)

st.text('This is a test version to combine the data report (strengths and to improve) with the traditional scout report with tactical aspects of the player (On and Off ball momentum and physical).')

st.text("Note that the value at the moment 'On Ball' are predefined values and have nothing to do with the player.")

st.pyplot(figTraditionReport)