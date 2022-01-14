import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import math
from datetime import datetime
from scipy.stats import norm 

def ponderate_avg(avg,dig=None,bim=None):
    if (not dig and not bim) or (np.isnan(dig) and np.isnan(bim)) :
        return avg
    if not dig or np.isnan(dig) :
        dig = (avg*55+bim*20)/75
    elif not bim or np.isnan(bim):
        bim = (avg*55+dig*25)/80   

    return (avg*55 + dig*25 + bim*20)/100

def indicators_calculator(classe,notes):
    filter_notes = notes.loc[(notes['class'] == classe) & (notes['especial_type'] == 'no')][['indicator','note','bimester']]

    indicators = filter_notes.groupby(['indicator','bimester']).mean()
    indicators = indicators.merge(filter_notes.groupby(['indicator','bimester']).std(),left_index=True, right_index=True)
    indicators.columns = ['normal_note_avg','note_std']

    special_notes = notes.loc[(notes['especial_type'] != 'no')]
    special_notes = special_notes.pivot(index=['indicator','bimester'], columns='especial_type', values='note')
    for special_note in ['dignostic','bimonthly']:
        try:
            special_notes[special_note]
        except KeyError:
            special_notes.insert(0,special_note,[0]*len(special_notes))
    indicators = indicators.join(special_notes)

    indicators['ponderate_avg'] = indicators.apply(lambda row: ponderate_avg(row['normal_note_avg'],row['dignostic'],row['bimonthly']),axis=1)

    indicators = indicators.reset_index(['indicator','bimester']).sort_values('bimester')
    

    return indicators

def plot_distribution_indicator(classe,bimester, notes):
    filter_notes = notes.loc[(notes['class'] == classe) & (notes['bimester'] == bimester)]

    indicators = filter_notes['indicator'].unique()
    indicators_gen = (i for i in [[0,0],[0,1],[1,0],[1,1]])
    
    fig, ax = plt.subplots(2,2,figsize=(12, 6),sharey=True)
    fig.tight_layout(h_pad=2) #avoid overlaping
    plt.subplots_adjust(top=0.85) #ovoid overlaping titile

    for indicator in indicators:
        ind_x_distribution = filter_notes.loc[(filter_notes['indicator'] == indicator)]['note'].values
        ubication = next(indicators_gen)

        #bars
        values, dist = np.unique(ind_x_distribution , return_counts=True)
        ax[ubication[0],ubication[1]].bar(values, dist)

        #distrivution
        dist = norm(ind_x_distribution.mean(), ind_x_distribution.std())
        x = np.arange(10, 50, 0.1)
        y = [dist.pdf(value)*len(ind_x_distribution) for value in x]
        ax[ubication[0],ubication[1]].plot(x,y) 

        ax[ubication[0],ubication[1]].set_title(f'{indicator}° indicator')

    fig.suptitle(f'bimester: {bimester}',fontsize=15)

def how_much_to_pass(normal_avg,dig = 0,bim = 0):
    bim_dig_values = {'dig':0, 'bim':0}
    dig_notes = not ((dig == 0) or np.isnan(dig))
    bim_notes = not ((bim == 0) or np.isnan(bim))
    assert dig_notes and bim_notes == False, 'if dig and bim notes are already there you can do anything'

    try:
        if not dig_notes and not bim_notes:
            bim_dig_values['dig'] = bim_dig_values['bim'] = np.ceil((700-11*avg)/9)

        elif dig_notes and not bim_notes:
            bim_dig_values['bim'] = int(np.ceil((700-11*normal_avg-dig*5)/4))
        
        elif not dig_notes and bim_notes:
            bim_dig_values['dig'] = int(np.ceil((700-11*normal_avg-bim*4)/5))

        assert bim_dig_values['dig'] < 50 and bim_dig_values['bim'] < 50, 'the neded note exced the maximun posible note'

    except AssertionError as error:
        return 'is imposible to pass'

    bim_dig_values = {key:item for key, item in bim_dig_values.items() if item != 0}
    return bim_dig_values

def zptile(z_score):
    return .5 * (math.erf(z_score / 2 ** .5) + 1) *100

def prov_of_needed_note(notes_dictionay,indicator,notes):
        min_prov = []
        max_prov = []
        provs = []

        notes_ind_data = notes.loc[(notes['indicator'] == indicator)] 
        error_magen = (1.96*notes_ind_data['note'].std())/len(notes_ind_data)**0.5 #revise it

        for key, item in notes_dictionay.items():

            z_score =  (item - notes_ind_data['note'].mean())/notes_ind_data['note'].std()
            prov = 100-zptile(z_score) #prov of achiving the note or a highter
            error = prov/100*error_magen

            provs.append(prov)
            min_prov.append(prov-error)
            max_prov.append(prov+error)


        provs = np.mean(provs)
        min_prov = round(np.mean(min_prov),2) #le saco el promedio o los multiplico
        max_prov = round(np.mean(max_prov),2)
        return f'{max_prov}%-{min_prov}%', [provs,error_magen]

def situation_analizer(indicators,notes):
    situation = pd.DataFrame(columns=['def_note','pass','need_to_pass','need_note_provavility'] )

    for index, row in indicators.iterrows():
        situation_row = {'indicator':row['indicator'], 'bimester': row['bimester'], 'def_note':row['ponderate_avg'].round(),  
        'pass':0 ,'need_to_pass': 'already_end'}

        dig_notes = not((row['dignostic'] == 0) or (np.isnan(row['dignostic'])))
        bim_notes = not((row['bimonthly'] == 0) or (np.isnan(row['bimonthly']))) 


        if dig_notes and bim_notes:
            situation_row['pass'] = 'yes' if situation_row['def_note'] >= 35 else 'no'

        else:
            situation_row['pass'] = 'currenly_yes' if situation_row['def_note'] >= 35 else 'currenly_no'
            situation_row['need_to_pass'] = how_much_to_pass(row['normal_note_avg'],row['dignostic'],row['bimonthly'])

            if type(situation_row['need_to_pass']) == dict:
                situation_row['need_note_provavility'], situation_row['provability_and_error'] = prov_of_needed_note(
                    situation_row['need_to_pass'],row['indicator'],notes)

        situation = situation.append(situation_row, ignore_index=True)

    situation=situation.astype({'bimester':int,'indicator':int})
    situation = situation.set_index(['bimester','indicator'])

    #join situation to indicator
    indicators = indicators.set_index(['bimester','indicator'])
    indicators = indicators.join(situation)
    indicators = indicators.reset_index(['bimester','indicator'])

    return situation, indicators

def filter_report(full_report,debug=False):
    filtered_full_report = full_report[['bimester','indicator','dignostic','bimonthly','def_note','pass','need_to_pass','need_note_provavility']].copy()
    filtered_full_report = filtered_full_report.astype({'def_note':int})
    filtered_full_report = filtered_full_report.set_index(['bimester','indicator'])

    if debug:
        return filtered_full_report, full_report
    else:
        return filtered_full_report

def ask_int(text):
    try:
        num = int(input(str(text)))
    except ValueError as e:
        print(f'invalid input, try again')
        return ask_int(text)
    else:
        return num