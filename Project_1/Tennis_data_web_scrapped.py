#competitions_table
import requests
import json
import pandas as pd


url = "https://api.sportradar.com/tennis/trial/v3/en/competitions.json?api_key=bwIsHpjx6GuHZW7ofX3Pvr9ntmpy98XgMolo2CE8"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)

#json format --> python object --> dictionary format
data = json.loads(response.text)

#creating table 1 (competitor_table) by joining competition data and category data through left join
category_list = []
comdata = []
for i in data['competitions']:
    parent_id = i.get('parent_id')
    if parent_id:
        comdata.append({'comp_id': i['id'],
                        'comp_name': i['name'],
                        'parent_id': i['parent_id'],
                        'type': i['type'],
                        'gender': i['gender'],
                        'category_id': i.get('category')['id']})
    else:
        comdata.append({'comp_id': i['id'],
                        'comp_name': i['name'],
                        'type': i['type'],
                        'gender': i['gender'],
                        'category_id': i.get('category')['id']})

for i in data['competitions']:
    category = i.get('category')
    if category:
        category_list.append({'category_id': category['id'],
                          'category_name': category['name']})

df_category = pd.DataFrame(category_list).drop_duplicates(subset='category_id')
df_competition = pd.DataFrame(comdata)
df_competition['parent_id'].fillna('NA', inplace=True)
df_merged_1 = pd.merge(
    df_category,
    df_competition[['category_id', 'comp_id', 'comp_name', 'parent_id', 'type', 'gender']],
    on='category_id',
    how='left')
df_merged_1.to_csv("competitor.csv", index=False)
#-------------------------------------------------------------------------x-----------------------------------------------x-----------------------------

#Table 2 complexes table
url = "https://api.sportradar.com/tennis/trial/v3/en/complexes.json?api_key=bwIsHpjx6GuHZW7ofX3Pvr9ntmpy98XgMolo2CE8"
headers = {"accept": "application/json"}
complex_response = requests.get(url, headers=headers)

#json file --> python object
complex_data = json.loads(complex_response.text) 

# Creating Table 2(Complexes table) by joining complex table and venue table through left join
complex_table = []
venues_table = []
for i in complex_data['complexes']:
    complex_table.append({'complex_id': i['id'],
                    'complex_name': i['name']})

for i in complex_data['complexes']:
    venues = i.get('venues')
    if venues:
        if isinstance(venues, list):
            for venue in venues:
                venues_table.append({
                    'venue_id': venue.get('id'),
                    'venue_name': venue.get('name'),
                    'city_name': venue.get('city_name'),
                    'country_name': venue.get('country_name'),
                    'country_code': venue.get('country_code'),
                    'timezone': venue.get('timezone'),
                    'complex_id': i['id']
                })
        elif isinstance(venues, dict):
            venues_table.append({
                'venue_id': venues.get('id'),
                'venue_name': venues.get('name'),
                'city_name': venues.get('city_name'),
                'country_name': venues.get('country_name'),
                'country_code': venues.get('country_code'),
                'timezone': venues.get('timezone'),
                'complex_id': i['id']
            })

df_venues = pd.DataFrame(venues_table)
df_complex = pd.DataFrame(complex_table)
df_merged_2 = pd.merge(df_venues, df_complex[['complex_id', 'complex_name']], on='complex_id', how='left')
df_merged_2.to_csv("complexes.csv",index=False)
#--------------------------------------------------------------------------x--------------------------------------------------x--------------------------

#Table 3 doubles data
url = "https://api.sportradar.com/tennis/trial/v3/en/double_competitors_rankings.json?api_key=bwIsHpjx6GuHZW7ofX3Pvr9ntmpy98XgMolo2CE8"
headers = {"accept": "application/json"}
doubles_response = requests.get(url, headers=headers)

#json file --> python object
doubles_data = json.loads(doubles_response.text)

#Creating doubles_data table by joining competion ranks and competitior table through left join
comp_ranks_table = []
competitor_table = []

for i in doubles_data['rankings']:
    comp_ranks = i.get('competitor_rankings', [])
    for comp_rank in comp_ranks:
        competitor = comp_rank.get('competitor', {})

        # Save comp_rank info
        comp_ranks_table.append({
            'ranks': comp_rank.get('rank'),
            'movement': comp_rank.get('movement'),
            'points': comp_rank.get('points'),
            'competitions_played': comp_rank.get('competitions_played'),
            'competitor_id': competitor.get('id')
        })

        # Save competitor info safely
        competitor_table.append({
            'competitor_id': competitor.get('id'),
            'competitor_name': competitor.get('name'),
            'country': competitor.get('country'),
            'country_code': competitor.get('country_code'),  # will be None if missing
            'abbreviation': competitor.get('abbreviation')
        }) 

df_comp_rank = pd.DataFrame(comp_ranks_table)
df_competitor_d = pd.DataFrame(competitor_table)
df_merged_3 = pd.merge(df_comp_rank, df_competitor_d[['competitor_id', 'competitor_name', 'country', 'country_code', 'abbreviation']], on='competitor_id', how='left')
df_merged_3.index = df_merged_3.index + 1
df_merged_3.index.name = 'rank_id'
df_merged_3.to_csv("doubles_data.csv", index=True)