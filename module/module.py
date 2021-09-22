import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


with open('datas/champion_filter.pkl', 'rb') as f:
    champion_datas = pickle.load(f)

palette_1 = sns.color_palette('hls',8)
palette_2 = sns.color_palette("Paired", 9)[1:]

def show_rateplot(df, criteria):
    """
    show_rateplot function : 특정 플레이 시간에 대해 오브젝트별 승률에 대한 barplot visualization
    Input Arguments : df(dataframe : 알고자하는 시간의 df), criteria(str : barplot의 title)
    return barplot
    """
    rate_data = {"name": [], "rate": []}
    for col in df.columns:
        if df[col].dtype == "bool":
            win_count = len(df[(df[col] == True) & (df['win'] == 'Win')])
            fail_count = len(df[(df[col] == True) & (df['win'] == 'Fail')])
            try:
                rate = np.round(win_count / (win_count + fail_count), 2)

                rate_data['name'].append(col)
                rate_data['rate'].append(rate)
            except:
                pass
            
    rate_df = pd.DataFrame(rate_data).sort_values('rate', ascending=False).reset_index(drop=True)
    
    g = sns.barplot(data=rate_df, x='name', y='rate', palette=palette_1)
    for idx, row in rate_df.iterrows():
        g.text(row.name, row.rate, f"{round(row.rate*100, 2)}%", color='black', ha='center')
    
    plt.title(f"First Object 선점에 따른 승률(%) - {criteria}", fontsize=15)
    plt.xticks(rotation=30, fontsize=10)
    plt.xlabel(None)
    plt.ylim((0.5, 1))


def preprocess_participant(df):
    """
    preprocess_participant function : participant_df의 비정상적으로 플레이 된 데이터를 삭제하는 함수
    Input Arguments : None
    return preprocessed participant_df
    """
    del_idx_1 = df[df['firstbloodkill'].isna()].index
    df.drop(index=del_idx_1, inplace=True)

    del_idx_2 = df[(df['firsttowerkill'].isna()) & (df['gameduration'] < 900)].sort_values('gameduration', ascending=False).index
    df.drop(index=del_idx_2, inplace=True)

    df['firsttowerassist'] = df['firsttowerassist'].fillna(False)
    df['firsttowerkill'] = df['firsttowerkill'].fillna(False)

    df['firstinhibitorkill'] = df['firstinhibitorkill'].fillna(False)
    df['firstinhibitorassist'] = df['firstinhibitorassist'].fillna(False)

    df['firstinhibitor'] = (df['firstinhibitorkill'] + df['firstinhibitorassist']).astype('bool')
    df['firsttower'] = (df['firsttowerkill'] + df['firsttowerassist']).astype('bool')
    df['firstblood'] = (df['firstbloodkill'] + df['firstbloodassist']).astype('bool')

    df.drop(columns=['firstinhibitorkill', 'firstinhibitorassist', 'firsttowerkill', 'firsttowerassist', 'firstbloodkill', 'firstbloodassist'], inplace=True)

    return df


def recommendation_obj(df, target, duration1, duration2=None):
    """
    recommendation_obj function : 특정 오브젝트 관여에 대한 포지션별 비율 및
                                  각 포지션에 대한 TOP2 챔피언 정보를 제공해주는 함수
    Input Arguments : df(dataframe), target(str : 특정 오브젝트 이름)
                     , duration1(int : start time), duration2(int : end time)
    return 포지션별 관여율, 각 포지션별 TOP2 챔피언
    """
    if (duration1 == 1200) & (duration2 is None):
        datas = df[(df['win'] == True) & (df[target] == True) & (df['gameduration'] < duration1)]
    elif (duration1 == 2400) & (duration2 is None):
        datas = df[(df['win'] == True) & (df[target] == True) & (df['gameduration'] >= duration1)]
    else:
        datas = df[(df['win'] == True) & (df[target] == True) & (df['gameduration'] >= duration1) & (df['gameduration'] < duration2)]
    
    result_1 = datas.groupby('position').size().reset_index(name='count')
    result_1['rate'] = np.round((result_1['count'] / np.sum(result_1['count']))*100, 2)
    
    result_2 = datas.groupby(['position', 'championid']).size().reset_index(name='count')
    for position in result_2['position'].unique():
        total = np.sum(result_2[result_2['position'] == position]['count'])
        data = result_2[result_2['position'] == position].sort_values('count', ascending=False).reset_index(drop=True)
        data['championid'] = data['championid'].map(champion_datas)

        print(f'{position}의 {target} 기여도: {result_1[result_1["position"] == position]["rate"].values[0]}%')
        print(f'{position} 중 TOP2 Champion: {data["championid"][0]}({np.round(data["count"][0]/total*100, 2)}%), \
{data["championid"][1]}({np.round(data["count"][1]/total*100, 2)}%)')
        print("--------------")


def show_label_plot(df, target_ls, palette=palette_2):
    """
    show_label_plot function : 군집별 특정 변수에 통계값(median)에 대한 barplot visualization
    Input Arguments : df(dataframe), target_ls(list)
    """
    plt.figure(figsize=(20, 4))
    
    plt.subplot(141)
    g = sns.barplot(data=df, x='label', y=target_ls[0], palette=palette)
    plt.ylabel(None)
    plt.title(target_ls[0], fontsize=15)
    for idx, row in df.iterrows():
        g.text(row.name, row[target_ls[0]], np.round(row[target_ls[0]], 2), color='black', ha='center')
        
    plt.subplot(142)
    g = sns.barplot(data=df, x='label', y=target_ls[1], palette=palette)
    plt.ylabel(None)
    plt.title(target_ls[1], fontsize=15)
    for idx, row in df.iterrows():
        g.text(row.name, row[target_ls[1]], np.round(row[target_ls[1]], 2), color='black', ha='center')
        
    plt.subplot(143)
    g = sns.barplot(data=df, x='label', y=target_ls[2], palette=palette)
    plt.ylabel(None)
    plt.title(target_ls[2], fontsize=15)
    for idx, row in df.iterrows():
        g.text(row.name, row[target_ls[2]], np.round(row[target_ls[2]], 2), color='black', ha='center')
        
    plt.subplot(144)
    g = sns.barplot(data=df, x='label', y=target_ls[3], palette=palette)
    plt.ylabel(None)
    plt.title(target_ls[3], fontsize=15)
    for idx, row in df.iterrows():
        g.text(row.name, row[target_ls[3]], np.round(row[target_ls[3]], 2), color='black', ha='center')
        
    plt.tight_layout()
    plt.show()


def show_rank_label(df, target):
    """
    show_rank_label function : 특정 변수에 대해 Best, Worst TOP2의 군집을 보여주는 함수
    Input Arguments : df(dataframe), target(str)
    return TOP2 Cluster of Best, Worst
    """
    high = {'label': [], 'value': []}
    low = {'label': [], 'value': []}
    for idx, row in df.iterrows():

        if row[target] >= df[target].mean():
            high['label'].append(row['label'])
            high['value'].append(row[target])
        else:
            low['label'].append(row['label'])
            low['value'].append(row[target])
            
    high_df = pd.DataFrame(high).sort_values('value', ascending=False).reset_index(drop=True)
    low_df = pd.DataFrame(low).sort_values('value').reset_index(drop=True)
    
    print(f'<{target}> : Mean={np.round(df[target].mean(), 2)}')
    for i in range(2):
        try:
            print(f'Best{i+1} >> Cluster: {int(high_df.loc[i]["label"])}, 평균보다 {np.round(high_df.loc[i]["value"] / df[target].mean(), 2)}배 높습니다')
        except:
            pass

    print("---------------------------")

    for i in range(2):
        try:
            if low_df.loc[i]["value"] == 0.0:
                print(f'Worst{i+1} >> Cluster: {int(low_df.loc[i]["label"])}, 값이 0입니다.')
                continue
            print(f'Worst{i+1} >> Cluster: {int(low_df.loc[i]["label"])}, 평균보다 {np.round(1-(low_df.loc[i]["value"] / df[target].mean()), 2)}배 낮습니다')
        except:
            pass


def show_champion_label(df, extract_one=None):
    """
    show_champion_label function : 군집별 픽률이 높은 TOP3 챔피언을 보여주는 함수
    Input Arguments : df(dataframe), extract_one(default : None)
    """
    datas = df.groupby(['label', 'championid']).size().reset_index(name='count')
    
    # 한 개의 변수에 대해서만 챔피언 정보를 알고 싶을 때 extract_one에 특정 군집의 번호를 입력
    if not extract_one is None:
        data = datas[datas['label'] == extract_one].sort_values('count', ascending=False).reset_index(drop=True)
        data['rate'] = np.round((data['count'] / np.sum(data['count']))*100, 2)
        data['championid'] = data['championid'].map(champion_datas)

        print(f'Cluster_{extract_one} 중 Pick Rate TOP3 >> {data.loc[0]["championid"]}({data.loc[0]["rate"]}%), \
{data.loc[1]["championid"]}({data.loc[1]["rate"]}%), {data.loc[2]["championid"]}({data.loc[2]["rate"]}%)')
    # 일반적으로 모든 군집에 대한 TOP3 챔피언 정보 제공
    else:
        for label in datas['label'].unique():
        
            data = datas[datas['label'] == label].sort_values('count', ascending=False).reset_index(drop=True)
            data['rate'] = np.round((data['count'] / np.sum(data['count']))*100, 2)
            data['championid'] = data['championid'].map(champion_datas)

            print(f'Cluster_{label} 중 Pick Rate TOP3 >> {data.loc[0]["championid"]}({data.loc[0]["rate"]}%), \
    {data.loc[1]["championid"]}({data.loc[1]["rate"]}%), {data.loc[2]["championid"]}({data.loc[2]["rate"]}%)')
            print()