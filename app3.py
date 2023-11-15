import streamlit as st # フロントエンドを扱うstreamlitの機能をインポート
import openai # openAIのchatGPTのAIを活用するための機能をインポート
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import folium
from streamlit_folium import folium_static
import requests
import urllib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

st.title("Bannai Bar Dashboard") # タイトルが出力される
st.write("本日の参加者一覧")
#--
SS_ID = ''
SHEET_NAME = 'アカウント情報_分析用'

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Credentials 情報を取得
credentials = ', scopes)

#OAuth2のクレデンシャルを使用してGoogleAPIにログイン
gc = gspread.authorize(credentials)

# IDを指定して、Googleスプレッドシートのワークブックを選択する
workbook = gc.open_by_key(SS_ID)

# シート名を指定して、ワークシートを選択
worksheet = workbook.worksheet(SHEET_NAME)

# スプレッドシートをDataFrameに取り込む
df = pd.DataFrame(worksheet.get_values()[1:], columns=worksheet.get_values()[0])

# Streamlitを使用してDataFrameを表示
st.dataframe(df)

#---------------
# Graph (Pie Chart in Sidebar)
df_target = df[['苗字', '性別']].groupby('性別').count() / len(df)
fig_target = go.Figure(data=[go.Pie(labels=df_target.index,
                                    values=df_target['苗字'],
                                    hole=.3)])
fig_target.update_layout(showlegend=False,
                         height=200,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
fig_target.update_traces(textposition='inside', textinfo='label+percent')

df_target2 = df[['苗字', '居住地（都道府県）']].groupby('居住地（都道府県）').count() / len(df)
fig_target2 = go.Figure(data=[go.Pie(labels=df_target2.index,
                                    values=df_target2['苗字'],
                                    hole=.3)])
fig_target2.update_layout(showlegend=False,
                         height=200,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
fig_target2.update_traces(textposition='inside', textinfo='label+percent')

df_target3 = df[['苗字', '出身地（都道府県）']].groupby('出身地（都道府県）').count() / len(df)
fig_target3 = go.Figure(data=[go.Pie(labels=df_target3.index,
                                    values=df_target3['苗字'],
                                    hole=.3)])
fig_target3.update_layout(showlegend=False,
                         height=200,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
fig_target3.update_traces(textposition='inside', textinfo='label+percent')

st.sidebar.markdown("参加者の属性情報")
st.sidebar.write("性別") # タイトル
st.sidebar.plotly_chart(fig_target, use_container_width=True)
st.sidebar.write("現住所") # タイトル
st.sidebar.plotly_chart(fig_target2, use_container_width=True)
st.sidebar.write("出身地") # タイトル
st.sidebar.plotly_chart(fig_target3, use_container_width=True)

# ------------------------original------------------------
import requests
import urllib
import pandas as pd

makeUrl = "https://msearch.gsi.go.jp/address-search/AddressSearch?q="

df['経度'] = None
df['緯度'] = None

for i, r in df.iterrows():
    s_quote = urllib.parse.quote(r['住所'])
    response = requests.get(makeUrl + s_quote)
    coordinates = response.json()[0]["geometry"]["coordinates"]
    df.loc[i, '経度'] = coordinates[0]
    df.loc[i, '緯度'] = coordinates[1]
    
print(df)

#latitude_list = [] # dfの住所から変換した経度を追加するための空配列
#longtude_list = [] # dfの住所から変換した経度を追加するための空配列

# 地図上に表示するマーカーを名前の一覧配列をdfから取得します。
# df["hotelName"].values.tolist() でdf["hotelName"]値を配列としてname_listに代入します。
name_list = df["苗字"].values.tolist() 
latitude_list = df["緯度"].values.tolist() 
longtude_list = df["経度"].values.tolist() 

import folium # 地図機能をインポートします
# APIの住所から緯度経度変換で得られた配列の１つ目を基準にマップを生成
map = folium.Map(location=[latitude_list[0], longtude_list[0]], zoom_start=15)

# 地図しか生成しておらず、マーカーが１つもないので、
# APIの住所から緯度経度変換で得られた配列の１つ目を基準に生成したマップにマーカを追加する
for i in range(0,len(df)):
    folium.Marker(location=[latitude_list[i], longtude_list[i]], popup=name_list[i]).add_to(map) # マーカー名にホテル名のname_listを指定する

# ------------------------画面作成------------------------

st.write("住所Plot一覧") # タイトル
folium_static(map) # 地図情報を表示
# ------------------------
st.write("職業興味検査") # タイトル


# 'object' 型の列を数値型に変換
numeric_columns = ['InterestTest1', 'InterestTest2', 'InterestTest3', 'InterestTest4', 'InterestTest5', 'InterestTest6']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# 欠損値の処理（例: 平均値で補完）
df.fillna(df.mean(), inplace=True)

# 使用する列を指定
selected_columns = ['InterestTest1', 'InterestTest2', 'InterestTest3', 'InterestTest4', 'InterestTest5', 'InterestTest6']

# 選択した列のデータを抽出
selected_data = df[selected_columns]

df_corr = selected_data.corr()

fig_corr = go.Figure([go.Heatmap(z=df_corr.values,
                                 x=df_corr.index.values,
                                 y=df_corr.columns.values)])
fig_corr.update_layout(height=300,
                       width=700,
                       margin={'l': 20, 'r': 20, 't': 0, 'b': 0})
st.plotly_chart(fig_corr)
#-------------

#-------------


import streamlit as st
import pandas as pd
import plotly.express as px

# データの準備（先ほどのデータを使用）
# ここにデータの読み込みや生成が必要です

# ユーザーが選択した条件
selected_gender = st.multiselect('性別を選択', df['性別'].unique())

# 選択された条件でデータを絞り込み
filtered_df = df[df['性別'].isin(selected_gender)]

# 性別ごとに興味テストのスコアの平均値を計算
average_scores = filtered_df.groupby('性別')[['InterestTest1', 'InterestTest2', 'InterestTest3', 'InterestTest4', 'InterestTest5', 'InterestTest6']].mean().reset_index()

# 箱ひげ図の描画
fig = px.box(
    average_scores.melt(id_vars='性別'),
    x='variable',
    y='value',
    color='性別',
    title='性別ごとの興味テストスコアの平均値の箱ひげ図'
)

# グラフを表示
st.plotly_chart(fig)

#----
import streamlit as st
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "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"

# GPT-3モデルのセットアップ
llm = OpenAI(model_name="text-davinci-003", temperature=0.2)

# Streamlitアプリケーションの開始
st.write("質問フォーム")

# 質問の入力フォーム
question = st.text_input("質問を入力してください:")

# クエリの処理
if st.button("実行"):
    # クエリを処理するためのエージェントを作成
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    
    # クエリの実行
    answer = agent.run(question)
    
    # 結果の表示
    st.write(f"質問: {question}")
    st.write(f"回答: {answer}")

#----ß

