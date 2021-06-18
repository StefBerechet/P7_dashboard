# Auteur : Stefan Berechet
# Date : 18 juin 2021

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
from PIL import Image

header = st.beta_container()
dataset = st.beta_container()


@st.cache
def chargement_donnees (file_name):
    data = joblib.load(file_name)    
    if 'SK_ID_CURR' in data.columns:
        data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)        
    if 'TARGET' in data.columns:
        data = data.drop(['TARGET'], axis=1)        
    data = data.reset_index(drop=True)
    return data

@st.cache
def chargement_modele (file_name):
    model = joblib.load(file_name)
    return model
mon_model = chargement_modele('datas_dashboard/best_model_GradientBoostingClassifier_undersampling_data.pkl')

@st.cache
def shap_explainer (model, X_val_std):  
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(X_val_std)   
    return explainer, shap_values

#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)

with header:
    logo_reduit = Image.open('datas_dashboard/logo_prêt_à_dépenser_réduit.png')
    st.image(logo_reduit, use_column_width=True)    
    st.title("Modèle de scoring - Défaut de paiement")    
    st.write("L'objectif est d'étayer la décision d'accorder ou non un prêt à un client en prédisant la probabilité de défaut de paiement.")
    st.write("Le modèle a été entrainé sur les données se trouvant à [cette adresse](https://www.kaggle.com/c/home-credit-default-risk/data).") 
    
with dataset:          
        
    ### CHARGEMENT DONNEES
       
    data_val = chargement_donnees('datas_dashboard/data_val.pkl')
    #data_val = chargement_donnees('data_val.pkl')
    data_info_clients = chargement_donnees('datas_dashboard/data_info_client.pkl')
    #data_info_clients = chargement_donnees('data_info_client.pkl')
    
    ### TRANSFORMATION DONNEES
    
    # Fusion data_val avec data_info_clients
    colonnes_info_clients = list(data_info_clients.columns.values)    
    liste_communs = ['SK_ID_CURR','AMT_CREDIT','DAYS_BIRTH','DAYS_EMPLOYED']
    colonnes_info_clients_uniques = [ele for ele in colonnes_info_clients if ele not in liste_communs]    
    data_merge_val_info = data_val.merge(data_info_clients, on=['SK_ID_CURR','AMT_CREDIT','DAYS_BIRTH','DAYS_EMPLOYED'], how='left')    
    data_merge_val_info = data_merge_val_info[data_merge_val_info['OCCUPATION_TYPE'].notna()]
    
    ### FILTRES SIDEBAR
    
    #logo = Image.open('datas_dashboard/logo_prêt_à_dépenser_réduit.png')
    #st.sidebar.image(logo, use_column_width=True)
    st.sidebar.header("Veuillez filtrer les clients : ")
    
    # Filtre 1 NAME_CONTRACT_TYPE contrat
    liste_contract = data_merge_val_info['NAME_CONTRACT_TYPE'].unique().tolist()
    liste_contract.insert(0, "All") 
    type_contract = st.sidebar.selectbox('Type contrat  :', liste_contract)
    
    # Filtre 2 CODE_GENDER
    liste_sexe = data_merge_val_info['CODE_GENDER'].unique().tolist()
    liste_sexe.remove('XNA')
    liste_sexe.insert(0, "All") 
    type_sexe = st.sidebar.selectbox('Genre (M/F) :', liste_sexe)    

    # Filtre 3 OCCUPATION_TYPE
    liste_occupation = data_merge_val_info['OCCUPATION_TYPE'].unique().tolist()    
    liste_occupation.insert(0, "All")
    type_occupation = st.sidebar.selectbox('Profession :', liste_occupation)    
    
    # Application filtres
    if (type_contract != 'All') & (type_sexe != 'All') & (type_occupation != 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[
            (data_merge_val_info['NAME_CONTRACT_TYPE']==type_contract) & 
            (data_merge_val_info['CODE_GENDER']==type_sexe) & 
            (data_merge_val_info['OCCUPATION_TYPE']==type_occupation)]
        
    if (type_contract == 'All') or (type_sexe != 'All') or (type_occupation != 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[ 
            (data_merge_val_info['CODE_GENDER']==type_sexe) & 
            (data_merge_val_info['OCCUPATION_TYPE']==type_occupation)]
        
    if (type_contract != 'All') & (type_sexe == 'All') & (type_occupation != 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[
            (data_merge_val_info['NAME_CONTRACT_TYPE']==type_contract) &
            (data_merge_val_info['OCCUPATION_TYPE']==type_occupation)]
    
    if (type_contract != 'All') & (type_sexe != 'All') & (type_occupation == 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[
            (data_merge_val_info['NAME_CONTRACT_TYPE']==type_contract) & 
            (data_merge_val_info['CODE_GENDER']==type_sexe)]
        
    if (type_contract == 'All') & (type_sexe == 'All') & (type_occupation != 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[
            (data_merge_val_info['OCCUPATION_TYPE']==type_occupation)]
        
    if (type_contract == 'All') & (type_sexe != 'All') & (type_occupation == 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[ 
            (data_merge_val_info['CODE_GENDER']==type_sexe)]
        
    if (type_contract != 'All') & (type_sexe == 'All') & (type_occupation == 'All') :
        data_info_clients_filtre = data_merge_val_info.loc[
            (data_merge_val_info['NAME_CONTRACT_TYPE']==type_contract)]
    
    if (type_contract == 'All') & (type_sexe == 'All') & (type_occupation == 'All') :
        data_info_clients_filtre = data_merge_val_info
        
    # Filtre age slider
    values = st.sidebar.slider("Tranche d'age :",
                               min(data_info_clients_filtre.AGE),
                               max(data_info_clients_filtre.AGE),
                               (min(data_info_clients_filtre.AGE), max(data_info_clients_filtre.AGE)),
                               step=1)    
    data_info_clients_filtre = data_info_clients_filtre[data_info_clients_filtre.AGE>=values[0]]
    data_info_clients_filtre = data_info_clients_filtre[data_info_clients_filtre.AGE<=values[1]]    
    data_info_clients_filtre = data_info_clients_filtre.reset_index(drop=True)   
   

   ### CHOIX CLIENT
    
    st.sidebar.header("Veuillez choisir le client : ")
    st.sidebar.write('Nombre de clients disponibles :',data_info_clients_filtre.shape[0]) 
    
    # ID Client sur liste déroulante    
    liste_ids_filtre = data_info_clients_filtre['SK_ID_CURR'].unique().tolist()
    client_id = st.sidebar.selectbox('IDs clients disponibles :', liste_ids_filtre)

    # Dataframe au format de l'input du modèle
    data_X_filtre = data_info_clients_filtre.drop(['AGE',
                                                   'NAME_HOUSING_TYPE',
                                                   'NAME_EDUCATION_TYPE',
                                                   'CNT_CHILDREN',
                                                   'NAME_FAMILY_STATUS',
                                                   'CODE_GENDER',
                                                   'AMT_INCOME_TOTAL',
                                                   'OCCUPATION_TYPE',
                                                   'NAME_CONTRACT_TYPE'], axis=1)
    
    if (client_id in data_info_clients_filtre['SK_ID_CURR'].values) == True:        
        
        ## INFORMATIONS CLIENT
        
        # Extraction Informations client
        index = data_info_clients_filtre.index
        condition = data_info_clients_filtre['SK_ID_CURR'] == client_id
        condition_indices = index[condition]
        position_client = condition_indices.tolist()[0]
        
        data_client = data_val[data_val.SK_ID_CURR == client_id]
        data_client_info = data_info_clients[data_info_clients.SK_ID_CURR == client_id]
        df = data_client_info.drop(['DAYS_BIRTH'], axis=1) 
        df = df[['NAME_CONTRACT_TYPE','AMT_CREDIT','AMT_INCOME_TOTAL',
                 'CODE_GENDER','AGE','CNT_CHILDREN',
                 'NAME_FAMILY_STATUS','OCCUPATION_TYPE','DAYS_EMPLOYED',
                 'NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE']]        
        df.index = [client_id] * len(df)
        
          
        # Affichage Informations client
        st.header("Informations au sujet du client choisi :") 
        st.table(df.T)
                
        ## PREDICITONS DU MODELE    
        
        # Input
        x_val = data_client.drop(['SK_ID_CURR'], axis=1)
        
        # Standard Scaler
        mon_scaler = joblib.load('datas_dashboard/scaler_StandardScaler.pkl')        
        x_val_std = mon_scaler.transform(x_val)
        
        # Intérrogation modèle
        y_val_pred_probas = mon_model.predict_proba(x_val_std)[:,1]
        threshold = 0.5
        y_val_pred = (y_val_pred_probas[:] >= threshold).astype('int')        

        # Affichage résultat prédiction sous forme d'image
        if  y_val_pred[0] == 0:
            img_result = Image.open('datas_dashboard/img_client_éligible.png')
            st.image(img_result, use_column_width=True)            
        if  y_val_pred[0] == 1:
            img_result = Image.open('datas_dashboard/img_client_non_éligible.png')
            st.image(img_result, use_column_width=True)
       
        ## FEATURES IMPORTANCE        
        
        X_val = data_X_filtre.drop(['SK_ID_CURR'], axis=1)
        X_val_std = mon_scaler.transform(X_val)        
        [explainer,shap_values] = shap_explainer(mon_model, X_val_std) 
        
        # sur un idividu 
        #st.header("Probabilité de défaut de paiement du client :", y_val_pred_probas)
        st.write("Probabilité de défaut de paiement du client :", round(y_val_pred_probas[0],2))
        
        #st_shap(shap.force_plot(explainer.expected_value[0],
        #               shap_values[position_client,:], X_val.iloc[position_client,:],
        #               link="logit"))
        #st.set_option('deprecation.showPyplotGlobalUse', False) 
        #st.pyplot(shap.force_plot(explainer.expected_value[0], shap_values[position_client,:], X_val.iloc[position_client,:], link="logit"))
        
        st.header("Données du client contribuant à la décision :")
        st.set_option('deprecation.showPyplotGlobalUse', False)        
        st.pyplot(shap.waterfall_plot(shap.Explanation(values=shap_values[position_client,:],
                                                     base_values=explainer.expected_value[0],
                                                     data=X_val.iloc[position_client,:],
                                                     feature_names=X_val.columns.tolist()),max_display=20))
                
        # Feature importance sur le corpus d'individus filtrés
        if st.checkbox('Comparaison avec les clients similaires précédamment filtrés :'):             
            st.write('Nombre de clients similaires :',data_info_clients_filtre.shape[0])                    
            st.pyplot(shap.summary_plot(shap_values, X_val))
        
    if (client_id in data_val['SK_ID_CURR'].values) == False:        
        st.write("L'ID client saisi n'existe pas, veuillez saisir un ID client valide")
        
        
        

    
    
    
