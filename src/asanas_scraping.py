import pandas as pd
import requests
import re
from bs4 import BeautifulSoup


from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By



def change_names(x):
    if x == "ashtanga namaskara":
        return "Ashtanga namaskara"
    elif x == "dwi pada viparita dandasana":
        return "Dwi pada viparita dandasana"
    elif x=="parsva bakasana":
        return "Parsva bakasana"
    elif x == "urdhva mukha svanasana":
        return "Urdhva mukha svanasana"
    elif x == "phalakasana":
        return "Kumbhakasana"
    else:
        return x.title()


def get_asanas_names(path):
    df = pd.read_csv(path)
    df["Name"] = df["Name"].apply(change_names)
    my_asanas = list(df["Name"])
    return my_asanas



def clean_df(df):
    df.loc[df["eng_name"] == "Happy Baby Pose", "asana_name"] = "Ananda Balasana"
    # remove duplicates
    df = df.drop(df[df["eng_name"] == "Low Lunge Pose"].index)
    df = df.drop(df[df["eng_name"] == "Arrow Lunge"].index)
    df = df.drop(df[df["eng_name"] == "Balancing Butterfly Pose"].index)
    df = df.drop(df[df["eng_name"] == "Noose Pose"].index)
    df = df.drop(df[df["eng_name"] == "Ragdoll Pose"].index)

    df.sort_values(by='asana_name',inplace=True,ignore_index=True)
    return df




# SCRAPING
def get_url(soup):
    s2 = soup.find_all("div",{"class":"position-card"})[0].find_all("a",{"class":"pic"})[0]
    return "https://www.yogapedia.com" + s2["href"]


def get_all_url():
    url_list = []
    driver = webdriver.Chrome()
    driver.get("https://www.yogapedia.com/yoga-poses#")
    html = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(html, "html.parser")
    all_positions = soup.find_all("div", {"id":"yoga-positions-wrapper"})[0].find_all("ul", {"id":"yoga-positions-listing"})[0].find_all("li",{"class":"item-card shown"})
    for el in all_positions:
        url = get_url(el)
        url_list.append(url)
    return url_list


def get_asana_info(asana_list,driver):
    d={}
    html = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(html, "html.parser")

    asana_name = soup.find_all("div", {"class":"headline"})[0].getText().split("\n")[-2]
    
    if asana_name in asana_list: # get only my asanas, not all the asanas in the website
        d["asana_name"]=asana_name
        
        img_url = soup.find_all("div",{"class":"photo"})[0].find('img')["src"]
        response = requests.get(img_url)
        with open(f'../images/{asana_name}.png', 'wb') as f:
            f.write(response.content)
            
        name = soup.find_all("div", {"class":"headline"})[0].getText().split("\n")[1]
        d["eng_name"]= name
        
        level = soup.find_all("div",{"class":"details"})[0].find('dt', text='Pose Level').find_next('dd').getText()
        d["level"]=level
        
        pose_type = soup.find_all("div",{"class":"details"})[0].find('dt', text='Pose Type').find_next('dd').getText()
        d["pose_type"]=pose_type
        
        instructions = soup.find_all("div", {"class":"description"})[0].find("h3", text="Instructions").find_next().getText()
        d["instructions"]=instructions
        
        drishti = soup.find_all("div",{"class":"details"})[0].find('dt', text='Drishti').find_next('dd').getText().split('(')[-1][:-1]
        d["drishti"]=drishti
        
        cautions = soup.find_all("div", {"class":"description"})[0].find("h3", text="Cautions").find_next().find_all('li')
        cautions = [el.getText() for el in cautions]
        d["cautions"] = '\n'.join(cautions)
        
        benefits = soup.find_all("div", {"class":"description"})[0].find("h3", text=re.compile(r'^Benefits.*')).find_next().find_all('li')
        benefits = [el.getText() for el in benefits]
        d["benefits"]= '\n'.join(benefits)
        
    return d


def get_all_asanas_info(names_path, save_path):

    asana_list = get_asanas_names(names_path)
    
    url_list = get_all_url() # urls to each position page
    all_poses = []

    for el in url_list:
        driver = webdriver.Chrome()
        driver.get(el)

        d = get_asana_info(asana_list, driver)
        if d!={}:
            all_poses.append(d) 

    asanas_df = pd.DataFrame(all_poses)
    asana_df = clean_df (asana_df)
    asanas_df.to_csv(save_path, index=False)