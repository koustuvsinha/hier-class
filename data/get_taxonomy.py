# Extract topic taxonomy from Wikipedia Vital Articles

import requests
from bs4 import BeautifulSoup as bs
import json

ENDPOINT = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles'
TAXONOMY_FILE = 'wiki_1000_taxonomy.json'

def recurse_links(rlist):
    """ Recurse through nested lists to get the pages
    """
    dic = {}
    if rlist:
        for li in rlist.find_all("li",recursive=False):
            #taxonomy[h_text][sub_head_t] = {}
            head_a = li.find_next("a")
            dic[head_a.text] = {}
            li_ul = li.find("ul")
            dic[head_a.text] = recurse_links(li_ul)
    return dic

# fetch the html
wiki_level_3 = bs(requests.get(ENDPOINT).text)
content = wiki_level_3.find("div", {"id": "mw-content-text"})
# create taxonomy
taxonomy = {}
avoid_headers = ['Current total:','Contents']
for heading in content.findAll("h2"):
    h_text = heading.text.split(' (')[0]
    if h_text not in avoid_headers:
        taxonomy[h_text] = {}
        h_content = heading.find_next("div")
        #print(h_content.findAll("h3"))
        if len(h_content.findAll("h3")) > 0:
            for sub_head in h_content.findAll("h3"):
                # subheading - Politicians and leaders
                sub_head_t = sub_head.text.split(' (')[0]
                taxonomy[h_text][sub_head_t] = {}
                c_list = sub_head.find_next("ul")
                taxonomy[h_text][sub_head_t] = recurse_links(c_list)
        else: # for mathematics
            c_list = h_content.find_all("ul",recursive=False)[0]
            taxonomy[h_text] = recurse_links(c_list)

# save taxonomy
json.dump(taxonomy, open('wiki_1000_taxonomy.json','w'))

