# Extract topic taxonomy from Wikipedia Vital Articles

import requests
from bs4 import BeautifulSoup as bs
import json
from gensim.models.doc2vec import Doc2Vec
import urllib
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm

ENDPOINT = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles'
TAXONOMY_FILE = 'wiki_1000_taxonomy.json'
WIKI_LOC = 'http://132.206.3.23:8020/enwikipedia/page/'
W_URL = 'https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json'

class DataCollector():
    def __init__(self, url_endpoint=ENDPOINT, taxonomy_file=TAXONOMY_FILE, wiki_elastic_loc=WIKI_LOC):
        self.url_endpoint = url_endpoint
        self.taxonomy_file = taxonomy_file
        self.wiki_elastic_loc = wiki_elastic_loc
        self.taxonomy = {}
        self.nodeID = {}

    def recurse_links(self,rlist):
        """ Recurse through nested lists to get the pages
        """
        dic = {}
        if rlist:
            for li in rlist.find_all("li",recursive=False):
                #taxonomy[h_text][sub_head_t] = {}
                head_a = li.find_next("a")
                dic[head_a.text] = {}
                li_ul = li.find("ul")
                dic[head_a.text] = self.recurse_links(li_ul)
        if len(dic) == 0:
            dic = 1
        return dic

    def get_taxonomy(self):
        # fetch the html
        wiki_level_3 = bs(requests.get(self.url_endpoint).text)
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
                        taxonomy[h_text][sub_head_t] = self.recurse_links(c_list)
                else: # for mathematics
                    c_list = h_content.find_all("ul",recursive=False)[0]
                    taxonomy[h_text] = self.recurse_links(c_list)
        self.taxonomy = taxonomy

    def get_trajectories(self):
        if len(self.taxonomy) < 1:
            self.get_taxonomy()

        # really bad code, was feeling too lazy
        all_trajs = []
        for k1, v1 in self.taxonomy.items():
            for k2, v2 in v1.items():
                if v2 == 1:
                    all_trajs.append([k1, k2])
                else:
                    for k3, v3 in v2.items():
                        if v3 == 1:
                            all_trajs.append([k1, k2, k3])
                        else:
                            for k4, v4 in v3.items():
                                if v4 == 1:
                                    all_trajs.append([k1, k2, k3, k4])
                                else:
                                    for k5, v5 in v4.items():
                                        if v5 == 1:
                                            all_trajs.append([k1, k2, k3, k4, k5])
        self.all_trajectories = all_trajs

    def get_leaf_nodes(self):
        if len(self.all_trajectories) < 1:
            self.get_trajectories()
        leaf_nodes = []
        for tr in self.all_trajectories:
            leaf_nodes.append(tr[-1])
        self.leaf_nodes = leaf_nodes
        return leaf_nodes

    def load_doc2vec(self, doc2vec_path='/home/ndg/projects/shared_datasets/wikipedia/Doc2Vec(dbow+w,d500,n10,hs,w8,mc19,t40).model'):
        self.doc2vec = Doc2Vec.load(doc2vec_path)

    ## Wikipedia ElasticSearch functions
    def search_by_id(self, query_id):
        xq = {
            "query": {
                "match": {
                    "_id": {
                        "query": query_id
                    }
                }
            }
        }
        res = requests.get(self.wiki_elastic_loc + '_search', data=json.dumps(xq))
        res_data = res.json()
        # print res_data['hits']['hits'][0]['_source']['external_link']
        page_hits = res_data['hits']['hits']
        if len(page_hits) < 1:
            return None
        pages = [p for p in page_hits]
        titles = [p['_source']['title'] for p in pages]
        if len(titles) > 0:
            page = pages[0]
            text = page['_source']['text']
            return text
        else:
            return None

    def get_pages(self, query_id, num_docs=5, sent_per_doc=10):
        """
        from a given query_id, extract the wiki page, then chunk it into top k j-sentence batch
        :param query_id:
        :return:
        """

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        page = self.search_by_id(query_id)
        if page:
            sents = sent_tokenize(page)
            sent_chunks = list(chunks(sents, sent_per_doc))
            cand = sent_chunks[:num_docs]
            cand = [' '.join(c) for c in cand]
            return cand
        else:
            return []


    def fetch_wiki_page_id(self, titles):
        """
        Get the ids of the pages (at most 50 at a time)
        """
        pruned = [t for t in titles if t not in self.nodeID]
        ptitles = [urllib.parse.quote(t.encode('utf8')) for t in pruned]
        if len(ptitles) > 0:
            pt = '|'.join(ptitles)
            res = requests.get(W_URL.format(pt))
            res_data = res.json()
            t = dict(res_data['query']['pages'])
            try:
                for k, v in t.items():
                    if 'pageid' in v:
                        self.nodeID[v['title']] = str(v['pageid'])
                    else:
                        # when the page is not found in wikipedia
                        self.nodeID[v['title']] = "-1"
                # check for normalization of titles
                if 'normalized' in res_data['query']:
                    norms = res_data['query']['normalized']
                    for n in norms:
                        id = titles.index(n['from'])
                        titles[id] = n['to']
            except Exception as e:
                print(e)

    def collect_ids(self, nodes):
        """
        Collect all the titles in bunch
        :param nodes:
        :return:
        """
        print('Collecting ids for {} nodes'.format(len(nodes)))
        for i in range(0, len(nodes), 50):
            bunch = nodes[i:i + 50]
            self.fetch_wiki_page_id(bunch)

        print('Nodes collected total : {}'.format(len(self.nodeID)))

    def collect_top_neighbours(self, topk=50):
        """
        policy: collect top k neighbors from each leaf node
        make sure that neighbor overlap is minimum. To do that, collect the neigbour similarities
        and break ties by higher similarity. but then we cannot fix topk neighbors
        :return:
        """
        node2sim = {}
        sim_max = {} # max similarity for sim
        for i,node in enumerate(self.leaf_nodes):
            # check if node is present
            if node in self.doc2vec.docvecs:
                sims = self.doc2vec.docvecs.most_similar(node, topn=topk)
                node2sim[node] = sims
                for s in sims:
                    if s[0] not in sim_max:
                        sim_max[s[0]] = 0
                    if sim_max[s[0]] < s[1]:
                        sim_max[s[0]] = s[1]
        # prune
        prune2sim = {}
        for node,sims in node2sim.items():
            prune2sim[node] = [s[0] for s in sims if s[1] == sim_max[s[0]]]

        self.node2sim = prune2sim


    def load_taxonomy(self):
        self.taxonomy = json.load(open(self.taxonomy_file))

    def save_taxonomy(self):
        # save taxonomy
        json.dump(self.taxonomy, open(self.taxonomy_file,'w'))


if __name__=='__main__':
    dc = DataCollector()
    dc.get_taxonomy()
    dc.get_trajectories()
    ln = dc.get_leaf_nodes()
    dc.load_doc2vec()
    dc.collect_top_neighbours()
    nd = dc.node2sim
    all_nodes = [v for k, v in nd.items()]
    all_nodes = [x for v in all_nodes for x in v]
    dc.collect_ids(all_nodes)
    node2id = dc.nodeID
    df = pd.DataFrame(columns=['text','l1','l2','l3'])
    doc_id = 0
    pb = tqdm(total=len(dc.all_trajectories))
    for i,traj in enumerate(dc.all_trajectories):
        leaf = traj[-1]
        if leaf in nd:
            leaf_sim = nd[leaf]
            doc_list = [leaf] + leaf_sim
            doc_id_list = [node2id[d] for d in doc_list if d in node2id]
            docs = [dc.get_pages(doc_id) for doc_id in doc_id_list]
            docs = [v for x in docs for v in x]
            for j,doc in enumerate(docs):
                if len(traj) >= 3:
                    df.loc[doc_id] = [doc, traj[0], traj[1], traj[2]]
                    doc_id += 1
        pb.update(1)
    pb.close()
    df.to_csv('full_docs.csv',index=None)





