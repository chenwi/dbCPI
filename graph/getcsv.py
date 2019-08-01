import pandas as pd
import uuid
import numpy as np
import pickle
import os

relations = {
    "['PART-OF']": 'PART-OF',
    "['REGULATOR', 'DIRECT-REGULATOR', 'INDIRECT-REGULATOR']": 'REGULATOR',
    "['UPREGULATOR', 'ACTIVATOR', 'INDIRECT-UPREGULATOR']": 'ACTIVATOR',
    "['DOWNREGULATOR', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR']": 'INHIBITOR',
    "['AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR']": 'AGONIST',
    "['ANTAGONIST']": "ANTAGONIST",
    "['MODULATOR', 'MODULATOR-ACTIVATOR', 'MODULATOR-INHIBITOR']": 'MODULATOR',
    "['COFACTOR']": 'COFACTOR',
    "['SUBSTRATE', 'PRODUCT-OF', 'SUBSTRATE_PRODUCT-OF']": 'SUBSTRATE_PRODUCT-OF',
    "['NOT', 'UNDEFINED']": 'NOT'
}


def is_trust(z, value=0.95):
    z = z[1:-1]
    z = z.split()
    z = list(map(float, z))
    z = np.asarray(z)
    p = np.exp(z) / sum(np.exp(z))
    return any(p > value)


num = 200

ann_path = r'E:\biorelation_extraction\testout\testout%s.txt' % num
with open(ann_path, 'r', encoding='utf-8') as f:
    data = f.read()
instances = data.strip().split('\n\n')
ann_list = []
count = 0
for x in instances:
    ann, *_, z, r = x.split('\n')
    if is_trust(z):
        count += 1
        ann_list.append([ann, r])
print(count, len(instances))
print(count / len(instances))


if not os.path.exists('pkl\chem_dict.pkl'):
    chem_dict, gene_dict = {}, {}
else:
    print('ok')
    chem_pkl = open('pkl\chem_dict.pkl', 'rb')
    gene_pkl = open('pkl\gene_dict.pkl', 'rb')
    chem_dict=pickle.load(chem_pkl)
    gene_dict = pickle.load(gene_pkl)

chem_data, gene_data, rel_data = [], [], []

for ann in ann_list:
    entity, rel = ann
    pmid, chem, gene, cst, cend, gst, gend = entity.split('\t')
    rel = relations[rel]
    if rel=='NOT':
        continue
    if chem not in chem_dict:
        chem_dict[chem] = uuid.uuid1()
    if gene not in gene_dict:
        gene_dict[gene] = uuid.uuid1()

    chemid = chem_dict[chem]
    geneid = gene_dict[gene]
    rel_data.append([chemid, geneid, rel, pmid, cst, cend, gst, gend])

for chem in chem_dict:
    chem_data.append([chem_dict[chem], chem, "Chemical"])
for gene in gene_dict:
    gene_data.append([gene_dict[gene], gene, "Protein"])
#
chem_pkl = open('pkl\chem_dict.pkl', 'wb')
gene_pkl = open('pkl\gene_dict.pkl', 'wb')
chem_dict=pickle.dump(chem_dict,chem_pkl)
gene_dict = pickle.dump(gene_dict,gene_pkl)

chem_path = r'data\chemical%s.csv' % num
pro_path = r'data\protein%s.csv' % num
rel_path = r'data\relation%s.csv' % num
chem_df = pd.DataFrame(chem_data, columns=['chemicalId:ID', 'name', 'label:LABEL'])
pro_df = pd.DataFrame(gene_data, columns=['proteinId:ID', 'name', 'label:LABEL'])
rel_df = pd.DataFrame(rel_data, columns=['chemicalId:START_ID', 'proteinId:END_ID', 'type:TYPE', 'pmid', 'cst', 'cend', 'gst', 'gend'])
# print(rel_df)
chem_df.to_csv(chem_path, index=False,header=None)
pro_df.to_csv(pro_path, index=False,header=None)
rel_df.to_csv(rel_path, index=False,header=None)
