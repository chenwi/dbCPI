# coding:utf-8
import re
from nltk import sent_tokenize


class GenerateData:
    def __init__(self, abstract_path, entities_path, relation_path, save_path):
        self.abstracts_dict = self.get_abstracts_dict(abstract_path)
        self.entities_dict = self.get_entities_dict(entities_path)
        self.relation_dict = self.get_relation_dict(relation_path)

        self.get_data(save_path)  # have 'not' label

    def cut_sentence(self, sent):
        temps = sent_tokenize(sent)
        splits = []
        for s in temps:
            s = re.sub('E\. coli', 'E  coli', s)
            s = re.sub('vs. ', 'vs  ', s)
            s = re.sub('\. |; ', '@ ', s)
            s = re.sub('\.T|; ', '@T', s)

            t = re.split('@', s)
            splits.extend(t)
        return splits

    def get_abstracts_dict(self, path):
        '''
        :param path: abstract.txt
        :return: abstracts_dict
            {pmid1: [s1,s2, ..], ...}
        '''
        abstracts_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.rstrip('\n')
                pmid, abstract = re.split('\t', line, maxsplit=1)
                title, abstract = re.split('\t', abstract, maxsplit=1)
                if pmid not in abstracts_dict:
                    # split_t = self.cut_sentence(title)
                    # split_a = self.cut_sentence(abstract)
                    abstracts_dict[pmid] = [title] + sent_tokenize(abstract)
        return abstracts_dict

    def get_entities_dict(self, path):
        '''
        :param path: entity.txt
        :return: entities_dict
        '''
        entities_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.rstrip('\n')
                pmid, tid, typ, start, end, entity = re.split('\t', line)

                if pmid not in entities_dict:
                    entities_dict[pmid] = {}
                if tid not in entities_dict[pmid]:
                    entities_dict[pmid][tid] = (typ, start, end, entity)
        return entities_dict

    def get_relation_dict(self, path):
        '''
        :param path: 'rel.txt'
        :return: relation_dict
        '''
        relation_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.rstrip('\n')
                pmid, *_, rel, arg1, arg2 = re.split('\t', line)
                arg = f'{arg1[5:]}_{arg2[5:]}'

                if pmid not in relation_dict:
                    relation_dict[pmid] = {}
                if arg not in relation_dict[pmid]:
                    relation_dict[pmid][arg] = rel
        # print(relation_dict)
        return relation_dict

    def filter(self, s):
        words = [',', '(', ')', '[', ']', 'and', 'or', 'as']
        for w in words:
            if w in s:
                return True
        return False

    def get_data(self, path):
        '''
        :param path: 'data.txt'
        '''
        count = 0
        _count = 0
        with open(path, 'w', encoding='utf-8') as f:
            for (pmid, sent_list) in self.abstracts_dict.items():
                pre_len = 0
                for sent in sent_list:
                    chem_dict, gene_dict = {}, {}
                    for (tid, ann) in self.entities_dict[pmid].items():
                        if int(ann[1]) >= pre_len and int(ann[2]) <= pre_len + len(sent):
                            # print(ann)
                            if ann[0] == 'CHEMICAL':
                                chem_dict[tid] = ann
                            else:
                                gene_dict[tid] = ann

                    for tid1, chem in chem_dict.items():
                        for tid2, gene in gene_dict.items():
                            arg = f"{tid1}_{tid2}"
                            if (pmid in self.relation_dict) and (arg in self.relation_dict[pmid]):
                                flag = True
                    for tid1, chem in chem_dict.items():
                        for tid2, gene in gene_dict.items():
                            chem_s, chem_e, gene_s, gene_e = int(chem[1]) - pre_len, int(chem[2]) - pre_len, \
                                                             int(gene[1]) - pre_len, int(gene[2]) - pre_len
                            p = [chem_s, chem_e, gene_s, gene_e]
                            p.sort()
                            d = p[2] - p[1]
                            # d = min(abs(gene_s - chem_e), abs(chem_s - gene_e))
                            if chem_e < gene_s:  # ('CHEMICAL', '41', '49', 'retinoid')
                                sent_ann = f"{sent[:chem_s]}<e1>{sent[chem_s:chem_e]}</e1>" \
                                           f"{sent[chem_e:gene_s]}<e2>{sent[gene_s:gene_e]}</e2>{sent[gene_e:]}"
                            elif gene_e < chem_s:
                                sent_ann = f"{sent[:gene_s]}<e1>{sent[gene_s:gene_e]}</e1>" \
                                           f"{sent[gene_e:chem_s]}<e2>{sent[chem_s:chem_e]}</e2>{sent[chem_e:]}"
                            else:
                                # print("ERROR!\n",pmid, sent)
                                # print(chem_s, chem_e, gene_s, gene_e, chem[-1], gene[-1])
                                continue
                            arg = f"{tid1}_{tid2}"

                            # if d < 10 and self.filter(sent[p[1]:p[2]]):
                            #     continue

                            if (pmid in self.relation_dict) and (arg in self.relation_dict[pmid]):
                                rel_type = self.relation_dict[pmid][arg]
                                count += 1

                            else:
                                rel_type = 'NOT'
                                _count += 1

                            instance = f"{pmid}@{chem[-1]}@{gene[-1]}@{rel_type}\n{sent_ann}\n{rel_type}(e1,e2)\n\n"
                            # print(instance)
                            f.write(instance)
                    pre_len += len(sent) + 1
        print(count)
        print(_count)


if __name__ == '__main__':
    dir = r'../../data/chemprot_training/'
    abstract_path = dir + r'chemprot_training_abstracts.tsv'
    entities_path = dir + r'chemprot_training_entities.tsv'
    relation_path = dir + r'chemprot_training_relations.tsv'
    save_path = r'../../data/format_data/ann_train_data.txt'
    # if not os.path.exists(save_path):
    GenerateData(abstract_path, entities_path, relation_path, save_path)

    #
    abstract_path = r'../../data/chemprot_test_gs/chemprot_test_abstracts_gs.tsv'
    entities_path = r'../../data/chemprot_test_gs/chemprot_test_entities_gs.tsv'
    relation_path = r'../../data/chemprot_test_gs/chemprot_test_relations_gs.tsv'
    save_path = r'../../data/format_data/ann_test_data.txt'
    # if not os.path.exists(save_path):
    GenerateData(abstract_path, entities_path, relation_path, save_path)

    abstract_path = r'../../data/chemprot_development/chemprot_development_abstracts.tsv'
    entities_path = r'../../data/chemprot_development/chemprot_development_entities.tsv'
    relation_path = r'../../data/chemprot_development/chemprot_development_relations.tsv'
    save_path = r'../../data/format_data/ann_dev_data.txt'
    # if not os.path.exists(save_path):
    GenerateData(abstract_path, entities_path, relation_path, save_path)
