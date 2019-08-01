# coding:utf-8
import re
nums=[str(i) for i in range(0,10)]
chars=[chr(i) for i in range(97,123)]+[' ']+nums
print(chars)
def sent_token(sen):
    words = sen.split()
    res = []
    for word in words:
        if len(word) < 3:
            if len(word) == 1:
                res.append("#" + word + "#")
            elif len(word) == 2:
                res.append("#" + word)
                res.append(word + "#")
        else:
            res.append("#" + word[:2])
            i = 0
            while i < len(word) - 2:
                res.append(word[i:i + 3])
                i += 1
            res.append(word[-2:] + "#")
    return " ".join(res)


def normalizer(text):
    text=text.lower()
    normal=''
    for i in text:
        if i not in build_vocab():
            # print(i)
            normal+='$'
        else:
            normal+=i
    return normal

def build_vocab():
    chars = [chr(i) for i in range(97, 123)]
    nums=[str(i) for i in range(0,10)]
    punks=[' ','-','/','(',')','α','β','γ','κ']
    return chars+nums+punks

with open(r'E:\BioNLP\xml\a0.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

with open('./token.txt', 'w', encoding='utf-8') as f:
    for line in data:
        abstract = line.split('\t', 1)[-1]
        # print(abstract)
        abstract=abstract.lower()
        abstract=normalizer(abstract)
        abstract = re.sub(',|\.|\?', "", abstract)
        abstract = re.sub('\t', " ", abstract)
        token = sent_token(abstract)
        # print(token.lower())
        f.write(token)
        f.write('\n')
