# coding:utf-8
import numpy as np


def evaluation_normal(path):
    base_relation_types = [
        ['PART-OF'],
        ['REGULATOR', 'DIRECT-REGULATOR', 'INDIRECT-REGULATOR'],
        ['UPREGULATOR', 'ACTIVATOR', 'INDIRECT-UPREGULATOR'],
        ['DOWNREGULATOR', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR'],
        ['AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR'],
        ['ANTAGONIST'],
        ['MODULATOR', 'MODULATOR-ACTIVATOR', 'MODULATOR-INHIBITOR'],
        ['COFACTOR'],
        ['SUBSTRATE', 'PRODUCT-OF', 'SUBSTRATE_PRODUCT-OF'],
        ['NOT', 'UNDEFINED']
    ]

    normal_relation_types = ['CPR:1', 'CPR:2', 'CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:7', 'CPR:8', 'CPR:9', 'CPR:10']

    relations2normal_dicts = {}
    for i, relations in enumerate(base_relation_types):
        relations2normal_dicts[str(relations)] = normal_relation_types[i]
        for relation in relations:
            relations2normal_dicts[relation] = normal_relation_types[i]

    print(relations2normal_dicts)

    N = {'CPR:1': 0, 'CPR:2': 0, 'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:7': 0, 'CPR:8': 0, 'CPR:9': 0,
         'CPR:10': 0}
    AP = {'CPR:1': 0, 'CPR:2': 0, 'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:7': 0, 'CPR:8': 0, 'CPR:9': 0,
          'CPR:10': 0}
    TP = {'CPR:1': 0, 'CPR:2': 0, 'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:7': 0, 'CPR:8': 0, 'CPR:9': 0,
          'CPR:10': 0}
    FP = {'CPR:1': 0, 'CPR:2': 0, 'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:7': 0, 'CPR:8': 0, 'CPR:9': 0,
          'CPR:10': 0}

    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.strip()
        instance = data.split('\n\n')
        results = [(i.split('\n')[2][:-7], i.split('\n')[-1]) for i in instance]

    results2normal = [(relations2normal_dicts[i[0]], relations2normal_dicts[i[1]]) for i in results]

    for item in results2normal:
        rel, pre = item
        AP[pre] += 1
        N[rel] += 1
        if pre == rel:
            TP[pre] += 1

    for i in AP:
        FP[i] = AP[i] - TP[i]
    print(f'N  {N}')
    print(f'AP {AP}')
    print(f'TP {TP}')
    print(f'FP {FP}')

    rec = [TP[i] / N[i] for i in N]
    prec = [TP[i] / (AP[i]) if AP[i] != 0 else 1 for i in N]
    f = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) if prec[i] != 0 else 1 for i in range(len(rec))]
    print(f'prec {prec}')
    print(f'rec {rec}')
    print(f'F {f}')

    # all
    N34569, TP34569, AP34569 = 0, 0, 0
    for i, item in enumerate(N, 1):
        if i == 3 or i == 4 or i == 5 or i == 6 or i == 9 or i == 1 or i == 2 or i == 7 or i == 8 or i == 10:
            N34569 += N[item]
            TP34569 += TP[item]
            AP34569 += AP[item]
    REC = TP34569 / N34569
    PREC = TP34569 / AP34569
    F = 2 * PREC * REC / (PREC + REC)
    print(f'PREC,REC,F {PREC, REC, F}')
    print()


######### gs
def evaluation_gs(path_gs):
    N_gs = {'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:9': 0}
    AP_gs = {'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:9': 0}
    TP_gs = {'CPR:3': 0, 'CPR:4': 0, 'CPR:5': 0, 'CPR:6': 0, 'CPR:9': 0}

    with open(path_gs, 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.strip()
        instance = data.split('\n\n')
        results = [(i.split('\n')[2], i.split('\n')[-1]) for i in instance]

    for item in results:
        rel, pre = item
        AP_gs[pre] += 1
        N_gs[rel] += 1
        if pre == rel:
            TP_gs[pre] += 1

    print(f'N_gs  {N_gs}')
    print(f'AP_ga {AP_gs}')
    print(f'TP_ga {TP_gs}')

    rec = [TP_gs[i] / N_gs[i] for i in N_gs]
    prec = [TP_gs[i] / AP_gs[i] for i in N_gs]
    f = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(len(rec))]
    print(f'prec_gs {prec}')
    print(f'rec_gs {rec}')
    print(f'F {f}')

    N, TP, AP = 0, 0, 0
    for i in N_gs:
        N += N_gs[i]
        TP += TP_gs[i]
        AP += AP_gs[i]
    REC = TP / N
    PREC = TP / AP
    F = 2 * PREC * REC / (PREC + REC)
    print(f'PREC,REC,F {PREC, REC, F}')
    print()


def evaluation(path):
    base_relation_types = [
        ['PART-OF'],
        ['REGULATOR', 'DIRECT-REGULATOR', 'INDIRECT-REGULATOR'],
        ['UPREGULATOR', 'ACTIVATOR', 'INDIRECT-UPREGULATOR'],
        ['DOWNREGULATOR', 'INHIBITOR', 'INDIRECT-DOWNREGULATOR'],
        ['AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR'],
        ['ANTAGONIST'],
        ['MODULATOR', 'MODULATOR-ACTIVATOR', 'MODULATOR-INHIBITOR'],
        ['COFACTOR'],
        ['SUBSTRATE', 'PRODUCT-OF', 'SUBSTRATE_PRODUCT-OF'],
        ['NOT', 'UNDEFINED']
    ]

    normal_relation_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    relations2normal_dicts = {}
    for i, relations in enumerate(base_relation_types):
        relations2normal_dicts[str(relations)] = normal_relation_types[i]
        for relation in relations:
            relations2normal_dicts[relation] = normal_relation_types[i]

    # print(relations2normal_dicts)
    matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.strip()
        instance = data.split('\n\n')
        results = [(i.split('\n')[2][:-7], i.split('\n')[-1]) for i in instance]

    results2normal = [(relations2normal_dicts[i[0]], relations2normal_dicts[i[1]]) for i in results]
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for item in results2normal:
        rel, pre = item
        total[rel] += 1
        matrix[rel][pre] += 1

    prec, rec, f1 = [], [], []
    tp,fp,fn=[],[],[]
    for i in range(len(matrix)):
        try:
            p = matrix[i][i] / sum([matrix[j][i] for j in range(len(matrix))])
            r = matrix[i][i] / sum([matrix[i][j] for j in range(len(matrix[0]))])
            tp.append(matrix[i][i])
            fp.append(sum([matrix[j][i] for j in range(len(matrix))])-matrix[i][i])
            fn.append(sum([matrix[i][j] for j in range(len(matrix[0]))])-matrix[i][i])
        except:
            p = 0
            r = 0
        prec.append(round(p,4))
        rec.append(round(r,4))
    # print(tp)
    # print(fp)
    # print(fn)
    # p=sum(tp)/(sum(tp)+sum(fp))
    # r=sum(tp)/(sum(tp)+sum(fn))
    # print(p,r,2*p*r/(p+r))
    for i in range(len(rec)):
        try:
            f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        except:
            f = 0
        f1.append(round(f,4))

    # all
    tp = [matrix[i][i] for i in range(len(matrix)-1)]
    matrix_gs = []
    for i in range(len(matrix)-1):
        m = matrix[i][0:len(matrix)-1]
        matrix_gs.append(m)
    tp, ap, n = sum(tp), 0, sum(total)
    for i in range(len(matrix_gs)):  # col
        ap += sum(matrix_gs[i])

    PREC, REC = tp / ap, tp / n
    F1 = 2 * PREC * REC / (PREC + REC)

    matrix1 = np.asarray(matrix)
    print(matrix1)

    print()
    print('total:', total)
    print('tp:   ', tp)
    print(f'prec:   {prec}')
    print(f'recall: {rec}')
    print(f'f1:     {f1}')
    print(f'all macro P,R,F {np.mean(prec), np.mean(rec), np.mean(f1)}')
    print(f'all PREC,REC,F {PREC, REC, F1}')
    print()

    # gs
    total_gs = total[2:6] + total[8:9]
    tp = [matrix[2][2], matrix[3][3], matrix[4][4], matrix[5][5], matrix[8][8]]


    matrix_gs = []
    for i in range(len(matrix)):
        m = matrix[i][2:6] + matrix[i][8:9]
        matrix_gs.append(m)
    tp, ap, n = sum(tp), 0, sum(total_gs)
    for i in range(len(matrix_gs)):  # col
        ap += sum(matrix_gs[i])

    PREC, REC = tp / ap, tp / n
    F1 = 2 * PREC * REC / (PREC + REC)

    print('total_gs:', total_gs)
    print('tp, ap, ap - tp, n:',tp, ap, ap - tp, n)
    print(f'PREC,REC,F {round(PREC,4), round(REC,4), round(F1,4)}')
    print()


if __name__ == '__main__':
    # path = r'./test_out.txt'  # test_pd
    # path=r'E:\biorelation_extraction\results\nores 2019-06-13 Thu 15_40_48 model acc 0.7329787235733465.txt'
    # path=r'E:\biorelation_extraction\results\2019-06-14 Fri 14_41_11 model acc 0.7006258692214155.txt'
    # path=r'E:\biorelation_extraction\results\2019-06-19 Wed 17_48_14 model acc 0.7153005471931455.txt'
    # path = r'testout2.txt'
    import sys
    path=sys.argv[1]
    evaluation(path)

    # evaluation_normal(path)
    # evaluation_normal(path1)
    # evaluation_normal(path2)

    # evaluation_gs(path_gs3)
    # evaluation_gs(path_gs2)
