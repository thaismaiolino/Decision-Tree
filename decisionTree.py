#coding: utf-8
from __future__ import print_function

import pandas

from sklearn.tree import DecisionTreeClassifier, export_graphviz
dataframe = None
file_name = None



def define_dataframe(csv_file):
    try:
        col_name = ['nu_ano','co_grupo','co_ies','co_catad','co_orgac','co_munic_curso','co_uf_curso','co_regiao_curso',
                    'co_curso','nu_idade','tp_sexo','ano_fim_2g','ano_in_grad','tp_semestre','in_matutino','in_vespertino',
                    'in_noturno','id_status','amostra','tp_inscricao','tp_def_fis','tp_def_vis','tp_def_aud','nu_item_ofg',
                    'nu_item_ofg_z','nu_item_ofg_x','nu_item_ofg_n','vt_gab_ofg_orig','vt_gab_ofg_fin','nu_item_oce',
                    'nu_item_oce_z','nu_item_oce_x','nu_item_oce_n','vt_gab_oce_orig','vt_gab_oce_fin','tp_pres',
                    'tp_pr_ger','tp_pr_ob_fg','tp_pr_di_fg','tp_pr_ob_ce','tp_pr_di_ce','tp_sfg_d1','tp_sfg_d2',
                    'tp_sce_d1','tp_sce_d2','tp_sce_d3','vt_esc_ofg','vt_ace_ofg','vt_esc_oce','vt_ace_oce',
                    'nt_obj_fg','nt_fg_d1_pt','nt_fg_d1_ct','nt_fg_d1','nt_fg_d2_pt','nt_fg_d2_ct','nt_fg_d2',
                    'nt_dis_fg','nt_fg','nt_obj_ce','nt_ce_d1','nt_ce_d2','nt_ce_d3','nt_dis_ce','nt_ce','nt_ger',
                    'qp_i1','qp_i2','qp_i3','qp_i4','qp_i5','qp_i6','qp_i7','qp_i8','qp_i9','qe_i1','qe_i2','qe_i3',
                    'qe_i4','qe_i5','qe_i6','qe_i7','qe_i8','qe_i9','qe_i10','qe_i11','qe_i12','qe_i13','qe_i14','qe_i15',
                    'qe_i16','qe_i17','qe_i18','qe_i19','qe_i20','qe_i21','qe_i22','qe_i23','qe_i24','qe_i25','qe_i26','qe_i27',
                    'qe_i28','qe_i29','qe_i30','qe_i31','qe_i32','qe_i33','qe_i34','qe_i35','qe_i36','qe_i37','qe_i38',
                    'qe_i39','qe_i40','qe_i41','qe_i42','qe_i43','qe_i44','qe_i45','qe_i46','qe_i47','qe_i48','qe_i49',
                    'qe_i50','qe_i51','qe_i52','qe_i53','qe_i54','qe_i55','qe_i56','qe_i57','qe_i58','qe_i59','qe_i60',
                    'qe_i61','qe_i62','qe_i63','qe_i64','qe_i65','qe_i66','qe_i67','qe_i68','qe_i69','qe_i70','qe_i71',
                    'qe_i72','qe_i73','qe_i74','qe_i75','qe_i76','qe_i77','qe_i78','qe_i79','qe_i80','qe_i81']
        global dataframe
        dataframe = pandas.read_csv(csv_file, sep=';', names=col_name)
        global file_name
        file_name = csv_file
        #Excluding NaN fields
        dataframe = dataframe.fillna(method='ffill')

    except pandas.io.common.EmptyDataError:
        raise Exception('Not a valid file.')


def print_dataframe():
    try:
        assert dataframe is not None, 'Dataframe was not created.'
        print('\n')
        print(dataframe)
        print('\n')
    except AssertionError, e:
        print(e.args[0])



def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


def decisionTree():
    data = dataframe[['nt_ger','ano_fim_2g','ano_in_grad','nu_idade','co_grupo', 'tp_sexo']].copy()
    df2, targets = encode_target(data, 'tp_sexo')
    print("* targets", targets, sep="\n", end="\n\n")
    features = list(df2.columns[:5])
    print("* features:", features, sep="\n")
    y = df2["Target"]
    X = df2[features]
    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X, y)
    return (dt, features)

def visualize_tree(tree, feature_names):

    try:
        dotfile = open("dt.dot", 'w')
        export_graphviz(tree, out_file = dotfile, feature_names = feature_names)
        dotfile.close()

    except Exception as e:
        print (e)
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def predict(info,dt):

    print (dt.predict(info))


define_dataframe("microdados_enade_2014_v2.csv")


dt, features = decisionTree()
visualize_tree(dt, features)
sample1 = [51.9, 2008, 2009, 24,21]#F
sample2 = [43.9, 2007, 2011, 25,21]#F
sample3 = [28.4, 2009, 2012, 23,79]#F
sample4 = [31.1, 2007, 2010, 27,79]#M
sample5 = [38.1, 2010, 2010, 22,702] #F
sample6 = [41.2,2008,2011,23,702] #M
sample7 = [61.6,2008,2011,30,904] #F
info = [sample1, sample2, sample3, sample4,sample5,sample6,sample7]
predict(info,dt)
