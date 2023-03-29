from math import floor
import os

import numpy as np
import pandas as pd
from joblib import dump, load

import config 

class Dev:
    def __init__(self, name: str, dataset_name: str, x: pd.DataFrame, y_class):

        self.config = config.get_config_dict()
        
        self.name = name
        self.dataset_name = dataset_name
        self.x = x.copy()
        self.y_class = y_class.copy()

        inlier_class = y_class.mode()[0]  # inlier class = most common class
        self.y_out = y_class.apply(lambda x: 0 if x == inlier_class else 1)

    def getX_y_class(self):
        return pd.concat(self.x, self.y)

    def model_fit(self, model, num_nodes_per_class=0, reset=False):
        clf_name = str(model).partition("(")[0]

        self.num_nodes_per_class = num_nodes_per_class
        
        root = self.config['step1_path']
        pth = "" if self.num_nodes_per_class == 0 else str(self.num_nodes_per_class)
        path = ( root + pth + "/" + self.dataset_name + "/" + clf_name + "/" )
        filename = str(self.name) + ".joblib"
        fullname = path + filename
        
        if not os.path.exists(fullname) or reset:
            #os.makedirs(os.path.dirname(fullname), exist_ok=True)
            os.makedirs(path, exist_ok=True) 
            print("Fitting and writing model", fullname)
            model.fit(self.x.to_numpy())
            dump(model, fullname)
        else:
            print("Reading model", fullname)
            model = load(fullname)
        self.model = model

    def write_other_models(self, devs: list, reset=False):
        clf_name = str(self.model).partition("(")[0]
        num_received = len(devs)  # mettiamo anche questo nel path
        root = self.config["step2_path"]
        for sender in devs:  # dev che inviano il proprio modello addestrato
            r = self.name  # receiver
            s = sender.name
            path = (root + self.dataset_name + "/" + str(num_received) + "/" + clf_name + "/")
            filename = r + "-" + s + ".joblib"
            fullname = path + filename
            if not os.path.exists(fullname) or reset:
                y_pred = self.senders[sender]
                print("Writing", fullname)
                os.makedirs(path, exist_ok=True)
                dump(y_pred, fullname)


    def set_other_models(self, devs: list, reset=False, write=True):
        self.senders = {}
        clf_name = str(self.model).partition("(")[0]
        num_received = len(devs)  # mettiamo anche questo nel path
        root = self.config["step2_path"]

        for sender in devs:  # dev che inviano il proprio modello addestrato
            r = self.name  # receiver
            s = sender.name
            pth = "" if self.num_nodes_per_class == 0 else str( self.num_nodes_per_class)
            path = ( root + pth + "/predictions/" + self.dataset_name + "/" + str(num_received) + "/" + clf_name + "/" )
            filename = r + "-" + s + ".joblib"
            fullname = path + filename
            if write:
                if not os.path.exists(fullname) or reset:
                    y_pred = sender.model.predict(self.x)
                    # print('Writing_comp', fullname)
                    os.makedirs(
                        path, exist_ok=True
                    )  # commenta da qui se vuoi separare write
                    dump(y_pred, fullname)
                else:
                    try:
                        # print('Reading', fullname)
                        y_pred = load(fullname)
                    except EOFError as e: # se fallisce la lettura
                        print("ERROR", e)
                        y_pred = sender.model.predict(self.x)
                        # print('Writing', fullname)
                        os.makedirs(path, exist_ok=True)
                        dump(y_pred, fullname)
            else :
                y_pred = sender.model.predict(self.x)
            # TODO ! stai provando oggetto dev (non name) come chiave. prima era self.senders[s]
            self.senders[sender] = y_pred
            if r == s:
                self.y_pred = y_pred
        print(r, "full pred", num_received, "received")

    def get_senders_perc_normal(self):
        assert bool(self.senders)  # falso=dizionario vuoto
        res_inlier = {}
        # res_outlier = {}
        for sender, y_pred in self.senders.items():
            x = y_pred
            res_inlier[sender] = np.count_nonzero(x == 0) / len(x)  # percentuale inlier
            # res_outlier[sender] = np.count_nonzero(x == 1)/len(x) # percentuale outlier
        return res_inlier

    def get_devs_federated(self, range_perc=0.10):
        res_inlier = self.get_senders_perc_normal()
        # y_pred_local = self.model.predict(self.x)
        y_pred_local = self.y_pred

        local_inlier = np.count_nonzero(y_pred_local == 0) / len(y_pred_local)

        devs_federated = []
        for sender, y_pred in self.senders.items():
            x = y_pred
            sender_inlier = np.count_nonzero(x == 0) / len(x)
            # if in_range(sender_inlier, local_inlier, range_perc):
            # devs_federated.append(sender)

            # cancella da qui
            sender_local_pred = np.count_nonzero(sender.y_pred == 0) / len(
                x
            )  # y_pred locale del sender
            # il mio modello con i dati del sender
            a = np.count_nonzero(sender.senders[self] == 0) / len(x)

            if in_range(sender_inlier, local_inlier, range_perc) and in_range(
                a, sender_local_pred, range_perc
            ):
                devs_federated.append(sender)

        return devs_federated

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def in_range(x, ref, range_perc):
    return ref - (ref * range_perc) <= x <= ref + (ref * range_perc)


# num_nodes_per_class
# IDEALE multipli di (num_class -1), cosi non c'e resto della divisione
# situazione di prima, = 9


def partition(x, y, num_nodes_per_class=20, outlier_fraction=0.1, random_state=42):
    df = pd.DataFrame(x)
    df["y_class"] = y
    class_list = list(np.unique(y))
    num_class = len(class_list)
    
    m1 = df.sample(frac=1.0 - outlier_fraction, random_state=random_state)
    m2 = df.drop(m1.index)
    # m1['y_class'].value_counts().plot.bar()
    # m2['y_class'].value_counts().plot.bar()
    inlier = {}
    out = {}
    datasets = {}
    for c in range(0, num_class):
        # inlier[0] dizionario: 20 split della classe 0
        inlier[c] = np.array_split(
            m1.loc[m1["y_class"] == c], num_nodes_per_class)
        out[c] = np.array_split(
            m2.loc[m2["y_class"] == c], num_nodes_per_class)

    div = num_nodes_per_class // (num_class - 1)
    rem = num_nodes_per_class % (num_class - 1)
    print("div,rem", div, rem)

    for c_in in range(0, num_class):
        count = 0
        for _ in range(div):
            # print(m)
            for c_out in range(0, num_class):
                if c_in != c_out:
                    # print(c_in,c_out)

                    # split_in = inlier[c_in].pop(0)
                    # split_out = out[c_out].pop(0)

                    dd = pd.concat([inlier[c_in].pop(0), out[c_out].pop(0)])
                    dd["y_out"] = dd["y_class"].apply(lambda x: 0 if x == c_in else 1)
                    dd = dd.sample(frac=1, random_state=random_state) #shuffle
                    key = str(c_in) + "_" + str(count)
                    datasets[key] = dd
                    count += 1
        # for c_in in range(0,num_class):
        # i restati li prendi TUTTI dalla classe successiva (modular counter)
        c_out = (c_in + 1) % num_class
        for _ in range(rem):
            # print(c_in,c_out)
            # split_in = inlier[c_in].pop(0)
            # split_out = out[c_out].pop(0)

            dd = pd.concat([inlier[c_in].pop(0), out[c_out].pop(0)])
            dd["y_out"] = dd["y_class"].apply(lambda x: 0 if x == c_in else 1)
            dd = dd.sample(frac=1, random_state=random_state) #shuffle
            key = str(c_in) + "_" + str(count)
            datasets[key] = dd
            count += 1
    return datasets


def partition2(x, y, num_nodes_per_class=20, outlier_fraction=0.1,  random_state=42):
    df = pd.DataFrame(x)
    df["y_class"] = y
    class_list = list(np.unique(y))
    num_class = len(class_list)

    m1 = df.sample(frac = 1.0 - outlier_fraction, random_state=random_state)
    m2 = df.drop(m1.index)
    #m1['y_class'].value_counts().plot.bar()
    #m2['y_class'].value_counts().plot.bar()
    inlier = {}
    
    datasets = {}
    for c in range(0, num_class):
        # inlier[0] list: num_nodes_per_class (es 9) split della classe 0
        inlier[c] = np.array_split(m1.loc[m1["y_class"] == c], num_nodes_per_class)

    for c in range(0, num_class):
        for j in range(0, num_nodes_per_class):
            key = str(c) + "_" + str(j)
            #print(key)
            ii = inlier[c].pop(0)
            n_ii = len(ii)
            n_oo = floor(outlier_fraction * (n_ii / (1-outlier_fraction)))
            #print('leng',len(oo),len(ii))
            try:
                oo = m2.loc[m2["y_class"] != c].sample(n = n_oo, random_state=random_state)
            except ValueError:
                print("request, remaining",len(oo),len(m2.loc[m2["y_class"] != c]))
                oo = m2.loc[m2["y_class"] != c].sample(frac=1,random_state=random_state) # prendili tutti
            m2=m2.drop(oo.index)
            len(m2)
            dd = pd.concat([ii, oo])
            dd = dd.sample(frac=1, random_state=random_state) #shuffle
            #len(out[c])
            dd["y_out"] = dd["y_class"].apply(lambda x: 0 if x == c else 1)
            datasets[key] = dd
    return datasets