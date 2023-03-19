import argparse
import os
import time
from pathlib import Path



import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np


from sorcery import dict_of

from auto import *
from datautil import *



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# cercalo! non occupa tutta la gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception(
            "Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"] # vedi la strategia su serverf.fit_config.

        # Train the model using hyperparameters from config
        history = self.model.fit(
                self.x_train, 
                self.x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=0.1,
                #validation_data=(x_test, x_test)
                verbose = 0,
                #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                )

        
        #history = self.model.fit(self.x_train,            self.y_train,            batch_size,            epochs,            validation_split=0.1,        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            #"accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            #"val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, = self.model.evaluate(self.x_test, self.x_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--num_clients", type=str, required=True) # cartella con numero client ideale, temporaneo
    parser.add_argument("--out_fold", type=str, required=True) # 
    args = parser.parse_args()

    # Load and compile Keras model # todo
    model = Autoencoder4(n_features=784) # hidden_neurons usa default in auto.py
    #model = Autoencoder3(n_features=784) # hidden_neurons usa default in auto.py
    model.build(input_shape=(None,784))
    model.compile(optimizer='adam', loss='mse')


    #model = tf.keras.applications.EfficientNetB0(input_shape=(32, 32, 3), weights=None, classes=10)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    x_train, x_test, y_train, y_test = load_partition(args.partition, args.data_name) # nel nostro caso Y_OUT. Le y non le usiamo in fed, le usiamo noi in ogni singolo thread"""

    # Start Flower client
    client = CifarClient(model, x_train, x_test)

    address = args.address
    fl.client.start_numpy_client(
        server_address="localhost:"+address,
        client=client,
    #    # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )
    
    
    # my stuff
    # model.save('saved_model/'+args.data_name+'/'+args.partition)
    
    #path = 'out'+args.num_clients+'_'+str(sum(model.hidden_neurons))+'/'+'output_'+args.out_fold+'/'+args.data_name+'/'
    path = 'out'+args.num_clients+'/'+'output_'+args.out_fold+'/'+args.data_name+'/'
    res_and_save(model,x_train,x_test,y_train, y_test, args.partition, path)

    # TODO 
    path = 'out'+args.num_clients+'/'+'output_local/'+args.data_name+'/'
    if not os.path.exists(path+args.partition+'.csv'):
        print('Start local training', args.partition, args.data_name)
        model_local = Autoencoder4(n_features=784) # hidden_neurons usa default in auto.py
        #model_local = Autoencoder3(n_features=784) # hidden_neurons usa default in auto.py
        model_local.build(input_shape=(None,784))
        model_local.compile(optimizer='adam', loss='mse') # TODO
        #model_local.compile(optimizer='adam', loss='binary_crossentropy')
        history = model_local.fit(
                    x_train, 
                    x_train,
                    epochs=40,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.1,
                    #validation_data=(x_test, x_test)
                    verbose =0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                    )

        # model_local.save('saved_model_local/'+args.data_name+'/'+args.partition)
        #path = 'out'+args.num_clients+'_'+str(sum(model.hidden_neurons))+ '/'+'output_local/'+args.data_name+'/'
        path = 'out'+args.num_clients+'/'+'output_local/'+args.data_name+'/'
        res_and_save(model_local,x_train,x_test,y_train, y_test, args.partition, path)




def res_and_save(model,x_train,x_test,y_train, y_test, partition, path):
    res = my_predict(model,x_train,x_test,y_train, y_test)
    res['dataset'] = partition
    df_res = pd.DataFrame.from_dict([res]).round(4)

    os.makedirs(path, exist_ok=True)
    df_res.to_csv(path+partition+'.csv')


def distancess(x,x_pred):
    euclidean_sq = np.square(x - x_pred)
    distances = np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
    return distances

def my_predict(model, x_train, x_test, y_train, y_test):
    x_train_pred = model.predict(x_train)
    distances = distancess(x_train, x_train_pred)
    contamination = 0.1  # TODO
    threshold = np.percentile(distances, 100 * (1 - contamination))

    x_test_pred = model.predict(x_test)
    
    #plots(x_test, x_test_pred, start = 10)


    distances = distancess(x_test, x_test_pred)
    labels = (distances > threshold).astype('int').ravel()

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

    y_true = y_test
    y_score = distances
    y_pred = labels
    aucroc = roc_auc_score(y_true, y_score)
    #aucpr = round(average_precision_score(y_true, y_score, pos_label=1)
    f1in = f1_score(y_true, y_pred, pos_label=0)
    f1out = f1_score(y_true, y_pred, pos_label=1)
    acc = accuracy_score(y_true, y_pred, normalize=True)
    return dict_of(aucroc, f1in, f1out, acc)




    



def load_partition(idx: str, data_name:str = 'mnist'):
    """Load 1/10th of the training and test data to simulate a partition."""
    print("LOAD PARTITION", idx, data_name)

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    if data_name=='mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif data_name=='fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif data_name=='kmnist':
        x_train = np.load('kmnist-train-imgs.npz')['arr_0']; x_test = np.load('kmnist-test-imgs.npz')['arr_0']; y_train = np.load('kmnist-train-labels.npz')['arr_0']; y_test = np.load('kmnist-test-labels.npz')['arr_0']
    else:
        raise ValueError(data_name,'not correct')

    n_features = np.prod(x_train.shape[1:])
 
    x_train = x_train.reshape(x_train.shape[0], n_features) / 255.0
    x_test = x_test.reshape(x_test.shape[0], n_features) / 255.0
    
    

    datasets = partition(x_train, y_train, num_nodes_per_class = 9)
    datasets_test = partition_old(x_test, y_test)

    
    x_train_local = datasets[idx].iloc[:,:n_features].to_numpy()
    y_train_out_local =  datasets[idx]['y_out'].to_numpy()

    maj_c = datasets[idx]['y_class'].value_counts().index[0] # majority class (value counts e' ordinato)
    min_c = datasets[idx]['y_class'].value_counts().index[-1] # minority class

    #maj_c = (int(maj_c) + 1) % 10
    #min_c = (int(min_c) + 1) % 10

    lab = str(maj_c) + '_' + str(min_c)
    #print(idx,'test set is',lab)
    #print(datasets_test[lab]['y_class'].value_counts())
    
    x_test_local = datasets_test[lab].iloc[:,:n_features].to_numpy()
    y_test_out_local =  datasets_test[lab]['y_out'].to_numpy()

    #return x_train_local, x_test_local
    return x_train_local, x_test_local, y_train_out_local, y_test_out_local



def partition_old(x,y, outlier_fraction=0.1,random_state=np.random.RandomState(42)):
    df = pd.DataFrame(x)
    df['y_class'] = y
    class_list = list(np.unique(y))
    print('Testset',)
    print(df['y_class'].value_counts())
    num_class = len(class_list)
    res = [(p1, p2) for p1 in class_list for p2 in class_list if p1 != p2] # tutte le possibili coppie escludendo duplicati
    
    datasets = {}  
    for i,j in res:
        dd = pd.concat([
            df.loc[df['y_class']==i].sample(frac=1.0-outlier_fraction, random_state=random_state),
            df.loc[df['y_class']==j].sample(frac=outlier_fraction, random_state=random_state)
        ])
        dd['y_out'] = dd['y_class'].apply(lambda x: 0 if x==i else 1)
        key = str(i)+'_'+str(j)
        datasets[key] = dd
    return datasets

def partition2(x,y):
    df = pd.DataFrame(x)
    df['y_class'] = y
    class_list = list(np.unique(y))
    num_class = len(class_list)
    res = [(p1, p2) for p1 in class_list for p2 in class_list if p1 != p2] # tutte le possibili coppie escludendo duplicati
    partitions = {}
    for i in range(0,num_class):
        partitions[i] = np.array_split(df.loc[df['y_class']==i], num_class*(num_class-1))  # 90 for mnist. TODO RIVEDI, RENDILO PARAMETRICO CON OUTLIER FRACTION
    datasets = {}
    for i,j in res:
        dd = pd.concat(
            partitions[i][:9]+
            partitions[j][0:1]
        )
        del partitions[i][:9]
        del partitions[j][0:1]
        dd['y_out'] = dd['y_class'].apply(lambda x: 0 if x==i else 1)
        key = str(i)+'_'+str(j)
        datasets[key] = dd
    
    for key in datasets.keys(): 
        datasets[key] = datasets[key].sample(frac=0.5, random_state=42)

    return datasets






if __name__ == "__main__":
    main()
