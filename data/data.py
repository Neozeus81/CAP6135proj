import os
from dotenv import load_dotenv
import pandas as pd 
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, max_error

# differe types of attacks and their labels
dos = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
r2l = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
u2r = ['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm']
probe = ['ipsweep','mscan','nmap','portsweep','saint','satan']

col_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level']


def map(attack):
    atype = 0
    if attack in dos:
        atype = 1
    if attack in r2l:
        atype = 2
    if attack in u2r:
        atype = 3
    if attack in probe:
        atype = 4
    return atype

def map_binary(attack):
    value = 0
    if attack in r2l or attack in dos or attack in u2r or attack in probe:
        value = 1
    return value

def normalize(col):
    return (1/(1+np.exp(-col)))

class Data: 
    def __init__(self, name, columns):
        # TODO normalize duration, src_bytes, dst_bytes
        load_dotenv("../.env")
        df = pd.read_csv(os.getenv(name))
        df.columns = col_names
        atype = df.attack.apply(map)
        btype = df.attack.apply(map_binary)
        df['duration'] = normalize(df['duration'])
        df['src_bytes'] = normalize(df['src_bytes'])
        df['dst_bytes'] = normalize(df['dst_bytes'])
        df['atype'] = atype
        df['btype'] = btype
        encode = pd.get_dummies(df, columns=columns)
        self.data = encode


    def get_train_data(self, columns):
        if not columns:
            return self.data.drop(columns=['attack', 'level', 'atype', 'btype'], axis=1)

        temp = self.data.drop(columns=columns, axis=1)
        return temp.drop(columns=['attack', 'level', 'atype', 'btype'], axis=1)

    def get_label_data(self, isBinary):
        if isBinary:
            return self.data['btype']
        else:
            return self.data['atype']

    def print_data(self):
        print(self.data)

    def get_Eval(pred, ground):
        return false

        


""" example usage
myData = data("KDD_TRAIN", ['protocol_type', 'service', 'flag'])
myData.print_data()
"""

