import h5py
from fuel.datasets import H5PYDataset

def parse_config(Cfile):
    
    f=open(Cfile)
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    values = [v.split()[1] for v in lines]
    
    config={}

    config["n_hidden"] = values[0]
    config["f_dim"] = values[1]
    config["learn_rate"] = values[2]
    config["num_epochs"] = values[3]
    config["n_speakers"] = values[4]
    config["model_name"] = values[5]
    config["save_path"] = values[6]
    #data
    config["train_file"] = values[7]
    config["v_split"] = values[8]
    
    f.close()

    return config

    
def load_data(t_split, v_split):
    
    train_set = H5PYDataset(t_split,  which_sets=('train',), subset=slice(0,1000))
    valid_set = H5PYDataset(v_split, which_sets=('test',), subset = slice(0,200)) 
    
    return train_set, valid_set