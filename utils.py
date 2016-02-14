import h5py
from fuel.datasets import H5PYDataset

def parse_config(Cfile):
    
    f=open(Cfile)
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    values = [v.split()[1] for v in lines]
    
    n_hidden = values[0]
    f_dim = values[1]
    learn_rate = values[2]
    num_epochs = values[3]
    model_name = values[4]
    save_path = values[5]
    #data
    t_split = values[6]
    v_split = values[7]
    
    f.close()

    return n_hidden, f_dim, learn_rate, num_epochs, model_name, save_path

    
def load_data(t_split, v_split):
    
    train_set = H5PYDataset(t_split,  which_sets=('train',))
    valid_set = H5PYDataset(v_split, which_sets=('test',)) 
    
    return train_set, valid_set
