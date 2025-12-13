from datasets import load_dataset

def load_cnn_dailymail(config):
    ds = load_dataset(config['dataset']['name'],
                      config['dataset']['version'])
    
    train_data = ds[config['dataset']['split_train']]
    val_data = ds[config['dataset']['split_val']]
    test_data = ds[config['dataset']['split_test']]
    return train_data, val_data, test_data