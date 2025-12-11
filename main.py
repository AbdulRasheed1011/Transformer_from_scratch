from src.utils.config import load_config
from src.data.dataset import load_cnn_dailymail
from src.data.tokenizer import load_tokenizer, tokenizer_batch

def main():
    config = load_config()
    train_ds, val_ds, test_ds = load_cnn_dailymail(config)
    
    tokenizer = load_tokenizer(config)
    print('Tokenizer loaded:', tokenizer)

    sample = train_ds[0]
    tokenized = tokenizer_batch(tokenizer, {
        'article':sample['article'],
        'highlights': sample['highlights']
    }, config)
    
    print( 'Input IDs length:', len(tokenized['input_ids']))
    print('Labels length:', len(tokenized['labels']))
if __name__ == '__main__':
    main()