### Download a pre-trained tokenizer (only the tokenizer, not the whole model)
from transformers import AutoTokenizer

def load_tokenizer(config):
    model_name = config['tokenizer']['model_name']

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast = True
    )
    return tokenizer
def tokenizer_batch(tokenizer, batch, config):
    max_inp = config['dataset']['max_input_len']
    max_tar = config['dataset']['max_target_len']


    model_inputs = tokenizer(
        batch['article'],
        max_length = max_inp,
        truncation = True,
        padding = 'max_length'
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch['highlights'],
            max_length = max_tar,
            truncation = True,
            padding = 'max_length'
        )

    model_inputs['labels'] = labels['input_ids']

    return model_inputs