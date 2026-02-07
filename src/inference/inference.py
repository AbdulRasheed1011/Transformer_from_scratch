
import tensorflow as tf
import os
import yaml
import numpy as np
from src.data_ingestion.preprocessing import load_tokenizer
from src.model.transformer_model import Transformer, create_masks

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Inference:
    def __init__(self, config_path="config/config.yaml", checkpoint_dir=None):
        self.config = load_config(config_path)
        
        # Load Tokenizer
        tokenizer_path = self.config['paths']['tokenizer_file']
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Initialize Model
        self.transformer = Transformer(
            num_layers=self.config['model']['num_layers'],
            d_model=self.config['model']['d_model'],
            num_heads=self.config['model']['num_heads'],
            dff=self.config['model']['dff'],
            input_vocab_size=self.config['data']['vocab_size'] + 100,
            target_vocab_size=self.config['data']['vocab_size'] + 100,
            pe_input=self.config['data'].get('max_input_length', 128) + 100,
            pe_target=self.config['data'].get('max_output_length', 128) + 100,
            rate=self.config['model']['dropout_rate']
        )
        
        # Load Checkpoint
        if checkpoint_dir is None:
            checkpoint_dir = self.config['training']['checkpoint_path']
            
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=5)
        
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Restored checkpoint from {self.ckpt_manager.latest_checkpoint}")
        else:
            print("No checkpoint found. Initializing with random weights.")

    def evaluate(self, sentence):
        # Tokenize input
        # Convert to python string if needed
        if isinstance(sentence, bytes):
            sentence = sentence.decode('utf-8')
            
        start_token = self.tokenizer.token_to_id("[CLS]")
        end_token = self.tokenizer.token_to_id("[SEP]")
        
        # Use tokenizer.encode
        ids = self.tokenizer.encode(sentence).ids
        max_input = self.config['data'].get('max_input_length', 128)
        if len(ids) > max_input:
            ids = ids[:max_input]
        encoder_input = tf.expand_dims(ids, 0) # (1, seq_len)

        decoder_input = [start_token]
        output = tf.expand_dims(decoder_input, 0) # (1, 1)

        for i in range(self.config['data'].get('max_output_length', 128)):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(
                encoder_input, 
                output, 
                False, 
                enc_padding_mask, 
                combined_mask, 
                dec_padding_mask
            )

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(predicted_id, tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_token:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def predict(self, sentence):
        result, attention_weights = self.evaluate(sentence)
        
        result_ids = result.numpy().tolist()
        
        # Decode
        predicted_sentence = self.tokenizer.decode(result_ids)
        
        return predicted_sentence

if __name__ == "__main__":
    inference = Inference()
    
    # Test
    sample_article = "The transformer model is a deep learning architecture that relies on the self-attention mechanism, weighing the significance of each part of the input data. It is used primarily in the field of natural language processing (NLP)."
    print("\nArticle:", sample_article)
    summary = inference.predict(sample_article)
    print("\nGenerated Summary:", summary)
