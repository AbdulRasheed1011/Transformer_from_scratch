
import tensorflow as tf
import os
import time
import yaml
import mlflow
import numpy as np
from src.data_ingestion.get_data import get_data
from src.data_ingestion.preprocessing import create_tf_dataset, load_tokenizer, train_tokenizer
from src.model.transformer_model import Transformer, CustomSchedule, create_masks
from rouge_score import rouge_scorer

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Loss Object
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.d_model = config['model']['d_model']
        
        # Setup Data
        print("Loading data...")
        self.dataset = get_data()
        
        tokenizer_path = config['paths']['tokenizer_file']
        if os.path.exists(tokenizer_path):
            self.tokenizer = load_tokenizer(tokenizer_path)
        else:
            self.tokenizer = train_tokenizer(self.dataset, vocab_size=config['data']['vocab_size'], save_path=tokenizer_path)
            
        # Handle new length config
        self.max_input_length = config['data'].get('max_input_length', config['data'].get('max_length', 128))
        self.max_output_length = config['data'].get('max_output_length', config['data'].get('max_length', 128))

        self.train_ds, _ = create_tf_dataset(self.dataset, self.tokenizer, self.max_input_length, self.max_output_length, self.batch_size)
        
        # Setup Model
        if isinstance(config['training']['learning_rate'], (float, int)):
             learning_rate = float(config['training']['learning_rate'])
        else:
             learning_rate = CustomSchedule(self.d_model)
             
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        self.transformer = Transformer(
            num_layers=config['model']['num_layers'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            dff=config['model']['dff'],
            input_vocab_size=config['data']['vocab_size'] + 100, # Safety margin
            target_vocab_size=config['data']['vocab_size'] + 100,
            pe_input=self.max_input_length + 100,
            pe_target=self.max_output_length + 100,
            rate=config['model']['dropout_rate']
        )
        
        # Checkpointing
        self.checkpoint_path = config['training']['checkpoint_path']
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, 
                                         training=True, 
                                         enc_padding_mask=enc_padding_mask, 
                                         look_ahead_mask=combined_mask, 
                                         dec_padding_mask=dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))

    def train(self):
        mlflow.set_experiment("Transformer_Summarization")
        
        with mlflow.start_run():
            # Log params
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])
            
            for epoch in range(self.epochs):
                start = time.time()
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()

                # Iterate over batches
                for (batch, (inp, tar)) in enumerate(self.train_ds):
                    self.train_step(inp, tar) # tar contains both input and target for decoder?
                    # In preprocessing, we returned: (enc, dec), dec
                    # Wait, preprocessing returns (enc_input, dec_input), dec_input
                    # But the dataset yields `(enc, dec)`.
                    # Let's double check `preprocessing.py`.
                    # `train_ds = train_ds.map(tf_encode_map)` returns `enc, dec`.
                    # So train_ds yields `(enc, dec)`.
                    
                    # My `train_step` expects `inp, tar`.
                    # So `inp` is `enc`, `tar` is `dec`.
                    
                    if batch % 50 == 0:
                        print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
                        mlflow.log_metric("batch_loss", self.train_loss.result().numpy(), step=epoch * len(self.dataset['train']) // self.batch_size + batch)

                if (epoch + 1) % 1 == 0:
                    val_acc = self.evaluate_accuracy(self.dataset['validation'])
                    print(f'Validation Accuracy: {val_acc:.4f}')
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                # ROUGE Evaluation (less frequent to save time)
                if (epoch + 1) % 5 == 0: 
                    self.evaluate_model()

                print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
                
                mlflow.log_metric("epoch_loss", self.train_loss.result().numpy(), step=epoch)
                mlflow.log_metric("epoch_accuracy", self.train_accuracy.result().numpy(), step=epoch)

        # Final Test Evaluation
        print("\nRunning Final Test Evaluation...")
        test_acc = self.evaluate_accuracy(self.dataset['test'])
        print(f'Test Accuracy: {test_acc:.4f}')
        mlflow.log_metric("test_accuracy", test_acc)

    def evaluate_accuracy(self, dataset):
        accuracy_metric = tf.keras.metrics.Mean(name='accuracy')
        # Create dataset for evaluation
        eval_ds, _ = create_tf_dataset(dataset, self.tokenizer, self.max_input_length, self.max_output_length, self.batch_size)
        
        for (batch, (inp, tar)) in enumerate(eval_ds):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            
            predictions, _ = self.transformer(inp, tar_inp, 
                                            training=False, 
                                            enc_padding_mask=enc_padding_mask, 
                                            look_ahead_mask=combined_mask, 
                                            dec_padding_mask=dec_padding_mask)
            
            accuracy_metric(accuracy_function(tar_real, predictions))
            
        return accuracy_metric.result().numpy()

    def evaluate_model(self, num_samples=3):
        print("\nRunning Evaluation...")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # Take a subset of validation data
        # Note: self.dataset['validation'] is a HF dataset
        val_subset = self.dataset['validation'].select(range(num_samples))
        
        for i, example in enumerate(val_subset):
            article = example['article']
            reference = example['highlights']
            
            prediction = self.predict(article)
            
            scores = scorer.score(reference, prediction)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
            
            if i == 0:
                print(f"Sample Prediction:\nArticle: {article[:100]}...\nReference: {reference}\nPrediction: {prediction}\n")
        
        avg_scores = {k: np.mean(v) for k, v in rouge_scores.items()}
        print(f"Validation ROUGE Scores: {avg_scores}")
        mlflow.log_metrics(avg_scores)

    def predict(self, sentence):
        # Tokenize input
        start_token = self.tokenizer.token_to_id("[CLS]")
        end_token = self.tokenizer.token_to_id("[SEP]")
        
        ids = self.tokenizer.encode(sentence).ids
        ids = self.tokenizer.encode(sentence).ids
        if len(ids) > self.max_input_length:
            ids = ids[:self.max_input_length]
        
        encoder_input = tf.expand_dims(ids, 0)

        decoder_input = [start_token]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_output_length): 
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            predictions, attention_weights = self.transformer(
                encoder_input, 
                output, 
                training=False, 
                enc_padding_mask=enc_padding_mask, 
                look_ahead_mask=combined_mask, 
                dec_padding_mask=dec_padding_mask
            )

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(predicted_id, tf.int32)

            if predicted_id == end_token:
                break
                
            output = tf.concat([output, predicted_id], axis=-1)
            
        result_ids = tf.squeeze(output, axis=0).numpy().tolist()
        # Remove start token
        if result_ids[0] == start_token:
            result_ids = result_ids[1:]
            
        return self.tokenizer.decode(result_ids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with small dataset')
    args = parser.parse_args()

    config = load_config()
    trainer = Trainer(config)
    
    if args.debug:
        print("DEBUG MODE: Training on small subset (100 samples)...")
        # Taking a small subset for debugging
        # Note: We need to handle the fact that dataset might be a DatasetDict or Dataset
        # The trainer init already loaded it.
        # But trainer.dataset is the dict.
        trainer.dataset['train'] = trainer.dataset['train'].select(range(100))
        trainer.dataset['validation'] = trainer.dataset['validation'].select(range(20))
        # Re-create the tf dataset with the new small subset
        trainer.train_ds, _ = create_tf_dataset(trainer.dataset, trainer.tokenizer, trainer.max_input_length, trainer.max_output_length, trainer.batch_size)
    
    trainer.train()
