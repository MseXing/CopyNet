import tempfile
import allennlp
from allennlp_models.generation.dataset_readers import CopyNetDatasetReader
from allennlp_models.generation.models import CopyNetSeq2Seq
from allennlp.data import Vocabulary, DataLoader
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.modules.attention import DotProductAttention
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import RnnSeq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

import torch
import torch.nn as nn
from torch.autograd import Variable


reader = CopyNetDatasetReader(target_namespace="trg")
train_dataset = reader.read('data/train.tsv')
train_loader = PyTorchDataLoader(train_dataset, batch_size=8, shuffle=True)
vocab = Vocabulary.from_instances(train_dataset)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
TARGET_EMBEDDING_DIM = 512

token_embedding = Embedding(embedding_dim=EMBEDDING_DIM, num_embeddings=vocab.get_vocab_size(namespace="tokens"))
word_embedding = BasicTextFieldEmbedder({"token": token_embedding})

bi_rnn_encoder = RnnSeq2SeqEncoder(EMBEDDING_DIM, HIDDEN_DIM, 2, bidirectional=True)
dot_attn = DotProductAttention()
model = CopyNetSeq2Seq(vocab, word_embedding, bi_rnn_encoder, dot_attn,
                       target_namespace="trg", target_embedding_dim=TARGET_EMBEDDING_DIM)

with tempfile.TemporaryDirectory() as serialization_dir:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=None,
        num_epochs=5,
        optimizer=optimizer,
    )
    trainer.train()

