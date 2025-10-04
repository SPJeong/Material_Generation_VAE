##### model.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from typing import List, Tuple, Union, Any

##### Model build #####
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, latent_dim, padding_value):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = hidden_dim
        self.encoder_num_layers = num_layers
        self.encoder_dropout = dropout
        self.latent_dim = latent_dim
        self.padding_value = padding_value

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx= self.padding_value)

        self.gru = nn.GRU(self.embedding_dim,
                          self.encoder_hidden_dim,
                          self.encoder_num_layers,
                          batch_first= True,
                          dropout= self.encoder_dropout if self.encoder_num_layers > 1 else 0)
        self.fc = nn.Linear(self.encoder_hidden_dim, self.latent_dim * 2) # latent_dim * 2 for z_mu and z_logvar for each

    def forward(self,
                input_ids,  # (batch_size, seq_len)
                lengths,  # (batch_size,)
                **kwargs):
        x = self.embeddings(input_ids)  # x: (batch_size, seq_len, embedding_dim)
        x = rnn_utils.pack_padded_sequence(x,
                                           lengths.cpu(),
                                           batch_first= True,
                                           enforce_sorted= False)
        _, hiddens = self.gru(x, None)  # hiddens: (num_layers, batch_size, encoder_hidden_dim)

        hiddens = hiddens[-1]  # hiddens: (batch_size, encoder_hidden_dim) for last layer

        z_mu, z_logvar = torch.split(self.fc(hiddens), self.latent_dim, dim= -1)
        # z_mu, z_logvar: (batch_size, latent_dim)

        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, latent_dim, padding_value):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dim = hidden_dim
        self.decoder_num_layers = num_layers
        self.decoder_dropout = dropout
        self.latent_dim = latent_dim

        self.input_dim = self.embedding_dim + self.latent_dim
        self.output_dim = vocab_size
        self.padding_value = padding_value
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx= self.padding_value)

        self.gru = nn.GRU(self.input_dim,
                          self.decoder_hidden_dim,
                          self.decoder_num_layers,
                          batch_first= True,
                          dropout= self.decoder_dropout if self.decoder_num_layers > 1 else 0)
        self.z2hidden = nn.Linear(self.latent_dim, self.decoder_hidden_dim)
        self.fc = nn.Linear(self.decoder_hidden_dim, self.output_dim)

    def forward(self,
                input_ids,  # (batch_size, seq_len)
                lengths,  # (batch_size,)
                z: torch.Tensor,  # (batch_size, latent_dim)
                **kwargs):
        x = self.embeddings(input_ids)  # x: (batch_size, seq_len, embedding_dim)
        hiddens = self.z2hidden(z)  # hiddens: (batch_size, decoder_hidden_dim)
        hiddens = hiddens.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1) # hiddens: (num_layers, batch_size, decoder_hidden_dim)

        z_ = z.unsqueeze(1).repeat(1, x.shape[1], 1)  # z: (batch_size, seq_len, latent_dim)
        x = torch.cat((x, z_), dim=-1)  # x: (batch_size, seq_len, embedding_dim + latent_dim)

        x = rnn_utils.pack_padded_sequence(x,
                                           lengths.cpu(),
                                           batch_first= True,
                                           enforce_sorted= False)
        x, _ = self.gru(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first= True,)  # x: (batch_size, seq_len, hidden_dim)
        outputs = self.fc(x)  # outputs: (batch_size, seq_len, vocab_size)

        return outputs


class VAEModel(nn.Module):
    def __init__(self,
                 vocab_size= None,
                 embedding_dim= None,
                 hidden_dim= None,
                 num_layers= None,
                 dropout= None,
                 latent_dim= None,
                 device = None,
                 padding_value= None):
        super(VAEModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.device = device
        self.padding_value = padding_value

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx= self.padding_value)

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers, self.dropout, self.latent_dim, self.padding_value)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers, self.dropout, self.latent_dim, self.padding_value)


    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(mean)
        z = epsilon * torch.exp(logvar * .5) + mean  # mean, logvar, z: (batch_size, latent_dim)

        return z

    def forward(self,
                input_ids,  # (batch_size, seq_len)
                lengths,  # (batch_size,)
                **kwargs ,):
        z_mu, z_logvar = self.encoder(input_ids, lengths)
        z = self.reparameterize(z_mu, z_logvar)  # z: (batch_size, latent_dim)
        y = self.decoder(input_ids, lengths, z)  # y: (batch_size, seq_len, vocab_size)

        return y, z_mu, z_logvar


    def sample_gaussian_dist(self, batch_size: int):
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    @torch.no_grad()
    def generate(self,
                 tokenizer= None,
                 max_length= None,
                 num_return_sequences: int = None, # batch_size_for_generate
                 start_token_id: int = None,
                 start_token_id_for_generation: int = None, # which token will be initiated for generation [<<SOS>> or '*']
                 end_token_id: int = None,
                 **kwargs):

        z = kwargs.pop('z', None)
        z = z if z is not None else self.sample_gaussian_dist(num_return_sequences)
        assert z.shape == (num_return_sequences, self.latent_dim)  # z: [batch_size, latent_dim]
        z_ = z.unsqueeze(1)  # z_: [batch_size, 1, latent_dim]


        if start_token_id_for_generation != start_token_id:
            # input for wildcard token start (*)
            initial_inputs = torch.tensor([[start_token_id, start_token_id_for_generation]] * num_return_sequences,
                                          dtype=torch.long,
                                          device=self.device)
            generated_sequences = initial_inputs
            input_ids = initial_inputs[:, -1].unsqueeze(1) # making wildcard token (*) is input for RNN

        elif start_token_id_for_generation == start_token_id:
            # input for <<SOS>>
            initial_inputs = torch.full((num_return_sequences, 1),
                                        start_token_id_for_generation,
                                        dtype=torch.long,
                                        device=self.device)

            generated_sequences = initial_inputs
            input_ids = initial_inputs  # input_ids: [batch_size, 1]


        # z -> initial hiddens
        hiddens = self.decoder.z2hidden(z)  # hiddens: [batch_size, hidden_dim]
        hiddens = hiddens.unsqueeze(0).repeat(self.decoder.decoder_num_layers, 1, 1) # hiddens: [decoder_num_layers, batch_size, hidden_dim]
        is_finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=self.device)

        for _ in range(max_length):
            if is_finished.all():
                break

            x = self.decoder.embeddings(input_ids)  # x: [batch_size, 1, embedding_dim]
            x = torch.cat((x, z_), dim=-1)  # x: [batch_size, 1, embedding_dim + latent_dim]
            x, hiddens = self.decoder.gru(x, hiddens)  # x: [batch_size, 1, hidden_dim]
            logits = self.decoder.fc(x)  # logits: [batch_size, 1, vocab_size]
            next_token_logits = logits.squeeze(1)  # next_token_logits: [batch_size, vocab_size]

            probabilities = F.softmax(next_token_logits, dim= -1)  # probabilities: [batch_size, vocab_size]
            next_tokens = torch.multinomial(probabilities, num_samples= 1)
            # next_tokens: [batch_size, 1]

            # Update finished sequences
            is_finished = is_finished | (next_tokens.squeeze(1) == end_token_id)

            input_ids = next_tokens
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)

        # Decode the sequences and handle special tokens
        generated_smiles_list = []
        for sequence in generated_sequences:
            decoded_smiles = tokenizer.decode(sequence.tolist())
            generated_smiles_list.append(decoded_smiles)

        return generated_smiles_list

