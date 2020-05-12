import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import random
from .attention import Attention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

IMG_ROWS = 32
IMG_COLS = 1024

class EncoderRNN(nn.Module):
    def __init__(self, dim_word_emb, en_dict_size, ch_dict_size,
        dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='lstm'):
        """
        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
      
        self.en_dict_size = en_dict_size
        self.en_embedding = nn.Embedding(en_dict_size, dim_word_emb)
 
        self.ch_dict_size = ch_dict_size
        self.ch_embedding = nn.Embedding(ch_dict_size, dim_word_emb)

        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        # self.cap2hid = nn.Linear(WORD_EMB_DIM, dim_hidden)
        # self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            print("LSTM")
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(input_size=dim_word_emb, hidden_size=dim_hidden,
                                num_layers=n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

    #     self._init_hidden()

    # def _init_hidden(self):
    #     if type(self.rnn) == nn.GRU:
    #         for i in range(len(self.rnn.all_weights[0])):
    #             nn.init.xavier_normal_(self.rnn.all_weights[0][i])
    #     else:
    #         nn.init.xavier_normal_(self.rnn.weight)

    def forward(self, language, cap1hot, lengths):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        word_emb = None
        if language == "ch":
            word_emb = self.ch_embedding(cap1hot)
        else:
            word_emb = self.en_embedding(cap1hot)

        packed_input = pack_padded_sequence(word_emb, lengths, batch_first=True)

        # batch_size, vid_len, dim_cap = word_emb.size()
        # cap_feats = self.cap2hid(word_emb.view(-1, dim_cap))
        # cap_feats = self.input_dropout(cap_feats)
        # cap_feats = cap_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        packed_output, (hidden, ct) = self.rnn(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        '''
        if(isinstance(hidden, tuple)):
            hidden = hidden[0]
        '''
        return output, hidden

class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)
    """

    def __init__(self,
                 # max_len,
                 dim_word,
                 dim_hidden,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output = IMG_ROWS*IMG_COLS
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + IMG_COLS,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                opt={}):
        """
        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences
        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """

        # sample_max = opt.get('sample_max', 1)
        # beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        print("ENCODER OUTPUTS SIZE:", encoder_outputs.shape)
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        # seq_logprobs = []
        seq_preds = []
        out_tensor = None
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs

            for i in range(IMG_ROWS):
                current_state = targets[:, i, :]
                if isinstance(decoder_hidden, tuple):
                    context = self.attention(decoder_hidden[0].squeeze(0), encoder_outputs)
                else:
                    context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)

                decoder_input = torch.cat([current_state, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                temp = self.out(decoder_output.squeeze(1))
                seq_preds.append(self.out(decoder_output.squeeze(1)))

            out_tensor = torch.stack(seq_preds)

        elif mode == 'inference':
            if beam_size > 1: # maybe comment this out?
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.IMG_ROWS - 1):
                if isinstance(decoder_hidden, tuple):
                    context = self.attention(decoder_hidden[0].squeeze(0), encoder_outputs)
                else:
                    context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)

                decoder_input = torch.cat([encoder_outputs, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)

                seq_preds.append(self.out(decoder_output.squeeze(1)))

            out_tensor = torch.stack(seq_preds)

        return out_tensor

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, language, cap1hot, lengths, target_variable=None,
                mode='train', opt={}):
        """
        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels
        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden = self.encoder(language, cap1hot, lengths)

        seq_preds = self.decoder(encoder_outputs, encoder_hidden,
            target_variable, mode, opt)
        return seq_preds

