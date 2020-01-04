#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab, dropout = 0.3, char_embed_size = 50, max_word_length = 21):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.char_embed_size = char_embed_size
        self.max_word_length = max_word_length

        self.char_embeddings = nn.Embedding(
            num_embeddings = len(vocab.char2id),
            embedding_dim = self.char_embed_size,
            padding_idx= vocab.char2id['<pad>']
        )

        self.conv_net = CNN(
            filters = self.embed_size,
            char_embed_size = self.char_embed_size,
            max_word_length = self.max_word_length
        )

        self.highway = Highway(embed_size= self.embed_size)

        self.dropout_layer = nn.Dropout(p=dropout)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_embeds = self.char_embeddings(input) #sentence_length X batch_size X max_word_length X char_embed_size
        sentence_length, batch_size, _, _ = char_embeds.size()
        x_reshaped = char_embeds.reshape((sentence_length * batch_size, self.char_embed_size, self.max_word_length))
        x_conv = self.conv_net(x_reshaped) #batch_size X embed_size
        x_high = self.highway(x_conv) #batch_size X embed_size
        output = self.dropout_layer(x_high) #batch_size X embed_size
        output = output.reshape((sentence_length, batch_size, self.embed_size))
        return output
        ### END YOUR CODE

