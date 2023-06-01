# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/10/24 14:30:00
@Author  :   Jian Zhang
@Contact :   zhangmingjian180@qq.com
HUST, Heilongjiang, China
'''

import torch
import torch.nn as nn

import models.layers as ml


class Transformer(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.conv1Dblock = nn.Sequential( 
                nn.Conv1d(1, 8, 3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(8, 8, 3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(8, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(32, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(12, stride=12)
        )

        self.add_positional_encoding = ml.AddPositionalEncoding(62, 32)
        self.encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block2 = ml.EncoderBlock(32, 32, 2) 
        self.encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.global_pool = ml.GlobalPool(32)
        self.decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block4 = ml.DecoderBlock(32, 32, 2)
         
        self.line = nn.Linear(32*4, classes)

        self.cel = nn.CrossEntropyLoss()
     
    def forward(self, x):
        x = x.reshape(-1, 1, 400)
        x = self.conv1Dblock(x)
        
        x = x.reshape(-1, 62, 32)

        x = self.add_positional_encoding(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        encoder_x = x
        x = self.global_pool(encoder_x)
        x1 = self.decoder_block1(x, encoder_x)
        x2 = self.decoder_block2(x, encoder_x) 
        x3 = self.decoder_block3(x, encoder_x)
        x4 = self.decoder_block4(x, encoder_x)

        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = x.squeeze(-2)
        x = self.line(x)

        predictions = torch.argmax(x, dim=1)
        
        return x, predictions

    def loss(self, output, label):
        cel_loss = self.cel(output, label)
        
        return cel_loss


class Transformer_DEAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1Dblock = nn.Sequential(
                
                nn.Conv1d(1, 8, 3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                
                #nn.Conv1d(8, 8, 3, padding=1),
                #nn.BatchNorm1d(8),
                #nn.ReLU(),
                #nn.MaxPool1d(2, stride=2),
                
                nn.Conv1d(8, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(32, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AvgPool1d(8, stride=8)
        )

        self.add_positional_encoding = ml.AddPositionalEncoding(32, 32)
        self.encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block2 = ml.EncoderBlock(32, 32, 2) 
        self.encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.global_pool = ml.GlobalPool(32)
        self.decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block4 = ml.DecoderBlock(32, 32, 2)
         
        self.line = nn.Linear(32*4, 1)

        self.loss_func = nn.BCELoss()
     
    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[2])
        x = self.conv1Dblock(x)
        
        x = x.reshape(-1, 32, 32)

        x = self.add_positional_encoding(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        encoder_x = x
        x = self.global_pool(encoder_x)
        x1 = self.decoder_block1(x, encoder_x)
        x2 = self.decoder_block2(x1, encoder_x) 
        x3 = self.decoder_block3(x2, encoder_x)
        x4 = self.decoder_block4(x3, encoder_x)

        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = x.squeeze(-2)
        x = self.line(x).squeeze(-1)
        x = torch.sigmoid(x)
       
        predictions = torch.trunc(x / 0.5)
        
        return x, predictions

    def loss(self, output, label):
        loss = self.loss_func(output, label)
        
        return loss


class Transformer_12_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1Dblock = nn.Sequential(
                nn.Conv1d(1, 8, 3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(8, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
        )

        self.add_positional_encoding_time = ml.AddPositionalEncoding(50, 32)
        self.time_encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block2 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.time_global_pool = ml.GlobalPool(32)
        self.time_decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block4 = ml.DecoderBlock(32, 32, 2)

        self.add_positional_encoding = ml.AddPositionalEncoding(62, 32)
        self.encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block2 = ml.EncoderBlock(32, 32, 2) 
        self.encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.global_pool = ml.GlobalPool(32)
        self.decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block4 = ml.DecoderBlock(32, 32, 2)
         
        self.line = nn.Linear(32, 3)

        self.cel = nn.CrossEntropyLoss()
     
    def forward(self, x):
        x = x.reshape(-1, 1, 400)
        x = self.conv1Dblock(x)
        x = x.reshape(-1, 62*32, 50).transpose(-1, -2).reshape(-1, 62, 32)

        x = self.add_positional_encoding(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        encoder_x = x
        x = self.global_pool(encoder_x)
        x = self.decoder_block1(x, encoder_x)
        x = self.decoder_block2(x, encoder_x) 
        x = self.decoder_block3(x, encoder_x)
        x = self.decoder_block4(x, encoder_x)

        x = x.reshape(-1, 50, 32)

        x = self.add_positional_encoding_time(x)
        x = self.time_encoder_block1(x)
        x = self.time_encoder_block2(x)
        x = self.time_encoder_block3(x)
        x = self.time_encoder_block4(x)
        encoder_x = x
        x = self.time_global_pool(encoder_x)
        x = self.time_decoder_block1(x, encoder_x)
        x = self.time_decoder_block2(x, encoder_x)
        x = self.time_decoder_block3(x, encoder_x)
        x = self.time_decoder_block4(x, encoder_x)

        x = x.squeeze(-2)
        x = self.line(x)

        predictions = torch.argmax(x, dim=1)
        
        return x, predictions

    def loss(self, output, label):
        cel_loss = self.cel(output, label)
        
        return cel_loss


class Transformer_binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1Dblock = nn.Sequential(
                nn.Conv1d(1, 8, 3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(8, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),

                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2)
        )

        self.add_positional_encoding_time = ml.AddPositionalEncoding(50, 32)
        self.time_encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block2 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.time_encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.time_global_pool = ml.GlobalPool(32)
        self.time_decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.time_decoder_block4 = ml.DecoderBlock(32, 32, 2)

        self.add_positional_encoding = ml.AddPositionalEncoding(62, 32)
        self.encoder_block1 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block2 = ml.EncoderBlock(32, 32, 2) 
        self.encoder_block3 = ml.EncoderBlock(32, 32, 2)
        self.encoder_block4 = ml.EncoderBlock(32, 32, 2)
        self.global_pool = ml.GlobalPool(32)
        self.decoder_block1 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block2 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block3 = ml.DecoderBlock(32, 32, 2)
        self.decoder_block4 = ml.DecoderBlock(32, 32, 2)
         
        self.line = nn.Linear(32, 1)

        self.loss_func = nn.BCELoss()
     
    def forward(self, x):
        x = x.reshape(-1, 1, 400)
        x = self.conv1Dblock(x)
        x = x.reshape(-1, 62*32, 50).transpose(-1, -2).reshape(-1, 62, 32)

        x = self.add_positional_encoding(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        encoder_x = x
        x = self.global_pool(encoder_x)
        x = self.decoder_block1(x, encoder_x)
        x = self.decoder_block2(x, encoder_x) 
        x = self.decoder_block3(x, encoder_x)
        x = self.decoder_block4(x, encoder_x)

        x = x.reshape(-1, 50, 32)

        x = self.add_positional_encoding_time(x)
        x = self.time_encoder_block1(x)
        x = self.time_encoder_block2(x)
        x = self.time_encoder_block3(x)
        x = self.time_encoder_block4(x)
        encoder_x = x
        x = self.time_global_pool(encoder_x)
        x = self.time_decoder_block1(x, encoder_x)
        x = self.time_decoder_block2(x, encoder_x)
        x = self.time_decoder_block3(x, encoder_x)
        x = self.time_decoder_block4(x, encoder_x)

        x = x.squeeze(-2)
        x = self.line(x).squeeze(-1)
        x = torch.sigmoid(x)
       
        predictions = torch.trunc(x / 0.5)

        return x, predictions

    def loss(self, output, label):
        loss_value = self.loss_func(output, torch.as_tensor(label, dtype=torch.float32))
        
        return loss_value











