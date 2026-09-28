import time
import os
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
import sys
import re
from pathlib import Path
#import torchvision.transforms as transforms
sys.path.append(str(Path(__file__).resolve().parents[1] / 'Chg2Cap'))
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.WHU_CDC.WHUCDC import WHUCDCDataset
from model.model_encoder import Encoder, SWSCEncoder
from model.model_decoder import DecoderTransformer
from utils import *

def get_segformer_dims(network):
    if 'segformer' not in network:
        raise ValueError('Only segformer backbones are supported, e.g. segformer-mit_b1.')
    backbone = network.split('-')[-1]
    if backbone == 'mit_b0':
        return [32, 64, 160, 256]
    return [64, 128, 320, 512]


def get_backbone_out_dim(network):
    return get_segformer_dims(network)[-1]


def resize_raw_frequency_input(shallow1, shallow2, wavelet_stage, raw_frequency_size):
    if wavelet_stage == 0 and raw_frequency_size is not None and raw_frequency_size > 0:
        size = (raw_frequency_size, raw_frequency_size)
        shallow1 = F.interpolate(shallow1, size=size, mode='bilinear', align_corners=False)
        shallow2 = F.interpolate(shallow2, size=size, mode='bilinear', align_corners=False)
    return shallow1, shallow2

def get_wavelet_in_dim(network, wavelet_stage):
    if wavelet_stage == 0:
        return 3
    dims = get_segformer_dims(network)
    if wavelet_stage in {1, 2, 3, 4}:
        return dims[wavelet_stage - 1]
    raise ValueError('wavelet_stage must be one of {0, 1, 2, 3, 4}; 0 means raw image.')


def evaluate_split(data_loader, encoder, encoder_trans, decoder, word_vocab, wavelet_stage, raw_frequency_size, split_name='Validation'):
    decoder.eval()
    encoder_trans.eval()
    encoder.eval()

    references = []
    hypotheses = []
    start_time = time.time()

    with torch.no_grad():
        for imgA, imgB, token_all, token_all_len, _, _, _ in data_loader:
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()

            feat1, feat2, shallow1, shallow2 = encoder(imgA, imgB)
            shallow1, shallow2 = resize_raw_frequency_input(shallow1, shallow2, wavelet_stage, raw_frequency_size)
            enc1, enc2 = encoder_trans(feat1, feat2, shallow1, shallow2)
            seq = decoder.sample(enc1, enc2, k=1)

            img_token = token_all.tolist()
            img_tokens = list(map(
                lambda c: [w for w in c if w not in {
                    word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']
                }],
                img_token,
            ))
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {
                word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']
            }]
            hypotheses.append(pred_seq)

    elapsed = time.time() - start_time
    score_dict = get_eval_score(references, hypotheses)
    average = (
        score_dict['Bleu_4']
        + score_dict['METEOR']
        + score_dict['ROUGE_L']
        + score_dict['CIDEr']
    ) / 4.0
    print(f'{split_name}:')
    print('Time: {0:.3f}\t'
          'BLEU-1: {1:.4f}\t'
          'BLEU-2: {2:.4f}\t'
          'BLEU-3: {3:.4f}\t'
          'BLEU-4: {4:.4f}\t'
          'Meteor: {5:.4f}\t'
          'Rouge: {6:.4f}\t'
          'Cider: {7:.4f}\t'
          'Average: {8:.4f}'.format(
              elapsed,
              score_dict['Bleu_1'],
              score_dict['Bleu_2'],
              score_dict['Bleu_3'],
              score_dict['Bleu_4'],
              score_dict['METEOR'],
              score_dict['ROUGE_L'],
              score_dict['CIDEr'],
              average,
          ))
    return score_dict


def main(args):
    """
    Training and validation.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    best_val_bleu4 = float('-inf')
    if args.data_name == 'LEVIR_CC':
        data_folder = 'dataset/LEVIR_CC/images'
        list_path = 'project/SWSC-Net/data/LEVIR_CC/'
        token_folder = 'project/SWSC-Net/data/LEVIR_CC/tokens/'
        max_length = 41
    elif args.data_name == 'WHU_CDC':
        data_folder = 'dataset/WHU_CDC/images'
        list_path = 'project/SWSC-Net/data/WHU_CDC/'
        token_folder = 'project/SWSC-Net/data/WHU_CDC/tokens/'
        max_length = 26
    else:
        raise ValueError(f'Unsupported data_name: {args.data_name}')
    with open(os.path.join(list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize models from scratch
    encoder = Encoder(args.network, wavelet_stage=args.wavelet_stage)
    encoder.fine_tune(args.fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                        lr=args.encoder_lr) if args.fine_tune_encoder else None
    backbone_out_dim = args.backbone_out_dim if args.backbone_out_dim > 0 else get_backbone_out_dim(args.network)
    encoder_trans = SWSCEncoder(
        in_dim=backbone_out_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        feat_size=args.feat_size,
        wavelet_in_dim=get_wavelet_in_dim(args.network, args.wavelet_stage),
        wavelet_bands=args.wavelet_bands,
    )
    encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()), lr=args.encoder_lr)
    decoder = DecoderTransformer(encoder_dim=args.hidden_dim, feature_dim=args.hidden_dim, vocab_size=len(word_vocab), max_lengths=max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                n_layers=args.decoder_n_layers, dropout=args.dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                        lr=args.decoder_lr)
    # Move to GPU, if available
    encoder = encoder.cuda()
    encoder_trans = encoder_trans.cuda()
    decoder = decoder.cuda()
    # Loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'train', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'val', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'WHU_CDC':
        train_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'train', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'val', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.5) if args.fine_tune_encoder else None
    encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_trans_optimizer, step_size=5, gamma=0.5)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs

    for epoch in range(args.num_epochs):        
        # Batches
        for id, (imgA, imgB, token_all, token_all_len, token, token_len, names) in enumerate(train_loader):
            #if id == 20:
            #    break
            start_time = time.time()
            decoder.train()  # train mode (dropout and batchnorm is used)
            encoder.train()
            encoder_trans.train()
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            token_all = token_all.cuda()

            imgA_a, imgB_a = imgA, imgB

            # Forward prop.
            feat1, feat2, shallow1, shallow2 = encoder(imgA_a, imgB_a)
            shallow1, shallow2 = resize_raw_frequency_input(shallow1, shallow2, args.wavelet_stage, args.raw_frequency_size)
            enc1_a, enc2_a = encoder_trans(feat1, feat2, shallow1, shallow2)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(
                enc1_a, enc2_a, token, token_len
            )
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # Calculate loss
            caption_loss = criterion(scores, targets)
            loss = caption_loss
            # Back prop.
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)

            # Update weights                      
            decoder_optimizer.step()
            encoder_trans_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            index_i += 1   
            # Print status
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs*len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])))
        # Evaluate validation data after each epoch.
        decoder.eval()
        encoder_trans.eval()
        if encoder is not None:
            encoder.eval()

        val_score_dict = evaluate_split(
            val_loader, encoder, encoder_trans, decoder, word_vocab, args.wavelet_stage, args.raw_frequency_size, split_name='Validation'
        )
        val_bleu4 = val_score_dict['Bleu_4']

        #Adjust learning rate
        decoder_lr_scheduler.step()
        print(decoder_optimizer.param_groups[0]['lr'])
        encoder_trans_lr_scheduler.step()
        if encoder_lr_scheduler is not None:
            encoder_lr_scheduler.step()
            print(encoder_optimizer.param_groups[0]['lr'])
        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            print('Save Model')
            state = {'epoch': epoch,
                    'val_bleu4': val_bleu4,
                    'encoder_dict': encoder.state_dict(), 
                    'SWSCEncoder': encoder_trans.state_dict(),
                    'decoder_dict': decoder.state_dict()
                    }
            model_name = (
                str(args.data_name) + '_batchsize_' + str(args.train_batchsize)
                + '_' + str(args.network)
                + '_epo_' + str(epoch)
                + '_Bleu4_' + str(round(10000 * val_bleu4)) + '.pth'
            )
            save_full_path = os.path.join(args.savepath, model_name)
            torch.save(state, save_full_path)


if __name__ == '__main__':
    band_choices = [
        'll', 'lh', 'hl', 'hh',
        'll_lh', 'll_hl', 'll_hh', 'lh_hl', 'lh_hh', 'hl_hh',
        'll_lh_hl', 'll_lh_hh', 'll_hl_hh', 'lh_hl_hh',
        'll_lh_hl_hh', 'all',
    ]
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default='LEVIR_CC', choices=['LEVIR_CC', 'WHU_CDC'], help='dataset name')

    parser.add_argument('--print_freq',type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=8, help='batch_size for training')
    parser.add_argument('--network', default='segformer-mit_b1', help='define the SegFormer encoder backbone, e.g. segformer-mit_b1')
    parser.add_argument('--backbone_out_dim', type=int, default=0, help='raw backbone output channels; 0 infers from --network')
    parser.add_argument('--feat_size', type=int, default=8, help='maximum backbone feature map size for SWSC position embeddings')
    parser.add_argument('--wavelet_stage', type=int, default=2, choices=[0, 1, 2, 3, 4], help='WSRM input source: 0 raw image, 1/2/3/4 SegFormer stage')
    parser.add_argument('--raw_frequency_size', type=int, default=128, help='resize raw image WSRM input when wavelet_stage=0; <=0 keeps original size')
    parser.add_argument('--wavelet_bands', default='lh_hl_hh', choices=band_choices, help='selected wavelet bands refined by WSRM; all means ll_lh_hl_hh')
    parser.add_argument('--num_epochs', type=int, default=11, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default='project/SWSC-Net/models_ckpt')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='number of transformer blocks in SWSC DCIB')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512, help='SWSC hidden/model dimension used by visual neck and decoder')
    args = parser.parse_args()
    main(args)
