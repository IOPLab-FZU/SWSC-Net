import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils import data

sys.path.append(str(Path(__file__).resolve().parents[1] / 'Chg2Cap'))
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.WHU_CDC.WHUCDC import WHUCDCDataset
from model.model_decoder import DecoderTransformer
from model.model_encoder import Encoder, SWSCEncoder
from utils import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resize_raw_frequency_input(shallow1, shallow2, wavelet_stage, raw_frequency_size):
    if wavelet_stage == 0 and raw_frequency_size is not None and raw_frequency_size > 0:
        size = (raw_frequency_size, raw_frequency_size)
        shallow1 = F.interpolate(shallow1, size=size, mode='bilinear', align_corners=False)
        shallow2 = F.interpolate(shallow2, size=size, mode='bilinear', align_corners=False)
    return shallow1, shallow2

def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if os.path.exists(args.savepath) is False:
        os.makedirs(args.savepath)
    if args.data_name == 'LEVIR_CC':
        data_folder = 'dataset/LEVIR_CC/images'
        list_path = 'project/SWSC-Net/data/LEVIR_CC/'
        token_folder = 'project/SWSC-Net/data/LEVIR_CC/tokens/'
        max_length = 41
        data_name = 'LEVIR_CC'
    elif args.data_name == 'WHU_CDC':
        data_folder = 'dataset/WHU_CDC/images'
        list_path = 'project/SWSC-Net/data/WHU_CDC/'
        token_folder = 'project/SWSC-Net/data/WHU_CDC/tokens/'
        max_length = 26
        data_name = 'WHU_CDC'
    with open(os.path.join(list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    snapshot_full_path = args.checkpoint
    if not os.path.isabs(snapshot_full_path):
        snapshot_full_path = os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(snapshot_full_path, map_location='cpu')

    if 'segformer' not in args.network:
        raise ValueError('Only segformer backbones are supported, e.g. segformer-mit_b1.')
    backbone = args.network.split('-')[-1]
    segformer_dims = [32, 64, 160, 256] if backbone == 'mit_b0' else [64, 128, 320, 512]
    backbone_out_dim = args.backbone_out_dim if args.backbone_out_dim > 0 else segformer_dims[-1]
    if args.wavelet_stage == 0:
        wavelet_in_dim = 3
    elif args.wavelet_stage in {1, 2, 3, 4}:
        wavelet_in_dim = segformer_dims[args.wavelet_stage - 1]
    else:
        raise ValueError('wavelet_stage must be one of {0, 1, 2, 3, 4}; 0 means raw image.')

    encoder = Encoder(args.network, wavelet_stage=args.wavelet_stage)
    encoder_trans = SWSCEncoder(
        in_dim=backbone_out_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        feat_size=args.feat_size,
        wavelet_in_dim=wavelet_in_dim,
        wavelet_bands=args.wavelet_bands,
    )
    decoder = DecoderTransformer(
        encoder_dim=args.hidden_dim,
        feature_dim=args.hidden_dim,
        vocab_size=len(word_vocab),
        max_lengths=max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout,
    )

    encoder.load_state_dict(checkpoint['encoder_dict'], strict=False)
    encoder_trans.load_state_dict(checkpoint['SWSCEncoder'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_dict'], strict=False)

    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    if args.data_name == 'LEVIR_CC':
        nochange_list = [
            'the scene is the same as before ',
            'there is no difference ',
            'the two scenes seem identical ',
            'no change has occurred ',
            'almost nothing has changed ',
        ]
        data_folder = 'dataset/LEVIR_CC/images'
        list_path = 'project/SWSC-Net/data/LEVIR_CC/'
        token_folder = 'project/SWSC-Net/data/LEVIR_CC/tokens/'
        test_loader = data.DataLoader(
            LEVIRCCDataset(data_folder, list_path, 'test', token_folder, args.vocab_file, max_length, args.allow_unk),            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    elif args.data_name == 'WHU_CDC':
        nochange_list = [
            'the scene is the same as before ',
            'there is no difference ',
            'the two scenes seem identical ',
            'no change has occurred ',
            'almost nothing has changed ',
        ]
        data_folder = 'dataset/WHU_CDC/images'
        list_path = 'project/SWSC-Net/data/WHU_CDC/'
        token_folder = 'project/SWSC-Net/data/WHU_CDC/tokens/'
        test_loader = data.DataLoader(
            WHUCDCDataset(data_folder, list_path, 'test', token_folder, args.vocab_file, max_length, args.allow_unk),
            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        raise ValueError(f'Unsupported data_name: {args.data_name}')

    test_start_time = time.time()
    references = []
    hypotheses = []
    change_references = []
    change_hypotheses = []
    nochange_references = []
    nochange_hypotheses = []
    change_acc = 0
    nochange_acc = 0

    with torch.no_grad():
        for _, (imgA, imgB, token_all, token_all_len, _, _, names) in enumerate(test_loader):
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()

            feat1, feat2, shallow1, shallow2 = encoder(imgA, imgB)
            shallow1, shallow2 = resize_raw_frequency_input(shallow1, shallow2, args.wavelet_stage, args.raw_frequency_size)
            enc1, enc2 = encoder_trans(feat1, feat2, shallow1, shallow2)
            seq = decoder.sample(enc1, enc2)

            img_token = token_all.tolist()
            img_tokens = list(
                map(
                    lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                    img_token,
                )
            )
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)

            pred_caption = ''
            for i in pred_seq:
                pred_caption += list(word_vocab.keys())[i] + ' '
            ref_caption = ''
            for i in img_tokens[0]:
                ref_caption += list(word_vocab.keys())[i] + ' '

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc += 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc += 1

        test_time = time.time() - test_start_time
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print(
                'BLEU-1: {0:.4f}	BLEU-2: {1:.4f}	BLEU-3: {2:.4f}	BLEU-4: {3:.4f}	Meteor: {4:.4f}	Rouge: {5:.4f}	Cider: {6:.4f}	'.format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print('nochange_acc:', nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print(
                'BLEU-1: {0:.4f}	BLEU-2: {1:.4f}	BLEU-3: {2:.4f}	BLEU-4: {3:.4f}	Meteor: {4:.4f}	Rouge: {5:.4f}	Cider: {6:.4f}	'.format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print('change_acc:', change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        Average = (Bleu_4 + Meteor + Rouge + Cider) / 4.0
        print(
            'Testing:\nTime: {0:.3f}	BLEU-1: {1:.4f}	BLEU-2: {2:.4f}	BLEU-3: {3:.4f}	BLEU-4: {4:.4f} Meteor: {5:.4f}	Rouge: {6:.4f}	Cider: {7:.4f}	Average: {8:.4f} '.format(
                test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider, Average
            )
        )


if __name__ == '__main__':
    band_choices = [
        'll', 'lh', 'hl', 'hh',
        'll_lh', 'll_hl', 'll_hh', 'lh_hl', 'lh_hh', 'hl_hh',
        'll_lh_hl', 'll_lh_hh', 'll_hl_hh', 'lh_hl_hh',
        'll_lh_hl_hh', 'all',
    ]
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='path of the data lists')
    parser.add_argument('--data_name', default='LEVIR_CC', choices=['LEVIR_CC', 'WHU_CDC'], help='dataset name')
    parser.add_argument('--checkpoint', default='LEVIR_CC_batchsize_32_segformer.pth', help='path to checkpoint')

    parser.add_argument('--network', default='segformer-mit_b1', help='define the SegFormer encoder backbone, e.g. segformer-mit_b1')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading')
    parser.add_argument('--backbone_out_dim', type=int, default=0, help='raw backbone output channels; 0 infers from --network')
    parser.add_argument('--feat_size', type=int, default=8, help='position embedding feature map size for SWSC visual neck')
    parser.add_argument('--wavelet_stage', type=int, default=4, choices=[0, 1, 2, 3, 4], help='WSRM input source: 0 raw image, 1/2/3/4 SegFormer stage')
    parser.add_argument('--raw_frequency_size', type=int, default=128, help='resize raw image WSRM input when wavelet_stage=0; <=0 keeps original size')
    parser.add_argument('--wavelet_bands', default='lh_hl', choices=band_choices, help='selected wavelet bands refined by WSRM; all means ll_lh_hl_hh')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=1, help='number of transformer blocks in SWSC DCIB')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512, help='SWSC hidden/model dimension used by visual neck and decoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--test_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default='project/SWSC-Net/models_ckpt')

    args = parser.parse_args()
    main(args)
