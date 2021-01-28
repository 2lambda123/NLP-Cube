import sys
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

sys.path.append('')
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.config import LemmatizerConfig
from cube3.networks.modules import LinearNorm, ConvNorm, Attention


class Lemmatizer(pl.LightningModule):
    encodings: Encodings
    config: LemmatizerConfig

    def __init__(self, config: LemmatizerConfig, encodings: Encodings, language_codes: [] = None):
        super(Lemmatizer, self).__init__()
        self._config = config
        self._encodings = encodings
        self._num_languages = encodings.num_langs
        self._language_codes = language_codes

        self._char_list = ['' for char in encodings.char2int]
        for char in encodings.char2int:
            self._char_list[encodings.char2int[char]] = char
        self._lang_emb = nn.Embedding(self._num_languages + 1, config.lang_emb_size, padding_idx=0)
        self._upos_emb = nn.Embedding(len(encodings.upos2int), config.upos_emb_size, padding_idx=0)
        self._char_emb = nn.Embedding(len(encodings.char2int) + 2, config.char_emb_size,
                                      padding_idx=0)  # start/stop index
        self._case_emb = nn.Embedding(4, 16, padding_idx=0)  # 0-pad 1-symbol 2-upper 3-lower
        convolutions = []
        cs_inp = config.char_emb_size + config.lang_emb_size + config.upos_emb_size + 16
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
            cs_inp = 512
        self._char_conv = nn.ModuleList(convolutions)
        encoder_layers = []
        for ii in range(config.encoder_layers):
            encoder_layers.append(nn.LSTM(cs_inp, config.encoder_size, 1, batch_first=True, bidirectional=True))
            cs_inp = config.encoder_size * 2 + config.lang_emb_size + config.upos_emb_size + 16

        self._encoder_layers = nn.ModuleList(encoder_layers)
        self._decoder = nn.LSTM(cs_inp + config.char_emb_size, config.decoder_size, config.decoder_layers,
                                batch_first=True, bidirectional=False)
        self._attention = Attention(cs_inp // 2, config.decoder_size, config.att_proj_size)

        self._output_char = LinearNorm(config.decoder_size, len(self._encodings.char2int) + 2)
        self._output_case = LinearNorm(config.decoder_size, 4)
        self._start_frame = nn.Embedding(1,
                                         config.encoder_size * 2 + config.char_emb_size + config.lang_emb_size + config.upos_emb_size + 16)

        self._res = {}
        for language_code in self._language_codes:
            self._res[language_code] = {"loss": 0., "acc": 0.}
        self._early_stop_meta_val = 0
        self._epoch_results = None

    def forward(self, X):
        x_char = X['x_char']
        x_case = X['x_case']
        x_lang = X['x_lang']
        x_upos = X['x_upos']
        if 'y_char' in X:
            gs_output = X['y_char']
        else:
            gs_output = None
        char_emb = self._char_emb(x_char)
        case_emb = self._case_emb(x_case)
        upos_emb = self._upos_emb(x_upos).unsqueeze(1).repeat(1, char_emb.shape[1], 1)
        lang_emb = self._lang_emb(x_lang).unsqueeze(1).repeat(1, char_emb.shape[1], 1)
        conditioning = torch.cat((case_emb, upos_emb, lang_emb), dim=-1)
        if gs_output is not None:
            output_idx = gs_output

        x = torch.cat((char_emb, conditioning), dim=-1)
        x = x.permute(0, 2, 1)
        for conv in self._char_conv:
            x = torch.dropout(torch.relu(conv(x)), 0.5, self.training)
        x = x.permute(0, 2, 1)

        output = x
        for ii in range(self._config.encoder_layers):
            output, _ = self._encoder_layers[ii](output)
            tmp = torch.cat((output, conditioning), dim=-1)
            output = tmp

        encoder_output = output

        step = 0
        done = np.zeros(encoder_output.shape[0])
        start_frame = self._start_frame(
            torch.tensor([0], dtype=torch.long, device=self._get_device())).unsqueeze(1).repeat(encoder_output.shape[0],
                                                                                                1, 1)
        decoder_output, decoder_hidden = self._decoder(start_frame)

        out_char_list = []
        out_case_list = []
        while True:
            if gs_output is not None:
                if step == output_idx.shape[1]:
                    break
            elif np.sum(done) == encoder_output.shape[0]:
                break
            elif step == encoder_output.shape[1] * 20:  # failsafe
                break

            att = self._attention(decoder_hidden[-1][-1, :, :], encoder_output)
            context = torch.bmm(att.unsqueeze(1), encoder_output)

            if step == 0:
                prev_char_emb = torch.zeros((encoder_output.shape[0], 1, self._config.char_emb_size),
                                            device=self._get_device())

            decoder_input = torch.cat((context, prev_char_emb), dim=-1)
            decoder_output, decoder_hidden = self._decoder(decoder_input,
                                                           hx=(torch.dropout(decoder_hidden[0], 0.5, self.training),
                                                               torch.dropout(decoder_hidden[1], 0.5, self.training)))

            output_char = self._output_char(decoder_output)
            output_case = self._output_case(decoder_output)
            out_char_list.append(output_char)
            out_case_list.append(output_case)
            selected_chars = torch.argmax(output_char, dim=-1)
            for ii in range(selected_chars.shape[0]):
                if selected_chars[ii].squeeze() == 0:
                    done[ii] = 1
            if gs_output is not None:
                prev_char_emb = self._char_emb(output_idx[:, step]).unsqueeze(1)
            else:
                prev_char_emb = self._char_emb(selected_chars)
            step += 1

        return torch.cat(out_char_list, dim=1), torch.cat(out_case_list, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self._target_device))

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))

    def process(self, sequences, lang_id):
        self.eval()
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx):
        y_char_pred, y_case_pred = self.forward(batch)
        y_char_target, y_case_target = batch['y_char'], batch['y_case']
        loss_char = F.cross_entropy(y_char_pred.view(-1, y_char_pred.shape[2]), y_char_target.view(-1), ignore_index=0)
        loss_case = F.cross_entropy(y_case_pred.view(-1, y_case_pred.shape[2]), y_case_target.view(-1), ignore_index=0)
        return loss_char + loss_case

    def validation_step(self, batch, batch_idx):
        y_char_pred, y_case_pred = self.forward(batch)
        y_char_target, y_case_target = batch['y_char'], batch['y_case']
        loss_char = F.cross_entropy(y_char_pred.view(-1, y_char_pred.shape[2]), y_char_target.view(-1), ignore_index=0)
        loss_case = F.cross_entropy(y_case_pred.view(-1, y_case_pred.shape[2]), y_case_target.view(-1), ignore_index=0)
        loss = loss_char + loss_case
        ok = 0

        language_result = {lang_id: {'total': 0, 'ok': 0}
                           for lang_id in range(self._num_languages)}

        y_char_target = y_char_target.detach().cpu().numpy()
        y_char_pred = torch.argmax(y_char_pred.detach(), dim=-1).cpu().numpy()
        lang = batch['x_lang'].detach().cpu().numpy()
        for lang_id, y_pred, y_target in zip(lang, y_char_pred, y_char_target):
            valid = True
            for y_p, y_t in zip(y_pred, y_target):
                if y_t != 0 and y_p != y_t:
                    valid = False
                    break
            if valid:
                language_result[lang_id - 1]['ok'] += 1
            language_result[lang_id - 1]['total'] += 1

        return {'loss': loss, 'acc': language_result}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        language_result = {lang_id: {'total': 0, 'ok': 0}
                           for lang_id in
                           range(self._num_languages)}
        loss = 0
        for result in outputs:
            loss += result['loss'].item()
            for lang_id in result['acc']:
                language_result[lang_id]['ok'] += result['acc'][lang_id]['ok']
                language_result[lang_id]['total'] += result['acc'][lang_id]['total']
        loss = loss / len(outputs)

        res = {}
        for lang_index in language_result:
            total = language_result[lang_index]['total']
            if total == 0:
                total = 1
            if self._language_codes is None:
                lang = lang_index
            else:
                lang = self._language_codes[lang_index]
            res[lang] = {
                "acc": language_result[lang_index]['ok'] / total,
            }

            self.log('val/ACC/{0}'.format(lang), language_result[lang_index]['ok'] / total)
        self.log('val/LOSS'.format(lang), loss)

        # single value for early stopping
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["acc"] > self._res[lang]["acc"]:
                self._early_stop_meta_val += 1
                self._res[lang]["acc"] = res[lang]["acc"]
                res[lang]["acc_best"] = True
        return res

    class PrintAndSaveCallback(pl.callbacks.Callback):
        def __init__(self, store_prefix):
            super().__init__()
            self.store_prefix = store_prefix

        def on_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch

            for lang in pl_module._epoch_results:
                res = pl_module._epoch_results[lang]
                if "acc_best" in res:
                    trainer.save_checkpoint(self.store_prefix + "." + lang + ".best")

            trainer.save_checkpoint(self.store_prefix + ".last")

            s = "{0:30s}\tACC".format("Language")
            print("\n\n\t" + s)
            print("\t" + ("=" * (len(s) + 16)))
            for lang in pl_module._language_codes:
                acc = metrics["val/ACC/{0}".format(lang)]
                msg = "\t{0:30s}:\t{1:.4f}".format(lang, acc)
                print(msg)
            print("\n")