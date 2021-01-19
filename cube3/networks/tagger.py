import sys

sys.path.append('')
import os, yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from cube3.io_utils.objects import Document, Sentence, Token, Word
from cube3.io_utils.encodings import Encodings
from cube3.io_utils.config import TaggerConfig
import numpy as np
from cube3.networks.modules import ConvNorm, LinearNorm
import random

from cube3.networks.utils import LMHelper, MorphoCollate, MorphoDataset

from cube3.networks.modules import WordGram


class Tagger(pl.LightningModule):
    def __init__(self, config: TaggerConfig, encodings: Encodings, language2id: [] = None):
        super().__init__()
        self._config = config
        self._encodings = encodings

        self._word_net = WordGram(len(encodings.char2int), num_langs=encodings.num_langs + 1,
                                  num_filters=config.char_filter_size, char_emb_size=config.char_emb_size,
                                  lang_emb_size=config.lang_emb_size, num_layers=config.char_layers)
        self._zero_emb = nn.Embedding(1, config.char_filter_size // 2)
        self._num_langs = encodings.num_langs
        self._language2id = language2id

        conv_layers = []
        cs_inp = config.char_filter_size // 2 + config.lang_emb_size + config.word_emb_size + 768
        NUM_FILTERS = config.cnn_filter
        for _ in range(config.cnn_layers):
            conv_layer = nn.Sequential(
                ConvNorm(cs_inp,
                         NUM_FILTERS,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(NUM_FILTERS))
            conv_layers.append(conv_layer)
            cs_inp = NUM_FILTERS // 2 + config.lang_emb_size
        self._word_emb = nn.Embedding(len(encodings.word2int), config.word_emb_size, padding_idx=0)
        self._lang_emb = nn.Embedding(encodings.num_langs + 1, config.lang_emb_size, padding_idx=0)
        self._convs = nn.ModuleList(conv_layers)
        self._upos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._xpos = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._attrs = LinearNorm(NUM_FILTERS // 2 + config.lang_emb_size, len(encodings.attrs2int))

        self._aupos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.upos2int))
        self._axpos = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.xpos2int))
        self._aattrs = LinearNorm(config.char_filter_size // 2 + config.lang_emb_size, len(encodings.attrs2int))

        self._res = {}
        for language_code in self._language2id:
            self._res[language_code] = {"upos": 0., "xpos": 0., "attrs": 0.}
        self._early_stop_meta_val = 0

    def _compute_early_stop(self, res):
        for lang in res:
            if res[lang]["upos"] > self._res[lang]["upos"]:
                self._early_stop_meta_val += 1
                self._res[lang]["upos"] = res[lang]["upos"]
                res[lang]["upos_best"] = True
            if res[lang]["xpos"] > self._res[lang]["xpos"]:
                self._early_stop_meta_val += 1
                self._res[lang]["xpos"] = res[lang]["xpos"]
                res[lang]["xpos_best"] = True
            if res[lang]["attrs"] > self._res[lang]["attrs"]:
                self._early_stop_meta_val += 1
                self._res[lang]["attrs"] = res[lang]["attrs"]
                res[lang]["attrs_best"] = True
        return res

    def forward(self, X):
        x_sents = X['x_sent']
        x_lang_sent = X['x_lang_sent']
        x_words_chars = X['x_word']
        x_words_case = X['x_word_case']
        x_lang_word = X['x_lang_word']
        x_sent_len = X['x_sent_len']
        x_word_len = X['x_word_len']
        x_sent_masks = X['x_sent_masks']
        x_word_masks = X['x_word_masks']
        x_word_emb_packed = X['x_word_embeddings']
        char_emb_packed = self._word_net(x_words_chars, x_words_case, x_lang_word, x_word_masks, x_word_len)

        blist_char = []
        blist_emb = []
        sl = x_sent_len.cpu().numpy()
        pos = 0
        for ii in range(x_sents.shape[0]):
            slist_char = []
            slist_emb = []
            for jj in range(sl[ii]):
                slist_char.append(char_emb_packed[pos, :].unsqueeze(0))
                slist_emb.append(x_word_emb_packed[pos, :].unsqueeze(0))
                pos += 1
            for jj in range(x_sents.shape[1] - sl[ii]):
                slist_char.append(torch.zeros((1, self._config.char_filter_size // 2),
                                              device=self._get_device(), dtype=torch.float))
                slist_emb.append(torch.zeros((1, 768),
                                             device=self._get_device(), dtype=torch.float))
            sent_emb = torch.cat(slist_char, dim=0)
            word_emb = torch.cat(slist_emb, dim=0)
            blist_char.append(sent_emb.unsqueeze(0))
            blist_emb.append(word_emb.unsqueeze(0))

        char_emb = torch.cat(blist_char, dim=0)
        word_emb_ext = torch.cat(blist_emb, dim=0)
        lang_emb = self._lang_emb(x_lang_sent)
        lang_emb = lang_emb.unsqueeze(1).repeat(1, char_emb.shape[1], 1)

        aupos = self._aupos(torch.cat([char_emb, lang_emb], dim=-1))
        axpos = self._axpos(torch.cat([char_emb, lang_emb], dim=-1))
        aattrs = self._aattrs(torch.cat([char_emb, lang_emb], dim=-1))

        word_emb = self._word_emb(x_sents)

        if self.training:
            mask_1 = np.ones((char_emb.shape[0], char_emb.shape[1]), dtype=np.long)
            mask_2 = np.ones((word_emb.shape[0], word_emb.shape[1]), dtype=np.long)
            mask_3 = np.ones((word_emb.shape[0], word_emb.shape[1]), dtype=np.long)

            for ii in range(mask_1.shape[0]):
                for jj in range(mask_1.shape[1]):
                    mult = 1
                    p = random.random()
                    if p < 0.33:
                        mult += 1
                        mask_1[ii, jj] = 0
                    p = random.random()
                    if p < 0.33:
                        mult += 1
                        mask_2[ii, jj] = 0
                    p = random.random()
                    if p < 0.33:
                        mult += 1
                        mask_3[ii, jj] = 0
                    mask_1[ii, jj] *= mult
                    mask_2[ii, jj] *= mult
                    mask_3[ii, jj] *= mult
            mask_1 = torch.tensor(mask_1, device=self._get_device())
            mask_2 = torch.tensor(mask_2, device=self._get_device())
            mask_3 = torch.tensor(mask_3, device=self._get_device())
            word_emb = word_emb * mask_1.unsqueeze(2)
            char_emb = char_emb * mask_2.unsqueeze(2)
            word_emb_ext = word_emb_ext * mask_3.unsqueeze(2)

        x = torch.cat([word_emb, lang_emb, char_emb, word_emb_ext], dim=-1)
        x = x.permute(0, 2, 1)
        lang_emb = lang_emb.permute(0, 2, 1)
        half = self._config.cnn_filter // 2
        res = None
        cnt = 0
        for conv in self._convs:
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid((conv_out[:, half:, :]))
            if res is None:
                res = tmp
            else:
                res = res + tmp
            x = torch.dropout(tmp, 0.2, self.training)
            cnt += 1
            if cnt != self._config.cnn_layers:
                x = torch.cat([x, lang_emb], dim=1)
        x = x + res
        x = torch.cat([x, lang_emb], dim=1)
        x = x.permute(0, 2, 1)
        x = torch.tanh(x)
        return self._upos(x), self._xpos(x), self._attrs(x), aupos, axpos, aattrs

    def _get_device(self):
        if self._lang_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._lang_emb.weight.device.type, str(self._lang_emb.weight.device.index))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)
        y_upos = batch['y_upos']
        y_xpos = batch['y_xpos']
        y_attrs = batch['y_attrs']

        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.view(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        loss_aupos = F.cross_entropy(a_upos.view(-1, a_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_axpos = F.cross_entropy(a_xpos.view(-1, a_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_aattrs = F.cross_entropy(a_attrs.view(-1, a_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)

        step_loss = ((loss_upos + loss_attrs + loss_xpos) / 3.) * 1.0 + (
                (loss_aupos + loss_aattrs + loss_axpos) / 3.) * 1.0

        return {'loss': step_loss}

    def validation_step(self, batch, batch_idx):
        p_upos, p_xpos, p_attrs, a_upos, a_xpos, a_attrs = self.forward(batch)
        y_upos = batch['y_upos']
        y_xpos = batch['y_xpos']
        y_attrs = batch['y_attrs']
        x_sent_len = batch['x_sent_len']
        x_lang = batch['x_lang_sent']
        loss_upos = F.cross_entropy(p_upos.view(-1, p_upos.shape[2]), y_upos.view(-1), ignore_index=0)
        loss_xpos = F.cross_entropy(p_xpos.view(-1, p_xpos.shape[2]), y_xpos.view(-1), ignore_index=0)
        loss_attrs = F.cross_entropy(p_attrs.view(-1, p_attrs.shape[2]), y_attrs.view(-1), ignore_index=0)
        loss = (loss_upos + loss_attrs + loss_xpos) / 3
        language_result = {lang_id: {'total': 0, 'upos_ok': 0, 'xpos_ok': 0, 'attrs_ok': 0} for lang_id in
                           range(self._num_langs)}

        pred_upos = torch.argmax(p_upos, dim=-1).detach().cpu().numpy()
        pred_xpos = torch.argmax(p_xpos, dim=-1).detach().cpu().numpy()
        pred_attrs = torch.argmax(p_attrs, dim=-1).detach().cpu().numpy()
        tar_upos = y_upos.detach().cpu().numpy()
        tar_xpos = y_xpos.detach().cpu().numpy()
        tar_attrs = y_attrs.detach().cpu().numpy()
        sl = x_sent_len.detach().cpu().numpy()
        x_lang = x_lang.detach().cpu().numpy()
        for iSent in range(p_upos.shape[0]):
            for iWord in range(sl[iSent]):
                lang_id = x_lang[iSent] - 1
                language_result[lang_id]['total'] += 1
                if pred_upos[iSent, iWord] == tar_upos[iSent, iWord]:
                    language_result[lang_id]['upos_ok'] += 1
                if pred_xpos[iSent, iWord] == tar_xpos[iSent, iWord]:
                    language_result[lang_id]['xpos_ok'] += 1
                if pred_attrs[iSent, iWord] == tar_attrs[iSent, iWord]:
                    language_result[lang_id]['attrs_ok'] += 1

        return {'loss': loss, 'acc': language_result}

    def validation_epoch_end(self, outputs):
        language_result = {lang_id: {'total': 0, 'upos_ok': 0, 'xpos_ok': 0, 'attrs_ok': 0} for lang_id in
                           range(self._num_langs)}

        valid_loss_total = 0
        total = 0
        attrs_ok = 0
        upos_ok = 0
        xpos_ok = 0
        for out in outputs:
            valid_loss_total += out['loss']
            for lang_id in language_result:
                valid_loss_total += out['loss']
                language_result[lang_id]['total'] += out['acc'][lang_id]['total']
                language_result[lang_id]['upos_ok'] += out['acc'][lang_id]['upos_ok']
                language_result[lang_id]['xpos_ok'] += out['acc'][lang_id]['xpos_ok']
                language_result[lang_id]['attrs_ok'] += out['acc'][lang_id]['attrs_ok']
                # global
                total += out['acc'][lang_id]['total']
                upos_ok += out['acc'][lang_id]['upos_ok']
                xpos_ok += out['acc'][lang_id]['xpos_ok']
                attrs_ok += out['acc'][lang_id]['attrs_ok']

        self.log('val/loss', valid_loss_total / len(outputs))
        self.log('val/UPOS/total', upos_ok / total)
        self.log('val/XPOS/total', xpos_ok / total)
        self.log('val/ATTRS/total', attrs_ok / total)

        res = {}
        for lang_id in language_result:
            total = language_result[lang_id]['total']
            if total == 0:
                total = 1
            if self._id2lang is None:
                lang = lang_id
            else:
                lang = self._id2lang[lang_id]
            res[lang] = {
                "upos": language_result[lang_id]['upos_ok'] / total,
                "xpos": language_result[lang_id]['xpos_ok'] / total,
                "attrs": language_result[lang_id]['attrs_ok'] / total
            }

            self.log('val/UPOS/{0}'.format(lang), language_result[lang_id]['upos_ok'] / total)
            self.log('val/XPOS/{0}'.format(lang), language_result[lang_id]['xpos_ok'] / total)
            self.log('val/ATTRS/{0}'.format(lang), language_result[lang_id]['attrs_ok'] / total)

        # single value for early stopping
        self._epoch_results = self._compute_early_stop(res)
        self.log('val/early_meta', self._early_stop_meta_val)

        # print("\n\n\n", upos_ok / total, xpos_ok / total, attrs_ok / total,
        #      aupos_ok / total, axpos_ok / total, aattrs_ok / total, "\n\n\n")


class PrintAndSaveCallback(pl.callbacks.Callback):
    def __init__(self, args, language2id):
        super().__init__()
        self.args = args
        self.language2id = language2id

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # from pprint import pprint
        # pprint(metrics)
        for lang in pl_module._epoch_results:
            res = pl_module._epoch_results[lang]
            if "upos_best" in res:
                trainer.save_checkpoint(self.args.store + "." + lang + ".upos")
            if "xpos_best" in res:
                trainer.save_checkpoint(self.args.store + "." + lang + ".xpos")
            if "attrs_best" in res:
                trainer.save_checkpoint(self.args.store + "." + lang + ".attrs")

        trainer.save_checkpoint(self.args.store + ".last")

        s = "{0:30s}\tUPOS\tXPOS\tATTRS".format("Language")
        print("\n\n\t" + s)
        print("\t" + ("=" * (len(s) + 9)))
        for lang in self.language2id:
            upos = metrics["val/UPOS/{0}".format(lang)]
            xpos = metrics["val/XPOS/{0}".format(lang)]
            attrs = metrics["val/ATTRS/{0}".format(lang)]
            msg = "\t{0:30s}:\t{1:.4f}\t{2:.4f}\t{3:.4f}".format(lang, upos, xpos, attrs)
            print(msg)
        print("\n")


if __name__ == '__main__':
    from cube3.io_utils.misc import ArgParser

    argparser = ArgParser()
    # run argparser
    args = argparser()
    print(args)  # example

    """
    import json

    langs = json.load(open(args.train_file))
    doc_train = Document()
    doc_dev = Document()
    id2lang = {}
    for ii in range(len(langs)):
        lang = langs[ii]
        print(lang[1], ii)
        doc_train.load(lang[1], lang_id=ii)
        doc_dev.load(lang[2], lang_id=ii)
        id2lang[ii] = lang[0]

    """
    with open(args.train_file) as file:
        train_config = yaml.full_load(file)
    doc_train = Document()
    doc_dev = Document()
    doc_test = Document()


    for d in train_config["train_files"]:
        lang_code = list(d.keys())[0]
        file = d[lang_code]
        print("Reading train file for language code [{}] : {}".format(lang_code, file))
        doc_train.load(file, lang_id=train_config["language2id"].index(lang_code))
    for d in train_config["dev_files"]:
        lang_code = list(d.keys())[0]
        file = d[lang_code]
        print("Reading dev file for language code [{}] : {}".format(lang_code, file))
        doc_dev.load(file, lang_id=train_config["language2id"].index(lang_code))
    for d in train_config["test_files"]:
        lang_code = list(d.keys())[0]
        file = d[lang_code]
        print("Reading test file for language code [{}] : {}".format(lang_code, file))
        doc_test.load(file, lang_id=train_config["language2id"].index(lang_code))


    # ensure target dir exists
    target = args.store
    i = args.store.rfind("/")
    if i > 0:
        target = args.store[:i]
        os.makedirs(target, exist_ok=True)

    enc = Encodings()
    enc.compute(doc_train, None)
    enc.save('{0}.encodings'.format(args.store))

    config = TaggerConfig()
    config.lm_model = args.lm_model
    if args.config_file:
        config.load(args.config_file)
        if args.lm_model is not None:
            config.lm_model = args.lm_model
    config.save('{0}.config'.format(args.store))

    helper = LMHelper(device=args.lm_device, model=config.lm_model)
    helper.apply(doc_dev)
    helper.apply(doc_train)
    trainset = MorphoDataset(doc_train)
    devset = MorphoDataset(doc_dev)

    collate = MorphoCollate(enc)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate.collate_fn, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(devset, batch_size=args.batch_size, collate_fn=collate.collate_fn,
                            num_workers=args.num_workers)

    model = Tagger(config=config, encodings=enc, id2lang=train_config["language2id"])

    # training

    early_stopping_callback = EarlyStopping(
        monitor='val/early_meta',
        patience=args.patience,
        verbose=True,
        mode='max'
    )
    if args.gpus == 0:
        acc = 'ddp_cpu'
    else:
        acc = 'ddp'
    trainer = pl.Trainer(
        gpus=args.gpus,
        #accelerator=acc,
        #num_nodes=1,
        default_root_dir='data/',
        callbacks=[early_stopping_callback, PrintAndSaveCallback(args, train_config["language2id"])]
        # limit_train_batches=5,
        # limit_val_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)
