# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:59:27 2021

@author: CdrSonan
"""

class LocaleDict:
    def __init__(self, lang):
        self.locale = dict()
        if lang == "en":
            self.locale["version_label"] = "NovaVox Devkit PREALPHA 0.1.2"
            self.locale["no_vb"] = "no Voicebank loaded"
            self.locale["unsaved_vb"] = "unsaved Voicebank"
            self.locale["metadat_btn"] = "edit Metadata"
            self.locale["metadat_lbl"] = "Metadata Editor"
            self.locale["phon_btn"] = "edit Phonemes"
            self.locale["phon_lbl"] = "Phoneme Editor"
            self.locale["crfai_btn"] = "edit Phoneme Crossfade Ai"
            self.locale["crfai_lbl"] = "Phoneme Crossfade Ai Editor"
            self.locale["param_btn"] = "edit Ai-driven Parameters"
            self.locale["param_lbl"] = "Ai-driven Parameter Editor"
            self.locale["dict_btn"] = "edit Dictionary"
            self.locale["dict_lbl"] = "Dictionary Editor"
            self.locale["save_as"] = "Save as..."
            self.locale["open"] = "Open..."
            self.locale["new"] = "New..."
            self.locale["warning"] = "Warning"
            self.locale["vb_discard_msg"] = "Creating a new Voicebank will discard all unsaved changes to the currently opened one. Continue?"
            self.locale[".nvvb_desc"] = "NovaVox Voicebanks"
            self.locale[".wav_desc"] = "wavesound audio files"
            self.locale["all_files_desc"] = "All files"
            self.locale["name"] = "Name:"
            self.locale["smp_rate"] = "Sample Rate:"
            self.locale["load_other_VB"] = "load from other VB"
            self.locale["additive_msg"] = "load data in addition to existing data (yes) or overwrite existing data (no)?"
            self.locale["ok"] = "OK"
            self.locale["phon_list"] = "Phoneme List"
            self.locale["add"] = "add"
            self.locale["remove"] = "remove"
            self.locale["per_ph_set"] = "per-phoneme settings"
            self.locale["phon_key"] = "phoneme key:"
            self.locale["est_pit"] = "estimated pitch:"
            self.locale["psearchr"] = "pitch search range:"
            self.locale["fwidth"] = "spectral filter width:"
            self.locale["viter"] = "voiced excitation filter iterations:"
            self.locale["uviter"] = "unvoiced excitation filter iterations:"
            self.locale["cng_file"] = "change file"
            self.locale["finalize"] = "finalize"
            self.locale["new_phon"] = "new Phoneme"
            self.locale["phon_key_sel"] = "please select a key for the new phoneme:"
            self.locale["ai_samp_list"] = "AI training sample list"
            self.locale["ai_settings"] = "Sample preprocessing and AI training settings"
            self.locale["epochs"] = "AI training epochs:"
            self.locale["train"] = "train AI"

            