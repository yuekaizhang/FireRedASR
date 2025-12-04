import os
import time

import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_aed_tensorrt import FireRedAsrAedTensorRT
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, tensorrt_model_dir=None):
        assert asr_type in ["aed", "llm", "aed_tensorrt"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path =os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "aed_tensorrt":
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
            engine_dir = tensorrt_model_dir if tensorrt_model_dir else model_dir
            model = FireRedAsrAedTensorRT.from_model_dir(engine_dir, tokenizer)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, args={}):
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)
        if args.get("use_gpu", False):
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        if self.asr_type == "aed":
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav,
                    "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "aed_tensorrt":
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"]] # No need for .cpu() as it is already on cpu
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav,
                    "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if args.get("use_gpu", False):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results



def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])

    # The model architecture has been refactored to align with Whisper's style.
    # This requires remapping the state dict keys from the original checkpoint for the decoder only.
    original_state_dict = package["model_state_dict"]
    new_state_dict = {}
    for key, value in original_state_dict.items():
        if key == "decoder.positional_encoding.pe":
            # Manually remap the sinusoidal positional encoding buffer to the new learnable parameter.
            # The original buffer has a shape of (1, max_len, d_model), so we squeeze it.
            new_state_dict["decoder.positional_embedding"] = value.squeeze(0)
            continue

        if key.startswith("decoder."):
            new_key = key
            # Top-level decoder module renames
            new_key = new_key.replace("decoder.tgt_word_emb.", "decoder.token_embedding.")
            new_key = new_key.replace("decoder.layer_stack.", "decoder.blocks.")
            new_key = new_key.replace("decoder.layer_norm_out.", "decoder.ln.")
            new_key = new_key.replace("decoder.tgt_word_prj.", "decoder.output_projection.")

            # ResidualAttentionBlock internal layer renames
            new_key = new_key.replace(".self_attn_norm.", ".attn_ln.")
            new_key = new_key.replace(".self_attn.", ".attn.")
            new_key = new_key.replace(".cross_attn_norm.", ".cross_attn_ln.")
            new_key = new_key.replace(".mlp_norm.", ".mlp_ln.")

            # Inlined PositionwiseFeedForward renames
            new_key = new_key.replace(".mlp.w_1.", ".mlp.0.")
            new_key = new_key.replace(".mlp.w_2.", ".mlp.2.")

            # MultiHeadAttention submodule renames (from old custom MHA to whisper.model.MultiHeadAttention)
            new_key = new_key.replace(".w_qs.", ".query.")
            new_key = new_key.replace(".w_ks.", ".key.")
            new_key = new_key.replace(".w_vs.", ".value.")
            new_key = new_key.replace(".fc.", ".out.")

            new_state_dict[new_key] = value
        else:
            # Keep encoder keys unchanged
            new_state_dict[key] = value
    # delete decoder related keys
    new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("decoder.")}
    model.load_state_dict(new_state_dict, strict=False)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer
