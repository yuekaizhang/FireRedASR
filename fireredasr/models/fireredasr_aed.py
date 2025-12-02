import torch
import json
from pathlib import Path

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig, GenerationSession
from tensorrt_llm.bindings import KVCacheType
from collections import OrderedDict
from fireredasr.models.module.conformer_encoder import ConformerEncoder
from fireredasr.models.module.transformer_decoder import TransformerDecoder

def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] !=
            pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor

class TrtLlmDecoder:

    def __init__(self, engine_dir, sos_id, eos_id, pad_id, runtime_mapping, debug_mode=False):
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config['plugin_config']['paged_kv_cache'] == True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_output_lengths,
                 max_new_tokens=40,
                 num_beams=1,
                 length_penalty=0.0):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([
            batch_size, decoder_max_input_length + max_new_tokens,
            encoder_max_input_length
        ]).int().cuda()
        # generation config
        sampling_config = SamplingConfig(end_id=self.eos_id,
                                         pad_id=self.pad_id,
                                         num_beams=num_beams,
                                         length_penalty=length_penalty)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # For this ASR task, the initial input is just [SOS], so no padding is present.
            # The logic from run.py is kept for completeness.
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=self.pad_id)
            if encoder_outputs.dim() == 3:
                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lengths)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_output_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        return output_ids

class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        # Allow passing the new flag via from_args
        return cls(args, use_trt_decoder=getattr(args, 'use_trt_decoder', True))

    def __init__(self, args, use_trt_decoder=True):
        super().__init__()
        self.use_trt_decoder = use_trt_decoder
        
        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        if self.use_trt_decoder:
            runtime_mapping = tensorrt_llm.Mapping() # Assuming single GPU
            self.decoder = TrtLlmDecoder(
                engine_dir=Path("/workspace_yuekai/asr/FireRedASR/examples/trt_engine"),
                sos_id=args.sos_id,
                eos_id=args.eos_id,
                pad_id=args.pad_id,
                runtime_mapping=runtime_mapping
            )
        else:
            self.decoder = TransformerDecoder(
                args.sos_id, args.eos_id, args.pad_id, args.odim,
                args.n_layers_dec, args.n_head, args.d_model, args.pe_maxlen)


    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        
        if self.use_trt_decoder:
            batch_size = padded_input.size(0)
            device = padded_input.device

            decoder_input_ids = torch.full((batch_size, 1), self.decoder.sos_id, dtype=torch.int32).to(device)
            encoder_max_input_length = enc_outputs.size(1)
            encoder_output_lengths = enc_mask.sum(dim=-1).to(torch.int32).squeeze(-1)
            # breakpoint()
            max_new_tokens = decode_max_len if decode_max_len > 0 else enc_outputs.size(1)

            output_ids = self.decoder.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=enc_outputs,
                encoder_max_input_length=encoder_max_input_length,
                encoder_output_lengths=encoder_output_lengths,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                length_penalty=length_penalty
            )
            print(output_ids, 2333333)
            # breakpoint()
            
            # Format the output tensor back to the expected structure
            if nbest > beam_size:
                nbest = beam_size
            
            nbest_hyps = []
            for i in range(batch_size):
                n_hyps = []
                for j in range(nbest):
                    token_ids = output_ids[i, j, decoder_input_ids.size(1):].cpu()
                    eos_index = (token_ids == self.decoder.eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_index) > 0:
                        token_ids = token_ids[:eos_index[0]]
                    n_hyps.append({"yseq": token_ids})
                nbest_hyps.append(n_hyps)

            return nbest_hyps
        else:
            return self.decoder.batch_beam_search(
                enc_outputs, enc_mask,
                beam_size, nbest, decode_max_len,
                softmax_smoothing, length_penalty, eos_penalty)
