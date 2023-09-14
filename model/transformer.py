import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.modelutils import generate_original_PE, generate_regular_PE, generate_sampling_PE


# 位置编码这一块还需要进一步优化
class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 enc_in: int,
                 dec_in: int,
                 c_out: int,
                 d_model: int,
                 dff: int,
                 q: int,
                 v: int,
                 h: int,
                 e_layers: int,
                 d_layers: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 24,
                 sampling_rate: int = 6):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dff=dff,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(e_layers)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dff=dff,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(d_layers)])

        self._embedding = nn.Linear(enc_in, d_model)
        self._embedding_y = nn.Linear(dec_in, d_model)
        self._linear = nn.Linear(d_model, c_out)
        # self._linear1 = nn.Linear(d_model * 100, d_output)
        self._recon = nn.Linear(d_model, enc_in)

        self._samping_rate_embedding = nn.Embedding(sampling_rate, d_model)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask_x_map, mask_pred_map, enc_types, pred_types, IFALL=1) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        y:
            :class:`torch.Tensor` of shape (batch_size, Y, d_y).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]
        Y = y.shape[1]
        # print(x.shape, y.shape)

        # Embeddin module
        encoding = self._embedding(x)
        decoding = self._embedding_y(y)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # 添加多采样率编码
        # enc_index = generate_sampling_PE(mask_x_map)

        sampling_rate_encoding = self._samping_rate_embedding(enc_types)
        sampling_rate_encoding = sampling_rate_encoding.to(encoding.device)
        encoding.add_(sampling_rate_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # encoding重构原始输入
        if not IFALL:
            recon_x = self._recon(encoding)
            return recon_x

        else:
            # Add position encoding
            if self._generate_PE is not None:
                positional_decoding = self._generate_PE(Y, self._d_model)
                positional_decoding = positional_decoding.to(decoding.device)
                decoding.add_(positional_decoding)

            # 添加多采样率编码
            # dec_index = generate_sampling_PE(torch.cat((mask_x_map, mask_pred_map), dim=1))

            dec_sampling_rate_encoding = self._samping_rate_embedding(torch.cat((enc_types, pred_types), dim=1))
            dec_sampling_rate_encoding = dec_sampling_rate_encoding.to(decoding.device)
            decoding.add_(dec_sampling_rate_encoding)

            for layer in self.layers_decoding:
                decoding = layer(decoding, encoding)

            # Output module
            output = self._linear(decoding)
            return output



