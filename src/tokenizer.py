"""
Custom Chess Tokenizer for the Chess Challenge.

This tokenizer treats each move as a single token using the extended UCI notation
from the Lichess dataset (e.g., WPe2e4, BNg8f6).

The dataset format uses:
- W/B prefix for White/Black
- Piece letter: P=Pawn, N=Knight, B=Bishop, R=Rook, Q=Queen, K=King
- Source and destination squares (e.g., e2e4)
- Special suffixes: (x)=capture, (+)=check, (+*)=checkmate, (o)/(O)=castling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class ChessTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer for chess moves using extended UCI notation.
    

    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initialize the chess tokenizer.
        
        Args:
            vocab_file: Path to a JSON file containing the vocabulary mapping.
            vocab: Dictionary mapping tokens to IDs (alternative to vocab_file).
            **kwargs: Additional arguments passed to PreTrainedTokenizer.
        """
        # Initialize special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        # Remove any duplicate special-token entries passed through kwargs
        # to avoid "multiple values for keyword" errors when loading from disk.
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)
        
        # Load or create vocabulary
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            # Create a minimal vocabulary with just special tokens
            # The full vocabulary should be built from the dataset
            self._vocab = self._create_fixed_vocab()
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )
    
    def _create_fixed_vocab(self) -> Dict[str, int]:

        vocab_list = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        # Separate moves
        vocab_list.append(" ")
        
        # Colors
        vocab_list.extend(["W", "B"])
        
        # Pieces
        vocab_list.extend(["P", "N", "B", "R", "Q", "K"])
        
        # 3. Squares
        files = "abcdefgh"
        ranks = "12345678"
        squares = [f"{f}{r}" for f in files for r in ranks]
        vocab_list.extend(sorted(squares))

        # Suffixes
        suffixes = ["(x)", "(+)", "(+*)", "(o)", "(O)", "(Q)", "(K)", "(x*)", "(x+*)"]
        vocab_list.extend(suffixes)

        vocab_list = list(dict.fromkeys(vocab_list))
        
        
        return {token: idx for idx, token in enumerate(vocab_list)}
    
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into atomic components.
        """
        import re
        tokens = []
        moves = text.strip().split()
        
        pattern = re.compile(r"^([WB])([PNBRQK])([a-h][1-8])([a-h][1-8])(.*)$")
        
        for move in moves:
            match = pattern.match(move)
            if match:
                for i in range(1,6):
                    if  match.group(i) in self._vocab:
                        tokens.append(match.group(i))
                tokens.append(' ')
            else:
                tokens.append(self.UNK_TOKEN)
                
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return "".join(t for t in tokens if t not in special)
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory to save the vocabulary.
            filename_prefix: Optional prefix for the filename.
        
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)


