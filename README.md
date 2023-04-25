# split-texts
split-texts is a command-line tool for splitting large texts into smaller chunks with JSON output for easy processing.

## Usage

To use split-texts, simply pipe or redirect a text file to the script, like so:

```
$ cat input.txt | split-texts > output.json
```

This will split the input text into smaller chunks, and write the resulting chunks to the output file in JSON format.

## Parameters

split-texts provides several parameters that can be customized to adjust the text splitting behavior, including:

- `encoding_name`: The tokenizer's encoding name. Default: `cl100k_base`.
- `desired_min_token_fraction`: The desired minimum fraction of tokens in a chunk. Default: `0.35`.
- `min_chunk_length_to_embed`: The minimum length of chunks to be included in the final list. Default: `5`.
- `avg_chars_per_token`: The average number of characters per token (e.g., in English). Default: `5`.
- `chunk_size`: The target size of each text chunk in tokens. Default: `3000`.
- `chunk_overlap`: The number of tokens to overlap between adjacent chunks. Default: `0`.

## Output

Split Texts outputs a list of text chunks, each of which is a string of approximately `chunk_size` tokens. The output is formatted as a JSON array for easy processing by other programs or scripts.

## Installation

To use Split Texts, you need to have Python 3.x installed on your system, as well as the following dependencies:

- TikToken (`pip install tiktoken`)

To install Split Texts, simply download or clone this repository and run the script `split_texts.py`.

## License

Split Texts is licensed under the MIT License. See `LICENSE` for more information.