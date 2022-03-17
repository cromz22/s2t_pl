import yaml
import sys
from pathlib import Path
import subprocess
from argparse import ArgumentParser


def split_wav(elem, wav_path, out_path, counter: int):
    duration = elem["duration"]
    offset = elem["offset"]
    wavfile = elem["wav"]
    stem = wavfile.split(".")[0]
    command = f"sox {wav_path}/{wavfile} {out_path}/{stem}_{counter}.wav trim {offset} {duration}"
    subprocess.call(command.split(" "))


def divide_wavs_in_yaml(yaml_file):
    yaml_file = Path(yaml_file)
    wav_path = yaml_file.parents[1] / "wav"
    out_path = wav_path / "divided"
    out_path.mkdir()

    try:
        with open(yaml_file) as f:
            dicts = yaml.safe_load(f)

            counter = 1
            current_wavfile = ""
            for elem in dicts:
                split_wav(elem, wav_path, out_path, counter)
                if current_wavfile == elem["wav"]:
                    counter = 1
                else:
                    counter += 1

    except Exception as e:
        print(f"Error while loading yaml file {yaml_file}", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()
    lang = args.lang
    assert lang in ["ja", "zh", "de"]

    for split in ["dev", "tst-COMMON", "tst-HE", "train"]:
        MuSTC= "/Users/shimizu/Desktop/research/iwslt2022/mustc_v2/dummy"
        yaml_file = f"{MuSTC}/en-{lang}/data/{split}/txt/{split}.yaml"
        print(f"splitting wav files in {yaml_file}...")
        divide_wavs_in_yaml(yaml_file)


if __name__ == "__main__":
    main()
