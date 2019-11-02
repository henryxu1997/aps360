import re

def preprocess_lines(lines):
    """
    Parse lines of text into paragraphs, doing some filtering.
    """
    p = re.compile('chapter', re.I)
    current_paragraph = []
    paragraphs = []
    for line in lines:
        # Skip chapter lines
        if p.match(line[:8]):
            continue
        if line == '\n':
            # Gutenberg saves their text files weirdly
            # A paragraph can span multiple lines and paragraphs are delimited by newline.
            if len(current_paragraph) > 0:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line.strip())
    return paragraphs

def parse_narration_speech(paragraphs):
    speech = []
    for para in paragraphs:
        # Regex to match text between quotes
        # TODO: Might need to add additional rules in the regex
        tokens = re.findall(r'("|“)(.*?)("|”)', para)
        # tokens is a list of 3-tuples where tup[0] and tup[2] are the quotation marks
        if len(tokens) > 0:
            speech.append([s[1] for s in tokens])
    return speech


def main(path):
    with open(path) as f:
        lines = f.readlines()
        paragraphs = preprocess_lines(lines)
        speech = parse_narration_speech(paragraphs)
        print(speech[:20])

if __name__ == '__main__':
    main('data/christmas_carol.txt')
