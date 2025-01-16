from random import choice, randint, choices, random
import resources

def get_char():
    return choice(resources.CHARACTERS)

def get_char_num():
    return choice(resources.CHARACTERS + resources.NUMBERS)

def get_word():
    return choice(resources.WORDS)

def get_number():
    return choice(resources.NUMBERS)

def get_space():
    return " "

def get_phone_number():
    return get_format("0nn.nnn.nnnn")

def get_email(max_chars_per_parts: tuple[int, int, int] = [10, 7, 4]):
    chars_per_parts = [choice(range(1, v + 1)) for v in max_chars_per_parts]
    return get_format(f"{'C'*chars_per_parts[0]}@{'C'*chars_per_parts[1]}.{'c'*chars_per_parts[2]}", case = None)

def get_delimiter(space_chance = 0.4):
    return ' ' if random() < space_chance else get_special()

def get_multi_words(max_words=5, delimiter=None):
    num_words = randint(2, max_words)
    words = choices(population=resources.WORDS, k = num_words)
    combined = words[0]
    for w in words[1:]:
        d = get_delimiter() if delimiter is None else delimiter 
        combined = combined + d + w
    return combined

def get_date():
    return get_format("nn-nn-nnnn")

def get_date_dash():
    return get_format("nn/nn/nnnn")

def next_line():
    return "\n"

def get_special():
    return choice(resources.SPECIALS)

parser = {"c": get_char,
          "w": get_word,
          "m": get_multi_words,
          "n": get_number,
          "C": get_char_num,
          "s": get_space,
          "p": get_phone_number,
          "e": get_email,
          "d": get_date,
          "D": get_date_dash,
          "l": next_line,}

def case_format(text:str, case:str = "random", probs = [5, 1, 1]):
    if case is None:
        return text
    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    if case == "title":
        return text.title()
    if case == "random":
        options = ["lower"] * probs[0] + ["upper"] * probs[1] + ["title"] * probs[2]
        return case_format(text, choice(options))
    raise ValueError(f"Unsupported case: {case}.")


def get_format(format: str, case = "random"):
    """Return a text match the given format."""
    output = ""
    for char in format:
        output += case_format(parser[char](), case) if char in parser.keys() else char
    return output

if __name__ == "__main__":
    format = "d D p c C e"
    print(get_format(format))