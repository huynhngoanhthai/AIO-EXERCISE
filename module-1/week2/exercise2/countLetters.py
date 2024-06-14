def countLetters(word: str) -> dict:

    letter_count = {}

    for letter in word:
        if letter in letter_count:
            letter_count[letter] += 1
        else:
            letter_count[letter] = 1

    return letter_count


input_word = "Happiness"
print(countLetters(input_word))
