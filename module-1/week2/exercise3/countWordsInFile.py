import re


def count_words_in_file(file_path: str) -> dict:
    word_count = {}

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                line = line.lower()

                words = re.findall(r'\b[a-z]+\b', line)

                for word in words:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
    except FileNotFoundError:
        print(f"File {file_path} not found.")

    return word_count


file_path = '/home/anhthai/AIO/Exercise/data/P1_data.txt'
print(count_words_in_file(file_path))
