def levenshtein_distance(source: str, target: str) -> int:

    if len(source) < len(target):
        return levenshtein_distance(target, source)

    previous_row = range(len(target) + 1)

    for source_index, source_char in enumerate(source):
        current_row = [source_index + 1]
        for target_index, target_char in enumerate(target):
            insertions = previous_row[target_index + 1] + 1
            deletions = current_row[target_index] + 1
            substitutions = previous_row[target_index] + \
                (source_char != target_char)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


source_string = "kitten"
target_string = "sitting"
distance = levenshtein_distance(source_string, target_string)
print(
    f"The Levenshtein distance between '{source_string}' and '{target_string}' is {distance}.")
