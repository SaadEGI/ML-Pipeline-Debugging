


def read_code_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


# read code from file and only return the ocde starting from the first comment
def read_code_from_file_starting_from_comment(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()

            # Find the index of the first comment
            comment_start = code.find("#")

            # If a comment is found, find the end of the line
            if comment_start != -1:
                comment_end = code.find("\n", comment_start)
                if comment_end != -1:
                    return code[comment_end + 1:]
                else:
                    return ""  # Return empty string if comment is the last line
            else:
                return ""  # Return empty string if no comment is found
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def summarize_usage_test(history_item: dict[str, any]):
    usage = history_item['response']['usage']
    print(
        f"LLM usage: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion, {usage['total_tokens']} total")

