import re

class TextCleaner:
    def __init__(self, remove_extra_whitespace: bool = True, lowercase: bool = False):
        self.remove_extra_whitespace = remove_extra_whitespace
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """
        Applies a series of cleaning steps to the input text, such as removing extra whitespace and converting to lowercase.
        """
        if not text:
            return ""
        
        text = text.replace("\xa0", " ")  # Replace non-breaking spaces with regular spaces
        if self.remove_extra_whitespace:
            # Replace multiple consecutive newlines with just a double newline
            text = re.sub(r'\n{3,}', '\n\n', text)

            # Replace multiple consecutive spaces with a single space
            text = re.sub(r'[ \t]+', ' ', text)

            # Trim whitespace from the start and end of the text
            text = "\n".join([line.strip() for line in text.splitlines()])

        if self.lowercase:
            text = text.lower()

        return text.strip()