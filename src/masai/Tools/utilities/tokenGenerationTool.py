# Define a color map dictionary
COLOR_MAP = {
    'black': "\033[30m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\u001b[33m",
    'blue': "\033[34m",
    'magenta': "\u001b[35m",
    'cyan': "\033[36m",
    'white': "\033[37m",
    'reset': "\033[0m"
}

# Example usage:
import re
import time
import sys
from typing import Dict, Callable, List

def token_stream(text, color='white', delay=0.08, token_type='word'):
    """Stream text with a specified color using a tokenization method."""
    # Use the provided color from our COLOR_MAP. Default to white if not found.
    color_code = COLOR_MAP.get(color.lower(), COLOR_MAP['white'])
    
    # Split text into tokens (words and newlines)
    tokens = []
    current = []
    for char in text:
        if (char == '\n' or (token_type == 'word' and char == ' ')) and current:
            tokens.append(''.join(current))
            current = []
        if char == '\n':
            tokens.append('\n')  # Preserve newlines
        elif char != ' ' or token_type != 'word':
            current.append(char)
    if current:
        tokens.append(''.join(current))
    
    # Set the color and start streaming tokens
    sys.stdout.write(color_code)
    for i, token in enumerate(tokens):
        if token == '\n':
            sys.stdout.write('\n')
            sys.stdout.flush()
            continue
            
        sys.stdout.write(token)
        sys.stdout.flush()
        time.sleep(delay)
        
        # If the next token is not a newline and we're tokenizing by word, add a space.
        if token_type == 'word' and i < len(tokens) - 1 and tokens[i+1] != '\n':
            sys.stdout.write(' ')
            sys.stdout.flush()
        if token_type=="word" and i==len(tokens) - 1:
            sys.stdout.write('\n\n')
            sys.stdout.flush()
    
    # # Reset the terminal color at the end.
    # sys.stdout.write(COLOR_MAP['reset'] + "\n")
class MarkupProcessor:
    """Process text markup with enhanced LaTeX block/inline handling and terminal formatting"""
    
    MATH_PATTERNS = re.compile(
        r'(?:\\\w+|\^|_|\{|\}|\\[{}]|'
        r'\b(?:sin|cos|tan|log|lim|sum|prod|int|sqrt|frac)\b|'
        r'\d+[.,]?\d*[eE]?[+-]?\d*|'
        r'[α-ωΑ-Ω]|π|∞|±|≠|≈|≡|≤|≥|'
        r'[+=<>×÷¬∧∨]|'
        r'(?<!\\)[_\^]|'
        r'\$(?!\$)|\\\(|\\\)|\\\[|\\\]|\\begin\{.*?\}|\\end\{.*?\})'
    )
    
    def __init__(self):
        self.handlers = {
            'latex_equation': {
                'pattern': r'(\\\[.*?\\\]|\$\$(.*?)\$\$|\\\(.*?\\\)|\$(.*?)\$)',
                'handler': self._handle_latex_equation
            },
            'bold': {
                'pattern': r'\*\*(.*?)\*\*',
                'handler': self._handle_bold
            },
            'italic': {
                'pattern': r'\*(.*?)\*',
                'handler': self._handle_italic
            },
            'color_tag': {
                'pattern': r'<color:(.*?)>(.*?)</color>',
                'handler': self._handle_color_tag
            }
        }
        self.unicode_map = {
            r'\nabla': '∇', r'\sum': 'Σ', r'\partial': '∂', r'\frac': '⁄',
            r'\cdot': '·', r'\times': '×', r'\infty': '∞', r'\sqrt': '√',
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\pi': 'π',
            r'\hbar': 'ℏ', r'\rightarrow': '→', r'\leftarrow': '←',
            r'\geq': '≥', r'\leq': '≤', r'\neq': '≠', r'\approx': '≈',
            r'\equiv': '≡', r'\pm': '±', r'\cdot': '·'
        }

    def _replace_unicode(self, match: re.Match) -> str:
        content = match.group(1)
        for latex, unicode_char in self.unicode_map.items():
            content = content.replace(latex, unicode_char)
        return content

    def _validate_latex(self, content: str) -> str:
        """Auto-detect and repair missing LaTeX delimiters"""
        has_dollar = ('$' in content) and (content.count('$') % 2 == 0)
        has_brackets = ('\\(' in content or '\\[' in content)
        math_confidence = len(self.MATH_PATTERNS.findall(content)) / max(len(content.split()), 1)

        if math_confidence > 0.3 and not (has_dollar or has_brackets):
            return f"${content}$" if '\n' not in content else f"$$\n{content}\n$$"
        if content.startswith('$') != content.endswith('$'):
            return f"${content.strip('$')}$"
        return content

    def _process_text_blocks(self, text: str) -> str:
        """Enhanced block processing with $$ delimiter support"""
        processed = []
        current_block = []
        in_math = False
        math_delimiter = None
        
        for line in text.split('\n'):
            stripped = line.lstrip()
            if any(stripped.startswith(d) for d in ('\\[', '$$', '\\begin{')):
                in_math = True
                math_delimiter = '\\]' if stripped.startswith('\\[') else '$$'
                current_block.append(line)
                continue
                
            if in_math:
                current_block.append(line)
                if math_delimiter in line.rstrip():
                    processed.append('\n'.join(current_block))
                    current_block = []
                    in_math = False
                continue
                
            processed_line = []
            for segment in re.split(r'(\\\(|\\\)|\\\[|\\\]|\$\$|\$)', line):
                if segment in ('\\[', '\\]', '\\', '$$', '$'):
                    processed_line.append(segment)
                else:
                    processed_line.append(self._validate_latex(segment))
            processed.append(''.join(processed_line))
        
        return '\n'.join(processed)

    def _handle_latex_equation(self, match: re.Match) -> str:
        """Process all LaTeX equation types with proper delimiter handling"""
        equation = match.group(0)
        
        # Determine delimiter type
        if equation.startswith('$$') and equation.endswith('$$'):
            content = equation[2:-2].strip()
            delimiter = '$$'
        elif equation.startswith('$') and equation.endswith('$'):
            content = equation[1:-1].strip()
            delimiter = '$'
        elif equation.startswith('\\[') and equation.endswith('\\]'):
            content = equation[2:-2].strip()
            delimiter = '\\['
        elif equation.startswith('\\(') and equation.endswith('\\)'):
            content = equation[2:-2].strip()
            delimiter = '\\('
        else:
            content = equation  # Fallback for malformed equations
        
        # Convert LaTeX to Unicode
        content = re.sub(r'\\(.*?)(\W|$)', self._replace_unicode, content)
        
        # Format based on delimiter type
        if delimiter in ('$$', '\\['):
            return f"{COLOR_MAP['cyan']}\n  {content}\n{COLOR_MAP['reset']}"
        return f"{COLOR_MAP['cyan']} {content} {COLOR_MAP['reset']}"

    def _handle_bold(self, match: re.Match) -> str:
        return f"\033[1m{match.group(1)}\033[0m"

    def _handle_italic(self, match: re.Match) -> str:
        return f"\033[3m{match.group(1)}\033[0m"

    def _handle_color_tag(self, match: re.Match) -> str:
        color = match.group(1).lower()
        content = match.group(2)
        return f"{COLOR_MAP.get(color, '')}{content}{COLOR_MAP['reset']}"

    def process(self, text: str) -> str:
        """Main processing method with LaTeX validation"""
        text = self._process_text_blocks(text)
        for handler in self.handlers.values():
            text = re.sub(
                handler['pattern'],
                handler['handler'],
                text,
                flags=re.DOTALL
            )
        return text