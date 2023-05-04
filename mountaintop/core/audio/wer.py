import re
import sys
import unicodedata


class WerCalculator :
    def __init__(self) :
        self.data = {}
        self.space = []
        self.cost = {
            'cor': 0, 
            'sub': 1, 
            'del': 1, 
            'ins': 1
        }
    
    def calculate(self, ref, hyp) :
        # Initialization
        ref.insert(0, '')
        hyp.insert(0, '')
        while len(self.space) < len(ref) :
            self.space.append([])
        for row in self.space :
            for element in row :
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(hyp) :
                row.append({'dist' : 0, 'error' : 'non'})
        for i in range(len(ref)) :
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(hyp)) :
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in ref :
            if token not in self.data and len(token) > 0 :
                self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
        for token in hyp :
            if token not in self.data and len(token) > 0 :
                self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
        # Computing edit distance
        for i, lab_token in enumerate(ref) :
            for j, rec_token in enumerate(hyp) :
                if i == 0 or j == 0 :
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i-1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist :
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j-1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist :
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token :
                    dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
                    error = 'cor'
                else :
                    dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist :
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error
        # Tracing back
        result = {'ref':[], 'hyp':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
        i = len(ref) - 1
        j = len(hyp) - 1
        while True :
            if self.space[i][j]['error'] == 'cor' : # correct
                if len(ref[i]) > 0 :
                    self.data[ref[i]]['all'] = self.data[ref[i]]['all'] + 1
                    self.data[ref[i]]['cor'] = self.data[ref[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['ref'].insert(0, ref[i])
                result['hyp'].insert(0, hyp[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub' : # substitution
                if len(ref[i]) > 0 :
                    self.data[ref[i]]['all'] = self.data[ref[i]]['all'] + 1
                    self.data[ref[i]]['sub'] = self.data[ref[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['ref'].insert(0, ref[i])
                result['hyp'].insert(0, hyp[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del' : # deletion
                if len(ref[i]) > 0 :
                    self.data[ref[i]]['all'] = self.data[ref[i]]['all'] + 1
                    self.data[ref[i]]['del'] = self.data[ref[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['ref'].insert(0, ref[i])
                result['hyp'].insert(0, "")
                i = i - 1
            elif self.space[i][j]['error'] == 'ins' : # insertion
                if len(hyp[j]) > 0 :
                    self.data[hyp[j]]['ins'] = self.data[hyp[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['ref'].insert(0, "")
                result['hyp'].insert(0, hyp[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non' : # starting point
                break
            else : # shouldn't reach here
                print(f"this should not happen , i = {i}, j = {j}, error = {self.space[i][j]['error']}")
        return result
    
    def overall(self) :
        result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
        for token in self.data :
            result['all'] = result['all'] + self.data[token]['all']
            result['cor'] = result['cor'] + self.data[token]['cor']
            result['sub'] = result['sub'] + self.data[token]['sub']
            result['ins'] = result['ins'] + self.data[token]['ins']
            result['del'] = result['del'] + self.data[token]['del']
        return result
    
    def cluster(self, data) :
        result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
        for token in data :
            if token in self.data :
                result['all'] = result['all'] + self.data[token]['all']
                result['cor'] = result['cor'] + self.data[token]['cor']
                result['sub'] = result['sub'] + self.data[token]['sub']
                result['ins'] = result['ins'] + self.data[token]['ins']
                result['del'] = result['del'] + self.data[token]['del']
        return result
    
    def keys(self) :
        return list(self.data.keys())


special_token_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")


def characterize(
    text, 
    puncts = [
        '!', ',', '?', '、', '。', '！', '，', '；', '？', 
        '：', '「', '」', '︰', '『', '』', '《', '》'
    ],
    spacelist= [' ', '\t', '\r', '\n']
) :
    res = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
        if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: # space or not assigned
            i += 1
            continue
        if cat1 == 'Lo': # letter-other
            res.append(char)
            i += 1
        else:
            # some input looks like: <unk><noise>, we want to separate it to two words.
            sep = ' '
            if char == '<': sep = '>'
            j = i+1
            while j < len(text):
                c = text[j]
                if ord(c) >= 128 or (c in spacelist) or (c==sep):
                    break
                j += 1
            if j < len(text) and text[j] == '>':
                j += 1
            res.append(text[i:j])
            i = j
    return res


def _stripoff_tags(text):
    if text is None or len(text) < 1: 
        return ''
    
    text_norm = ''
    sub_parts = special_token_pattern.split(text)
    for sub_part in sub_parts:
        if len(sub_part) < 1:
            continue
        if special_token_pattern.fullmatch(sub_part) is not None:
            # special token like: <unk><noise>
            continue
        text_norm += sub_part
    return text_norm


def normalize(sentence, ignore_words, case_sensitive, split=None, remove_tag=True):
    """ 
    sentence, ignore_words are both in unicode
    """
    new_sentence = []
    for token in sentence:
        x = token
        if not case_sensitive:
            x = x.lower()
        if x in ignore_words:
            continue
        if remove_tag:
            x = _stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


def width(text):
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in text)


def default_cluster(word) :
    unicode_names = [ unicodedata.name(char) for char in word ]
    for i in reversed(range(len(unicode_names))) :
        if unicode_names[i].startswith('DIGIT') :  # 1
            unicode_names[i] = 'Number'  # 'DIGIT'
        elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
                    unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
            # 明 / 郎
            unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
        elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
                    unicode_names[i].startswith('LATIN SMALL LETTER')) :
            # A / a
            unicode_names[i] = 'English'  # 'LATIN LETTER'
        elif unicode_names[i].startswith('HIRAGANA LETTER') :  # は こ め
            unicode_names[i] = 'Japanese'  # 'GANA LETTER'
        elif (unicode_names[i].startswith('AMPERSAND') or
                    unicode_names[i].startswith('APOSTROPHE') or
                    unicode_names[i].startswith('COMMERCIAL AT') or
                    unicode_names[i].startswith('DEGREE CELSIUS') or
                    unicode_names[i].startswith('EQUALS SIGN') or
                    unicode_names[i].startswith('FULL STOP') or
                    unicode_names[i].startswith('HYPHEN-MINUS') or
                    unicode_names[i].startswith('LOW LINE') or
                    unicode_names[i].startswith('NUMBER SIGN') or
                    unicode_names[i].startswith('PLUS SIGN') or
                    unicode_names[i].startswith('SEMICOLON')) :
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            del unicode_names[i]
        else :
            return 'Other'
    if len(unicode_names) == 0 :
            return 'Other'
    if len(unicode_names) == 1 :
            return unicode_names[0]
    for i in range(len(unicode_names)-1) :
        if unicode_names[i] != unicode_names[i+1] :
            return 'Other'
    return unicode_names[0]

