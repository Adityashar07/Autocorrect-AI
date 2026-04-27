from flask import Flask, render_template, request, jsonify
import os
import re
from spellchecker import SpellChecker

app = Flask(__name__)
spell = SpellChecker()

SUBJECT_VERB_MAP = {
    "i": "am", "he": "is", "she": "is", "it": "is",
    "we": "are", "they": "are", "you": "are",
    "dog":"is","cat":"is","boy":"is","girl":"is","man":"is","woman":"is",
    "child":"is","baby":"is","teacher":"is","student":"is","doctor":"is",
    "player":"is","bird":"is","horse":"is","cow":"is","lion":"is",
    "king":"is","queen":"is","father":"is","mother":"is","brother":"is",
    "sister":"is","friend":"is","boss":"is","soldier":"is",
    "dogs":"are","cats":"are","boys":"are","girls":"are","men":"are",
    "women":"are","children":"are","students":"are","teachers":"are",
    "players":"are","birds":"are","people":"are","kids":"are",
    "friends":"are","workers":"are","soldiers":"are","doctors":"are",
}

PAST_MARKERS = {
    "yesterday","last","ago","previously","earlier",
    "once","before","already","recently","formerly",
}

FUTURE_MARKERS = {
    "tomorrow","soon","later","next","tonight","eventually","shortly",
}

THIRD_PERSON_SINGULAR = {"he","she","it"}
ALL_SUBJECTS = {"i","he","she","it","we","they","you"}

IRREGULAR_VERBS = {
    "go":"went","come":"came","run":"ran","eat":"ate","drink":"drank",
    "see":"saw","do":"did","have":"had","make":"made","take":"took",
    "give":"gave","get":"got","sit":"sat","stand":"stood","sleep":"slept",
    "speak":"spoke","write":"wrote","read":"read","drive":"drove",
    "fly":"flew","swim":"swam","sing":"sang","ring":"rang","win":"won",
    "buy":"bought","bring":"brought","think":"thought","catch":"caught",
    "teach":"taught","sell":"sold","tell":"told","feel":"felt","keep":"kept",
    "leave":"left","meet":"met","send":"sent","spend":"spent","build":"built",
    "hold":"held","hear":"heard","know":"knew","grow":"grew","throw":"threw",
    "break":"broke","choose":"chose","forget":"forgot","hide":"hid",
    "ride":"rode","rise":"rose","shake":"shook","steal":"stole","wake":"woke",
    "wear":"wore","bite":"bit","blow":"blew","draw":"drew","fall":"fell",
    "fight":"fought","find":"found","lose":"lost","pay":"paid","say":"said",
    "shoot":"shot","shut":"shut","tear":"tore","understand":"understood",
    "begin":"began","bend":"bent","bleed":"bled","burst":"burst",
    "cut":"cut","deal":"dealt","dig":"dug","feed":"fed","flee":"fled",
    "forgive":"forgave","hang":"hung","hit":"hit","hurt":"hurt",
    "kneel":"knelt","lay":"laid","lead":"led","lend":"lent","let":"let",
    "light":"lit","mean":"meant","put":"put","quit":"quit","set":"set",
    "shine":"shone","show":"showed","shrink":"shrank","sink":"sank",
    "slide":"slid","spread":"spread","spring":"sprang","stick":"stuck",
    "sting":"stung","strike":"struck","swear":"swore","sweep":"swept",
    "swing":"swung","weep":"wept","withdraw":"withdrew","wring":"wrung",
}

IRREGULAR_3RD = {
    "go":"goes","do":"does","have":"has","be":"is",
    "say":"says","get":"gets","make":"makes","come":"comes",
    "see":"sees","know":"knows","take":"takes","give":"gives",
    "find":"finds","think":"thinks","tell":"tells","become":"becomes",
    "show":"shows","leave":"leaves","feel":"feels","put":"puts",
    "bring":"brings","begin":"begins","keep":"keeps","hold":"holds",
    "write":"writes","stand":"stands","hear":"hears","let":"lets",
    "mean":"means","set":"sets","meet":"meets","run":"runs",
    "pay":"pays","sit":"sits","speak":"speaks","lead":"leads",
    "read":"reads","grow":"grows","lose":"loses","fall":"falls",
    "send":"sends","build":"builds","stay":"stays","win":"wins",
    "buy":"buys","drive":"drives","break":"breaks","spend":"spends",
    "cut":"cuts","rise":"rises","eat":"eats","drink":"drinks",
    "sleep":"sleeps","play":"plays","walk":"walks","talk":"talks",
    "work":"works","live":"lives","love":"loves","learn":"learns",
    "help":"helps","move":"moves","open":"opens","close":"closes",
    "turn":"turns","ask":"asks","seem":"seems","need":"needs",
    "try":"tries","carry":"carries","study":"studies","fly":"flies",
    "cry":"cries","die":"dies","swim":"swims","sing":"sings",
    "dance":"dances","catch":"catches","watch":"watches","wash":"washes",
    "push":"pushes","reach":"reaches","teach":"teaches",
    "finish":"finishes","miss":"misses","pass":"passes",
    "fix":"fixes","mix":"mixes","lie":"lies",
}

CONSONANT_SOUND_H = ("hotel","historic","historical","hysterical","habitat","harbor","harvest","hazard")
CONSONANT_SOUND_VOWEL_LETTER = ("uni","eu","use","used","useful","usual","unique","university","unit","union","uniform","user","european","eucalyptus","euphoria")
VOWEL_SOUND_H = ("hour","hours","honest","honestly","honesty","honor","honour","heir","heiress","heirloom")

SKIP_WORDS = {
    "to","the","a","an","very","quite","just","also","not",
    "always","never","often","usually","sometimes","at","in",
    "on","by","for","of","with","from","up","down","out",
    "into","onto","upon","about","over","under","through",
}

HELPING_VERBS = {
    "am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would",
    "shall","should","may","might","must","can","could",
}

def is_present_participle(word):
    return word.endswith("ing") and len(word) > 4

# ── 1. SPELLING ───────────────────────────────────────────────────────────────
def correct_spelling(words):
    corrected = []
    spell_changes = []
    for word in words:
        # Strip ALL punctuation for checking
        stripped = re.sub(r'[^\w]', '', word).strip()
        stripped_lower = stripped.lower()
        if not stripped_lower or not stripped_lower.isalpha():
            corrected.append(word)
            continue
        if word[0].isupper():
            corrected.append(word)
            continue
        if len(spell.unknown([stripped_lower])) == 0:
            corrected.append(word)
        else:
            correction = spell.correction(stripped_lower)
            if correction and correction != stripped_lower:
                corrected.append(correction)
                spell_changes.append({
                    "type": "spelling",
                    "original": word,
                    "correction": correction,
                    "detail": f"'{word}' corrected to '{correction}'"
                })
            else:
                corrected.append(word)
    return corrected, spell_changes

# ── 2. HELPING VERBS ──────────────────────────────────────────────────────────
def fix_helping_verb(words):
    result = list(words)
    i = 0
    changes = []
    while i < len(result):
        word = result[i]
        word_lower = word.lower()
        if word_lower in SUBJECT_VERB_MAP:
            needed_verb = SUBJECT_VERB_MAP[word_lower]
        elif word[0].isupper() and len(word) > 1 and word.isalpha():
            needed_verb = "is"
        else:
            i += 1
            continue
        if i + 1 < len(result):
            next_word = result[i + 1].lower()
            if is_present_participle(next_word) and next_word not in HELPING_VERBS:
                result.insert(i + 1, needed_verb)
                changes.append({
                    "type": "grammar",
                    "detail": f"Added missing helping verb '{needed_verb}' after '{word}'"
                })
                i += 2
                continue
        i += 1
    return result, changes

# ── 3. TENSE ──────────────────────────────────────────────────────────────────
def fix_tense(words):
    result = list(words)
    changes = []
    text_lower = " ".join(words).lower().split()

    detected_tense = None
    for marker in PAST_MARKERS:
        if marker in text_lower:
            detected_tense = "past"
            break
    if not detected_tense:
        for marker in FUTURE_MARKERS:
            if marker in text_lower:
                detected_tense = "future"
                break

    # Present: he/she/it + base verb → verb+s/es
    if detected_tense is None:
        for i in range(len(result)):
            word_lower = result[i].lower()
            if word_lower not in THIRD_PERSON_SINGULAR:
                continue
            for j in range(i + 1, min(i + 6, len(result))):
                next_lower = result[j].lower().strip(".,!?")
                if next_lower in HELPING_VERBS:
                    break
                if next_lower in SKIP_WORDS:
                    continue
                if not next_lower.isalpha() or len(next_lower) < 2:
                    break
                if next_lower.endswith("ing") or next_lower.endswith("ed") or next_lower.endswith("s"):
                    break
                if next_lower in IRREGULAR_3RD:
                    fixed = IRREGULAR_3RD[next_lower]
                    changes.append({"type": "tense", "detail": f"'{result[j]}' → '{fixed}' (3rd person singular)"})
                    result[j] = fixed
                    break
                if next_lower.endswith(("sh","ch","x","z","o")):
                    fixed = next_lower + "es"
                elif next_lower.endswith("y") and next_lower[-2] not in "aeiou":
                    fixed = next_lower[:-1] + "ies"
                else:
                    fixed = next_lower + "s"
                changes.append({"type": "tense", "detail": f"'{result[j]}' → '{fixed}' (3rd person singular)"})
                result[j] = fixed
                break

    # Past: irregular verbs
    if detected_tense == "past":
        for i, word in enumerate(result):
            w = word.lower()
            if w in IRREGULAR_VERBS:
                past = IRREGULAR_VERBS[w]
                if past != w:
                    changes.append({"type": "tense", "detail": f"'{word}' → '{past}' (past tense)"})
                    result[i] = past

    # Future: add "will" before verb
    if detected_tense == "future":
        for i in range(len(result)):
            word_lower = result[i].lower()
            is_subject = word_lower in ALL_SUBJECTS or result[i] == "I"
            if not is_subject:
                continue
            for j in range(i + 1, min(i + 6, len(result))):
                next_lower = result[j].lower().strip(".,!?")
                if next_lower in {"will","shall","going","would"}:
                    break
                if next_lower in HELPING_VERBS:
                    break
                if next_lower in {"the","a","an","to"}:
                    continue
                if next_lower in FUTURE_MARKERS:
                    continue
                if not next_lower.isalpha() or len(next_lower) < 2:
                    break
                if next_lower.endswith("ing"):
                    break
                result.insert(j, "will")
                changes.append({"type": "tense", "detail": f"Added 'will' before '{next_lower}' (future tense)"})
                break

    return result, changes

# ── 4. ARTICLES ───────────────────────────────────────────────────────────────
def fix_articles(words):
    result = list(words)
    changes = []
    for i in range(len(result) - 1):
        word = result[i].lower()
        next_word = result[i + 1].lower().lstrip("\"'(")
        if not next_word:
            continue
        if word in ("a","an"):
            starts_with_vowel = next_word[0] in "aeiou"
            if next_word in CONSONANT_SOUND_H:
                starts_with_vowel = False
            for prefix in CONSONANT_SOUND_VOWEL_LETTER:
                if next_word.startswith(prefix):
                    starts_with_vowel = False
                    break
            for prefix in VOWEL_SOUND_H:
                if next_word.startswith(prefix):
                    starts_with_vowel = True
                    break
            correct_article = "an" if starts_with_vowel else "a"
            if word != correct_article:
                if result[i][0].isupper():
                    correct_article = correct_article.capitalize()
                changes.append({"type": "article", "detail": f"'{result[i]}' → '{correct_article}' before '{result[i+1]}'"})
                result[i] = correct_article
    return result, changes

# ── 5. PUNCTUATION ────────────────────────────────────────────────────────────
def fix_punctuation(text):
    changes = []
    text = re.sub(r' {2,}', ' ', text)
    fixed = re.sub(r'\s([?.!,;:])', r'\1', text)
    if fixed != text:
        changes.append({"type": "punctuation", "detail": "Removed extra space before punctuation"})
        text = fixed
    fixed = re.sub(r'([?.!,;:])([A-Za-z])', r'\1 \2', text)
    if fixed != text:
        changes.append({"type": "punctuation", "detail": "Added missing space after punctuation"})
        text = fixed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        changes.append({"type": "punctuation", "detail": "Capitalized first letter of sentence"})
    if text and text[-1] not in ".!?":
        text = text + "."
        changes.append({"type": "punctuation", "detail": "Added missing full stop at end"})
    fixed = re.sub(r'\b i \b', ' I ', text)
    fixed = re.sub(r'^i ', 'I ', fixed)
    fixed = re.sub(r' i$', ' I', fixed)
    if fixed != text:
        changes.append({"type": "punctuation", "detail": "Capitalized standalone 'i' → 'I'"})
        text = fixed
    fixed = re.sub(r'([.!?])\1+', r'\1', text)
    if fixed != text:
        changes.append({"type": "punctuation", "detail": "Removed duplicate punctuation"})
        text = fixed
    fixed = re.sub(r'\(\s+', '(', text)
    if fixed != text:
        changes.append({"type": "punctuation", "detail": "Removed space after opening bracket"})
        text = fixed
    return text, changes

# ── PIPELINE ──────────────────────────────────────────────────────────────────
def process_sentence(text):
    all_changes = []
    text, c = fix_punctuation(text)
    all_changes.extend(c)
    words = text.split()
    words, c = correct_spelling(words)
    all_changes.extend(c)
    words, c = fix_articles(words)
    all_changes.extend(c)
    words, c = fix_tense(words)
    all_changes.extend(c)
    words, c = fix_helping_verb(words)
    all_changes.extend(c)
    return " ".join(words), all_changes

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/correct", methods=["POST"])
def correct():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.")
    corrected_text, changes = process_sentence(text)
    changed = corrected_text != text
    return render_template("result.html",
        original_text=text, corrected_text=corrected_text,
        changes=changes, change_count=len(changes), changed=changed)

@app.route("/api/correct", methods=["POST"])
def api_correct():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    corrected_text, changes = process_sentence(text)
    return jsonify({"corrected": corrected_text, "details": changes})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=False)