from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
import nltk

# classifier = '/usr/local/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
# jar = '/usr/local/share/stanford-ner/stanford-ner.jar'
# stan_tagger = StanfordNERTagger(classifier, jar)

class utterance(object):
    def __init__(self, speaker = None,  addressee =None, sentence = None,):
        self.speaker = speaker
        self.sentence = sentence
        self.addressee = addressee


def interactionScore(context, participants, window):

    pairs = dict()

    l = list(participants)[:]
    l.sort()
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            pairs[(l[i], l[j])] = 0

    for cur_utterance in context:
        if cur_utterance.addressee:
            k = (cur_utterance.speaker, cur_utterance.addressee)
            if k in pairs:
                pairs[k] += 1

            elif (cur_utterance.addressee, cur_utterance.speaker) in pairs:
                pairs[(cur_utterance.addressee, cur_utterance.speaker)] += 1

    score = 0
    for key in pairs:
        score += pairs[key]

    return score


def ruleBasedVanilla(context):
    prev = None
    prev_addressee = None
    for current_utterance in context:
        if current_utterance.addressee is None and prev is not None:
            if prev.speaker == current_utterance.speaker:
                current_utterance.addressee = prev_addressee
            else:
                current_utterance.addressee = prev.speaker

            prev_addressee = current_utterance.addressee

        prev = current_utterance

    return context


def preprocessData(file_name):
    fp = open(file_name)
    data = fp.readlines()
    context = []
    participants = set()
    for item in data:
        info = item.rstrip().split('\t')
        participants.add(info[0])
        s = filter(lambda x: x.lower() in info[1].lower(), participants)
        if len(s) == 0:
            context.append(utterance(info[0], None, info[1]))
        else:
            context.append(utterance(info[0], s[0], info[1]))

    # print ('done creating sequence')
    return context, participants


def main():
    context, participants = preprocessData('chat_basic.txt')
    context = ruleBasedVanilla(context)
    score = interactionScore(context, participants, 10)
    print score/float(len(context))

if __name__ == '__main__':
    main()