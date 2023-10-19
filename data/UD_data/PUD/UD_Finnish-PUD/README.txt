# Summary

This is a part of the Parallel Universal Dependencies (PUD) treebanks created
for the [CoNLL 2017 shared task on Multilingual Parsing from Raw Text to
Universal Dependencies](http://universaldependencies.org/conll17/).


# Introduction

There are
1000 sentences in each language, always in the same order. (The sentence
alignment is 1-1 but occasionally a sentence-level segment actually consists
of two real sentences.) The sentences are taken from the news domain (sentence
id starts in ‘n’) and from Wikipedia (sentence id starts with ‘w’). There are
usually only a few sentences from each document, selected randomly, not
necessarily adjacent. The digits on the second and third position in the
sentence ids encode the original language of the sentence. The first 750
sentences are originally English (01). The remaining 250 sentences are
originally German (02), French (03), Italian (04) or Spanish (05) and they
were translated to other languages via English. Translation into German,
French, Italian, Spanish, Arabic, Hindi, Chinese, Indonesian, Japanese,
Korean, Portuguese, Russian, Thai and Turkish has been provided by DFKI and
performed (except for German) by professional translators. Then the data has
been annotated morphologically and syntactically by Google according to Google
universal annotation guidelines; finally, it has been converted by members of
the UD community to UD v2 guidelines.

Additional languages have been provided (both translation and native UD v2
annotation) by other teams: Czech by Charles University, Finnish by University
of Turku and Swedish by Uppsala University.

The entire treebank is labeled as test set (and was used for testing in the
shared task). If it is used for training in future research, the users should
employ ten-fold cross-validation.

# Changelog

* 2017-11-15 v2.1
  * First official release after it was used as a surprise dataset in the
    CoNLL 2017 shared task.
    
* CHANGELOG 2.1 -> 2.2

- No changes

* CHANGELOG 2.2 -> 2.3

- No changes

* CHANGELOG 2.3 -> 2.4

- Parataxis punctuation now follow the official UD style
- Basic dependencies are repeated in the enhanced graph
- Multiword token annotation for tokens like "ettei"
- More consistent lemma annotation
- Various validation errors fixed
- Various fixes of individual annotation errors

=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.1
License: CC BY-SA 4.0
Includes text: yes
Genre: news wiki
Lemmas: manual native
UPOS: manual native
XPOS: not available
Features: manual native
Relations: manual native
Contributors: Kanerva, Jenna; Ginter, Filip; Ojala, Stina; Missilä, Anna
Contributing: here
Contact: jmnybl@utu.fi, figint@utu.fi
===============================================================================
