# Summary

Swedish-PUD is the Swedish part of the Parallel Universal Dependencies (PUD) treebanks.

# Introduction

Swedish-PUD was created (together with the other parallel treebanks) for the CoNLL 
2017 shared task on Multilingual Parsing from Raw Text to Universal Dependencies 
(http://universaldependencies.org/conll17/). It consists of Swedish translations 
of the 1000 sentences from the news domain and from Wikipedia, annotated according
to the principles of the Swedish-PT treebank. The syntactic annotation has been 
manually validated, but the morphological annotation is automatically predicted.

# Acknowledgments

Translations were produced by Jacob Nolskog at Teknotrans AB and checked by Joakim 
Nivre. The automatic annotation was carried out using SwePipe, a tool suite trained
on the Stockholm-Umeå Corpus and the Swedish-TP treebank, developed by Robert Östling, 
Aaron Smith and Joakim, and available from https://github.com/robertostling/efselab.
The syntactic annotation was checked and corrected manually by Joakim Nivre. 

# PUD Treebanks

There are 1000 sentences in each language, always in the same order. (The sentence
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

From v2.0 to v2.1, no changes have been made.

From v2.1 to v2.2:
- Harmonization with other Swedish treebanks:
  - Possessives retagged DET -> PRON
  - Negations ("inte", "icke", "ej") retagged ADV -> PART
  - Comparative markers ("som", "än") retagged CCONJ -> SCONJ
  - Comparative with nominal complement relabeled advcl -> obl [mark -> case, SCONJ -> ADP]
  - Clefts reanalyzed as copula constructions and relabeled acl:relcl -> acl:cleft
  - Temporal subordinating conjunctions ("när", "då") retagged ADV -> SCONJ and relabeled advmod -> mark
- Added enhanced dependencies

From v2.2 to v2.3:
- Fixed a few errors in enhanced dependencies

From v2.4 to v2.5
- Corrected morphological annotation (LEMMA, UPOS, XPOS, FEATS)

From v2.10 to v2.11:
- Removed incorrect case markers in enhanced dependencies (mostly English words).

=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.1
License: CC BY-SA 4.0
Includes text: yes
Genre: news wiki
Lemmas: automatic with corrections
UPOS: automatic with corrections
XPOS: automatic with corrections
Features: automatic with corrections
Relations: manual native
Contributors: Nivre, Joakim; Griciūtė, Bernadeta
Contributing: here
Contact: joakim.nivre@lingfil.uu.se
===============================================================================



